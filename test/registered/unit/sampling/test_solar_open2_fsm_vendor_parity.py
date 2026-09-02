"""Vendor-parity rules of the Solar-Open2 FSM.

The vendor's reference is the UpstageAI vLLM logits processor (Solar Pro 4
parser/LP patch set for vLLM 0.25.0, 2026-09-01). Three of its rules are
checked here against ``solar_open2_fsm``:

* the reasoning budget is a fixed table keyed by the request's reasoning
  effort (low 4K / medium 16K / high 32K / xhigh 64K / max 128K, default
  high, none/minimal close the block at once, nothing above the hard limit),
  and does not depend on ``max_tokens``;
* a fresh CONTENT state may not end the turn (``content_mask`` on by default);
* the token right after ``<|think:start|>`` may not be a bare newline.

Pure CPU: small float logits tensors and duck-typed requests.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.sampling import solar_open2_fsm as fsm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

THINK_START, THINK_END, IM_END, TOOL_START = 100, 101, 102, 103
EOS = 2
NL, NLNL = 50, 51
VOCAB = 128
NEG_INF = float("-inf")


def _cfg(**overrides):
    c = fsm.CFG
    c.enabled = True
    c.think_start, c.think_end = THINK_START, THINK_END
    c.im_end, c.tool_call_start, c.tool_call_end = IM_END, TOOL_START, None
    c.all_controls = frozenset({THINK_START, THINK_END, IM_END, TOOL_START})
    c.reasoning_forbidden = (EOS, IM_END, TOOL_START)
    c.leading_newline_forbidden = (NL, NLNL)
    c.reasoning_open_forbidden = tuple(sorted({EOS, IM_END, TOOL_START, NL, NLNL}))
    c.content_fresh_forbidden = (EOS, IM_END, THINK_START, THINK_END)
    c.content_done_forbidden = (THINK_START, THINK_END)
    c.content_fresh_forbidden_notools = (EOS, IM_END, THINK_START, THINK_END, TOOL_START)
    c.content_done_forbidden_notools = (THINK_START, THINK_END, TOOL_START)
    c.budget_policy = "effort"
    c.effort_budgets = dict(fsm._EFFORT_BUDGETS)
    c.default_effort = "high"
    c.hard_limit = fsm._HARD_LIMIT
    c.budget_abs, c.budget_ratio = 3072, 0.75
    c.content_mask = True
    c.spec_always_eager = False
    c._mask_cache.clear()
    for name, value in overrides.items():
        setattr(c, name, value)
    fsm._EFFORT_LOG["last"] = 0.0
    fsm._EFFORT_LOG["num_suppressed"] = 0


def _req(
    output_ids=(),
    *,
    prompt=(1, 2, 3, THINK_START),
    max_new_tokens=4096,
    effort=None,
    tools=None,
    rid="r0",
):
    custom = {}
    if effort is not None:
        custom[fsm.EFFORT_PARAM] = effort
    if tools is not None:
        custom[fsm.TOOLS_PARAM] = tools
    custom = custom or None
    return SimpleNamespace(
        rid=rid,
        retraction_count=0,
        origin_input_ids=list(prompt),
        output_ids=list(output_ids),
        sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens, custom_params=custom),
    )


def _apply(*reqs):
    logits = torch.zeros(len(reqs), VOCAB)
    fsm.apply(logits, SimpleNamespace(solar_fsm_rows=list(reqs)))
    return logits


def _masked(logits, row=0):
    return {i for i in range(VOCAB) if logits[row, i] == NEG_INF}


class TestEffortBudget(unittest.TestCase):
    def setUp(self):
        _cfg()

    def test_budget_follows_the_vendor_table(self):
        for effort, expected in fsm._EFFORT_BUDGETS.items():
            with self.subTest(effort=effort):
                self.assertEqual(fsm._req_fsm(_req(effort=effort)).budget, expected)

    def test_no_effort_means_high(self):
        self.assertEqual(fsm._req_fsm(_req()).budget, 32 * 1024)

    def test_effort_is_case_and_space_insensitive(self):
        self.assertEqual(fsm._req_fsm(_req(effort=" Medium ")).budget, 16 * 1024)

    def test_budget_ignores_max_new_tokens(self):
        self.assertEqual(fsm._req_fsm(_req(effort="low", max_new_tokens=100)).budget, 4 * 1024)
        self.assertEqual(fsm._req_fsm(_req(effort="max", max_new_tokens=None)).budget, 128 * 1024)

    def test_none_and_minimal_close_the_block_at_once(self):
        for effort in ("none", "minimal"):
            with self.subTest(effort=effort):
                req = _req(effort=effort)
                self.assertEqual(fsm._req_fsm(req).budget, 0)
                logits = _apply(req)
                # Forced: only <|think:end|> survives on the very first step.
                self.assertEqual(set(range(VOCAB)) - _masked(logits), {THINK_END})

    def test_hard_limit_caps_every_effort(self):
        _cfg(hard_limit=1000)
        self.assertEqual(fsm._req_fsm(_req(effort="max")).budget, 1000)
        self.assertEqual(fsm._req_fsm(_req(effort="low")).budget, 1000)

    def test_unknown_effort_uses_the_default_and_warns_rate_limited(self):
        with self.assertLogs(fsm.logger, level="WARNING") as captured:
            self.assertEqual(fsm._req_fsm(_req(effort="ultra")).budget, 32 * 1024)
        self.assertIn("unknown reasoning effort", captured.output[0])
        # Inside the interval: counted, not logged.
        with self.assertNoLogs(fsm.logger, level="WARNING"):
            fsm._req_fsm(_req(effort="ultra", rid="r1"))
        self.assertEqual(fsm._EFFORT_LOG["num_suppressed"], 1)
        # After the interval: logged again, with the suppressed count.
        fsm._EFFORT_LOG["last"] = 0.0
        with self.assertLogs(fsm.logger, level="WARNING") as captured:
            fsm._req_fsm(_req(effort="ultra", rid="r2"))
        self.assertIn("1 earlier occurrence(s) suppressed", captured.output[0])

    def test_legacy_policy_keeps_the_old_formula(self):
        _cfg(budget_policy="legacy")
        self.assertEqual(fsm._req_fsm(_req(effort="low", max_new_tokens=1000)).budget, 750)
        self.assertEqual(fsm._req_fsm(_req(max_new_tokens=None)).budget, 3072)

    def test_non_string_effort_is_unknown(self):
        """A value the entrypoint would never write (an int) is a client
        misuse: default budget, and the same warning as an unknown string."""
        req = _req()
        req.sampling_params.custom_params = {fsm.EFFORT_PARAM: 3}
        with self.assertLogs(fsm.logger, level="WARNING") as captured:
            self.assertEqual(fsm._req_fsm(req).budget, 32 * 1024)
        self.assertIn("unknown reasoning effort 3", captured.output[0])

    def test_rebuild_after_retraction_keeps_the_effort_budget(self):
        req = _req(effort="low")
        first = fsm._req_fsm(req)
        req.retraction_count = 1
        rebuilt = fsm._req_fsm(req)
        self.assertIsNot(rebuilt, first)
        self.assertEqual(rebuilt.budget, 4 * 1024)
        self.assertTrue(rebuilt.at_think_open)


class TestLeadingNewline(unittest.TestCase):
    def setUp(self):
        _cfg()

    def test_first_token_after_the_prompt_think_start_cannot_be_a_newline(self):
        self.assertEqual(_masked(_apply(_req())), {EOS, IM_END, TOOL_START, NL, NLNL})

    def test_rule_lifts_after_one_token(self):
        self.assertEqual(_masked(_apply(_req([7]))), {EOS, IM_END, TOOL_START})

    def test_prompt_inside_reasoning_but_not_at_its_start_has_no_rule(self):
        req = _req(prompt=(1, THINK_START, 5))
        self.assertEqual(_masked(_apply(req)), {EOS, IM_END, TOOL_START})

    def test_a_generated_think_start_reopens_the_rule(self):
        opened = _req([7, THINK_START], prompt=(1, 2, 3))
        self.assertIn(NL, _masked(_apply(opened)))
        moved_on = _req([7, THINK_START, 8], prompt=(1, 2, 3))
        self.assertNotIn(NL, _masked(_apply(moved_on)))

    def test_verify_plan_applies_the_rule_along_the_draft_chain(self):
        req = _req([7], prompt=(1, 2, 3))  # committed: CONTENT with content
        fsm._req_fsm(req)
        chain = torch.tensor([[7, THINK_START, 9]])  # anchor, draft_1, draft_2
        plan = fsm.plan_verify([req], chain, stride=3)
        # row 0: committed CONTENT (has content) ; row 1: after the drafted
        # <|think:start|> -> leading-newline set ; row 2: after one reasoning
        # token -> plain set.
        self.assertEqual(plan.mask_rows[fsm.CFG.content_done_forbidden], [0])
        self.assertEqual(plan.mask_rows[fsm.CFG.reasoning_open_forbidden], [1])
        self.assertEqual(plan.mask_rows[fsm.CFG.reasoning_forbidden], [2])
        self.assertEqual(plan.force_rows, [])

    def test_verify_plan_writes_the_open_set(self):
        req = _req([7], prompt=(1, 2, 3))
        fsm._req_fsm(req)
        plan = fsm.plan_verify([req], torch.tensor([[7, THINK_START, 9]]), stride=3)
        logits = torch.zeros(3, VOCAB)
        plan.apply(logits)
        self.assertEqual(logits[1, NL].item(), NEG_INF)
        self.assertEqual(logits[1, NLNL].item(), NEG_INF)
        self.assertEqual(logits[2, NL].item(), 0.0)
        self.assertEqual(logits[2, EOS].item(), NEG_INF)
        # Rows past verify_lens are padding and stay untouched.
        logits = torch.zeros(3, VOCAB)
        plan.apply(logits, verify_lens=torch.tensor([2]))
        self.assertEqual(logits[1, NL].item(), NEG_INF)
        self.assertTrue(torch.isfinite(logits[2]).all())

    def test_commit_run_ending_in_think_start_arms_the_rule(self):
        """The speculative commit path, not only the prompt walk."""
        req = _req([], prompt=(1, 2, 3))
        state = fsm._req_fsm(req)
        state.commit([7, THINK_START])
        self.assertTrue(state.at_think_open)
        req.output_ids.extend([7, THINK_START])
        state.advance(req.output_ids)  # already consumed: a no-op
        self.assertTrue(state.at_think_open)
        state.commit([8])
        self.assertFalse(state.at_think_open)

    def test_sim_state_copies_the_open_flag(self):
        state = fsm._req_fsm(_req())
        self.assertTrue(state.at_think_open)
        self.assertTrue(fsm._SimState(state).at_think_open)

    def test_pre_closed_prompt_starts_in_fresh_content(self):
        """What the served template renders for none/minimal/low: the block
        is already closed, so the first token is fresh CONTENT."""
        req = _req([], prompt=(1, 2, 3, THINK_START, THINK_END), effort="none")
        state = fsm._req_fsm(req)
        self.assertFalse(state.in_reasoning)
        self.assertFalse(state.at_think_open)
        self.assertEqual(_masked(_apply(req)), set(fsm.CFG.content_fresh_forbidden))

    def test_each_batch_row_gets_its_own_set(self):
        _cfg(effort_budgets={**fsm._EFFORT_BUDGETS, "low": 2})
        rows = [
            _req([], rid="open"),
            _req([7], rid="mid"),
            _req([7, THINK_END], rid="fresh"),
            _req([7, 8], effort="low", rid="forced"),
        ]
        logits = _apply(*rows)
        self.assertEqual(_masked(logits, 0), set(fsm.CFG.reasoning_open_forbidden))
        self.assertEqual(_masked(logits, 1), set(fsm.CFG.reasoning_forbidden))
        self.assertEqual(_masked(logits, 2), set(fsm.CFG.content_fresh_forbidden))
        self.assertEqual(set(range(VOCAB)) - _masked(logits, 3), {THINK_END})

    def test_disabled_rule_leaves_the_plain_set(self):
        _cfg(leading_newline_forbidden=(), reasoning_open_forbidden=(EOS, IM_END, TOOL_START))
        self.assertEqual(_masked(_apply(_req())), {EOS, IM_END, TOOL_START})


class TestContentMask(unittest.TestCase):
    def setUp(self):
        _cfg()

    def test_fresh_content_cannot_end_the_turn(self):
        req = _req([7, THINK_END])
        self.assertEqual(_masked(_apply(req)), set(fsm.CFG.content_fresh_forbidden))

    def test_content_with_progress_may_end_the_turn(self):
        req = _req([7, THINK_END, 8])
        masked = _masked(_apply(req))
        self.assertEqual(masked, set(fsm.CFG.content_done_forbidden))
        self.assertNotIn(EOS, masked)

    def test_forced_close_is_followed_by_the_fresh_content_mask(self):
        _cfg(effort_budgets={**fsm._EFFORT_BUDGETS, "low": 2})
        req = _req([7, 8], effort="low")
        logits = _apply(req)
        self.assertEqual(set(range(VOCAB)) - _masked(logits), {THINK_END})
        req.output_ids.append(THINK_END)
        self.assertIn(EOS, _masked(_apply(req)))

    def test_no_tools_forbids_a_tool_call_in_content(self):
        """With EOS shut in fresh CONTENT a model that wanted to stop takes
        <|tool_call:start|> as the exit; a request without tools loses it."""
        fresh = _req([7, THINK_END], tools=False)
        self.assertEqual(_masked(_apply(fresh)), set(fsm.CFG.content_fresh_forbidden_notools))
        done = _req([7, THINK_END, 8], tools=False)
        self.assertEqual(_masked(_apply(done)), set(fsm.CFG.content_done_forbidden_notools))
        self.assertIn(TOOL_START, _masked(_apply(done)))

    def test_tools_present_or_unstated_keep_the_vendor_sets(self):
        for tools in (True, None):
            with self.subTest(tools=tools):
                req = _req([7, THINK_END], tools=tools)
                masked = _masked(_apply(req))
                self.assertEqual(masked, set(fsm.CFG.content_fresh_forbidden))
                self.assertNotIn(TOOL_START, masked)

    def test_no_tools_applies_along_the_verify_chain(self):
        req = _req([7], tools=False)
        fsm._req_fsm(req)
        plan = fsm.plan_verify([req], torch.tensor([[7, THINK_END, 9]]), stride=3)
        self.assertEqual(plan.mask_rows[fsm.CFG.reasoning_forbidden], [0])
        self.assertEqual(plan.mask_rows[fsm.CFG.content_fresh_forbidden_notools], [1])
        self.assertEqual(plan.mask_rows[fsm.CFG.content_done_forbidden_notools], [2])

    def test_switch_off_restores_the_unmasked_content(self):
        _cfg(content_mask=False)
        self.assertEqual(_masked(_apply(_req([7, THINK_END]))), set())


class TestPlanGate(unittest.TestCase):
    """The fold escape stays row-conditional with content_mask on."""

    STRIDE = 4

    def setUp(self):
        _cfg()

    def _gate(self, req):
        fsm._req_fsm(req)
        return fsm.plan_gate([req], self.STRIDE)

    def test_reasoning_far_from_budget_keeps_the_folded_path(self):
        self.assertFalse(self._gate(_req([7] * 10, effort="high")))

    def test_right_after_think_start_goes_eager(self):
        self.assertTrue(self._gate(_req()))

    def test_fresh_content_goes_eager_only_with_content_mask(self):
        self.assertTrue(self._gate(_req([7, THINK_END])))
        _cfg(content_mask=False)
        self.assertFalse(self._gate(_req([7, THINK_END])))

    def test_content_with_progress_keeps_the_folded_path(self):
        self.assertFalse(self._gate(_req([7, THINK_END, 8])))

    def test_near_budget_and_zero_budget_go_eager(self):
        _cfg(effort_budgets={**fsm._EFFORT_BUDGETS, "low": 12})
        self.assertTrue(self._gate(_req([7] * 5, effort="low")))
        self.assertTrue(self._gate(_req([7] * 5, effort="none")))

    def test_zero_budget_does_not_pin_a_closed_block_eager(self):
        """A none/minimal request whose block is closed (pre-closed by the
        template, or forced) must not keep the batch off the folded path."""
        pre_closed = _req([7, 8], prompt=(1, 2, 3, THINK_START, THINK_END), effort="none")
        self.assertFalse(self._gate(pre_closed))
        forced_then_content = _req([THINK_END, 7], effort="minimal")
        self.assertFalse(self._gate(forced_then_content))


class TestInitFromEnv(unittest.TestCase):
    """The env surface: defaults and overrides as read by init_from_env."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        files = {
            "tokenizer_config.json": {
                "added_tokens_decoder": {
                    str(THINK_START): {"content": "<|think:start|>"},
                    str(THINK_END): {"content": "<|think:end|>"},
                    str(IM_END): {"content": "<|im:end|>"},
                    str(TOOL_START): {"content": "<|tool_call:start|>"},
                }
            },
            "generation_config.json": {"eos_token_id": [EOS]},
            "tokenizer.json": {"model": {"vocab": {"Ċ": NL, "ĊĊ": NLNL, "a": 9}}},
        }
        for name, body in files.items():
            with open(os.path.join(self.dir, name), "w", encoding="utf-8") as f:
                json.dump(body, f)
        self.base_env = {"SOLAR_FSM": "1", "SOLAR_FSM_TOKENIZER_DIR": self.dir}
        self.clear = [
            k for k in os.environ if k.startswith("SOLAR_FSM_") and k != "SOLAR_FSM_TOKENIZER_DIR"
        ]

    def _init(self, **env):
        with mock.patch.dict(os.environ, {**self.base_env, **env}):
            for k in self.clear:
                os.environ.pop(k, None)
            for k in list(os.environ):
                if k.startswith("SOLAR_FSM_") and k not in env and k not in self.base_env:
                    os.environ.pop(k)
            fsm.CFG.enabled = False
            fsm.init_from_env()

    def tearDown(self):
        _cfg()

    def test_defaults_are_the_vendor_rules(self):
        self._init()
        c = fsm.CFG
        self.assertTrue(c.content_mask)
        self.assertEqual(c.budget_policy, "effort")
        self.assertEqual(c.effort_budgets, fsm._EFFORT_BUDGETS)
        self.assertEqual(c.default_effort, "high")
        self.assertEqual(c.hard_limit, 128 * 1024)
        self.assertEqual(c.leading_newline_forbidden, (NL, NLNL))
        self.assertEqual(
            set(c.reasoning_open_forbidden), set(c.reasoning_forbidden) | {NL, NLNL}
        )
        self.assertIn(EOS, c.reasoning_forbidden)

    def test_content_sets_are_built_from_the_vocab(self):
        """The vendor sets allow a tool call; the no-tools variants forbid it.
        EOS is shut only while the turn has no content."""
        self._init()
        c = fsm.CFG
        self.assertNotIn(TOOL_START, c.content_fresh_forbidden)
        self.assertNotIn(TOOL_START, c.content_done_forbidden)
        self.assertIn(TOOL_START, c.content_fresh_forbidden_notools)
        self.assertIn(TOOL_START, c.content_done_forbidden_notools)
        for name in ("content_fresh_forbidden", "content_fresh_forbidden_notools"):
            self.assertIn(EOS, getattr(c, name))
            self.assertIn(IM_END, getattr(c, name))
        for name in ("content_done_forbidden", "content_done_forbidden_notools"):
            self.assertNotIn(EOS, getattr(c, name))
            self.assertNotIn(IM_END, getattr(c, name))
        self.assertEqual(
            set(c.content_done_forbidden_notools), set(c.content_done_forbidden) | {TOOL_START}
        )

    def test_per_effort_override_and_hard_limit(self):
        self._init(SOLAR_FSM_BUDGET_MEDIUM="777", SOLAR_FSM_HARD_LIMIT="50000")
        self.assertEqual(fsm.CFG.effort_budgets["medium"], 777)
        self.assertEqual(fsm.CFG.effort_budgets["max"], 50000)
        self.assertEqual(fsm.CFG.effort_budgets["high"], 32 * 1024)

    def test_content_mask_and_newline_rule_can_be_switched_off(self):
        self._init(SOLAR_FSM_CONTENT_MASK="0", SOLAR_FSM_LEADING_NEWLINE_IDS="")
        self.assertFalse(fsm.CFG.content_mask)
        self.assertEqual(fsm.CFG.leading_newline_forbidden, ())
        self.assertEqual(fsm.CFG.reasoning_open_forbidden, fsm.CFG.reasoning_forbidden)

    def test_missing_newline_tokens_fail_loud(self):
        with open(os.path.join(self.dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump({"model": {"vocab": {"a": 9}}}, f)
        with self.assertRaisesRegex(RuntimeError, "leading-newline tokens"):
            self._init()
        # A Unigram-style layout (vocab as a list) counts as not found too.
        with open(os.path.join(self.dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump({"model": {"vocab": [["Ċ", -1.0]]}}, f)
        with self.assertRaisesRegex(RuntimeError, "leading-newline tokens"):
            self._init()

    def test_bad_values_fail_loud_by_name(self):
        with self.assertRaisesRegex(RuntimeError, "SOLAR_FSM_HARD_LIMIT"):
            self._init(SOLAR_FSM_HARD_LIMIT="abc")
        with self.assertRaisesRegex(RuntimeError, "SOLAR_FSM_HARD_LIMIT must be positive"):
            self._init(SOLAR_FSM_HARD_LIMIT="0")
        with self.assertRaisesRegex(RuntimeError, "SOLAR_FSM_BUDGET_LOW must be >= 0"):
            self._init(SOLAR_FSM_BUDGET_LOW="-1")
        with self.assertRaisesRegex(RuntimeError, "SOLAR_FSM_LEADING_NEWLINE_IDS"):
            self._init(SOLAR_FSM_LEADING_NEWLINE_IDS="7,x")
        with self.assertRaisesRegex(RuntimeError, "negative id"):
            self._init(SOLAR_FSM_LEADING_NEWLINE_IDS="-7")

    def test_explicit_override_above_the_hard_limit_warns_and_clamps(self):
        with self.assertLogs(fsm.logger, level="WARNING") as captured:
            self._init(SOLAR_FSM_BUDGET_MAX="200000", SOLAR_FSM_HARD_LIMIT="131072")
        self.assertIn("SOLAR_FSM_BUDGET_MAX=200000 exceeds", captured.output[0])
        self.assertEqual(fsm.CFG.effort_budgets["max"], 131072)
        # A table default clamped by a lower hard limit is silent.
        with self.assertNoLogs(fsm.logger, level="WARNING"):
            self._init(SOLAR_FSM_HARD_LIMIT="50000")
        self.assertEqual(fsm.CFG.effort_budgets["max"], 50000)

    def test_explicit_newline_ids_override_the_vocab(self):
        self._init(SOLAR_FSM_LEADING_NEWLINE_IDS="7, 9")
        self.assertEqual(fsm.CFG.leading_newline_forbidden, (7, 9))

    def test_bad_policy_or_default_effort_fails_loud(self):
        with self.assertRaises(RuntimeError):
            self._init(SOLAR_FSM_BUDGET_POLICY="ratio")
        with self.assertRaises(RuntimeError):
            self._init(SOLAR_FSM_DEFAULT_EFFORT="ultra")

    def test_none_as_default_effort_is_allowed(self):
        self._init(SOLAR_FSM_DEFAULT_EFFORT="none")
        self.assertEqual(fsm._budget_for(None, 4096), 0)


if __name__ == "__main__":
    unittest.main()
