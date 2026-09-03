"""Vendor-parity rules of the Solar-Open2 FSM.

The vendor's reference is the UpstageAI vLLM logits processor (Solar Pro 4
parser/LP patch set for vLLM 0.25.0, 2026-09-01). The rules checked here
against ``solar_open2_fsm``:

* the forbidden tables per state (all sentinels minus the state's allowed
  set; EOS masked everywhere except CONTENT-with-progress and TOOL_CALL_END;
  sentinels never counted as EOS);
* the tool-call envelope walk -- transitions and auto-advance -- checked
  differentially against a transcript of the vendor's ``_process_token``;
* budget accounting (REASONING tokens only, reset at ``<|think:start|>``)
  and the env semantics of the budget/hard-limit overrides;
* the reasoning budget is a fixed table keyed by the request's reasoning
  effort (low 4K / medium 16K / high 32K / xhigh 64K / max 128K, default
  high, none/minimal close the block at once, nothing above the hard limit),
  and does not depend on ``max_tokens``;
* a fresh CONTENT state may not end the turn;
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
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

THINK_START, THINK_END, IM_END, TOOL_START = 100, 101, 102, 103
TOOL_END, ARG_START, ARG_VALUE, ARG_END, IM_START = 104, 105, 106, 107, 108
TOOL_ENVELOPE = (TOOL_END, ARG_START, ARG_VALUE, ARG_END)
EOS = 2
NL, NLNL = 50, 51
VOCAB = 128
NEG_INF = float("-inf")


IDS = {
    "im_start": IM_START,
    "im_end": IM_END,
    "think_start": THINK_START,
    "think_end": THINK_END,
    "tool_call_start": TOOL_START,
    "tool_call_end": TOOL_END,
    "tool_arg_start": ARG_START,
    "tool_arg_value": ARG_VALUE,
    "tool_arg_end": ARG_END,
}


def _cfg(**overrides):
    c = fsm.CFG
    c.enabled = True
    # The same builder the server uses: every set is derived from the vendor's
    # spec, so a test can only pass against what init_from_env would produce.
    fsm.configure_ids(IDS, eos=[EOS], leading_newline=[NL, NLNL])
    c.effort_budgets = dict(fsm._EFFORT_BUDGETS)
    c.default_effort = "high"
    c.hard_limit = fsm._HARD_LIMIT
    c.no_reasoning_efforts = fsm._NO_REASONING_EFFORTS
    c.spec_always_eager = False
    c._mask_cache.clear()
    for name, value in overrides.items():
        setattr(c, name, value)
    fsm._EFFORT_LOG["last"] = -fsm._LOG_INTERVAL
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
        sampling_params=SimpleNamespace(
            max_new_tokens=max_new_tokens,
            custom_params=custom,
            json_schema=None,
            regex=None,
            ebnf=None,
            structural_tag=None,
        ),
    )


def _apply(*reqs):
    logits = torch.zeros(len(reqs), VOCAB)
    fsm.apply(logits, SimpleNamespace(solar_fsm_rows=list(reqs)))
    return logits


def _masked(logits, row=0):
    return {i for i in range(VOCAB) if logits[row, i] == NEG_INF}


_CFG_FIELDS = (
    "enabled",
    "think_start",
    "think_end",
    "im_end",
    "im_start",
    "im_content",
    "tool_start",
    "tool_end",
    "tool_call_start",
    "tool_call_end",
    "tool_arg_start",
    "tool_arg_value",
    "tool_arg_end",
    "tool_response_start",
    "tool_response_end",
    "all_controls",
    "transitions",
    "forbidden",
    "forbidden_notools",
    "reasoning_forbidden",
    "leading_newline_forbidden",
    "reasoning_open_forbidden",
    "content_fresh_forbidden",
    "content_done_forbidden",
    "content_fresh_forbidden_notools",
    "content_done_forbidden_notools",
    "effort_budgets",
    "default_effort",
    "hard_limit",
    "no_reasoning_efforts",
    "spec_always_eager",
)


class _FsmCase(CustomTestCase):
    """Configures the module-global CFG for a test and restores it after, so
    this file leaves no live FSM (with fake ids) behind for the other sampler
    suites in the same process -- the same pattern as the sibling files."""

    def setUp(self):
        self._saved = {k: getattr(fsm.CFG, k) for k in _CFG_FIELDS}
        self._saved_warned = dict(fsm._WARNED)
        _cfg()

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(fsm.CFG, k, v)
        fsm._WARNED.clear()
        fsm._WARNED.update(self._saved_warned)
        fsm.CFG._mask_cache.clear()


class TestEffortBudget(_FsmCase):

    def test_budget_follows_the_vendor_table(self):
        for effort, expected in fsm._EFFORT_BUDGETS.items():
            with self.subTest(effort=effort):
                self.assertEqual(fsm._req_fsm(_req(effort=effort)).budget, expected)

    def test_no_effort_means_high(self):
        self.assertEqual(fsm._req_fsm(_req()).budget, 32 * 1024)

    def test_effort_is_case_and_space_insensitive(self):
        self.assertEqual(fsm._req_fsm(_req(effort=" Medium ")).budget, 16 * 1024)

    def test_budget_ignores_max_new_tokens(self):
        self.assertEqual(
            fsm._req_fsm(_req(effort="low", max_new_tokens=100)).budget, 4 * 1024
        )
        self.assertEqual(
            fsm._req_fsm(_req(effort="max", max_new_tokens=None)).budget, 128 * 1024
        )

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
        fsm._EFFORT_LOG["last"] = -fsm._LOG_INTERVAL
        with self.assertLogs(fsm.logger, level="WARNING") as captured:
            fsm._req_fsm(_req(effort="ultra", rid="r2"))
        self.assertIn("1 earlier occurrence(s) suppressed", captured.output[0])

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


class TestLeadingNewline(_FsmCase):

    def test_first_token_after_the_prompt_think_start_cannot_be_a_newline(self):
        self.assertEqual(_masked(_apply(_req())), set(fsm.CFG.reasoning_open_forbidden))

    def test_rule_lifts_after_one_token(self):
        self.assertEqual(_masked(_apply(_req([7]))), set(fsm.CFG.reasoning_forbidden))

    def test_prompt_inside_reasoning_but_not_at_its_start_has_no_rule(self):
        req = _req(prompt=(1, THINK_START, 5))
        self.assertEqual(_masked(_apply(req)), set(fsm.CFG.reasoning_forbidden))

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
        _cfg(
            leading_newline_forbidden=(),
            reasoning_open_forbidden=(EOS, IM_END, TOOL_START),
        )
        self.assertEqual(_masked(_apply(_req())), {EOS, IM_END, TOOL_START})


class TestContentMask(_FsmCase):

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
        self.assertEqual(
            _masked(_apply(fresh)), set(fsm.CFG.content_fresh_forbidden_notools)
        )
        done = _req([7, THINK_END, 8], tools=False)
        self.assertEqual(
            _masked(_apply(done)), set(fsm.CFG.content_done_forbidden_notools)
        )
        self.assertIn(TOOL_START, _masked(_apply(done)))

    def test_non_bool_tools_value_is_permissive_and_loud_once(self):
        req = _req([7, THINK_END])
        req.sampling_params.custom_params = {fsm.TOOLS_PARAM: 0}
        with self.assertLogs(fsm.logger, level="WARNING") as captured:
            masked = _masked(_apply(req))
        self.assertIn("not a bool", captured.output[0])
        self.assertNotIn(TOOL_START, masked)

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


class TestPlanGate(_FsmCase):
    """The fold escape stays row-conditional."""

    STRIDE = 4

    def _gate(self, req):
        fsm._req_fsm(req)
        return fsm.plan_gate([req], self.STRIDE)

    def test_reasoning_far_from_budget_keeps_the_folded_path(self):
        self.assertFalse(self._gate(_req([7] * 10, effort="high")))

    def test_right_after_think_start_goes_eager(self):
        self.assertTrue(self._gate(_req()))

    def test_fresh_content_goes_eager(self):
        self.assertTrue(self._gate(_req([7, THINK_END])))

    def test_fresh_content_under_a_grammar_folds_and_nothing_masks_it(self):
        """The third kind of row that reaches the folded path, and the only one
        with no mask at all. `_content_needs_eager` exempts a CONTENT row under
        a grammar -- the grammar owns CONTENT -- so plan_gate lets the step
        fold, and none of the three flag functions arm it either.

        What keeps that from mattering is in the worker: fold_eligible requires
        `not batch.has_grammar`. That reads `req.grammar` while this reads the
        sampling params, so the two can drift apart, and the day they do this
        row folds unmasked. Pinned here so the drift is a test failure rather
        than a control token in someone's answer.
        """
        req = _req([7, THINK_END])
        req.sampling_params.json_schema = '{"type": "object"}'
        self.assertFalse(self._gate(req), "a grammar row is not sent eager")
        for flags in (
            fsm.folded_mask_flags([req], 4),
            fsm.folded_content_mask_flags([req], 4),
            fsm.folded_content_notools_mask_flags([req], 4),
        ):
            self.assertEqual(flags, [False] * 4, "no in-graph mask arms it")

    def test_content_with_progress_keeps_the_folded_path(self):
        self.assertFalse(self._gate(_req([7, THINK_END, 8])))

    def test_near_budget_and_zero_budget_go_eager(self):
        _cfg(effort_budgets={**fsm._EFFORT_BUDGETS, "low": 12})
        self.assertTrue(self._gate(_req([7] * 5, effort="low")))
        self.assertTrue(self._gate(_req([7] * 5, effort="none")))

    def test_zero_budget_does_not_pin_a_closed_block_eager(self):
        """A none/minimal request whose block is closed (pre-closed by the
        template, or forced) must not keep the batch off the folded path."""
        pre_closed = _req(
            [7, 8], prompt=(1, 2, 3, THINK_START, THINK_END), effort="none"
        )
        self.assertFalse(self._gate(pre_closed))
        forced_then_content = _req([THINK_END, 7], effort="minimal")
        self.assertFalse(self._gate(forced_then_content))


class TestInitFromEnv(_FsmCase):
    """The env surface: defaults and overrides as read by init_from_env."""

    def setUp(self):
        super().setUp()
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
                    str(TOOL_END): {"content": "<|tool_call:end|>"},
                    str(ARG_START): {"content": "<|tool_arg:start|>"},
                    str(ARG_VALUE): {"content": "<|tool_arg:value|>"},
                    str(ARG_END): {"content": "<|tool_arg:end|>"},
                    str(IM_START): {"content": "<|im:start|>"},
                    "110": {"content": "<|im:content|>"},
                    "111": {"content": "<|tool:start|>"},
                    "112": {"content": "<|tool:end|>"},
                    "113": {"content": "<|tool_response:start|>"},
                    "114": {"content": "<|tool_response:end|>"},
                    "115": {"content": "ĊĊĊ"},
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
            k
            for k in os.environ
            if k.startswith(("SOLAR_FSM_", "SOLAR_REASONING_BUDGET_", "SOLAR_OPEN2_"))
            and k != "SOLAR_FSM_TOKENIZER_DIR"
        ]

    def _init(self, **env):
        with mock.patch.dict(os.environ, {**self.base_env, **env}):
            for k in self.clear:
                os.environ.pop(k, None)
            for k in list(os.environ):
                if (
                    k.startswith(
                        ("SOLAR_FSM_", "SOLAR_REASONING_BUDGET_", "SOLAR_OPEN2_")
                    )
                    and k not in env
                    and k not in self.base_env
                ):
                    os.environ.pop(k)
            fsm.CFG.enabled = False
            fsm.init_from_env()

    def test_defaults_are_the_vendor_rules(self):
        self._init()
        c = fsm.CFG
        self.assertEqual(c.effort_budgets, fsm._EFFORT_BUDGETS)
        self.assertEqual(c.default_effort, "high")
        self.assertEqual(c.hard_limit, 128 * 1024)
        self.assertEqual(c.leading_newline_forbidden, (NL, NLNL, 115))
        self.assertEqual(
            set(c.reasoning_open_forbidden),
            set(c.reasoning_forbidden) | {NL, NLNL, 115},
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
            set(c.content_done_forbidden_notools),
            set(c.content_done_forbidden) | {TOOL_START},
        )

    def test_eos_ids_override_is_a_json_list(self):
        self._init(SOLAR_OPEN2_EOS_TOKEN_IDS="[3, 5]")
        self.assertEqual(sorted(fsm.CFG.reasoning_forbidden[:2]), [3, 5][:2])
        self.assertIn(3, fsm.CFG.reasoning_forbidden)
        self.assertIn(5, fsm.CFG.reasoning_forbidden)
        with self.assertRaisesRegex(RuntimeError, "SOLAR_OPEN2_EOS_TOKEN_IDS"):
            self._init(SOLAR_OPEN2_EOS_TOKEN_IDS="3,5")

    def test_per_effort_override_and_hard_limit(self):
        self._init(
            SOLAR_REASONING_BUDGET_MEDIUM="777",
            SOLAR_REASONING_BUDGET_HARD_LIMIT="50000",
        )
        self.assertEqual(fsm.CFG.effort_budgets["medium"], 777)
        self.assertEqual(fsm.CFG.effort_budgets["max"], 50000)
        self.assertEqual(fsm.CFG.effort_budgets["high"], 32 * 1024)

    def test_newline_rule_can_be_switched_off(self):
        self._init(SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS="[]")
        self.assertEqual(fsm.CFG.leading_newline_forbidden, ())
        self.assertEqual(fsm.CFG.reasoning_open_forbidden, fsm.CFG.reasoning_forbidden)

    def test_blank_newline_ids_env_is_unset(self):
        self._init()
        from_tokenizer = fsm.CFG.leading_newline_forbidden
        self.assertTrue(from_tokenizer)
        self._init(SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS="")
        self.assertEqual(fsm.CFG.leading_newline_forbidden, from_tokenizer)

    def test_no_bare_eos_warns(self):
        with open(
            os.path.join(self.dir, "generation_config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump({}, f)
        with self.assertLogs("sglang.srt.sampling.solar_open2_fsm", "WARNING") as logs:
            self._init()
        self.assertTrue(any("no EOS id found" in line for line in logs.output))
        # Only the sentinels are masked: the bare EOS is not in any set.
        self.assertNotIn(EOS, fsm.CFG.reasoning_forbidden)

    def test_missing_newline_tokens_fail_loud(self):
        # Neither the vocab nor the added tokens may carry a newline run.
        with open(
            os.path.join(self.dir, "tokenizer_config.json"), encoding="utf-8"
        ) as f:
            cfg = json.load(f)
        cfg["added_tokens_decoder"].pop("115")
        with open(
            os.path.join(self.dir, "tokenizer_config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(cfg, f)
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
        with self.assertRaisesRegex(RuntimeError, "SOLAR_REASONING_BUDGET_HARD_LIMIT"):
            self._init(SOLAR_REASONING_BUDGET_HARD_LIMIT="abc")
        with self.assertRaisesRegex(
            RuntimeError, "SOLAR_REASONING_BUDGET_HARD_LIMIT must be >= 0"
        ):
            self._init(SOLAR_REASONING_BUDGET_HARD_LIMIT="-1")
        # Vendor: 0 disables the server-wide ceiling ...
        self._init(
            SOLAR_REASONING_BUDGET_HARD_LIMIT="0", SOLAR_REASONING_BUDGET_MAX="200000"
        )
        self.assertEqual(fsm._req_fsm(_req(effort="max")).budget, 200000)
        # ... and a negative per-effort budget warns and keeps the default.
        with self.assertLogs(fsm.logger, level="WARNING") as captured:
            self._init(SOLAR_REASONING_BUDGET_LOW="-1")
        self.assertIn("SOLAR_REASONING_BUDGET_LOW", captured.output[0])
        self.assertEqual(fsm.CFG.effort_budgets["low"], 4 * 1024)
        with self.assertRaisesRegex(
            RuntimeError, "SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS"
        ):
            self._init(SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS='[7, "x"]')
        with self.assertRaisesRegex(RuntimeError, "non-negative ints"):
            self._init(SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS="[-7]")

    def test_explicit_override_above_the_hard_limit_warns_and_clamps(self):
        with self.assertLogs(fsm.logger, level="WARNING") as captured:
            self._init(
                SOLAR_REASONING_BUDGET_MAX="200000",
                SOLAR_REASONING_BUDGET_HARD_LIMIT="131072",
            )
        self.assertIn("SOLAR_REASONING_BUDGET_MAX=200000 exceeds", captured.output[0])
        self.assertEqual(fsm.CFG.effort_budgets["max"], 131072)
        # A table default clamped by a lower hard limit is silent.
        with self.assertNoLogs(fsm.logger, level="WARNING"):
            self._init(SOLAR_REASONING_BUDGET_HARD_LIMIT="50000")
        self.assertEqual(fsm.CFG.effort_budgets["max"], 50000)

    def test_explicit_newline_ids_override_the_vocab(self):
        self._init(SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS="[7, 9]")
        self.assertEqual(fsm.CFG.leading_newline_forbidden, (7, 9))

    def test_bad_default_effort_fails_loud(self):
        with self.assertRaises(RuntimeError):
            self._init(SOLAR_REASONING_BUDGET_DEFAULT_EFFORT="ultra")

    def test_no_reasoning_efforts_are_configurable(self):
        self._init(SOLAR_REASONING_BUDGET_NO_REASONING_EFFORTS="none, low")
        self.assertEqual(fsm._budget_for("low"), 0)
        self.assertEqual(fsm._budget_for("minimal"), 32 * 1024)

    def test_retired_env_names_warn(self):
        for name, replacement in fsm._RETIRED_ENV.items():
            with self.subTest(name=name):
                with self.assertLogs(fsm.logger, level="WARNING") as captured:
                    self._init(**{name: "1000"})
                expected = f"{name} is not read any more" + (
                    f"; use {replacement}" if replacement else " (removed)"
                )
                self.assertIn(expected, "\n".join(captured.output))
        self.assertEqual(fsm.CFG.hard_limit, 128 * 1024)

    def test_none_as_default_effort_is_allowed(self):
        self._init(SOLAR_REASONING_BUDGET_DEFAULT_EFFORT="none")
        self.assertEqual(fsm._budget_for(None), 0)

    def test_all_fourteen_sentinels_are_controls_and_newline_added_token_is_leading(
        self,
    ):
        self._init()
        controls = fsm.CFG.all_controls
        self.assertEqual(len(controls), 14)
        for tid in (110, 111, 112, 113, 114):
            self.assertIn(tid, controls)
            # a control the state does not allow is forbidden everywhere
            self.assertIn(tid, fsm.CFG.forbidden[(fsm.CONTENT, True)])
            self.assertIn(tid, fsm.CFG.forbidden[(fsm.TOOL_CALL_NAME, False)])
            self.assertIn(tid, fsm.CFG.reasoning_forbidden)
        # a pure newline-run added token joins the leading-newline set
        self.assertIn(115, fsm.CFG.leading_newline_forbidden)
        self.assertIn(NL, fsm.CFG.leading_newline_forbidden)
        self.assertNotIn(9, fsm.CFG.leading_newline_forbidden)  # "a"


# The vendor's tables, verbatim (03-logits-processor.patch _MASK_SPEC_BY_STATE /
# _MASK_SPEC_CONTENT), spelled with this file's ids. If our derivation drifts
# from the vendor's, these literals catch it.
VENDOR_ALLOWED = {
    fsm.REASONING: ({THINK_END}, True),
    fsm.TOOL_CALL_BEGIN: (set(), True),
    fsm.TOOL_CALL_NAME: ({ARG_START, TOOL_END}, True),
    fsm.TOOL_ARG_BEGIN: (set(), True),
    fsm.TOOL_ARG_NAME: ({ARG_VALUE}, True),
    fsm.TOOL_ARG_VALUE_BEGIN: ({ARG_END}, True),
    fsm.TOOL_ARG_VALUE: ({ARG_END}, True),
    fsm.TOOL_ARG_END: ({ARG_START, TOOL_END}, True),
    fsm.TOOL_CALL_END: ({TOOL_START, IM_END}, False),
}
VENDOR_CONTENT = {True: ({TOOL_START, IM_END}, False), False: ({TOOL_START}, True)}
ALL_CONTROLS = set(IDS.values())


class TestVendorTables(_FsmCase):
    """The forbidden tables equal the vendor's spec: all controls minus the
    state's allowed sentinels, plus bare EOS where the turn may not end."""

    def test_tables_match_the_vendor_spec(self):
        for state, (allowed, eos_masked) in VENDOR_ALLOWED.items():
            expect = ALL_CONTROLS - allowed
            if eos_masked:
                expect.add(EOS)
            with self.subTest(state=fsm._STATE_NAMES[state]):
                self.assertEqual(set(fsm.CFG.forbidden[(state, False)]), expect)
                self.assertEqual(set(fsm.CFG.forbidden[(state, True)]), expect)
        for progress, (allowed, eos_masked) in VENDOR_CONTENT.items():
            expect = ALL_CONTROLS - allowed
            if eos_masked:
                expect.add(EOS)
            self.assertEqual(set(fsm.CFG.forbidden[(fsm.CONTENT, progress)]), expect)

    def test_no_tools_variant_forbids_tool_call_start_everywhere(self):
        for key, ids in fsm.CFG.forbidden.items():
            with self.subTest(key=key):
                self.assertEqual(
                    set(fsm.CFG.forbidden_notools[key]), set(ids) | {TOOL_START}
                )

    def test_sentinels_are_never_bare_eos(self):
        fsm.configure_ids(IDS, eos=[EOS, IM_END], leading_newline=[])
        # <|im:end|> is a sentinel: state-managed, not part of the EOS mask.
        self.assertNotIn(IM_END, fsm.CFG.forbidden[(fsm.TOOL_CALL_END, False)])

    def test_transitions_match_the_vendor(self):
        self.assertEqual(
            fsm.CFG.transitions,
            {
                THINK_START: fsm.REASONING,
                THINK_END: fsm.CONTENT,
                TOOL_START: fsm.TOOL_CALL_BEGIN,
                TOOL_END: fsm.TOOL_CALL_END,
                ARG_START: fsm.TOOL_ARG_BEGIN,
                ARG_VALUE: fsm.TOOL_ARG_VALUE_BEGIN,
                ARG_END: fsm.TOOL_ARG_END,
            },
        )


class TestToolCallEnvelope(_FsmCase):
    """A tool call walks the vendor's TOOL_CALL_* states; each step masks
    exactly that state's table (2026-09-02 fleet regression: the envelope
    sentinels were masked, so no tool call could complete)."""

    # <|think:end|> <|tool_call:start|> name <|tool_arg:start|> key
    # <|tool_arg:value|> val <|tool_arg:end|> <|tool_call:end|>
    CALL = (THINK_END, TOOL_START, 7, ARG_START, 8, ARG_VALUE, 9, ARG_END, TOOL_END)
    EXPECT = (
        fsm.CONTENT,  # after think_end: fresh content
        fsm.TOOL_CALL_BEGIN,
        fsm.TOOL_CALL_NAME,
        fsm.TOOL_ARG_BEGIN,
        fsm.TOOL_ARG_NAME,
        fsm.TOOL_ARG_VALUE_BEGIN,
        fsm.TOOL_ARG_VALUE,
        fsm.TOOL_ARG_END,
        fsm.TOOL_CALL_END,
    )

    def test_every_step_of_a_tool_call_masks_its_own_state(self):
        for n, state in enumerate(self.EXPECT, start=1):
            with self.subTest(step=n, state=fsm._STATE_NAMES[state]):
                req = _req(self.CALL[:n], tools=True)
                logits = _apply(req)
                self.assertEqual(req._solar_fsm.state, state)
                # step 1 is fresh CONTENT (no progress yet); tool states
                # ignore the flag.
                self.assertEqual(
                    _masked(logits), set(fsm.CFG.forbidden[(state, False)])
                )

    def test_the_function_name_may_be_followed_by_an_argument(self):
        masked = _masked(_apply(_req(self.CALL[:3], tools=True)))
        self.assertNotIn(ARG_START, masked)
        self.assertNotIn(TOOL_END, masked)  # a call without arguments
        for tok in (EOS, IM_END, TOOL_START, ARG_VALUE, ARG_END, THINK_START):
            self.assertIn(tok, masked)

    def test_a_completed_tool_call_may_end_the_turn(self):
        masked = _masked(_apply(_req(self.CALL, tools=True)))
        self.assertEqual(masked, set(fsm.CFG.forbidden[(fsm.TOOL_CALL_END, False)]))
        self.assertNotIn(IM_END, masked)
        self.assertNotIn(EOS, masked)
        self.assertNotIn(TOOL_START, masked)  # a second (parallel) call
        # The token after the closed call is content: the turn has progress.
        req = _req(self.CALL + (11,), tools=True)
        masked = _masked(_apply(req))
        self.assertEqual(req._solar_fsm.state, fsm.CONTENT)
        self.assertTrue(req._solar_fsm.content_progress)
        self.assertEqual(masked, set(fsm.CFG.forbidden[(fsm.CONTENT, True)]))

    def test_fresh_content_forbids_the_envelope_outside_a_call(self):
        masked = _masked(_apply(_req((THINK_END,), tools=True)))
        for tok in TOOL_ENVELOPE:
            self.assertIn(tok, masked)
        self.assertNotIn(TOOL_START, masked)

    def test_a_request_without_tools_cannot_open_a_tool_call(self):
        self.assertIn(TOOL_START, _masked(_apply(_req((THINK_END,), tools=False))))

    def test_structured_outputs_own_the_content_phase(self):
        req = _req((THINK_END,), tools=True)
        req.sampling_params.json_schema = "{}"
        self.assertEqual(_masked(_apply(req)), set())
        # ... but not the tool states (the vendor's rule; under a grammar the
        # JSON tool-call output never opens one).
        req = _req(self.CALL[:3], tools=True, rid="r1")
        req.sampling_params.json_schema = "{}"
        self.assertEqual(
            _masked(_apply(req)), set(fsm.CFG.forbidden[(fsm.TOOL_CALL_NAME, False)])
        )
        # A server-side grammar object alone (--enable-strict-thinking) is
        # not a structured-outputs request: the CONTENT rules stay on.
        req = _req((THINK_END,), tools=True, rid="r2")
        req.grammar = object()
        self.assertEqual(
            _masked(_apply(req)), set(fsm.CFG.forbidden[(fsm.CONTENT, False)])
        )
        # Every request-level constraint counts, on the eager and the
        # speculative (plan_verify) path alike.
        for field in ("regex", "ebnf", "structural_tag"):
            with self.subTest(field=field):
                req = _req((THINK_END,), tools=True, rid=f"g-{field}")
                setattr(req.sampling_params, field, "x")
                self.assertEqual(_masked(_apply(req)), set())
                fsm._req_fsm(req).advance(req.output_ids)
                plan = fsm.plan_verify([req], torch.tensor([[7, 8]]), stride=2)
                self.assertIsNotNone(plan)
                self.assertEqual(plan.mask_rows, {})
        # ... but not the reasoning phase.
        req = _req((), tools=True)
        req.sampling_params.json_schema = "{}"
        self.assertEqual(_masked(_apply(req)), set(fsm.CFG.reasoning_open_forbidden))

    def test_think_start_reopens_reasoning_from_inside_a_call(self):
        req = _req(self.CALL[:3] + (THINK_START,), tools=True)
        state = fsm._req_fsm(req)
        state.advance(req.output_ids)
        self.assertEqual(state.state, fsm.REASONING)
        self.assertEqual(state.count, 0)

    def test_a_completed_call_in_the_prompt_turn_counts_as_content(self):
        # ... <|im:start|> assistant <|tool_call:start|> f <|tool_call:end|>
        # <|tool_response:...|> then the continued turn: the turn may end.
        prompt = (1, IM_START, 2, TOOL_START, 7, TOOL_END, 3)
        req = _req((), prompt=prompt, tools=True)
        self.assertEqual(
            _masked(_apply(req)), set(fsm.CFG.forbidden[(fsm.CONTENT, True)])
        )
        # A call from an *earlier* turn does not.
        prompt = (1, TOOL_START, 7, TOOL_END, IM_START, 2)
        req = _req((), prompt=prompt, tools=True, rid="r1")
        self.assertEqual(
            _masked(_apply(req)), set(fsm.CFG.forbidden[(fsm.CONTENT, False)])
        )

    def test_sim_state_walks_the_envelope_like_the_committed_state(self):
        req = _req((THINK_END,), tools=True)
        state = fsm._req_fsm(req)
        state.advance(req.output_ids)
        sim = fsm._SimState(state)
        for tok, expect in zip(self.CALL[1:], self.EXPECT[1:]):
            sim.step(tok)
            self.assertEqual(sim.state, expect)
        self.assertTrue(sim.content_progress)
        self.assertEqual(state.state, fsm.CONTENT)  # the committed state is untouched
        self.assertEqual(
            fsm._forbidden_for(
                sim.state, content_progress=sim.content_progress, tools=True
            ),
            fsm.CFG.forbidden[(fsm.TOOL_CALL_END, False)],
        )


# ---------------------------------------------------------------------------
# Differential check against a transcript of the vendor's enforcer.
# ---------------------------------------------------------------------------
class _VendorTranscript:
    """SolarOpen2TokenFSMEnforcer's state walk and table lookup, transcribed
    from the vendor's logits processor (03-logits-processor.patch:
    _initial_state, _initial_completed_tool_call_state, _process_token,
    _forbidden_table, and the table lookup of advance_mask_ids) with this
    file's ids -- the budget force, the leading-newline set and the
    structured-outputs exemption are not modelled, and only 9 of the 14
    control sentinels are drawn. Kept as a separate, deliberately literal
    implementation so a drift in the port shows up as a per-step mismatch
    rather than a failed hand-written case."""

    TRANSITIONS = (
        ("think_start", fsm.REASONING),
        ("think_end", fsm.CONTENT),
        ("tool_call_start", fsm.TOOL_CALL_BEGIN),
        ("tool_call_end", fsm.TOOL_CALL_END),
        ("tool_arg_start", fsm.TOOL_ARG_BEGIN),
        ("tool_arg_value", fsm.TOOL_ARG_VALUE_BEGIN),
        ("tool_arg_end", fsm.TOOL_ARG_END),
    )

    def __init__(self, prompt):
        ts, te = IDS["think_start"], IDS["think_end"]
        last_start, last_end = fsm._rindex(prompt, needle=ts), fsm._rindex(
            prompt, needle=te
        )
        self.state = (
            fsm.CONTENT
            if last_start is None or (last_end is not None and last_start < last_end)
            else fsm.REASONING
        )
        last_call_end = fsm._rindex(prompt, needle=IDS["tool_call_end"])
        last_im_start = fsm._rindex(prompt, needle=IDS["im_start"])
        self.content_progress = (
            last_call_end is not None
            and last_im_start is not None
            and last_im_start < last_call_end
        )
        self.count = 0
        self.prev_think_start = bool(prompt) and prompt[-1] == ts
        self.controls = frozenset(IDS.values())
        self.transitions = {}
        for field, state in self.TRANSITIONS:
            self.transitions.setdefault(IDS[field], state)

    def process(self, tok):
        if tok not in self.controls:
            st = self.state
            if st == fsm.REASONING:
                self.count += 1
            elif st == fsm.CONTENT:
                self.content_progress = True
            elif st == fsm.TOOL_CALL_BEGIN:
                self.state = fsm.TOOL_CALL_NAME
            elif st == fsm.TOOL_ARG_BEGIN:
                self.state = fsm.TOOL_ARG_NAME
            elif st == fsm.TOOL_ARG_VALUE_BEGIN:
                self.state = fsm.TOOL_ARG_VALUE
            elif st == fsm.TOOL_CALL_END:
                self.state = fsm.CONTENT
                self.content_progress = True
            self.prev_think_start = False
            return
        prev = self.state
        nxt = self.transitions.get(tok)
        if nxt is not None:
            self.state = nxt
        elif self.state == fsm.TOOL_CALL_BEGIN:
            self.state = fsm.TOOL_CALL_NAME
        elif self.state == fsm.TOOL_ARG_BEGIN:
            self.state = fsm.TOOL_ARG_NAME
        elif self.state == fsm.TOOL_ARG_VALUE_BEGIN:
            self.state = fsm.TOOL_ARG_VALUE
        elif self.state == fsm.TOOL_CALL_END:
            self.state = fsm.CONTENT
        if tok == IDS["think_start"]:
            self.count = 0
        elif prev == fsm.REASONING and self.state == fsm.REASONING:
            self.count += 1
        if tok == IDS["tool_call_end"]:
            self.content_progress = True
        self.prev_think_start = tok == IDS["think_start"]

    def mask(self):
        allowed, eos_masked = (
            VENDOR_CONTENT[self.content_progress]
            if self.state == fsm.CONTENT
            else VENDOR_ALLOWED[self.state]
        )
        forbidden = ALL_CONTROLS - allowed
        if eos_masked:
            forbidden.add(EOS)
        return tuple(sorted(forbidden))


class TestDifferentialAgainstVendorTranscript(_FsmCase):
    """Random token walks (sentinel-heavy) over several prompt shapes: after
    every step the port and the transcript must agree on state, reasoning
    count, content progress, the leading-token flag and the mask."""

    ORDINARY = tuple(range(1000, 1040))
    PROMPTS = (
        (1, 5, THINK_START),  # reasoning open (the chat template's shape)
        (1, 5, THINK_START, THINK_END),  # pre-closed think block (low/none)
        (1, IM_START, 7, TOOL_START, 8, TOOL_END, 9),  # completed call this turn
        (1, TOOL_START, 8, TOOL_END, IM_START, 9),  # a call in an earlier turn
        (1, 5),  # no think block at all
    )

    def test_random_walks_agree_step_by_step(self):
        import random

        rng = random.Random(20260902)
        sentinels = list(IDS.values())
        steps = 0
        for run in range(150):
            prompt = list(self.PROMPTS[run % len(self.PROMPTS)])
            ref = _VendorTranscript(prompt)
            ours = fsm.SolarReqFSM(prompt, effort="high", tools=True)
            self.assertEqual(
                (ours.state, ours.content_progress, ours.at_think_open),
                (ref.state, ref.content_progress, ref.prev_think_start),
                prompt,
            )
            seq = []
            for _ in range(rng.randint(1, 40)):
                tok = (
                    rng.choice(sentinels)
                    if rng.random() < 0.45
                    else rng.choice(self.ORDINARY)
                )
                seq.append(tok)
                ref.process(tok)
                ours._step(tok)
                steps += 1
                self.assertEqual(
                    (ours.state, ours.count, ours.content_progress, ours.at_think_open),
                    (ref.state, ref.count, ref.content_progress, ref.prev_think_start),
                    (prompt, seq),
                )
                got = (
                    fsm.CFG.reasoning_forbidden
                    if ours.state == fsm.REASONING
                    else fsm._forbidden_for(
                        ours.state, content_progress=ours.content_progress, tools=True
                    )
                )
                self.assertEqual(got, ref.mask(), (prompt, seq))
        self.assertGreater(steps, 2000)


class TestBudgetCountingAndReset(_FsmCase):
    """Vendor accounting: a control token that keeps the FSM in REASONING
    counts toward the budget, <|think:start|> resets it, and a reopened block
    gets a fresh budget."""

    def test_reasoning_preserving_control_token_counts(self):
        req = _req((7, IM_START, 8), tools=True)
        state = fsm._req_fsm(req)
        state.advance(req.output_ids)
        self.assertTrue(state.in_reasoning)
        self.assertEqual(state.count, 3)  # 7, <|im:start|>, 8

    def test_think_start_resets_a_nonzero_count_and_the_budget_reapplies(self):
        req = _req((7, 8, 9, THINK_END, 10, THINK_START), tools=True, effort="low")
        state = fsm._req_fsm(req)
        state.advance(req.output_ids)
        self.assertTrue(state.in_reasoning)
        self.assertEqual(state.count, 0)
        state.commit([11] * (4 * 1024 - 1))
        self.assertFalse(state.budget_exhausted())  # the earlier 3 do not count
        state.commit([11])
        self.assertTrue(state.budget_exhausted())


class TestRetractionRebuildSeeds(_FsmCase):
    def test_rebuild_keeps_seeded_progress_and_tools(self):
        prompt = (1, IM_START, 2, TOOL_START, 7, TOOL_END, 3)
        req = _req((), prompt=prompt, tools=False)
        first = fsm._req_fsm(req)
        self.assertTrue(first.content_progress)
        self.assertFalse(first.tools)
        req.retraction_count += 1
        second = fsm._req_fsm(req)
        self.assertIsNot(first, second)
        self.assertTrue(second.content_progress)
        self.assertFalse(second.tools)
        self.assertIn(TOOL_START, _masked(_apply(req)))


class TestSpecPathToolStates(_FsmCase):
    """The speculative path with the content mask on: a committed tool state
    leaves the folded path, and plan_verify masks each chain row with the
    row's own tool-state table."""

    CALL = TestToolCallEnvelope.CALL

    def test_plan_gate_goes_eager_inside_a_tool_call(self):
        """With a committed FSM: a row in any TOOL_* state leaves the folded
        path (also after content, when content_progress is True);
        content-with-progress stays on it; fresh CONTENT under a grammar is
        not sent eager by this predicate (the grammar owns CONTENT; the
        worker forces eager for grammar batches anyway), without a grammar
        it leaves."""
        for output, state in (
            ((THINK_END, TOOL_START), fsm.TOOL_CALL_BEGIN),
            ((THINK_END, TOOL_START, 7), fsm.TOOL_CALL_NAME),
            (
                (THINK_END, TOOL_START, 7, ARG_START, 8, ARG_VALUE),
                fsm.TOOL_ARG_VALUE_BEGIN,
            ),
            (self.CALL[:-1], fsm.TOOL_ARG_END),
            # A second call after content: content_progress is True, so only
            # the tool-state clause can send the row eager (review round 3).
            (self.CALL + (11, TOOL_START), fsm.TOOL_CALL_BEGIN),
            (self.CALL + (11, TOOL_START, 7), fsm.TOOL_CALL_NAME),
        ):
            with self.subTest(state=fsm._STATE_NAMES[state]):
                req = _req(output, tools=True, rid=f"r{state}")
                committed = fsm._req_fsm(req)
                committed.advance(req.output_ids)
                self.assertEqual(committed.state, state)
                if output[: len(self.CALL)] == self.CALL:
                    self.assertTrue(committed.content_progress)
                self.assertTrue(fsm.plan_gate([req], stride=3))
        req = _req(self.CALL + (11,), tools=True, rid="done")
        committed = fsm._req_fsm(req)
        committed.advance(req.output_ids)
        self.assertEqual(committed.state, fsm.CONTENT)
        self.assertTrue(committed.content_progress)
        self.assertFalse(fsm.plan_gate([req], stride=3))
        # Fresh CONTENT under a grammar is the grammar's: no eager step.
        req = _req((THINK_END,), tools=True, rid="grammar")
        req.sampling_params.json_schema = "{}"
        fsm._req_fsm(req).advance(req.output_ids)
        self.assertFalse(fsm.plan_gate([req], stride=3))
        req = _req((THINK_END,), tools=True, rid="nogrammar")
        fsm._req_fsm(req).advance(req.output_ids)
        self.assertTrue(fsm.plan_gate([req], stride=3))

    def test_plan_verify_walks_the_envelope_per_row(self):
        req = _req((THINK_END, TOOL_START, 7), tools=True)
        fsm._req_fsm(req).advance(req.output_ids)
        chain = torch.tensor([[7, ARG_START, 8]])  # anchor, then two drafts
        plan = fsm.plan_verify([req], chain, stride=3)
        self.assertIsNotNone(plan)
        self.assertEqual(
            plan.mask_rows[fsm.CFG.forbidden[(fsm.TOOL_CALL_NAME, False)]], [0]
        )
        self.assertEqual(
            plan.mask_rows[fsm.CFG.forbidden[(fsm.TOOL_ARG_BEGIN, False)]], [1]
        )
        self.assertEqual(
            plan.mask_rows[fsm.CFG.forbidden[(fsm.TOOL_ARG_NAME, False)]], [2]
        )
        self.assertEqual(plan.force_rows, [])

    def test_the_documented_folded_gaps_stay_exactly_as_documented(self):
        """plan_gate records two folded-path gaps as accepted. Nothing pinned
        them, so a refactor could widen or close one and no test would notice --
        and this repo pins accepted trade-offs so that changing one is
        deliberate. This lives here, not next to the flag functions, because
        only this suite builds its tables with ``configure_ids``: a pin read off
        a hand-written fixture re-checks the fixture, not the spec.

        Not asserting the gaps are harmless -- each is a control token reaching
        the answer. Asserting their extent.
        """
        # <|think:end|>: REASONING allows <|think:end|> (its only legal exit) while
        # fresh CONTENT forbids it, so the rows after a drafted <|think:end|>
        # are fresh CONTENT wearing the reasoning set, and a second one lands.
        self.assertNotIn(THINK_END, fsm.CFG.reasoning_forbidden)
        self.assertIn(THINK_END, fsm.CFG.content_fresh_forbidden)

        # Not a gap, and the reason is worth pinning: <|think:start|> is
        # masked on every row that can reach the folded path, and the folded
        # accept is greedy, so a -inf logit is never the argmax and the token
        # never commits. If either half changes -- the sentinel leaving a
        # forbidden set, or the accept becoming non-greedy -- a real gap opens
        # here, and this fails.
        self.assertIn(THINK_START, fsm.CFG.reasoning_forbidden)
        self.assertIn(THINK_START, fsm.CFG.content_done_forbidden)
        self.assertIn(THINK_START, fsm.CFG.content_done_forbidden_notools)

        # <|tool_call:start|>: the tool states forbid the turn from ending, but the rows a
        # legally drafted <|tool_call:start|> puts there carry the CONTENT set,
        # which permits both.
        self.assertIn(EOS, fsm.CFG.forbidden[(fsm.TOOL_CALL_NAME, False)])
        self.assertIn(IM_END, fsm.CFG.forbidden[(fsm.TOOL_CALL_NAME, False)])
        self.assertNotIn(EOS, fsm.CFG.content_done_forbidden)
        self.assertNotIn(IM_END, fsm.CFG.content_done_forbidden)


if __name__ == "__main__":
    unittest.main()
