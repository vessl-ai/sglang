"""Unit tests for the Solar Open2 reasoning FSM's speculative-verify path
(srt/sampling/solar_open2_fsm.py).

Covers the two-part fix for the DSPARK + overlap-scheduling + xgrammar
interaction where the FSM's verify plan and the grammar bitmask were built
from different points in the committed-token timeline: the FSM commit-path
advance (``SolarReqFSM.commit`` / ``advance_committed``) that keeps the FSM on
the grammar's time base, and the defense-in-depth guard in ``VerifyPlan.apply``
/ ``apply`` that never forces a row the grammar has already closed to
``<|think:end|>``.
"""

import time
import types
import unittest

import torch

from sglang.srt.sampling import solar_open2_fsm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB_SIZE = 32
THINK_START = 10
THINK_END = 11
IM_END = 12
EOS = 2


def _req(
    rid="r0",
    output_ids=None,
    origin_input_ids=None,
    max_new_tokens=None,
    grammar=None,
    is_retracted=False,
    finished=False,
    fsm=None,
):
    req = types.SimpleNamespace(
        rid=rid,
        output_ids=list(output_ids or []),
        origin_input_ids=list(origin_input_ids or []),
        sampling_params=types.SimpleNamespace(max_new_tokens=max_new_tokens),
        grammar=grammar,
        is_retracted=is_retracted,
        inflight_middle_chunks=0,
        retraction_count=0,
        finished=lambda f=finished: f,
    )
    if fsm is not None:
        req._solar_fsm = fsm
    return req


def _fsm(budget, in_reasoning=True, count=0, consumed=0):
    """Build a SolarReqFSM with explicit state, bypassing the prompt walk."""
    fsm = solar_open2_fsm.SolarReqFSM.__new__(solar_open2_fsm.SolarReqFSM)
    fsm.state = solar_open2_fsm.REASONING if in_reasoning else solar_open2_fsm.CONTENT
    fsm.count = count
    fsm.consumed = consumed
    fsm.budget = budget
    fsm.forced = False
    fsm.forced_tok = None
    fsm.content_progress = False
    fsm.at_think_open = False
    fsm.tools = True
    return fsm


def _forced(plan):
    """Rows the plan forces, flattened.

    ``force_rows`` used to be a flat row list, all of them forced to
    ``<|think:end|>``. It is now {token id: rows}, because a budget-forced row
    may be told to emit a boundary first, so a test that asks "is row 0 forced"
    has to look inside the values.
    """
    return sorted(r for rows in plan.force_rows.values() for r in rows)


class SolarOpen2FsmVerifyTestBase(CustomTestCase):
    """Wires CFG to a tiny fake vocab and undoes it after every test."""

    _CFG_FIELDS = (
        "enabled",
        "effort_budgets",
        "default_effort",
        "hard_limit",
        "leading_newline_forbidden",
        "reasoning_open_forbidden",
        "content_fresh_forbidden_notools",
        "content_done_forbidden_notools",
        "think_start",
        "think_end",
        "im_end",
        "all_controls",
        "transitions",
        "reasoning_forbidden",
        "content_fresh_forbidden",
        "content_done_forbidden",
        "spec_always_eager",
        # The forced-close arm. init_from_env writes both, so a test that
        # turns the arm on leaks into the rest of the process unless they are
        # saved and restored like the rest.
        "force_seq",
        "force_newline",
    )

    def setUp(self):
        cfg = solar_open2_fsm.CFG
        self._saved = {name: getattr(cfg, name) for name in self._CFG_FIELDS}
        cfg.enabled = True
        cfg.think_start = THINK_START
        cfg.think_end = THINK_END
        cfg.im_end = IM_END
        cfg.all_controls = frozenset({THINK_START, THINK_END, IM_END})
        cfg.transitions = {
            THINK_START: solar_open2_fsm.REASONING,
            THINK_END: solar_open2_fsm.CONTENT,
        }
        cfg.reasoning_forbidden = (EOS, IM_END)
        cfg.leading_newline_forbidden = ()
        cfg.reasoning_open_forbidden = (EOS, IM_END)
        cfg.content_fresh_forbidden = (EOS,)
        cfg.content_done_forbidden = ()
        cfg.content_fresh_forbidden_notools = (EOS,)
        cfg.content_done_forbidden_notools = ()
        # These suites pin the budget by hand: one effort, 1000 tokens.
        cfg.effort_budgets = {"high": 1000}
        cfg.default_effort = "high"
        cfg.hard_limit = 1000
        cfg.spec_always_eager = False
        cfg._mask_cache.clear()
        # Both share _rate_limited, so a second test touching the promote path
        # within the interval would log nothing and fail on the assertLogs
        # rather than on what it is testing.
        for state in (solar_open2_fsm._CONFLICT_LOG, solar_open2_fsm._PROMOTE_LOG):
            state["last"] = -solar_open2_fsm._LOG_INTERVAL
            state["num_suppressed"] = 0

    def tearDown(self):
        cfg = solar_open2_fsm.CFG
        for name, value in self._saved.items():
            setattr(cfg, name, value)
        cfg._mask_cache.clear()


class TestPlanVerifyUsesCommittedState(SolarOpen2FsmVerifyTestBase):
    def test_verify_plan_does_not_force_after_committed_think_end(self):
        # The committed run [reason, reason, reason, think_end] landed via the
        # grammar barrier's commit() -- the FSM's in_reasoning has already
        # flipped False, so plan_verify must not force row 0.
        fsm = _fsm(budget=3)
        fsm.commit([20, 21, 22, THINK_END])
        self.assertFalse(fsm.in_reasoning)

        # req.output_ids is stale: it has not caught up to the committed run yet.
        req = _req(output_ids=[], fsm=fsm)
        chain = torch.tensor([[THINK_END, 30, 31]])
        plan = solar_open2_fsm.plan_verify([req], chain, stride=3)

        self.assertIsNotNone(plan)
        self.assertNotIn(0, _forced(plan))

    def test_uncommitted_stale_state_forces_row0(self):
        # Negative control: without the commit() feed, the FSM is still sitting
        # at the old exhausted-reasoning state and (wrongly) forces row 0 even
        # though the chain anchor is already past think_end.
        fsm = _fsm(budget=3, in_reasoning=True, count=3, consumed=0)
        req = _req(output_ids=[], fsm=fsm)
        chain = torch.tensor([[THINK_END, 30, 31]])
        plan = solar_open2_fsm.plan_verify([req], chain, stride=3)

        self.assertIsNotNone(plan)
        # The token matters as much as the row: since [H27-FORCE-SEQ] a forced
        # row may be told to emit a boundary first, and flattening would pass
        # either way.
        self.assertIn(0, plan.force_rows.get(THINK_END, []))


class TestVerifyPlanForceGuard(SolarOpen2FsmVerifyTestBase):
    def test_force_skipped_when_grammar_forbids_think_end(self):
        logits = torch.zeros(1, VOCAB_SIZE)
        logits[0, THINK_END] = float("-inf")
        original = logits.clone()

        plan = solar_open2_fsm.VerifyPlan(
            force_rows={THINK_END: [0]}, mask_rows={}, stride=1, bs=1, rids=["r0"]
        )
        with self.assertLogs(solar_open2_fsm.logger, level="WARNING"):
            plan.apply(logits)

        self.assertTrue(torch.isfinite(logits[0]).any())
        non_think_end = [i for i in range(VOCAB_SIZE) if i != THINK_END]
        self.assertTrue(
            torch.equal(logits[0, non_think_end], original[0, non_think_end])
        )

    def test_force_fires_when_grammar_allows_think_end(self):
        logits = torch.zeros(1, VOCAB_SIZE)
        original_think_end = logits[0, THINK_END].item()

        plan = solar_open2_fsm.VerifyPlan(
            force_rows={THINK_END: [0]}, mask_rows={}, stride=1, bs=1, rids=["r0"]
        )
        plan.apply(logits)

        for i in range(VOCAB_SIZE):
            if i == THINK_END:
                self.assertEqual(logits[0, i].item(), original_think_end)
            else:
                self.assertEqual(logits[0, i].item(), float("-inf"))


class TestNonSpecApplyForceGuard(SolarOpen2FsmVerifyTestBase):
    def test_non_spec_apply_force_guard(self):
        fsm0 = _fsm(budget=1, in_reasoning=True, count=1, consumed=0)
        fsm1 = _fsm(budget=1, in_reasoning=True, count=1, consumed=0)
        req0 = _req(rid="r0", output_ids=[], fsm=fsm0)
        req1 = _req(rid="r1", output_ids=[], fsm=fsm1)

        logits = torch.zeros(2, VOCAB_SIZE)
        logits[0, THINK_END] = float("-inf")  # grammar already closed row 0
        original_row0 = logits[0].clone()

        sampling_info = types.SimpleNamespace(solar_fsm_rows=[req0, req1])
        with self.assertLogs(solar_open2_fsm.logger, level="INFO") as logs:
            solar_open2_fsm.apply(logits, sampling_info)
        # One warning for the row left to the grammar, one info for the forced
        # row; a second pass on the same request does not log the force again.
        self.assertEqual(
            [l.split(":")[0] for l in logs.output], ["WARNING", "INFO"], logs.output
        )
        self.assertIn("r1", logs.output[1])
        self.assertFalse(fsm0.forced)
        self.assertTrue(fsm1.forced)
        with self.assertNoLogs(solar_open2_fsm.logger, level="INFO"):
            solar_open2_fsm.apply(logits, sampling_info)
        # The conflict warning is rate-limited per row: with both rows now
        # closed by the grammar, one suppressed pass counts two, and the next
        # report carries that count.
        logits[1, THINK_END] = float("-inf")
        with self.assertNoLogs(solar_open2_fsm.logger, level="INFO"):
            solar_open2_fsm.apply(logits, sampling_info)
        self.assertEqual(solar_open2_fsm._CONFLICT_LOG["num_suppressed"], 3)
        solar_open2_fsm._CONFLICT_LOG["last"] = -solar_open2_fsm._LOG_INTERVAL
        with self.assertLogs(solar_open2_fsm.logger, level="WARNING") as logs:
            solar_open2_fsm.apply(logits, sampling_info)
        self.assertIn("3 earlier occurrence(s) suppressed", logs.output[0])
        self.assertEqual(solar_open2_fsm._CONFLICT_LOG["num_suppressed"], 0)
        logits[1, THINK_END] = 0.0

        # Row 0: the force was skipped, so the row keeps the reasoning mask it
        # would have had anyway and still has finite entries to sample from.
        # Leaving it untouched would open EOS and every sentinel for that step.
        self.assertTrue(torch.isfinite(logits[0]).any())
        for tid in solar_open2_fsm.CFG.reasoning_forbidden:
            self.assertEqual(logits[0, tid].item(), float("-inf"), tid)
        for i in range(VOCAB_SIZE):
            if i not in solar_open2_fsm.CFG.reasoning_forbidden:
                self.assertEqual(logits[0, i].item(), original_row0[i].item(), i)
        # Row 1: forced -- -inf everywhere except think_end.
        for i in range(VOCAB_SIZE):
            if i == THINK_END:
                self.assertNotEqual(logits[1, i].item(), float("-inf"))
            else:
                self.assertEqual(logits[1, i].item(), float("-inf"))


class TestPlanVerifyReasoningMask(SolarOpen2FsmVerifyTestBase):
    def test_reasoning_sentinel_mask_unchanged(self):
        fsm = _fsm(budget=100, in_reasoning=True, count=0, consumed=0)
        req = _req(output_ids=[], fsm=fsm)
        stride = 3
        chain = torch.tensor([[5, 6, 7]])  # ordinary anchor + drafts, no controls
        plan = solar_open2_fsm.plan_verify([req], chain, stride=stride)

        self.assertIsNotNone(plan)
        self.assertEqual(_forced(plan), [])
        self.assertEqual(
            sorted(plan.mask_rows[solar_open2_fsm.CFG.reasoning_forbidden]),
            [0, 1, 2],
        )

        logits = torch.zeros(stride, VOCAB_SIZE)
        plan.apply(logits)

        for row in range(stride):
            for idx in range(VOCAB_SIZE):
                if idx in solar_open2_fsm.CFG.reasoning_forbidden:
                    self.assertEqual(logits[row, idx].item(), float("-inf"))
                else:
                    self.assertEqual(logits[row, idx].item(), 0.0)


class TestSolarReqFsmCommitAdvance(SolarOpen2FsmVerifyTestBase):
    def test_commit_then_advance_does_not_double_count(self):
        fsm = _fsm(budget=100, in_reasoning=True, count=0, consumed=0)
        fsm.commit([50, 51])
        self.assertEqual(fsm.count, 2)
        self.assertEqual(fsm.consumed, 2)

        # output_ids now ends with the same run commit() already consumed.
        fsm.advance([50, 51])
        self.assertEqual(fsm.count, 2)
        self.assertEqual(fsm.consumed, 2)

        # Genuinely new tokens are still consumed afterwards.
        fsm.advance([50, 51, 52, 53])
        self.assertEqual(fsm.count, 4)
        self.assertEqual(fsm.consumed, 4)


class TestRetractionRebuildsFsm(SolarOpen2FsmVerifyTestBase):
    def test_discarded_commit_after_retraction_does_not_skip_tokens(self):
        # The barrier committed a run that the scheduler then threw away when
        # the request was retracted, so `consumed` is ahead of output_ids.
        # Without the rebuild, advance() would skip the tokens the request
        # regenerates -- including the think_end below.
        req = _req(rid="r0", origin_input_ids=[THINK_START], max_new_tokens=100)
        fsm = solar_open2_fsm._req_fsm(req)
        fsm.commit([50, 51, 52])  # run dropped by the retraction
        self.assertEqual(fsm.consumed, 3)
        self.assertTrue(fsm.in_reasoning)

        req.retraction_count = 1
        req.output_ids = [60, THINK_END, 61]

        fsm = solar_open2_fsm._req_fsm(req)
        fsm.advance(req.output_ids)
        self.assertEqual(fsm.consumed, 3)
        self.assertFalse(fsm.in_reasoning)

    def test_plan_gate_is_conservative_after_a_retraction(self):
        # One token past <|think:start|>: at the block's first token the gate
        # fires for the leading-newline set regardless of a retraction.
        req = _req(rid="r0", origin_input_ids=[THINK_START, 5], max_new_tokens=100)
        solar_open2_fsm._req_fsm(req)  # seed the persistent FSM
        self.assertFalse(solar_open2_fsm.plan_gate([req], 3))

        req.retraction_count = 1
        self.assertTrue(solar_open2_fsm.plan_gate([req], 3))


class TestPlanGate(SolarOpen2FsmVerifyTestBase):
    def test_plan_gate_covers_a_pending_run(self):
        stride = 3
        window = 2 * stride

        # count + stride < budget <= count + 2*stride -> conservative True.
        fsm_near = _fsm(budget=10, in_reasoning=True, count=5, consumed=0)
        req_near = _req(output_ids=[], fsm=fsm_near)
        self.assertTrue(solar_open2_fsm.plan_gate([req_near], stride))

        # Budget far away -> False.
        fsm_far = _fsm(budget=1000, in_reasoning=True, count=0, consumed=0)
        req_far = _req(output_ids=[], fsm=fsm_far)
        self.assertFalse(solar_open2_fsm.plan_gate([req_far], stride))

        # No FSM yet -> conservative True.
        req_unseen = _req(output_ids=[])
        self.assertTrue(solar_open2_fsm.plan_gate([req_unseen], stride))


class TestAdvanceCommitted(SolarOpen2FsmVerifyTestBase):
    def _decode_batch(
        self, reqs, next_token_ids, accept_lens, stride, grammar_retained_tokens=None
    ):
        batch = types.SimpleNamespace(
            reqs=reqs,
            forward_mode=types.SimpleNamespace(
                is_decode=lambda: True, is_extend=lambda: False
            ),
        )
        result = types.SimpleNamespace(
            solar_fsm_advanced=False,
            copy_done=None,
            next_token_ids=torch.tensor(next_token_ids, dtype=torch.int64),
            accept_lens=torch.tensor(accept_lens, dtype=torch.int64),
            speculative_num_draft_tokens=stride,
            grammar_retained_tokens=grammar_retained_tokens,
        )
        return batch, result

    def test_advance_committed_uses_grammar_retained_tokens(self):
        stride = 3
        req_grammar = _req(
            rid="g0",
            origin_input_ids=[THINK_START],
            grammar=object(),
        )
        req_no_grammar = _req(
            rid="ng0",
            origin_input_ids=[THINK_START],
            grammar=None,
        )
        reqs = [req_grammar, req_no_grammar]
        next_token_ids = [101, 102, 103, 201, 202, 203]
        accept_lens = [3, 3]
        # Grammar truncates req_grammar's run to a strict prefix of the accepted run.
        retained = [[101, 102], None]
        batch, result = self._decode_batch(
            reqs, next_token_ids, accept_lens, stride, grammar_retained_tokens=retained
        )

        solar_open2_fsm.advance_committed(result, batch)

        self.assertTrue(result.solar_fsm_advanced)
        fsm_g = solar_open2_fsm._req_fsm(req_grammar)
        fsm_ng = solar_open2_fsm._req_fsm(req_no_grammar)
        self.assertEqual(fsm_g.count, 2)
        self.assertEqual(fsm_g.consumed, 2)
        # No grammar -> the raw accepted run (all 3 tokens) is used.
        self.assertEqual(fsm_ng.count, 3)
        self.assertEqual(fsm_ng.consumed, 3)

        # Idempotent: a second call must not double-advance.
        solar_open2_fsm.advance_committed(result, batch)
        self.assertEqual(fsm_g.count, 2)
        self.assertEqual(fsm_ng.count, 3)

    def test_advance_committed_extend_commits_single_token(self):
        req = _req(rid="e0", origin_input_ids=[THINK_START])
        batch = types.SimpleNamespace(
            reqs=[req],
            forward_mode=types.SimpleNamespace(
                is_decode=lambda: False, is_extend=lambda: True
            ),
        )
        result = types.SimpleNamespace(
            solar_fsm_advanced=False,
            copy_done=None,
            next_token_ids=torch.tensor([77], dtype=torch.int64),
            accept_lens=None,
            speculative_num_draft_tokens=None,
            grammar_retained_tokens=None,
        )

        solar_open2_fsm.advance_committed(result, batch)

        self.assertTrue(result.solar_fsm_advanced)
        fsm = solar_open2_fsm._req_fsm(req)
        self.assertEqual(fsm.count, 1)
        self.assertEqual(fsm.consumed, 1)


class TestForcedCloseBoundary(SolarOpen2FsmVerifyTestBase):
    """What a budget-spent row is forced to emit, and where that travels."""

    # This suite has no vocabulary table; _arm sets the ids directly.
    NL = 5

    def _arm(self):
        cfg = solar_open2_fsm.CFG
        cfg.force_seq = "nl_te"
        cfg.force_newline = self.NL

    def _decide(self, last_tok, at_think_open=False):
        state = _fsm(budget=100, count=100)
        state.at_think_open = at_think_open
        return solar_open2_fsm._force_token(state, last_tok=last_tok)

    def test_the_think_open_row_is_never_forced_to_a_newline(self):
        # reasoning_open_forbidden already shuts the leading-newline ids on the
        # token right after <|think:start|>, so forcing one there would leave
        # the row fully -inf and the sampler would return an id the grammar
        # rejects. That row closes on <|think:end|> instead.
        self._arm()
        self.assertEqual(self._decide(last_tok=99, at_think_open=True), THINK_END)

    def test_the_arm_stays_off_unless_the_env_turns_it_on(self):
        # The whole gate: with force_seq off the module forces the sentinel and
        # nothing before it, whatever newline id happens to be resolved.
        cfg = solar_open2_fsm.CFG
        cfg.force_seq, cfg.force_newline = "off", self.NL
        self.assertEqual(self._decide(last_tok=99), THINK_END)

    def test_non_spec_apply_does_not_force_the_boundary_twice(self):
        # req.output_ids lags a step on this path, and advance() is monotonic,
        # so the same anchor would be read again and the boundary forced a
        # second time -- the block would commit "word \n \n TE". The token
        # this row was told to emit last step is what the next one must read.
        self._arm()
        req = _req(rid="r0", output_ids=[99], fsm=_fsm(budget=1, count=1))
        info = types.SimpleNamespace(solar_fsm_rows=[req])
        first = torch.zeros(1, VOCAB_SIZE)
        solar_open2_fsm.apply(first, info)
        self.assertEqual(first[0, self.NL].item(), 0.0)
        second = torch.zeros(1, VOCAB_SIZE)
        solar_open2_fsm.apply(second, info)
        self.assertEqual(second[0, THINK_END].item(), 0.0)
        self.assertEqual(second[0, self.NL].item(), float("-inf"))
        # And once the close is out, nothing more is forced: the row keeps the
        # mask until advance() sees it leave the block.
        third = torch.zeros(1, VOCAB_SIZE)
        solar_open2_fsm.apply(third, info)
        self.assertEqual(third[0, THINK_END].item(), 0.0)
        self.assertEqual(third[0, self.NL].item(), 0.0)

    def test_a_blocked_row_is_offered_the_boundary_again(self):
        # forced_tok records what was written, not what was decided. A row the
        # grammar shut on both tokens took no write, so the next step has to
        # offer it the same boundary rather than move on as if it had closed.
        self._arm()
        req = _req(rid="r0", output_ids=[99], fsm=_fsm(budget=1, count=1))
        info = types.SimpleNamespace(solar_fsm_rows=[req])
        shut = torch.zeros(1, VOCAB_SIZE)
        shut[0, self.NL] = float("-inf")
        shut[0, THINK_END] = float("-inf")
        with self.assertLogs(solar_open2_fsm.logger, level="WARNING"):
            solar_open2_fsm.apply(shut, info)
        self.assertIsNone(req._solar_fsm.forced_tok)
        open_again = torch.zeros(1, VOCAB_SIZE)
        solar_open2_fsm.apply(open_again, info)
        self.assertEqual(open_again[0, self.NL].item(), 0.0)

    def test_a_reopened_think_block_is_forced_again(self):
        # <|think:start|> puts count back to zero, so one request can open the
        # block twice. Without the reset the first block's close stays recorded
        # and the second block is never closed at all.
        self._arm()
        # The counter is back to zero and the budget is not spent, which is
        # what the second block looks like on its first steps.
        fsm = _fsm(budget=5, count=0)
        fsm.forced_tok = THINK_END  # ... left over from the first block
        req = _req(rid="r0", output_ids=[99], fsm=fsm)
        info = types.SimpleNamespace(solar_fsm_rows=[req])
        solar_open2_fsm.apply(torch.zeros(1, VOCAB_SIZE), info)
        self.assertIsNone(fsm.forced_tok)
        fsm.count = 5  # the second block now spends its own budget
        logits = torch.zeros(1, VOCAB_SIZE)
        solar_open2_fsm.apply(logits, info)
        self.assertEqual(logits[0, self.NL].item(), 0.0)

    def test_off_still_repeats_the_close_while_output_ids_lag(self):
        # The unprescribed arm is left exactly as it was: it has no sequence to
        # finish, and the repeat is what supplies the second sentinel that the
        # non-speculative path has always closed on.
        cfg = solar_open2_fsm.CFG
        cfg.force_seq, cfg.force_newline = "off", self.NL
        req = _req(rid="r0", output_ids=[99], fsm=_fsm(budget=1, count=1))
        info = types.SimpleNamespace(solar_fsm_rows=[req])
        for _ in range(2):
            logits = torch.zeros(1, VOCAB_SIZE)
            solar_open2_fsm.apply(logits, info)
            self.assertEqual(logits[0, THINK_END].item(), 0.0)
            self.assertEqual(logits[0, self.NL].item(), float("-inf"))

    def test_the_plan_log_names_the_token_it_planned(self):
        # An operator reading "planning token id=..." has to be able to trust
        # it: naming think_end where the plan holds the boundary sends them
        # after the wrong id.
        self._arm()
        req = _req(output_ids=[99], fsm=_fsm(budget=100, count=100))
        with self.assertLogs(solar_open2_fsm.logger, level="INFO") as logs:
            solar_open2_fsm.plan_verify([req], torch.tensor([[99, 30, 31]]), stride=3)
        self.assertIn(f"id={self.NL}", logs.output[0])

    def test_nl_te_emits_newline_then_think_end(self):
        # Without the "last_tok is already the newline" guard the newline is
        # forced forever and the block never closes.
        self._arm()
        self.assertEqual(self._decide(last_tok=99), self.NL)
        self.assertEqual(self._decide(last_tok=self.NL), THINK_END)

    def test_blocked_boundary_is_promoted_and_logged(self):
        # The promote path had a NameError for four review rounds because no
        # test ever made _mask_and_force return a non-empty promoted list. It
        # only fires when the grammar forbids the boundary, which is exactly
        # the case an arm cannot see from its own configuration.
        self._arm()
        logits = torch.zeros(1, VOCAB_SIZE)
        logits[0, self.NL] = float("-inf")  # the grammar forbids the boundary
        forced, blocked, promoted = solar_open2_fsm._mask_and_force(
            logits, mask_rows={}, force_rows={self.NL: [0]}
        )
        self.assertEqual(promoted, [(0, self.NL)])
        self.assertEqual(forced, [(0, THINK_END)])
        self.assertEqual(blocked, [])
        # The row closes on think_end rather than being left to the grammar,
        # which would re-force the same forbidden token every step.
        self.assertEqual(logits[0, THINK_END].item(), 0.0)

    def test_row_blocked_on_both_tokens_is_left_to_the_grammar(self):
        self._arm()
        logits = torch.zeros(1, VOCAB_SIZE)
        logits[0, self.NL] = float("-inf")
        logits[0, THINK_END] = float("-inf")
        forced, blocked, promoted = solar_open2_fsm._mask_and_force(
            logits, mask_rows={}, force_rows={self.NL: [0]}
        )
        self.assertEqual(promoted, [])
        self.assertEqual(forced, [])
        self.assertEqual(blocked, [(0, self.NL)])

    def test_plan_verify_carries_the_decided_token_into_force_rows(self):
        # The decision function has its own tests above; this one pins where
        # the decision travels. Without a check on the key of force_rows,
        # plan_verify could throw the answer away and force think_end anyway
        # with the suite still green, which deletes the nl_te arm.
        self._arm()
        fsm = _fsm(budget=100, count=100)
        req = _req(output_ids=[99], fsm=fsm)
        plan = solar_open2_fsm.plan_verify(
            [req], torch.tensor([[99, 30, 31]]), stride=3
        )
        self.assertIsNotNone(plan)
        self.assertIn(0, plan.force_rows.get(self.NL, []))
        self.assertNotIn(THINK_END, plan.force_rows)

    def test_verify_plan_reads_the_chain_anchor_not_stale_output_ids(self):
        # req.output_ids lags a step behind what the FSM has consumed, so at
        # w == 0 the anchor is row_ids[0]. Reading the stale value here forces
        # a newline onto a row whose last committed token already was one,
        # putting a second newline into the block.
        self._arm()
        fsm = _fsm(budget=100, count=100)
        req = _req(output_ids=[99], fsm=fsm)
        chain = torch.tensor([[self.NL, 30, 31]])
        plan = solar_open2_fsm.plan_verify([req], chain, stride=3)
        self.assertIsNotNone(plan)
        self.assertIn(0, plan.force_rows.get(THINK_END, []))
        self.assertNotIn(0, plan.force_rows.get(self.NL, []))

    def test_a_draft_that_already_carries_the_boundary_closes_on_it(self):
        # The rows past w == 0 are planned against draft tokens, so they only
        # ever commit when the draft matched what was forced. When it did, the
        # anchor for the next row is the boundary itself and that row has to
        # close rather than force a second one.
        self._arm()
        fsm = _fsm(budget=100, count=100)
        req = _req(output_ids=[99], fsm=fsm)
        chain = torch.tensor([[99, self.NL, 31]])
        plan = solar_open2_fsm.plan_verify([req], chain, stride=3)
        self.assertIn(0, plan.force_rows.get(self.NL, []))
        self.assertIn(1, plan.force_rows.get(THINK_END, []))

    def test_non_spec_apply_forces_the_decided_token(self):
        # The same guard on the non-speculative entry point, read off the
        # logits rather than the plan: the boundary is the only column left
        # finite, so forcing think_end here instead would fail.
        self._arm()
        fsm = _fsm(budget=1, count=1)
        req = _req(rid="r0", output_ids=[99], fsm=fsm)
        logits = torch.zeros(1, VOCAB_SIZE)
        solar_open2_fsm.apply(logits, types.SimpleNamespace(solar_fsm_rows=[req]))
        self.assertEqual(logits[0, self.NL].item(), 0.0)
        self.assertEqual(logits[0, THINK_END].item(), float("-inf"))

    def test_a_row_blocked_on_both_tokens_keeps_its_envelope(self):
        # A forced row that the grammar closes on both tokens gets no tensor
        # write from the force loop. Without the reasoning mask underneath it,
        # that step leaves EOS and every sentinel open and the model can end
        # the turn inside the think block, which the parser then drops whole.
        self._arm()
        fsm = _fsm(budget=1, count=1)
        req = _req(rid="r0", output_ids=[99], fsm=fsm)
        logits = torch.zeros(1, VOCAB_SIZE)
        logits[0, self.NL] = float("-inf")
        logits[0, THINK_END] = float("-inf")
        with self.assertLogs(solar_open2_fsm.logger, level="WARNING"):
            solar_open2_fsm.apply(logits, types.SimpleNamespace(solar_fsm_rows=[req]))
        for tid in solar_open2_fsm.CFG.reasoning_forbidden:
            self.assertEqual(logits[0, tid].item(), float("-inf"), tid)

    def test_two_tokens_are_forced_in_one_batch(self):
        # One batch forces a boundary on one row and the close on another, so
        # _mask_and_force has to carry more than one key.
        self._arm()
        logits = torch.zeros(2, VOCAB_SIZE)
        forced, blocked, promoted = solar_open2_fsm._mask_and_force(
            logits, mask_rows={}, force_rows={self.NL: [0], THINK_END: [1]}
        )
        self.assertEqual(sorted(forced), [(0, self.NL), (1, THINK_END)])
        self.assertEqual((blocked, promoted), ([], []))
        self.assertEqual(logits[0, self.NL].item(), 0.0)
        self.assertEqual(logits[0, THINK_END].item(), float("-inf"))
        self.assertEqual(logits[1, THINK_END].item(), 0.0)
        self.assertEqual(logits[1, self.NL].item(), float("-inf"))

    def test_the_promote_log_is_wired_into_both_entry_points(self):
        # The wiring, not the two functions: deleting either `if promoted:`
        # block leaves the promotion silent, which is the whole point of it.
        self._arm()
        for build in (
            lambda lg: solar_open2_fsm.VerifyPlan(
                force_rows={self.NL: [0]}, mask_rows={}, stride=1, bs=1, rids=["r0"]
            ).apply(lg),
            lambda lg: solar_open2_fsm.apply(
                lg,
                types.SimpleNamespace(
                    solar_fsm_rows=[
                        _req(rid="r0", output_ids=[99], fsm=_fsm(budget=1, count=1))
                    ]
                ),
            ),
        ):
            # The conflict limit is held shut so the promotion has to reach
            # the log on its own state: the two say different things about
            # which arm is live, and a burst of one must not swallow the other.
            solar_open2_fsm._CONFLICT_LOG["last"] = time.monotonic()
            solar_open2_fsm._CONFLICT_LOG["num_suppressed"] = 0
            solar_open2_fsm._PROMOTE_LOG["last"] = -solar_open2_fsm._LOG_INTERVAL
            solar_open2_fsm._PROMOTE_LOG["num_suppressed"] = 0
            logits = torch.zeros(1, VOCAB_SIZE)
            logits[0, self.NL] = float("-inf")
            with self.assertLogs(solar_open2_fsm.logger, level="WARNING") as logs:
                build(logits)
            self.assertTrue(
                any("the boundary token the arm wanted" in l for l in logs.output),
                logs.output,
            )


class TestFileShape(CustomTestCase):
    """The CI runner executes each registered file with ``python3 <file>``, so
    ``__name__`` is ``"__main__"`` and anything defined after the
    ``unittest.main()`` call never exists. Eight tests in this file were dead
    for four review rounds that way, and the registration sanity check only
    asks whether a main block is present, not whether it is last."""

    def test_registered_files_put_the_main_block_last(self):
        # Checked across the directory, not just this file: a file whose own
        # main block is misplaced stops executing before its checks run, so it
        # cannot catch its own defect. A sibling has to.
        import ast as _ast
        import pathlib

        here = pathlib.Path(__file__).resolve().parent
        checked = 0
        for path in sorted(here.glob("test_*.py")):
            body = _ast.parse(path.read_text(encoding="utf-8")).body
            mains = [
                i
                for i, node in enumerate(body)
                if isinstance(node, _ast.If) and "__name__" in _ast.dump(node.test)
            ]
            if not mains:
                continue
            checked += 1
            self.assertEqual(
                mains[-1],
                len(body) - 1,
                f"{path.name}: the __main__ block must be the last statement. "
                "The CI runner executes each registered file directly, so "
                "anything defined after unittest.main() never exists and its "
                "tests are silently absent.",
            )
        self.assertGreater(checked, 1, "expected sibling files to check")


if __name__ == "__main__":
    unittest.main()
