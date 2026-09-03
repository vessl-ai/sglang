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
    fsm.content_progress = False
    fsm.at_think_open = False
    fsm.tools = True
    return fsm


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
        solar_open2_fsm._CONFLICT_LOG["last"] = -solar_open2_fsm._LOG_INTERVAL
        solar_open2_fsm._CONFLICT_LOG["num_suppressed"] = 0

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
        self.assertNotIn(0, plan.force_rows)

    def test_uncommitted_stale_state_forces_row0(self):
        # Negative control: without the commit() feed, the FSM is still sitting
        # at the old exhausted-reasoning state and (wrongly) forces row 0 even
        # though the chain anchor is already past think_end.
        fsm = _fsm(budget=3, in_reasoning=True, count=3, consumed=0)
        req = _req(output_ids=[], fsm=fsm)
        chain = torch.tensor([[THINK_END, 30, 31]])
        plan = solar_open2_fsm.plan_verify([req], chain, stride=3)

        self.assertIsNotNone(plan)
        self.assertIn(0, plan.force_rows)


class TestVerifyPlanForceGuard(SolarOpen2FsmVerifyTestBase):
    def test_force_skipped_when_grammar_forbids_think_end(self):
        logits = torch.zeros(1, VOCAB_SIZE)
        logits[0, THINK_END] = float("-inf")
        original = logits.clone()

        plan = solar_open2_fsm.VerifyPlan(
            force_rows=[0], mask_rows={}, stride=1, bs=1, rids=["r0"]
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
            force_rows=[0], mask_rows={}, stride=1, bs=1, rids=["r0"]
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

        # Row 0: forced skipped, logits unchanged, still has finite entries.
        self.assertTrue(torch.equal(logits[0], original_row0))
        self.assertTrue(torch.isfinite(logits[0]).any())
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
        self.assertEqual(plan.force_rows, [])
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


if __name__ == "__main__":
    unittest.main()
