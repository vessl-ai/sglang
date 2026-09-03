"""The in-graph FSM masks as the epilogue wires them, not as the FSM plans them.

``apply_folded_mask`` is unit-tested next to the flag functions in
test_solar_open2_fsm_mask_gate.py. What neither covers is
``DsparkVerifyEpilogue``: delete the second ``apply_folded_mask`` call in
``_apply_fsm_mask`` and the CONTENT mask silently stops existing while every
other test in the repo stays green. Same for the staging setter and the buffer
the ids are snapshotted into.

Pure CPU: the epilogue's constructor touches only torch and a tp_sync whose
only use here is ``world_size``.
"""

import ast
import inspect
import os
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.sampling import solar_open2_fsm as fsm
from sglang.srt.speculative.dspark_components import dspark_worker_v2
from sglang.srt.speculative.dspark_components.dspark_tp import DsparkTpSync
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DsparkVerifyEpilogue,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

THINK_START, THINK_END, EOS, TOOL_START = 100, 101, 2, 102
TOOL_ARG_START, TOOL_ARG_VALUE, TOOL_ARG_END, TOOL_CALL_END = 103, 104, 105, 106
TOOL_INTERNALS = (TOOL_ARG_START, TOOL_ARG_VALUE, TOOL_ARG_END, TOOL_CALL_END)
VOCAB = 128

_CFG_FIELDS = (
    "enabled",
    "think_start",
    "think_end",
    "all_controls",
    "transitions",
    "reasoning_forbidden",
    "reasoning_open_forbidden",
    "leading_newline_forbidden",
    "content_done_forbidden",
    "content_fresh_forbidden",
    "content_done_forbidden_notools",
    "content_fresh_forbidden_notools",
    "tool_call_start",
    "tool_arg_start",
    "tool_arg_value",
    "tool_arg_end",
    "tool_call_end",
    "spec_always_eager",
    "effort_budgets",
    "default_effort",
    "hard_limit",
)


def _cfg():
    fsm.CFG.enabled = True
    fsm.CFG.think_start, fsm.CFG.think_end = THINK_START, THINK_END
    fsm.CFG.all_controls = frozenset(
        {THINK_START, THINK_END, TOOL_START, TOOL_ARG_VALUE}
    )
    fsm.CFG.transitions = {THINK_START: fsm.REASONING, THINK_END: fsm.CONTENT}
    # Disjoint on purpose: it is the only way this suite can tell the two masks
    # apart by looking at the logits.
    fsm.CFG.reasoning_forbidden = (EOS,)
    fsm.CFG.reasoning_open_forbidden = (EOS,)
    fsm.CFG.leading_newline_forbidden = ()
    fsm.CFG.content_done_forbidden = (THINK_START, THINK_END, *TOOL_INTERNALS)
    fsm.CFG.content_fresh_forbidden = (THINK_START, THINK_END, *TOOL_INTERNALS, EOS)
    # The no-tools tables differ from the above by the opener alone, which is
    # what lets the third buffer hold one id.
    fsm.CFG.tool_call_start = TOOL_START
    fsm.CFG.tool_arg_start = TOOL_ARG_START
    fsm.CFG.tool_arg_value = TOOL_ARG_VALUE
    fsm.CFG.tool_arg_end = TOOL_ARG_END
    fsm.CFG.tool_call_end = TOOL_CALL_END
    fsm.CFG.content_done_forbidden_notools = (
        THINK_START,
        THINK_END,
        TOOL_START,
        *TOOL_INTERNALS,
    )
    fsm.CFG.content_fresh_forbidden_notools = (
        THINK_START,
        THINK_END,
        TOOL_START,
        *TOOL_INTERNALS,
        EOS,
    )
    fsm.CFG.spec_always_eager = False
    fsm.CFG.effort_budgets = {"high": 3072}
    fsm.CFG.default_effort = "high"
    fsm.CFG.hard_limit = 3072


class TestEpilogueFsmMasks(CustomTestCase):
    max_bs, stride = 2, 4

    def setUp(self):
        self._saved = {k: getattr(fsm.CFG, k) for k in _CFG_FIELDS}
        _cfg()
        # After _cfg(): both forbid buffers are snapshotted in __init__.
        self.ep = DsparkVerifyEpilogue(
            max_bs=self.max_bs,
            verify_num_draft_tokens=self.stride,
            device="cpu",
            tp_sync=DsparkTpSync(SimpleNamespace(world_size=1)),
        )

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(fsm.CFG, k, v)
        fsm.CFG._mask_cache.clear()

    def _logits(self):
        self.ep.strided_logits = torch.zeros(
            self.max_bs * self.stride, VOCAB, dtype=torch.float32
        )
        return self.ep.strided_logits

    def test_the_content_forbid_buffer_holds_the_content_set(self):
        """Not the reasoning set, and not the [0] placeholder -- a placeholder
        left in place masks token id 0, which looks masked and is not."""
        self.assertEqual(
            self.ep.fsm_content_forbid_buf.tolist(),
            [i for i in fsm.CFG.content_done_forbidden if i not in TOOL_INTERNALS],
        )
        self.assertEqual(
            self.ep.fsm_forbid_buf.tolist(), list(fsm.CFG.reasoning_forbidden)
        )
        # int32 indexes fine today and may not tomorrow; the row buffer's dtype
        # is asserted for the same reason.
        self.assertEqual(self.ep.fsm_content_forbid_buf.dtype, torch.long)

    def test_the_two_row_buffers_are_separate_and_full_size(self):
        """One buffer for two masks would alias them; a short one would leave
        the tail rows of a max_bs batch permanently unarmed."""
        self.assertIsNot(self.ep.fsm_row_buf, self.ep.fsm_content_row_buf)
        self.assertEqual(
            self.ep.fsm_content_row_buf.shape, (self.max_bs * self.stride,)
        )
        self.assertEqual(self.ep.fsm_content_row_buf.dtype, torch.bool)
        self.assertFalse(self.ep.fsm_content_row_buf.any())

    def test_set_fsm_content_rows_stages_exactly_the_flags(self):
        flags = [False] * self.stride + [True] * self.stride
        self.ep.set_fsm_content_rows(flags)
        self.assertEqual(self.ep.fsm_content_row_buf.tolist(), flags)

    def test_a_short_flag_list_zeroes_the_tail(self):
        """A bs=2 step arms rows 4..7; the next step is bs=1 and passes four
        flags. Without the tail zero those rows stay armed and mask a request
        that is not in CONTENT at all."""
        self.ep.set_fsm_content_rows([True] * (self.max_bs * self.stride))
        self.ep.set_fsm_content_rows([True] * self.stride)
        self.assertEqual(
            self.ep.fsm_content_row_buf.tolist(),
            [True] * self.stride + [False] * self.stride,
        )

    def test_none_and_empty_disarm_the_content_mask(self):
        """folded_content_mask_flags returns None while the FSM is inactive --
        the contract every deployment that does not run this model rests on."""
        for flags in (None, []):
            with self.subTest(flags=flags):
                self.ep.set_fsm_content_rows([True] * self.stride)
                self.ep.set_fsm_content_rows(flags)
                self.assertFalse(self.ep.fsm_content_row_buf.any())

    def test_set_fsm_content_rows_does_not_touch_the_reasoning_buffer(self):
        self.ep.set_fsm_rows([True] * self.stride * self.max_bs, (EOS,))
        self.ep.set_fsm_content_rows([False] * self.stride * self.max_bs)
        self.assertTrue(self.ep.fsm_row_buf.all())

    def test_a_content_armed_row_gets_the_content_set_to_neg_inf(self):
        """The commit's whole claim, end to end. Delete the second
        apply_folded_mask call in _apply_fsm_mask and this is what fails."""
        logits = self._logits()
        # rows 0..3 = request 0 (REASONING), rows 4..7 = request 1 (CONTENT).
        self.ep.set_fsm_rows([True] * self.stride + [False] * self.stride, (EOS,))
        self.ep.set_fsm_content_rows([False] * self.stride + [True] * self.stride)
        self.ep._apply_fsm_mask(bs=self.max_bs)

        content = logits[self.stride :]
        reasoning = logits[: self.stride]
        self.assertTrue(
            torch.isinf(content[:, THINK_START]).all(),
            "content-armed rows must have <|think:start|> shut",
        )
        self.assertTrue(
            torch.isfinite(content[:, EOS]).all(),
            "EOS is free once content exists -- the CONTENT mask must not "
            "borrow the reasoning set",
        )
        self.assertTrue(
            torch.isinf(reasoning[:, EOS]).all(),
            "the reasoning mask must still run; the second call is additional",
        )
        self.assertTrue(
            torch.isfinite(reasoning[:, THINK_START]).all(),
            "a reasoning row must not pick up the content set",
        )

    def test_an_unarmed_step_moves_nothing(self):
        """Both masks are captured into the graph unconditionally, so an
        FSM-off step runs both kernels and writes every logit back unchanged."""
        logits = self._logits()
        logits.copy_(torch.randn_like(logits))
        before = logits.clone()
        self.ep.set_fsm_rows(None)
        self.ep.set_fsm_content_rows(None)
        self.ep._apply_fsm_mask(bs=self.max_bs)
        self.assertTrue(torch.equal(logits, before))

    def test_a_smaller_batch_does_not_reach_stale_rows(self):
        """_apply_fsm_mask slices bs*stride; that slice is what keeps a
        shrinking batch from masking the previous step's rows."""
        logits = self._logits()
        self.ep.set_fsm_content_rows([True] * (self.max_bs * self.stride))
        self.ep._apply_fsm_mask(bs=1)
        self.assertTrue(torch.isinf(logits[: self.stride, THINK_START]).all())
        self.assertTrue(torch.isfinite(logits[self.stride :, THINK_START]).all())

    def test_the_masks_compose_on_a_row_armed_by_both(self):
        """The flag functions are what keep the two sets apart; the epilogue
        does not enforce it. Pin what happens if that is ever violated -- both
        sets shut, no exception -- so the failure surfaces in the flag tests."""
        logits = self._logits()
        n = self.max_bs * self.stride
        self.ep.set_fsm_rows([True] * n, (EOS,))
        self.ep.set_fsm_content_rows([True] * n)
        self.ep._apply_fsm_mask(bs=self.max_bs)
        self.assertTrue(torch.isinf(logits[:, EOS]).all())
        self.assertTrue(torch.isinf(logits[:, THINK_START]).all())

    def test_the_placeholder_is_used_while_the_fsm_is_off(self):
        """The [0] fallback is what keeps the captured shape static on a
        deployment that does not run this model; it is only ever paired with
        an all-False row buffer, so it writes nothing.

        is_active() re-resolves from SOLAR_FSM, so clearing CFG.enabled alone
        measures nothing on a host that sets it -- which is every engine pod.
        """
        prev = os.environ.get("SOLAR_FSM")
        os.environ["SOLAR_FSM"] = "0"
        fsm.CFG.enabled = False
        try:
            ep = DsparkVerifyEpilogue(
                max_bs=self.max_bs,
                verify_num_draft_tokens=self.stride,
                device="cpu",
                tp_sync=DsparkTpSync(SimpleNamespace(world_size=1)),
            )
            self.assertEqual(ep.fsm_content_forbid_buf.tolist(), [0])
        finally:
            if prev is None:
                os.environ.pop("SOLAR_FSM", None)
            else:
                os.environ["SOLAR_FSM"] = prev
            _cfg()

    def test_a_stale_content_id_snapshot_is_reported(self):
        """__init__ snapshots the ids while the flags are decided per step. A
        buffer left behind by a late FSM resolution masks the wrong ids, and
        every armed content row then looks masked and is not."""
        fsm.CFG.content_done_forbidden = (THINK_START, 55)
        with self.assertLogs(
            "sglang.srt.speculative.dspark_components.dspark_verify", level="ERROR"
        ) as logs:
            self.ep.set_fsm_content_rows([True] * self.stride)
        self.assertTrue(any("CONTENT mask holds" in m for m in logs.output))

    def test_a_fresh_content_id_snapshot_is_not_reported(self):
        """Without this, a detector that logs unconditionally passes the test
        above."""
        with self.assertNoLogs(
            "sglang.srt.speculative.dspark_components.dspark_verify", level="ERROR"
        ):
            self.ep.set_fsm_content_rows([True] * self.stride)

    def test_a_disarming_call_does_not_burn_the_check(self):
        """The check fires once. If a None call spends it, the first step of a
        process with the FSM off silences every later divergence -- and that is
        the common case, since flags are None whenever the FSM is inactive."""
        self.ep.set_fsm_content_rows(None)
        fsm.CFG.content_done_forbidden = (THINK_START, 55)
        with self.assertLogs(
            "sglang.srt.speculative.dspark_components.dspark_verify", level="ERROR"
        ) as logs:
            self.ep.set_fsm_content_rows([True] * self.stride)
        self.assertTrue(any("CONTENT mask holds" in m for m in logs.output))

    def test_each_buffer_gets_its_own_check(self):
        """The one-shot check is keyed by a label string. Give two buffers the
        same label and the first staging call spends the shot for both, so the
        second is never checked for the life of the process -- silently, which
        is the exact failure the check exists to prevent. Every other test here
        builds a fresh epilogue and stages one buffer, so none of them can see
        it."""
        fsm.CFG.reasoning_forbidden = (EOS, 55)
        fsm.CFG.content_done_forbidden = (THINK_START, 56)
        fsm.CFG.content_done_forbidden_notools = (THINK_START, TOOL_START, 57)
        rows = [True] * self.stride
        with self.assertLogs(
            "sglang.srt.speculative.dspark_components.dspark_verify", level="ERROR"
        ) as logs:
            self.ep.set_fsm_rows(rows, fsm.CFG.reasoning_forbidden)
            self.ep.set_fsm_content_rows(rows)
            self.ep.set_fsm_content_notools_rows(rows)
        self.assertEqual(len(self.ep._ids_checked), 3, self.ep._ids_checked)
        self.assertEqual(len(logs.output), 3, logs.output)

    def test_the_worker_arms_the_content_mask_from_the_content_flags(self):
        """Both halves are unit-tested; nothing else tests that the worker
        connects them. Deleting the set_fsm_content_rows call leaves every
        other test in the repo green and the mask armed on no row at all --
        and handing it folded_mask_flags instead is a live copy-paste mutant.

        Source-shape, deliberately: there is no GPU-free way to drive the
        worker. A rename breaks this, which is the intended cost.
        """
        tree = ast.parse(inspect.getsource(dspark_worker_v2))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "set_fsm_content_rows"
        ]
        self.assertEqual(len(calls), 1, "the CONTENT mask must be armed once")
        arg = calls[0].args[0]
        self.assertIsInstance(arg, ast.Call)
        self.assertEqual(
            arg.func.attr,
            "folded_content_mask_flags",
            "the CONTENT setter must be given the CONTENT flags",
        )

    def test_the_notools_buffer_holds_the_whole_no_tools_set(self):
        """Not a difference against the shared set, and not the [0] placeholder
        (which masks token id 0 and looks armed). Carrying the whole set is what
        keeps the placeholder unreachable while a row is armed: with the FSM
        active this set always holds <|think:start|> at least, so it is never
        empty. A difference-set buffer lost that property."""
        self.assertEqual(
            self.ep.fsm_content_notools_forbid_buf.tolist(),
            list(fsm.CFG.content_done_forbidden_notools),
        )
        self.assertEqual(self.ep.fsm_content_notools_forbid_buf.dtype, torch.long)
        self.assertIsNot(
            self.ep.fsm_content_notools_row_buf, self.ep.fsm_content_row_buf
        )
        self.assertEqual(
            self.ep.fsm_content_notools_row_buf.shape, (self.max_bs * self.stride,)
        )
        self.assertFalse(self.ep.fsm_content_notools_row_buf.any())

    def test_a_notools_row_loses_the_opener_and_the_shared_set(self):
        """The layering, measured on the logits: the shared content mask and the
        opener both land on a no-tools row. Delete the third apply_folded_mask
        call and only the opener survives here."""
        logits = self._logits()
        self.ep.set_fsm_content_rows([True] * self.stride + [False] * self.stride)
        self.ep.set_fsm_content_notools_rows(
            [True] * self.stride + [False] * self.stride
        )
        self.ep._apply_fsm_mask(self.max_bs)
        self.assertTrue(torch.isneginf(logits[0, THINK_START]))
        self.assertTrue(torch.isneginf(logits[0, TOOL_START]))
        # And the tool-call internals, which the shared buffer drops for the
        # sake of a chain that may legally open a call. A no-tools chain cannot.
        for tid in TOOL_INTERNALS:
            self.assertTrue(torch.isneginf(logits[0, tid]), tid)
        # The tools-available row keeps both open: it may be inside a real call.
        self.assertFalse(torch.isneginf(logits[self.stride, TOOL_START]))
        for tid in TOOL_INTERNALS:
            self.assertFalse(torch.isneginf(logits[self.stride, tid]), tid)

    def test_the_notools_setter_stages_exactly_the_flags(self):
        flags = [True, False, True, False, False, False, False, True]
        self.ep.set_fsm_content_notools_rows(flags)
        self.assertEqual(self.ep.fsm_content_notools_row_buf.tolist(), flags)
        self.ep.set_fsm_content_notools_rows(None)
        self.assertFalse(self.ep.fsm_content_notools_row_buf.any())

    def test_the_notools_setter_does_not_touch_the_other_buffers(self):
        """Three staging calls into three buffers; a shared one would alias the
        masks and this is the cheapest way to see it."""
        self.ep.set_fsm_content_notools_rows([True] * self.stride * self.max_bs)
        self.assertFalse(self.ep.fsm_row_buf.any())
        self.assertFalse(self.ep.fsm_content_row_buf.any())

    def test_a_fresh_notools_id_snapshot_is_not_reported(self):
        """The pair for the staleness test below: a detector that logs
        unconditionally passes that one and would cry wolf on every engine."""
        with self.assertNoLogs(
            "sglang.srt.speculative.dspark_components.dspark_verify", level="ERROR"
        ):
            self.ep.set_fsm_content_notools_rows([True] * self.stride)

    def test_the_notools_staleness_check_sees_a_config_that_moved(self):
        """The realistic shape, matching the CONTENT sibling: the CFG changes
        after construction rather than the buffer being swapped by hand."""
        fsm.CFG.content_done_forbidden_notools = (THINK_START, TOOL_START, 42)
        with self.assertLogs(
            "sglang.srt.speculative.dspark_components.dspark_verify", level="ERROR"
        ) as cm:
            self.ep.set_fsm_content_notools_rows([True] * self.stride)
        self.assertIn("no-tools", "\n".join(cm.output))

    def test_the_worker_arms_the_notools_mask_with_the_notools_flags(self):
        """Source-shape guard, like the CONTENT one: the call is invisible to
        every behavioural test in this file if it is simply absent."""
        tree = ast.parse(inspect.getsource(dspark_worker_v2))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "set_fsm_content_notools_rows"
        ]
        self.assertEqual(len(calls), 1, "the no-tools mask must be armed once")
        arg = calls[0].args[0]
        self.assertIsInstance(arg, ast.Call)
        self.assertEqual(arg.func.attr, "folded_content_notools_mask_flags")


if __name__ == "__main__":
    unittest.main()
