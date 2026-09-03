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

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.sampling import solar_open2_fsm as fsm
from sglang.srt.speculative.dspark_components.dspark_tp import DsparkTpSync
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DsparkVerifyEpilogue,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

THINK_START, THINK_END, EOS = 100, 101, 2
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
    "spec_always_eager",
    "effort_budgets",
    "default_effort",
    "hard_limit",
)


def _cfg():
    fsm.CFG.enabled = True
    fsm.CFG.think_start, fsm.CFG.think_end = THINK_START, THINK_END
    fsm.CFG.all_controls = frozenset({THINK_START, THINK_END})
    fsm.CFG.transitions = {THINK_START: fsm.REASONING, THINK_END: fsm.CONTENT}
    # Disjoint on purpose: it is the only way this suite can tell the two masks
    # apart by looking at the logits.
    fsm.CFG.reasoning_forbidden = (EOS,)
    fsm.CFG.reasoning_open_forbidden = (EOS,)
    fsm.CFG.leading_newline_forbidden = ()
    fsm.CFG.content_done_forbidden = (THINK_START,)
    fsm.CFG.content_fresh_forbidden = (THINK_START, EOS)
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
            list(fsm.CFG.content_done_forbidden),
        )
        self.assertEqual(
            self.ep.fsm_forbid_buf.tolist(), list(fsm.CFG.reasoning_forbidden)
        )

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


if __name__ == "__main__":
    unittest.main()
