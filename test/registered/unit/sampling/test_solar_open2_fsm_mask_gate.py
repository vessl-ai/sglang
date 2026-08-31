"""The mask gate must fire for any row the mask would touch.

``plan_gate`` decides two things at once, and they are one decision: the folded
in-graph accept path cannot take a mask written into ``next_token_logits``, so a
step that needs a mask must run eager. The caller
(``dspark_worker_v2``) reads the same boolean to decide whether to call
``plan_verify`` at all, so a gate narrower than "would any row be masked?"
leaves reasoning rows unmasked on the steps it excludes -- and an unmasked
reasoning row can emit EOS mid-think, which is the failure the mask exists to
prevent.

Pure CPU: ``plan_gate`` reads ``CFG`` and duck-typed request attributes only.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.sampling import solar_open2_fsm as fsm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

THINK_START, THINK_END, EOS = 100, 101, 2


def _cfg(**over):
    fsm.CFG.enabled = True
    fsm.CFG.think_start, fsm.CFG.think_end = THINK_START, THINK_END
    fsm.CFG.all_controls = frozenset({THINK_START, THINK_END})
    fsm.CFG.reasoning_forbidden = (EOS,)
    fsm.CFG.content_mask = False
    fsm.CFG.spec_always_eager = False
    fsm.CFG.budget_abs, fsm.CFG.budget_ratio = 3072, 0.75
    for k, v in over.items():
        setattr(fsm.CFG, k, v)


def _req(output_ids, *, in_think=True, max_new_tokens=4096, primed=True):
    """A request whose FSM state is already built, which is what plan_gate
    judges. Without priming it takes the "no committed state yet" branch and
    gates unconditionally, so an unprimed fixture cannot measure the predicate
    under test — see test_no_committed_state_gates for that branch itself.
    """
    prompt = [1, 2, 3] + ([THINK_START] if in_think else [])
    req = SimpleNamespace(
        rid="r0",
        retraction_count=0,
        origin_input_ids=prompt,
        output_ids=list(output_ids),
        sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
    )
    if primed:
        fsm._req_fsm(req).advance(req.output_ids)
    return req


class TestSolarFsmMaskGate(unittest.TestCase):
    def test_reasoning_row_far_from_the_budget_still_gates(self):
        """The case the old budget-window predicate answered wrongly: 44 tokens
        into a 3072-token budget is nowhere near the boundary, and the EOS ban
        is exactly as necessary there as at the boundary."""
        _cfg()
        req = _req([7] * 44)
        f = fsm._req_fsm(req)
        f.advance(req.output_ids)
        self.assertTrue(f.in_reasoning, "fixture must be in REASONING")
        self.assertIn(EOS, fsm.CFG.reasoning_forbidden)
        self.assertTrue(fsm.plan_gate([req], 8))

    def test_gate_agrees_with_what_plan_verify_would_mask(self):
        """Both directions, on one fixture: the gate is True exactly when
        plan_verify produces rows, so neither can drift from the other."""
        import torch

        _cfg()
        for label, req in (
            ("reasoning", _req([7] * 44)),
            ("content", _req([7, THINK_END, 7], in_think=True)),
        ):
            with self.subTest(label):
                gated = fsm.plan_gate([req], 4)
                plan = fsm.plan_verify([req], torch.tensor([[7] * 4]), 4)
                produces = bool(plan and (plan.mask_rows or plan.force_rows))
                self.assertEqual(
                    gated,
                    produces,
                    f"{label}: gate={gated} but plan_verify produced={produces}",
                )

    def test_content_row_does_not_gate_unless_content_mask_is_on(self):
        _cfg()
        req = _req([7, THINK_END, 7])
        self.assertFalse(fsm.plan_gate([req], 8))
        _cfg(content_mask=True)
        req = _req([7, THINK_END, 7])
        self.assertTrue(fsm.plan_gate([req], 8))

    def test_one_reasoning_row_gates_the_whole_batch(self):
        """The mask is per-row but the fold is per-step, so any masked row
        forces the whole step eager."""
        _cfg()
        batch = [_req([7, THINK_END, 7]), _req([7] * 44)]
        self.assertTrue(fsm.plan_gate(batch, 8))

    def test_no_committed_state_gates(self):
        _cfg()
        req = _req([], primed=False)
        self.assertTrue(fsm.plan_gate([req], 8))

    def test_inactive_never_gates(self):
        _cfg(enabled=False)
        try:
            req = _req([7] * 44)
            self.assertFalse(fsm.plan_gate([req], 8))
        finally:
            _cfg()


if __name__ == "__main__":
    unittest.main()
