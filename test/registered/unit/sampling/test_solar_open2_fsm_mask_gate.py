"""Every reasoning row is masked, on whichever accept path the step takes.

The reasoning mask forbids the EOS ids while a row is inside the think block.
An unmasked reasoning row can emit EOS mid-think; the block never closes, the
parser has no ``<|think:end|>`` to split on, and the whole output comes back as
reasoning with an empty answer. That is the customer-reported shape.

Two mechanisms cover it, and the invariant is that together they leave no gap:

* ``plan_gate`` sends the step to the eager path, where ``plan_verify`` writes
  the mask. It fires only for what the eager path is *needed* for -- a forced
  ``<|think:end|>`` at a spent budget, and the ``content_mask`` sets -- because
  forcing eager on every thinking step would cost the folded in-graph accept
  for most of a generation.
* ``folded_mask_flags`` carries the reasoning mask into the graph instead, one
  flag per (request, chain position) row.

``test_no_reasoning_row_is_left_unmasked`` is the pairing: for every row
``plan_verify`` would mask with ``reasoning_forbidden``, either the gate fired
or the flag is set. The two cannot drift apart without that failing.

Pure CPU: both predicates read ``CFG`` and duck-typed request attributes only.
"""

import os
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
    def test_reasoning_row_far_from_the_budget_is_flagged(self):
        """The defect's own case. 44 tokens into a 3072-token budget is nowhere
        near the boundary, so the gate correctly declines -- nothing there needs
        the eager path -- and the flag is what must carry the EOS ban."""
        _cfg()
        req = _req([7] * 44)
        f = fsm._req_fsm(req)
        f.advance(req.output_ids)
        self.assertTrue(f.in_reasoning, "fixture must be in REASONING")
        self.assertIn(EOS, fsm.CFG.reasoning_forbidden)
        self.assertFalse(fsm.plan_gate([req], 8), "nothing here needs eager")
        self.assertEqual(fsm.folded_mask_flags([req], 8), [True] * 8)

    def test_no_reasoning_row_is_left_unmasked(self):
        """The pairing invariant, over the state space that matters: whatever
        plan_verify would mask as a reasoning row is covered by the gate or by
        the flag. A gap here is the defect."""
        import torch

        for label, req, stride in (
            ("early reasoning", _req([7] * 44), 8),
            ("reasoning at the budget boundary", _req([7] * 3070), 8),
            ("content", _req([7, THINK_END, 7]), 4),
            ("fresh reasoning", _req([7]), 4),
        ):
            with self.subTest(label):
                _cfg()
                gated = fsm.plan_gate([req], stride)
                flags = fsm.folded_mask_flags([req], stride) or []
                plan = fsm.plan_verify([req], torch.tensor([[7] * stride]), stride)
                masked = set()
                if plan:
                    for ids, rows in plan.mask_rows.items():
                        if ids == fsm.CFG.reasoning_forbidden:
                            masked.update(rows)
                for row in masked:
                    self.assertTrue(
                        gated or (row < len(flags) and flags[row]),
                        f"{label}: row {row} would be masked by plan_verify but "
                        f"the step is not eager (gate={gated}) and the in-graph "
                        f"flag is not set",
                    )

    def test_a_spent_budget_still_takes_the_eager_path(self):
        """The force is plan_verify's alone -- nothing in the graph writes
        <|think:end|> -- so the gate must still fire near the boundary, and the
        flag must not claim that row."""
        _cfg()
        req = _req([7] * 3070)
        self.assertTrue(fsm.plan_gate([req], 8))

    def test_content_row_is_not_flagged(self):
        _cfg()
        req = _req([7, THINK_END, 7])
        self.assertEqual(fsm.folded_mask_flags([req], 8), [False] * 8)
        self.assertFalse(fsm.plan_gate([req], 8))

    def test_content_mask_still_takes_the_eager_path(self):
        """The content sets have no in-graph carrier, so content_mask must gate
        unconditionally or those rows go unmasked."""
        _cfg(content_mask=True)
        req = _req([7, THINK_END, 7])
        self.assertTrue(fsm.plan_gate([req], 8))

    def test_flags_are_per_request_not_per_batch(self):
        """The fold is per-step but the mask is per-row: one thinking request
        beside one answering request must flag only its own rows."""
        _cfg()
        batch = [_req([7, THINK_END, 7]), _req([7] * 44)]
        self.assertEqual(fsm.folded_mask_flags(batch, 4), [False] * 4 + [True] * 4)

    def test_no_committed_state_gates_and_is_not_flagged(self):
        """plan_gate sends it eager, where plan_verify judges it with fresh
        state; a flag from stale state would be a guess."""
        _cfg()
        req = _req([], primed=False)
        self.assertTrue(fsm.plan_gate([req], 8))
        self.assertEqual(fsm.folded_mask_flags([req], 8), [False] * 8)

    def test_inactive_never_gates_and_never_flags(self):
        # is_active() resolves lazily and re-enables itself from SOLAR_FSM=1, so
        # clearing CFG.enabled alone measures nothing where that variable is set
        # -- which is every engine pod. Clear the environment too, and build the
        # fixture first, since _req_fsm is one of the calls that re-resolves.
        _cfg()
        req = _req([7] * 44)
        prev = os.environ.get("SOLAR_FSM")
        os.environ["SOLAR_FSM"] = "0"
        fsm.CFG.enabled = False
        try:
            self.assertFalse(fsm.plan_gate([req], 8))
            self.assertIsNone(fsm.folded_mask_flags([req], 8))
        finally:
            if prev is None:
                os.environ.pop("SOLAR_FSM", None)
            else:
                os.environ["SOLAR_FSM"] = prev
            _cfg()

    def test_spec_always_eager_still_forces_the_eager_path(self):
        _cfg(spec_always_eager=True)
        req = _req([7] * 44)
        self.assertTrue(fsm.plan_gate([req], 8))


if __name__ == "__main__":
    unittest.main()
