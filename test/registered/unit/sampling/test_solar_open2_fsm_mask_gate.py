"""Every reasoning row is masked, on whichever accept path the step takes.

The reasoning mask forbids the EOS ids while a row is inside the think block.
An unmasked reasoning row can emit EOS mid-think; the block never closes, the
parser has no ``<|think:end|>`` to split on, and the whole output comes back as
reasoning with an empty answer. That is the customer-reported shape.

Two mechanisms cover it, and the invariant is that together they leave no gap:

* ``plan_gate`` sends the step to the eager path, where ``plan_verify`` writes
  the mask. It fires only for what the eager path is *needed* for -- a forced
  ``<|think:end|>`` at a spent budget, the content sets (a fresh
  CONTENT row and every step inside a tool call), and the leading-newline set
  right after ``<|think:start|>`` -- because forcing eager on every thinking
  step would cost the folded in-graph accept for most of a generation.
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
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

THINK_START, THINK_END, EOS = 100, 101, 2


def _cfg(**over):
    fsm.CFG.enabled = True
    fsm.CFG.think_start, fsm.CFG.think_end = THINK_START, THINK_END
    fsm.CFG.all_controls = frozenset({THINK_START, THINK_END})
    fsm.CFG.transitions = {THINK_START: fsm.REASONING, THINK_END: fsm.CONTENT}
    fsm.CFG.reasoning_forbidden = (EOS,)
    fsm.CFG.leading_newline_forbidden = ()
    fsm.CFG.reasoning_open_forbidden = (EOS,)
    fsm.CFG.spec_always_eager = False
    # Budgets here are pinned by hand: one effort, 3072 tokens.
    fsm.CFG.effort_budgets = {"high": 3072}
    fsm.CFG.default_effort = "high"
    fsm.CFG.hard_limit = 3072
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


class TestSolarFsmMaskGate(CustomTestCase):
    def setUp(self):
        # CFG is module-global. Configuring it here rather than inside each test
        # keeps a fixture built in a loop header -- which Python evaluates before
        # the body runs -- from being built against whatever the previous test
        # left behind, and restoring it in tearDown keeps this file from leaving
        # a live FSM behind for the other sampler suites in the same process.
        self._saved = {
            k: getattr(fsm.CFG, k)
            for k in (
                "enabled",
                "think_start",
                "think_end",
                "all_controls",
                "transitions",
                "reasoning_forbidden",
                "spec_always_eager",
                "effort_budgets",
                "default_effort",
                "hard_limit",
                "leading_newline_forbidden",
                "reasoning_open_forbidden",
            )
        }
        _cfg()

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(fsm.CFG, k, v)
        fsm.CFG._mask_cache.clear()

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
        """The pairing invariant, as an equality: the rows plan_verify would
        mask with reasoning_forbidden are exactly the rows the in-graph flag
        claims, unless the step is eager anyway. Equality, not implication —
        an all-True implementation satisfies implication and would overmask
        every answer."""
        import torch

        for label, req, stride, chain in (
            ("early reasoning", _req([7] * 44), 8, None),
            ("reasoning at the budget boundary", _req([7] * 3070), 8, None),
            ("content", _req([7, THINK_END, 7]), 4, None),
            ("fresh reasoning", _req([7]), 4, None),
            ("spent budget", _req([7] * 3072), 8, None),
            # A chain that leaves the block: plan_verify walks it and stops
            # masking at the drafted <|think:end|>, the flags cannot see it.
            ("chain drafts think_end", _req([7] * 44), 4, [7, THINK_END, 7, 7]),
        ):
            with self.subTest(label):
                gated = fsm.plan_gate([req], stride)
                flags = fsm.folded_mask_flags([req], stride) or []
                ids = chain if chain is not None else [7] * stride
                plan = fsm.plan_verify([req], torch.tensor([ids]), stride)
                masked = set()
                if plan:
                    for forbidden, rows in plan.mask_rows.items():
                        if forbidden == fsm.CFG.reasoning_forbidden:
                            masked.update(rows)
                flagged = {r for r, on in enumerate(flags) if on}
                if label in ("content", "spent budget"):
                    # Nothing to mask: content is the parser's, and a spent
                    # budget is forced rather than masked.
                    self.assertEqual(masked, set(), label)
                else:
                    self.assertTrue(
                        masked,
                        f"{label}: plan_verify masked nothing — the fixture "
                        f"never reached REASONING, so this subtest would pass "
                        f"without asserting anything",
                    )
                for row in masked:
                    self.assertTrue(
                        gated or row in flagged,
                        f"{label}: row {row} would be masked by plan_verify but "
                        f"the step is not eager (gate={gated}) and the in-graph "
                        f"flag is not set",
                    )
                if not gated and chain is None:
                    # Where the chain does not move the state, the two must
                    # agree exactly; a chain that transitions is the documented
                    # divergence and only the one-directional check applies.
                    self.assertEqual(flagged, masked, label)

    def test_flags_and_plan_verify_agree_row_for_row_across_a_batch(self):
        """Row numbering is request-major, chain-minor. At bs=1 that convention
        is indistinguishable from any other, so a batch is the only fixture
        that can catch the two sides disagreeing about it."""
        import torch

        stride = 4
        batch = [_req([7, THINK_END, 7]), _req([7] * 44)]
        flags = fsm.folded_mask_flags(batch, stride)
        self.assertEqual(len(flags), len(batch) * stride)
        plan = fsm.plan_verify(batch, torch.tensor([[7] * stride] * 2), stride)
        masked = set(plan.mask_rows.get(fsm.CFG.reasoning_forbidden, []))
        self.assertEqual({r for r, on in enumerate(flags) if on}, masked)

    def test_a_spent_budget_takes_the_eager_path_and_is_not_flagged(self):
        """The force is plan_verify's alone — nothing in the graph writes
        <|think:end|> — so a spent budget must go eager, and the flag must not
        claim the row. Overclaiming it would forbid EOS after the block closed
        and run the turn to max_tokens."""
        req = _req([7] * 3072)  # budget = min(3072, 4096*0.75) = 3072, so spent
        f = fsm._req_fsm(req)
        f.advance(req.output_ids)
        self.assertTrue(f.budget_exhausted(), "fixture must be spent")
        self.assertTrue(fsm.plan_gate([req], 8))
        self.assertEqual(fsm.folded_mask_flags([req], 8), [False] * 8)

    def test_a_row_near_the_boundary_still_gates(self):
        """The window is 2*stride wide and covers the run that has not been
        fed yet, so the step before exhaustion is eager too."""
        req = _req([7] * 3070)
        self.assertTrue(fsm.plan_gate([req], 8))

    def test_stale_state_gates_and_is_not_flagged(self):
        """A retraction leaves the FSM describing a prefix that no longer
        exists. plan_gate sends the step eager so plan_verify can rebuild it;
        a flag from that state would be a guess."""
        req = _req([7] * 44)
        req.retraction_count = 1
        self.assertTrue(fsm.plan_gate([req], 8))
        self.assertEqual(fsm.folded_mask_flags([req], 8), [False] * 8)

    def test_content_row_is_not_flagged(self):
        _cfg()
        req = _req([7, THINK_END, 7])
        self.assertEqual(fsm.folded_mask_flags([req], 8), [False] * 8)
        self.assertFalse(fsm.plan_gate([req], 8))

    def test_fresh_content_row_gates(self):
        """The content sets have no in-graph carrier, so a row whose turn has
        no content yet must go eager; once it has content the fold is kept
        (its rows then only lack the content_done set, a documented gap)."""
        _cfg()
        self.assertTrue(fsm.plan_gate([_req([7, THINK_END])], 8))
        self.assertFalse(fsm.plan_gate([_req([7, THINK_END, 7])], 8))

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


class TestApplyFoldedMask(CustomTestCase):
    """The row-wise mask itself. It is the piece the folded accept path depends
    on, and the piece a live boot cannot check: a row/column transposition or a
    broadcast mistake writes -inf somewhere plausible and the engine keeps
    serving."""

    def setUp(self):
        import torch

        self.torch = torch
        self.stride, self.vocab = 4, 128  # wide enough for THINK_START/THINK_END
        self.forbid = torch.tensor([2, 5], dtype=torch.long)

    def _logits(self, rows):
        # logits[r, c] = r*100 + c, so a transposition is visible in the values
        t = self.torch.arange(rows * self.vocab, dtype=self.torch.float32)
        return (t.view(rows, self.vocab) % self.vocab) + (
            self.torch.arange(rows, dtype=self.torch.float32).unsqueeze(1) * 100
        )

    def test_only_the_flagged_rows_and_forbidden_columns_move(self):
        rows = 2 * self.stride
        before = self._logits(rows)
        after = before.clone()
        flags = self.torch.tensor([False] * self.stride + [True] * self.stride)
        fsm.apply_folded_mask(after, flags, self.forbid)

        self.assertTrue(
            self.torch.isinf(after[self.stride :, self.forbid]).all(),
            "every flagged row's forbidden ids must be -inf",
        )
        self.assertTrue(
            self.torch.equal(after[: self.stride], before[: self.stride]),
            "an unflagged row must not move at all",
        )
        keep = [c for c in range(self.vocab) if c not in (2, 5)]
        self.assertTrue(
            self.torch.equal(after[:, keep], before[:, keep]),
            "no column outside the forbidden set may move",
        )

    def test_an_all_false_buffer_writes_nothing(self):
        """The claim the unarmed step rests on, and with it every deployment
        that does not run this model."""
        before = self._logits(self.stride)
        after = before.clone()
        flags = self.torch.zeros(self.stride, dtype=self.torch.bool)
        fsm.apply_folded_mask(
            after, flags, self.torch.tensor([0], dtype=self.torch.long)
        )
        self.assertTrue(self.torch.equal(after, before))

    def test_dtype_is_preserved(self):
        """A float32 -inf operand would promote and then fail the index_put_."""
        for dtype in (self.torch.float16, self.torch.bfloat16, self.torch.float32):
            with self.subTest(str(dtype)):
                logits = self._logits(self.stride).to(dtype)
                flags = self.torch.ones(self.stride, dtype=self.torch.bool)
                fsm.apply_folded_mask(logits, flags, self.forbid)
                self.assertEqual(logits.dtype, dtype)
                self.assertTrue(self.torch.isinf(logits[:, self.forbid]).all())

    def test_think_end_is_never_masked(self):
        """plan_verify's force branch reads whether <|think:end|> is still
        finite before forcing it, so this mask must leave that column alone —
        it is what ties the two mechanisms together."""
        _cfg()
        logits = self._logits(self.stride)
        flags = self.torch.ones(self.stride, dtype=self.torch.bool)
        forbid = self.torch.tensor(
            list(fsm.CFG.reasoning_forbidden), dtype=self.torch.long
        )
        self.assertNotIn(THINK_END, fsm.CFG.reasoning_forbidden)
        fsm.apply_folded_mask(logits, flags, forbid)
        self.assertTrue(self.torch.isfinite(logits[:, THINK_END]).all())


if __name__ == "__main__":
    unittest.main()
