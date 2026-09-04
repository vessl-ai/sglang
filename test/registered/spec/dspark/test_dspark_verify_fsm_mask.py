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
    _fsm_content_forbidden_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_LOG = "sglang.srt.speculative.dspark_components.dspark_verify"
# (staging setter, the flag function the worker must hand it)
_MASKS = (
    ("set_fsm_rows", "folded_mask_flags"),
    ("set_fsm_content_rows", "folded_content_mask_flags"),
    ("set_fsm_content_notools_rows", "folded_content_notools_mask_flags"),
)

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
    # configure_ids puts every sentinel in all_controls; a fixture that lists
    # fewer than its forbidden tables describes a world it cannot build.
    fsm.CFG.all_controls = frozenset(
        {THINK_START, THINK_END, TOOL_START, *TOOL_INTERNALS}
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

    def _epilogue(self):
        """A fresh epilogue, which snapshots the forbid buffers off the CFG
        _cfg() has already set."""
        return DsparkVerifyEpilogue(
            max_bs=self.max_bs,
            verify_num_draft_tokens=self.stride,
            device="cpu",
            tp_sync=DsparkTpSync(SimpleNamespace(world_size=1)),
        )

    def setUp(self):
        self._saved = {k: getattr(fsm.CFG, k) for k in _CFG_FIELDS}
        _cfg()
        self.ep = self._epilogue()

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
        self.ep.set_fsm_rows([True] * self.stride * self.max_bs)
        self.ep.set_fsm_content_rows([False] * self.stride * self.max_bs)
        self.assertTrue(self.ep.fsm_row_buf.all())

    def test_a_content_armed_row_gets_the_content_set_to_neg_inf(self):
        """The commit's whole claim, end to end. Delete the second
        apply_folded_mask call in _apply_fsm_mask and this is what fails."""
        logits = self._logits()
        # rows 0..3 = request 0 (REASONING), rows 4..7 = request 1 (CONTENT).
        self.ep.set_fsm_rows([True] * self.stride + [False] * self.stride)
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
        self.ep.set_fsm_rows([True] * n)
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
        with self.assertLogs(_LOG, level="ERROR") as logs:
            self.ep.set_fsm_content_rows([True] * self.stride)
        self.assertTrue(any("CONTENT mask holds" in m for m in logs.output))

    def test_a_disarming_call_does_not_burn_the_check(self):
        """The check fires once. If a None call spends it, the first step of a
        process with the FSM off silences every later divergence -- and that is
        the common case, since flags are None whenever the FSM is inactive."""
        self.ep.set_fsm_content_rows(None)
        fsm.CFG.content_done_forbidden = (THINK_START, 55)
        with self.assertLogs(_LOG, level="ERROR") as logs:
            self.ep.set_fsm_content_rows([True] * self.stride)
        self.assertTrue(any("CONTENT mask holds" in m for m in logs.output))

    def test_the_worker_arms_each_mask_from_its_own_flags(self):
        """Source-shape guard for all three. A setter the worker never calls is
        invisible to every behavioural test here, and handing one the wrong
        flags is a live copy-paste mutant -- the content flags on the reasoning
        setter puts EOS out of reach once the answer is done, which is the shape
        this port exists to fix."""
        self.assertEqual(len(_MASKS), 3, "a mask lost its guards with its row")
        calls = [
            n
            for n in ast.walk(ast.parse(inspect.getsource(dspark_worker_v2)))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]
        for setter, flags in _MASKS:
            with self.subTest(setter):
                armed = [c for c in calls if c.func.attr == setter]
                self.assertEqual(len(armed), 1, f"{setter} must be armed once")
                arg = armed[0].args[0]
                self.assertIsInstance(arg, ast.Call)
                self.assertEqual(arg.func.attr, flags)

    def test_the_worker_refuses_to_fold_a_sampling_batch(self):
        """Leg A of what keeps the folded path greedy. The epilogue accepts by
        argmax alone, so a batch that wants sampling must not reach it.

        Structural because fold_eligible is an inline expression with no handle
        to call, and because a grep survives a negation.
        """
        worker = ast.parse(inspect.getsource(dspark_worker_v2))
        binds = [
            n
            for n in ast.walk(worker)
            if isinstance(n, (ast.Assign, ast.AugAssign, ast.NamedExpr))
            and any(
                isinstance(t, ast.Name) and t.id == "fold_eligible"
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
            )
        ]
        self.assertEqual(len(binds), 1, "fold_eligible is bound once")
        self.assertIsInstance(binds[0], ast.Assign, "and not amended after")
        conj = binds[0].value
        self.assertIsInstance(conj, ast.BoolOp)
        self.assertIsInstance(conj.op, ast.And)
        greedy = [v for v in conj.values if "is_all_greedy" in ast.dump(v)]
        self.assertEqual(len(greedy), 1, "one conjunct tests it")
        # `x is None or x.is_all_greedy` and nothing else: a third disjunct
        # reads the flag without requiring it, and a negation or comparison
        # inverts what it requires.
        self.assertIsInstance(greedy[0], ast.BoolOp)
        self.assertIsInstance(greedy[0].op, ast.Or)
        self.assertEqual(len(greedy[0].values), 2, "no escape hatch disjunct")
        for n in ast.walk(greedy[0]):
            if isinstance(n, ast.UnaryOp) or (
                isinstance(n, ast.Compare) and "is_all_greedy" in ast.dump(n)
            ):
                self.fail("the greedy test is negated or compared, not required")

    def test_the_in_graph_mask_is_not_gated(self):
        """It fires on every compact step, and `VerifyPlan.apply` may write over
        the same tensor on the same one -- `plan_gate` decides whether the eager
        plan is built, not whether these kernels run. Both only write -inf, so
        the applied mask is the union.

        Putting `_apply_fsm_mask` behind a condition would look like a saving
        and would silently unmask a replayed graph, because a graph is captured
        once: the kernels have to be in it whether or not the FSM is on. It
        would also make `SOLAR_FSM_SPEC_ALWAYS_EAGER` mean what a reader expects
        it to mean, which it does not -- that flag adds the eager plan and
        cannot take this one away.
        """
        # The class, not the module: both the call and the method live in it,
        # and scoping here keeps an unrelated helper elsewhere in the file from
        # answering for the one the epilogue runs.
        epilogue = ast.parse(inspect.getsource(DsparkVerifyEpilogue))
        calls = [
            n
            for n in ast.walk(epilogue)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_apply_fsm_mask"
        ]
        self.assertEqual(len(calls), 1, "_apply_fsm_mask is called once")
        guarded = [
            n
            for n in ast.walk(epilogue)
            if isinstance(n, (ast.If, ast.IfExp))
            and any(
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and c.func.attr == "_apply_fsm_mask"
                for c in ast.walk(n)
            )
        ]
        self.assertEqual(
            guarded, [], "_apply_fsm_mask must not sit under a condition"
        )

    def test_the_fold_gate_requires_the_fsm_to_stand_down(self):
        """Leg B. `plan_gate` decides, per step, that the committed state is
        too close to a boundary for the in-graph masks to be right -- a spent
        budget, fresh CONTENT, a tool state. That verdict only binds if
        `fold_eligible` requires it; delete the conjunct and every one of those
        steps folds anyway, wearing a mask built for the state it just left.

        Structural for the same reason as leg A: `fold_eligible` is an inline
        expression, and a grep survives a negation.
        """
        worker = ast.parse(inspect.getsource(dspark_worker_v2))
        binds = [
            n.value
            for n in ast.walk(worker)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "fold_eligible" for t in n.targets
            )
        ]
        self.assertEqual(len(binds), 1, "fold_eligible is bound once")
        conj = binds[0]
        self.assertIsInstance(conj, ast.BoolOp)
        self.assertIsInstance(conj.op, ast.And)
        gated = [
            v
            for v in conj.values
            if isinstance(v, ast.UnaryOp)
            and isinstance(v.op, ast.Not)
            and isinstance(v.operand, ast.Name)
            and v.operand.id == "_solar_fsm_gate"
        ]
        self.assertEqual(
            len(gated),
            1,
            "fold_eligible must carry exactly `and not _solar_fsm_gate`; the "
            "gate is advisory without it",
        )

    def test_the_folded_accept_is_gated_on_fold_eligible(self):
        """Leg A decides whether a batch may fold; this is where that decision
        is used. Both legs pass with `fold_eligible and` deleted from here --
        the flag would still be computed correctly and the epilogue would still
        hold only the greedy accept, while every sampling batch took it."""
        worker = ast.parse(inspect.getsource(dspark_worker_v2))
        gates = [
            n.value
            for n in ast.walk(worker)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "folded_accept" for t in n.targets
            )
        ]
        self.assertEqual(len(gates), 1, "folded_accept is assigned once")
        self.assertIsInstance(gates[0], ast.BoolOp)
        self.assertIsInstance(gates[0].op, ast.And)
        self.assertIn(
            "fold_eligible",
            [v.id for v in gates[0].values if isinstance(v, ast.Name)],
            "the folded accept must require fold_eligible",
        )

    def test_the_epilogue_has_no_accept_but_the_greedy_one(self):
        """Leg B. Scoped to the class -- the eager path in the same module has
        its own accept and is not folded. Reached by name, too: a computed
        callee leaves nothing to read while the accept changes underneath.
        """
        epilogue = ast.parse(inspect.getsource(DsparkVerifyEpilogue))
        callees = [
            (n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", ""))
            for n in ast.walk(epilogue)
            if isinstance(n, ast.Call)
        ]
        self.assertEqual(
            [c for c in callees if c.startswith("accept_")],
            ["accept_greedy_triton"],
        )
        self.assertNotIn("getattr", callees, "no dynamic dispatch in here")
        for n in ast.walk(epilogue):
            if isinstance(n, ast.Call):
                self.assertIsInstance(n.func, (ast.Name, ast.Attribute))

    def test_a_fresh_id_snapshot_is_not_reported(self):
        """The negative control for all three divergence checks: a detector that
        logs unconditionally passes the staleness tests and cries wolf on every
        engine."""
        for setter, _ in _MASKS:
            # A fresh epilogue each, as the three separate tests had: sharing one
            # makes subtests 2 and 3 pass vacuously under a mutation that
            # collapses the per-label one-shot into a per-process one.
            ep = self._epilogue()
            with self.subTest(setter), self.assertNoLogs(_LOG, level="ERROR"):
                getattr(ep, setter)([True] * self.stride)

    def test_a_content_set_of_only_tool_internals_refuses_to_boot(self):
        """The placeholder here is not the safe one. The flags never consult
        this set, so a CONTENT set the subtraction empties leaves rows armed
        against token id 0 with every sentinel open -- for the life of the
        process, since the buffer is captured into the verify graph. The
        sibling helper keeps a misconfiguration raising for the same reason;
        this must too, or the only symptom is wrong output nobody can trace."""
        fsm.CFG.content_done_forbidden = TOOL_INTERNALS
        with self.assertRaises(RuntimeError) as caught:
            _fsm_content_forbidden_ids()
        self.assertIn("token id 0", str(caught.exception))

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
        with self.assertLogs(_LOG, level="ERROR") as logs:
            self.ep.set_fsm_rows(rows)
            self.ep.set_fsm_content_rows(rows)
            self.ep.set_fsm_content_notools_rows(rows)
        self.assertEqual(len(self.ep._ids_checked), 3, self.ep._ids_checked)
        self.assertEqual(len(logs.output), 3, logs.output)

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
        # Both rows carry the shared mask; only the first is no-tools. Giving
        # the two setters the same flags would leave row `stride` armed by
        # neither, and then every assertion below about it holds vacuously --
        # including under a mutation that points the third mask at the shared
        # row buffer, which would mask the opener on every tools request.
        self.ep.set_fsm_content_rows([True] * self.stride * self.max_bs)
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
        # ...while still carrying the shared set, which is what makes the row a
        # control for the no-tools mask rather than an unarmed row.
        self.assertTrue(torch.isneginf(logits[self.stride, THINK_START]))

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

    def test_the_notools_staleness_check_sees_a_config_that_moved(self):
        """The realistic shape, matching the CONTENT sibling: the CFG changes
        after construction rather than the buffer being swapped by hand."""
        fsm.CFG.content_done_forbidden_notools = (THINK_START, TOOL_START, 42)
        with self.assertLogs(_LOG, level="ERROR") as cm:
            self.ep.set_fsm_content_notools_rows([True] * self.stride)
        self.assertIn("no-tools", "\n".join(cm.output))


if __name__ == "__main__":
    unittest.main()
