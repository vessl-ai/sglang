"""The Solar-Open2 FSM keeps its per-row ``Req`` handles on ``SamplingBatchInfo``
as ``solar_fsm_rows``. Before every forward the scheduler substitutes a
forward-only copy made with ``dataclasses.replace`` (``copy_for_forward``);
that rebuilds the object from its declared fields, so an ad-hoc attribute did
not survive it and ``solar_open2_fsm.apply`` returned before masking on every
non-speculative sampler step. Pinning the rows down as a field is the fix;
the tests below check the field survives the copy *and* that the mask is
actually written afterwards, which is the behaviour that was lost.

Pure CPU: a 1-row float logits tensor and duck-typed requests.
"""

import dataclasses
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.sampling import solar_open2_fsm as fsm
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

THINK_START, THINK_END, EOS = 100, 101, 2
VOCAB = 128


def _cfg():
    fsm.CFG.enabled = True
    fsm.CFG.think_start, fsm.CFG.think_end = THINK_START, THINK_END
    fsm.CFG.all_controls = frozenset({THINK_START, THINK_END})
    fsm.CFG.transitions = {THINK_START: fsm.REASONING, THINK_END: fsm.CONTENT}
    fsm.CFG.reasoning_forbidden = (EOS,)
    fsm.CFG.leading_newline_forbidden = ()
    fsm.CFG.reasoning_open_forbidden = (EOS,)
    fsm.CFG.content_mask = False
    fsm.CFG.spec_always_eager = False
    fsm.CFG.budget_abs, fsm.CFG.budget_ratio = 3072, 0.75
    # These tests are about the rows surviving the copy, not the budget rule;
    # pin the legacy formula so the budget below is min(3072, 0.75 * 4096).
    fsm.CFG.budget_policy = "legacy"
    fsm.CFG._mask_cache.clear()


def _req(output_ids, *, max_new_tokens=4096):
    """A request inside its think block with ``output_ids`` already emitted."""
    return SimpleNamespace(
        rid="r0",
        retraction_count=0,
        origin_input_ids=[1, 2, 3, THINK_START],
        output_ids=list(output_ids),
        sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
    )


def _minimal_sampling_info() -> SamplingBatchInfo:
    """Build a SamplingBatchInfo with placeholder values for every field that
    has no default. The buckets below cover today's no-default fields (tensors,
    bools, one int); a new no-default field of another type lands in ``None``,
    so keep this in step with the dataclass if that ever changes."""
    kwargs = {}
    for field in dataclasses.fields(SamplingBatchInfo):
        if (
            field.default is not dataclasses.MISSING
            or field.default_factory is not dataclasses.MISSING
        ):
            continue
        annotation = str(field.type)
        if "Tensor" in annotation:
            kwargs[field.name] = torch.zeros(1)
        elif "bool" in annotation:
            kwargs[field.name] = False
        elif "int" in annotation:
            kwargs[field.name] = 1
        else:
            kwargs[field.name] = None
    return SamplingBatchInfo(**kwargs)


def _forward_copy(info: SamplingBatchInfo) -> SamplingBatchInfo:
    """``copy_for_forward`` as the scheduler calls it. ``update_penalties`` is
    bypassed: the minimal fixture has no penalizer orchestrator, and the copy
    itself -- ``dataclasses.replace`` -- is what these tests are about."""
    info.update_penalties = lambda: None
    return info.copy_for_forward()


_CFG_FIELDS = (
    "enabled",
    "think_start",
    "think_end",
    "all_controls",
    "transitions",
    "reasoning_forbidden",
    "content_mask",
    "spec_always_eager",
    "budget_abs",
    "budget_ratio",
    "budget_policy",
    "leading_newline_forbidden",
    "reasoning_open_forbidden",
)


class TestSolarFsmRowsSurviveCopy(unittest.TestCase):
    def setUp(self):
        # CFG and _WARNED are module-global; restoring them in tearDown keeps
        # this file from leaving a live FSM (or a spent once-only warning)
        # behind for the other sampler suites in the same process.
        self._saved = {k: getattr(fsm.CFG, k) for k in _CFG_FIELDS}
        self._saved_warned = dict(fsm._WARNED)
        _cfg()

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(fsm.CFG, k, v)
        fsm._WARNED.clear()
        fsm._WARNED.update(self._saved_warned)
        fsm.CFG._mask_cache.clear()

    def test_rows_is_a_declared_field(self):
        names = {f.name for f in dataclasses.fields(SamplingBatchInfo)}
        self.assertIn("solar_fsm_rows", names)

    def test_replace_keeps_rows(self):
        info = _minimal_sampling_info()
        info.solar_fsm_rows = ["req-a", "req-b"]
        copied = dataclasses.replace(info, penalizer_orchestrator=None)
        self.assertEqual(copied.solar_fsm_rows, ["req-a", "req-b"])

    def test_copy_for_forward_keeps_rows(self):
        info = _minimal_sampling_info()
        info.solar_fsm_rows = ["req-a"]
        copied = _forward_copy(info)
        self.assertEqual(copied.solar_fsm_rows, ["req-a"])
        self.assertIsNone(copied.penalizer_orchestrator)

    def test_mask_is_written_on_the_forward_copy(self):
        """The behaviour that was lost: after the copy, a reasoning row still
        gets EOS forbidden by ``apply``."""
        info = _minimal_sampling_info()
        info.solar_fsm_rows = [_req([10, 11, 12])]
        copied = _forward_copy(info)
        logits = torch.zeros(1, VOCAB)
        fsm.apply(logits, copied)
        self.assertEqual(logits[0, EOS].item(), float("-inf"))
        self.assertEqual(logits[0, 7].item(), 0.0)

    def test_budget_force_is_written_on_the_forward_copy(self):
        """With the reasoning budget spent, ``apply`` on the copy forces
        ``<|think:end|>`` -- everything else goes to -inf."""
        info = _minimal_sampling_info()
        budget = int(4096 * fsm.CFG.budget_ratio)
        info.solar_fsm_rows = [_req([10] * budget)]
        copied = _forward_copy(info)
        logits = torch.zeros(1, VOCAB)
        fsm.apply(logits, copied)
        self.assertEqual(logits[0, THINK_END].item(), 0.0)
        self.assertEqual(logits[0, EOS].item(), float("-inf"))
        self.assertEqual(logits[0, 7].item(), float("-inf"))

    def test_attach_rows_reaches_the_sampler_copy(self):
        """End to end through the real attach helper: rows attached at schedule
        time are the rows the sampler-side copy masks from."""
        req = _req([10, 11])
        info = _minimal_sampling_info()
        with mock.patch.dict(os.environ, {"SOLAR_FSM": "1"}):
            fsm.attach_rows(info, SimpleNamespace(reqs=[req]))
        copied = _forward_copy(info)
        self.assertEqual(copied.solar_fsm_rows, [req])
        logits = torch.zeros(1, VOCAB)
        fsm.apply(logits, copied)
        self.assertEqual(logits[0, EOS].item(), float("-inf"))

    def test_missing_rows_is_loud_once(self):
        info = _minimal_sampling_info()
        self.assertIsNone(info.solar_fsm_rows)
        fsm._WARNED["rows"] = False
        with self.assertLogs(fsm.logger, level="WARNING") as captured:
            fsm.apply(torch.zeros(1, VOCAB), info)
        self.assertTrue(any("no solar_fsm_rows" in m for m in captured.output))
        self.assertTrue(fsm._WARNED["rows"])

    def test_filter_and_merge_keep_working_on_the_field(self):
        info = _minimal_sampling_info()
        info.solar_fsm_rows = ["a", "b", "c"]
        fsm.filter_rows(info, [0, 2])
        self.assertEqual(info.solar_fsm_rows, ["a", "c"])
        other = _minimal_sampling_info()
        other.solar_fsm_rows = ["d"]
        fsm.merge_rows(info, other)
        self.assertEqual(info.solar_fsm_rows, ["a", "c", "d"])

    def test_merge_with_one_side_missing_keeps_rows_and_is_loud_once(self):
        """A batch that lost its rows must not erase the other side's rows on
        merge -- and, with the FSM on, the mismatch is reported once."""
        fsm._WARNED["merge"] = False
        for missing_side in ("self", "other"):
            with self.subTest(missing=missing_side):
                info, other = _minimal_sampling_info(), _minimal_sampling_info()
                (other if missing_side == "self" else info).solar_fsm_rows = ["a"]
                if missing_side == "self":
                    with self.assertLogs(fsm.logger, level="WARNING") as captured:
                        fsm.merge_rows(info, other)
                    self.assertIn("one side has no solar_fsm_rows", captured.output[0])
                else:
                    with self.assertNoLogs(fsm.logger, level="WARNING"):
                        fsm.merge_rows(info, other)  # warned once already
                self.assertEqual(info.solar_fsm_rows, ["a"])
        # Both sides missing is not a merge problem: nothing to warn about.
        info, other = _minimal_sampling_info(), _minimal_sampling_info()
        fsm._WARNED["merge"] = False
        with self.assertNoLogs(fsm.logger, level="WARNING"):
            fsm.merge_rows(info, other)
        self.assertIsNone(info.solar_fsm_rows)

    def test_empty_rows_under_a_nonempty_batch_is_loud_once(self):
        """Rows that fell out of step with the batch (here: an empty list under
        a one-row logits tensor) are reported, once, and the step is skipped
        rather than masked against the wrong rows."""
        info = _minimal_sampling_info()
        info.solar_fsm_rows = []
        logits = torch.zeros(1, VOCAB)
        fsm._WARNED["shape"] = False
        with self.assertLogs(fsm.logger, level="WARNING") as captured:
            fsm.apply(logits, info)
        self.assertIn("out of step", captured.output[0])
        self.assertTrue(torch.isfinite(logits).all())
        with self.assertNoLogs(fsm.logger, level="WARNING"):
            fsm.apply(logits, info)
        # A genuinely empty batch stays silent.
        with self.assertNoLogs(fsm.logger, level="WARNING"):
            fsm.apply(torch.zeros(0, VOCAB), info)

    def _orchestrator_stub(self):
        return SimpleNamespace(
            filter=lambda keep_indices_device: None,
            merge=lambda other: None,
            is_required=False,
        )

    def test_filter_batch_keeps_rows_in_step(self):
        """Through the real ``SamplingBatchInfo.filter_batch``, not just the
        helper: the rows list is filtered with the same indices as the
        per-row tensors."""
        info = _minimal_sampling_info()
        info.penalizer_orchestrator = self._orchestrator_stub()
        for item in ("temperatures", "top_ps", "top_ks", "min_ps"):
            setattr(info, item, torch.tensor([0.1, 0.2, 0.3]))
        reqs = [_req([10]), _req([11]), _req([12])]
        info.solar_fsm_rows = list(reqs)
        info.filter_batch([0, 2], torch.tensor([0, 2], dtype=torch.long))
        self.assertEqual(info.solar_fsm_rows, [reqs[0], reqs[2]])
        self.assertEqual(len(info), 2)

    def test_merge_batch_keeps_rows_in_step(self):
        """Through the real ``SamplingBatchInfo.merge_batch``: self's rows
        first, then other's, matching the tensor concatenation order."""
        left, right = _minimal_sampling_info(), _minimal_sampling_info()
        for info, temps in ((left, [0.1, 0.2]), (right, [0.3])):
            info.penalizer_orchestrator = self._orchestrator_stub()
            info.temperatures = torch.tensor(temps)
        lreqs, rreqs = [_req([10]), _req([11])], [_req([12])]
        left.solar_fsm_rows, right.solar_fsm_rows = list(lreqs), list(rreqs)
        left.merge_batch(right)
        self.assertEqual(left.solar_fsm_rows, lreqs + rreqs)
        self.assertEqual(len(left), 3)


if __name__ == "__main__":
    unittest.main()
