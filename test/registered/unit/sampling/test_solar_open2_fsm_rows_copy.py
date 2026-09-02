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
    fsm.CFG.reasoning_forbidden = (EOS,)
    fsm.CFG.content_mask = False
    fsm.CFG.spec_always_eager = False
    fsm.CFG.budget_abs, fsm.CFG.budget_ratio = 3072, 0.75
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


class TestSolarFsmRowsSurviveCopy(unittest.TestCase):
    def setUp(self):
        _cfg()

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

    def test_merge_keeps_rows_when_one_side_has_none(self):
        info = _minimal_sampling_info()
        info.solar_fsm_rows = ["a"]
        other = _minimal_sampling_info()
        fsm.merge_rows(info, other)
        self.assertEqual(info.solar_fsm_rows, ["a"])
        info2 = _minimal_sampling_info()
        other2 = _minimal_sampling_info()
        other2.solar_fsm_rows = ["b"]
        fsm.merge_rows(info2, other2)
        self.assertEqual(info2.solar_fsm_rows, ["b"])


if __name__ == "__main__":
    unittest.main()
