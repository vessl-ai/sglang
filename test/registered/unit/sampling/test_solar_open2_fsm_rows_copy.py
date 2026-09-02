"""The Solar-Open2 FSM hangs its per-row ``Req`` handles off ``SamplingBatchInfo``
as ``solar_fsm_rows``. The scheduler hands the sampler a forward-only copy made
with ``dataclasses.replace`` (``copy_for_forward``), which rebuilds the object
from its declared fields: an ad-hoc attribute does not survive it, and
``solar_open2_fsm.apply`` then returns before masking on every non-speculative
sampler step. These tests pin the field down so that copy keeps the rows."""

import dataclasses
import unittest

import torch

from sglang.srt.sampling import solar_open2_fsm
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


def _minimal_sampling_info() -> SamplingBatchInfo:
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


class TestSolarFsmRowsSurviveCopy(unittest.TestCase):
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
        info.update_penalties = lambda: None  # no orchestrator in this unit
        copied = info.copy_for_forward()
        self.assertEqual(copied.solar_fsm_rows, ["req-a"])
        self.assertIsNone(copied.penalizer_orchestrator)

    def test_filter_and_merge_keep_working_on_the_field(self):
        info = _minimal_sampling_info()
        info.solar_fsm_rows = ["a", "b", "c"]
        solar_open2_fsm.filter_rows(info, [0, 2])
        self.assertEqual(info.solar_fsm_rows, ["a", "c"])
        other = _minimal_sampling_info()
        other.solar_fsm_rows = ["d"]
        solar_open2_fsm.merge_rows(info, other)
        self.assertEqual(info.solar_fsm_rows, ["a", "c", "d"])


if __name__ == "__main__":
    unittest.main()
