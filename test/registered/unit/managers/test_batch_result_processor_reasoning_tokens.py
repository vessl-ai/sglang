"""Unit tests for reasoning-token usage accounting in the result processor.

`_maybe_update_reasoning_tokens` runs after `update_finish_state`, so tokens
past `finished_len` — a stop mid-verify-run, or the max_new_tokens cap — count
for neither completion (`output_ids_through_stop`) nor reasoning: both usage
numbers share one basis.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

THINK_END_IDS = [7, 8]


def _make_processor() -> SchedulerBatchResultProcessor:
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(enable_metrics=False),
        model_config=SimpleNamespace(think_end_ids=THINK_END_IDS),
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=SimpleNamespace(),
        draft_worker=None,
        model_worker=SimpleNamespace(on_verify_complete_cpu=lambda *a, **k: None),
        logprob_result_processor=None,
        output_streamer=SimpleNamespace(),
        abort_request=lambda *a, **k: None,
    )


def _make_req() -> Req:
    sp = SamplingParams(max_new_tokens=256, temperature=0)
    sp.normalize(None)
    return Req(
        rid="r0",
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=sp,
        require_reasoning=True,
    )


class TestReasoningTokenStopTrim(CustomTestCase):
    def test_full_run_counts_when_not_finished(self):
        req, proc = _make_req(), _make_processor()
        req.output_ids.extend([11, 12, 13, 14])
        proc._maybe_update_reasoning_tokens(req, [11, 12, 13, 14])
        self.assertEqual(req.reasoning_tokens, 4)

    def test_stop_mid_run_trims_reasoning_count(self):
        """A verify run accepted past the stop: only tokens that survive
        `output_ids_through_stop` count as reasoning."""
        req, proc = _make_req(), _make_processor()
        req.output_ids.extend([11, 12, 13, 14])
        req.finished_len = 2
        proc._maybe_update_reasoning_tokens(req, [11, 12, 13, 14])
        self.assertEqual(req.reasoning_tokens, 2)

    def test_run_entirely_past_stop_counts_nothing(self):
        req, proc = _make_req(), _make_processor()
        req.output_ids.extend([11, 12])
        proc._maybe_update_reasoning_tokens(req, [11, 12])
        req.output_ids.extend([13, 14])
        req.finished_len = 2
        proc._maybe_update_reasoning_tokens(req, [13, 14])
        self.assertEqual(req.reasoning_tokens, 2)

    def test_think_end_inside_kept_region_still_closes_counting(self):
        req, proc = _make_req(), _make_processor()
        run = [11, 7, 8, 14]
        req.output_ids.extend(run)
        req.finished_len = 4
        proc._maybe_update_reasoning_tokens(req, run)
        # counted up to and including the end-tag sequence, then closed
        self.assertEqual(req.reasoning_tokens, 3)
        self.assertTrue(req._is_reasoning_over)

    def test_scalar_token_prefill_path(self):
        req, proc = _make_req(), _make_processor()
        req.output_ids.append(11)
        proc._maybe_update_reasoning_tokens(req, 11)
        self.assertEqual(req.reasoning_tokens, 1)


if __name__ == "__main__":
    unittest.main()
