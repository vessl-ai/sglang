"""CPU coverage for SGLANG_OPT_PREFILL_MIDFORWARD_LAYERS (INF-681 k-gap).

The PD prefill overlap loop can run its between-step work (the previous batch's
result, transfer polling, request intake) from inside the next forward, at fixed
layers. These tests drive the real loop with the controlled scheduler from
test_prefill_result_polling and a fake forward that calls the hook at every layer,
and check that the same batches launch in the same order, every result is
processed exactly once and in order, the previous result is processed inside the
next forward, intake keeps message order, and nothing changes when the hooks do
not fire.
"""

import multiprocessing
import os
import sys
import tempfile
import unittest
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_prefill_result_polling as base  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.scheduler_components import midforward_hook  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_LAYERS = 8


class _HookScheduler(base._Scheduler):
    """base._Scheduler whose forward walks NUM_LAYERS layers and fires the hooks."""

    def __init__(self, *args, num_layers=NUM_LAYERS, mid_arrivals=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.device_module = SimpleNamespace(stream=lambda stream: nullcontext())
        self.num_layers = num_layers
        # {(iteration, layer): [inputs]} arriving while that layer is launched.
        self.mid_arrivals = mid_arrivals or {}
        self.in_forward = None
        self.launched = []
        # (what, batch or request name, forward it ran inside or None)
        self.timeline = []

    def run_batch(self, batch):
        result = super().run_batch(batch)
        # On the GPU this forward starts after every earlier one has finished.
        for name in self.launched:
            self.copy_events[name].ready = True
        self.launched.append(batch.forward_iter)
        self.in_forward = batch.forward_iter
        try:
            for layer in range(self.num_layers):
                self._incoming_requests.extend(
                    self.mid_arrivals.get((self.iteration, layer), [])
                )
                midforward_hook.maybe_fire(layer)
        finally:
            self.in_forward = None
        self.timeline.append(("forward_end", batch.forward_iter, None))
        return result

    def process_batch_result(self, batch, result):
        self.timeline.append(("result", batch.forward_iter, self.in_forward))
        super().process_batch_result(batch, result)

    def process_disagg_prefill_inflight_queue(self):
        self.timeline.append(("inflight", None, self.in_forward))
        super().process_disagg_prefill_inflight_queue()

    def process_input_requests(self, recv_reqs):
        for req in recv_reqs:
            name = (
                req.rid
                if isinstance(req, TokenizedGenerateReqInput)
                else type(req).__name__
            )
            self.timeline.append(("dispatch", name, self.in_forward))
        super().process_input_requests(recv_reqs)


def _schedule():
    arrivals = [base._batch(1), base._batch(2), base._batch(3), None, base._batch(4)]
    return arrivals + [None] * 5


def _run(layers, min_tokens=0, **kwargs):
    arrivals = kwargs.pop("arrivals", None) or _schedule()
    names = [b.forward_iter for b in arrivals if b is not None]
    scheduler = _HookScheduler(
        arrivals,
        {name: base._result() for name in names},
        {name: 10**6 for name in names},
        **kwargs,
    )
    with (
        envs.SGLANG_OPT_PREFILL_MIDFORWARD_LAYERS.override(layers),
        envs.SGLANG_OPT_PREFILL_MIDFORWARD_MIN_TOKENS.override(min_tokens),
    ):
        base._run_loop(scheduler, enabled=False)
    assert not midforward_hook.is_armed()
    return scheduler


def _launches(scheduler):
    return [(a[0], a[2]) for a in scheduler.actions if a[1] == "launch"]


def _results(scheduler):
    return [a[2] for a in scheduler.actions if a[1] == "result"]


class TestPrefillMidForwardHook(CustomTestCase):
    def test_parse_layers(self):
        self.assertEqual(midforward_hook.parse_layers(""), [])
        self.assertEqual(midforward_hook.parse_layers("12, 4,4"), [4, 12])
        with self.assertRaises(ValueError):
            midforward_hook.parse_layers("-1")

    def test_hooks_keep_batches_and_process_previous_result_inside_next_forward(self):
        baseline = _run("")
        hooked = _run("2,5")
        self.assertEqual(_launches(hooked), _launches(baseline))
        self.assertEqual(_results(hooked), [1, 2, 3, 4])
        self.assertEqual(_results(baseline), [1, 2, 3, 4])
        for scheduler in (baseline, hooked):
            self.assertEqual(scheduler.result_processes, {1: 1, 2: 1, 3: 1, 4: 1})
        # Baseline: every result after a forward returned, never inside one.
        self.assertTrue(
            all(f is None for what, _, f in baseline.timeline if what == "result")
        )
        # Hooks: the result of k-1 inside forward k whenever k followed k-1 directly.
        where = {name: f for what, name, f in hooked.timeline if what == "result"}
        self.assertEqual(where[1], 2)
        self.assertEqual(where[2], 3)
        # Batch 3's successor came two iterations later: processed after forward 3
        # on the next (idle) iteration, as before.
        self.assertIsNone(where[3])
        self.assertIsNone(where[4])
        # No transfer poll right after a forward whose hooks fired.
        for i, (what, name, f) in enumerate(hooked.timeline):
            if what == "forward_end":
                nxt = hooked.timeline[i + 1 : i + 2]
                self.assertNotIn(("inflight", None, None), nxt)
        inside = [f for what, _, f in hooked.timeline if what == "inflight" and f]
        self.assertEqual(sorted(set(inside)), [1, 2, 3, 4])
        for event in hooked.copy_events.values():
            self.assertEqual(event.synchronizations, 1)
        # The forward's successor result was always complete when its hook ran.
        self.assertEqual(hooked.copy_events[1].blocking_synchronizations, 0)
        self.assertEqual(hooked.copy_events[2].blocking_synchronizations, 0)

    def test_unfired_hooks_change_nothing(self):
        baseline = _run("")
        for layers, kwargs in (
            ("9", {}),  # past the last layer of this forward
            ("2,5", {"min_tokens": 1}),  # batches below the token floor
        ):
            with self.subTest(layers=layers, **kwargs):
                other = _run(layers, **kwargs)
                self.assertEqual(other.actions, baseline.actions)
                self.assertEqual(other.timeline, baseline.timeline)

    def test_midforward_intake_dispatches_only_a_leading_run_of_generate_requests(self):
        arrivals = [base._batch(1), None, None, None] + [None] * 6
        b2, b3 = base._batch(2), base._batch(3)
        mid = {
            (0, 3): [base._input(b2)],
            (0, 6): [PauseGenerationReqInput(mode="in_place"), base._input(b3)],
        }
        runs = {}
        for layers in ("", "2,5,7"):
            scheduler = _HookScheduler(
                arrivals,
                {1: base._result(), 2: base._result(), 3: base._result()},
                {1: 10**6, 2: 10**6, 3: 10**6},
                resume_at=4,
                mid_arrivals=mid,
            )
            scheduler.batches.update({2: b2, 3: b3})
            with envs.SGLANG_OPT_PREFILL_MIDFORWARD_LAYERS.override(layers):
                base._run_loop(scheduler, enabled=False)
            runs[layers] = scheduler
        hooked, baseline = runs["2,5,7"], runs[""]
        dispatch = [(n, f) for what, n, f in hooked.timeline if what == "dispatch"]
        # Request 2 arrived at layer 3: taken in by the layer-5 hook of forward 1.
        self.assertIn(("2", 1), dispatch)
        # The pause arrived at layer 6 with request 3 behind it: the layer-7 hook
        # leaves both, and the loop's own intake dispatches them in order.
        names = [n for n, _ in dispatch]
        self.assertLess(
            names.index("PauseGenerationReqInput"), names.index("3"), names
        )
        self.assertIn(("PauseGenerationReqInput", None), dispatch)
        self.assertIn(("3", None), dispatch)
        self.assertIn(("ContinueGenerationReqInput", None), dispatch)
        self.assertEqual(_launches(hooked), _launches(baseline))
        self.assertEqual(_results(hooked), _results(baseline))
        self.assertFalse(hooked._deferred_input_requests)

    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_two_ranks_stay_in_lockstep(self):
        with tempfile.TemporaryDirectory() as tmp:
            rendezvous = Path(tmp) / "rendezvous"
            ctx = multiprocessing.get_context("spawn")
            procs = [
                ctx.Process(target=_two_rank_worker, args=(rank, str(rendezvous)))
                for rank in range(2)
            ]
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join(60)
                self.assertEqual(proc.exitcode, 0)


def _two_rank_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        b2 = base._batch(2)
        arrivals = [base._batch(1), None, base._batch(3), base._batch(4)] + [None] * 6
        scheduler = _HookScheduler(
            arrivals,
            {name: base._result() for name in (1, 2, 3, 4)},
            {name: 10**6 for name in (1, 2, 3, 4)},
            mid_arrivals={(0, 4): [base._input(b2)]},
        )
        scheduler.batches[2] = b2
        scheduler.tp_cpu_group = dist.group.WORLD
        with envs.SGLANG_OPT_PREFILL_MIDFORWARD_LAYERS.override("1,6"):
            base._run_loop(
                scheduler, enabled=False, tp_size=2, ranks={"world_rank": rank}
            )
        assert _results(scheduler) == [1, 2, 3, 4], scheduler.actions
        assert scheduler.result_processes == {1: 1, 2: 1, 3: 1, 4: 1}
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
