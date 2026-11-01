"""CPU coverage for the INF-681 k-phost prefill-step switches.

All three extend SGLANG_OPT_PREFILL_MIDFORWARD_LAYERS (test_prefill_midforward_hook):
- SGLANG_OPT_PREFILL_MIDFORWARD_RESULT_SLICE: the previous batch's per-request result
  work is spread over the hooks a fixed number of requests at a time;
- SGLANG_OPT_PREFILL_EARLY_SEND_IN_HOOK: run_batch's cached-prefix early sends are
  decided before the launch but sent from the first hook;
- SGLANG_OPT_PREFILL_NEXT_BATCH_IN_FORWARD: the last hook drains the HiCache events
  and resolves the waiting queue's bootstraps ahead of the next batch.
The tests drive the real loop with the controlled scheduler and check that batches,
results, sends and their order stay what they are without the switches; only the
point in time (which forward, which layer) moves.
"""

import multiprocessing
import os
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_prefill_midforward_hook as hook_tests  # noqa: E402
import test_prefill_result_polling as base  # noqa: E402

from sglang.srt.disaggregation import prefill as prefill_mod  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.schedule_batch import Req  # noqa: E402
from sglang.srt.managers.scheduler_components import midforward_hook  # noqa: E402
from sglang.srt.managers.utils import GenerationBatchResult  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

REQS_PER_BATCH = 5
CACHED_PREFIX = 4


class _FakeCudaEvent:
    """Stands in for torch.cuda.Event in maybe_send_cached_prefix_chunk."""

    count = 0

    def __init__(self):
        _FakeCudaEvent.count += 1
        self.name = f"ev{_FakeCudaEvent.count}"
        self.stream = None

    def record(self, stream=None):
        self.stream = stream


def _batch(forward_iter, n=REQS_PER_BATCH):
    reqs = []
    for j in range(n):
        req = Req(f"{forward_iter}.{j}", "", [1, 2, 3, 4, 5, 6], SamplingParams())
        # A device-resident cached prefix, so run_batch early-sends it.
        req.prefix_len = CACHED_PREFIX
        req.disagg_kv_sender = SimpleNamespace()
        reqs.append(req)
    return base._batch(forward_iter, reqs)


def _result(n=REQS_PER_BATCH):
    return GenerationBatchResult(
        next_token_ids=torch.tensor([11] * n), copy_done=base._Event()
    )


class _StepScheduler(hook_tests._HookScheduler):
    """_HookScheduler whose run_batch early-sends like the real one, and which records
    every per-request result step and every send with the forward and layer it ran in."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cur_layer = None
        self.events = []  # (what, rid, forward, layer, detail)
        self.token_to_kv_pool_allocator = SimpleNamespace(page_size=1)
        self.send_kv_chunk = Mock(side_effect=self._record_send)
        self.hicache_drains = []  # (forward, layer)
        self.bootstrap_candidates = []  # (forward, layer, [rid])
        self.forward_ct = 0

    def _record_send(self, req, last_chunk=False, end_idx=None):
        detail = (
            "last"
            if last_chunk
            else (end_idx, getattr(req.disagg_kv_sender, "_early_send_wait_event").name)
        )
        self.events.append(("send", req.rid, self.in_forward, self.cur_layer, detail))

    def checkpoint_disagg_prefill(self, req):
        self.events.append(("result_req", req.rid, self.in_forward, self.cur_layer, None))

    def _process_hicache_events(self, should_retry_storage_prefetch=True):
        self.hicache_drains.append((self.in_forward, self.cur_layer))

    def run_batch(self, batch):
        result = base._Scheduler.run_batch(self, batch)
        self.forward_ct += 1
        # Real run_batch: early-send the cached prefixes before the launch.
        for req in batch.reqs:
            self.maybe_send_cached_prefix_chunk(req)
        for name in self.launched:
            self.copy_events[name].ready = True
        self.launched.append(batch.forward_iter)
        self.in_forward = batch.forward_iter
        try:
            for layer in range(self.num_layers):
                self._incoming_requests.extend(
                    self.mid_arrivals.get((self.iteration, layer), [])
                )
                self.cur_layer = layer
                midforward_hook.maybe_fire(layer)
        finally:
            self.in_forward = None
            self.cur_layer = None
        self.timeline.append(("forward_end", batch.forward_iter, None))
        return result


def _schedule():
    arrivals = [_batch(1), _batch(2), _batch(3), None, _batch(4)]
    return arrivals + [None] * 5


def _run(layers, slice_reqs=0, early_in_hook=False, next_in_forward=False, **kwargs):
    arrivals = kwargs.pop("arrivals", None) or _schedule()
    names = [b.forward_iter for b in arrivals if b is not None]
    scheduler = _StepScheduler(
        arrivals,
        {name: _result() for name in names},
        {name: 10**6 for name in names},
        **kwargs,
    )
    with (
        envs.SGLANG_OPT_PREFILL_MIDFORWARD_LAYERS.override(layers),
        envs.SGLANG_OPT_PREFILL_MIDFORWARD_RESULT_SLICE.override(slice_reqs),
        envs.SGLANG_OPT_PREFILL_EARLY_SEND_IN_HOOK.override(early_in_hook),
        envs.SGLANG_OPT_PREFILL_NEXT_BATCH_IN_FORWARD.override(next_in_forward),
        patch.object(prefill_mod.torch.cuda, "Event", _FakeCudaEvent),
    ):
        _FakeCudaEvent.count = 0
        base._run_loop(scheduler, enabled=False)
    assert not midforward_hook.is_armed()
    assert getattr(scheduler, "_midforward_early_send_sink", None) is None
    return scheduler


def _order(scheduler, what=None):
    """Event list without the time fields: what happened, to which request."""
    return [(e[0], e[1], e[4]) for e in scheduler.events if what in (None, e[0])]


def _where(scheduler, what):
    return {e[1]: (e[2], e[3]) for e in scheduler.events if e[0] == what}


class TestPrefillMidForwardStep(CustomTestCase):
    def test_result_slices_keep_order_and_spread_over_hooks(self):
        whole = _run("1,3,5")
        sliced = _run("1,3,5", slice_reqs=2)
        self.assertEqual(hook_tests._launches(sliced), hook_tests._launches(whole))
        # Same per-request work, same sends, same order.
        self.assertEqual(_order(sliced), _order(whole))
        self.assertEqual(
            [r.rid for r in sliced.disagg_prefill_inflight_queue],
            [r.rid for r in whole.disagg_prefill_inflight_queue],
        )
        self.assertEqual(len(whole.disagg_prefill_inflight_queue), 4 * REQS_PER_BATCH)
        self.assertEqual(sliced.result_processes, {1: 1, 2: 1, 3: 1, 4: 1})
        # Whole: batch 1's five requests all in forward 2's first hook (layer 1).
        w = _where(whole, "result_req")
        self.assertEqual({w[f"1.{j}"] for j in range(5)}, {(2, 1)})
        # Sliced: two at layer 1, two at layer 3, the last at layer 5.
        s = _where(sliced, "result_req")
        self.assertEqual([s[f"1.{j}"] for j in range(5)], [(2, 1), (2, 1), (2, 3), (2, 3), (2, 5)])
        self.assertEqual([s[f"2.{j}"] for j in range(5)], [(3, 1), (3, 1), (3, 3), (3, 3), (3, 5)])
        # Batch 3's successor came two iterations later: after the forward, as before.
        self.assertEqual({s[f"3.{j}"] for j in range(5)}, {(None, None)})
        # Host backups submitted after every slice.
        self.assertGreater(
            sliced.tree_cache.flush_pending_backups.call_count,
            whole.tree_cache.flush_pending_backups.call_count,
        )

    def test_result_slices_left_after_the_forward_finish_before_the_next_step(self):
        sliced = _run("6", slice_reqs=2)
        whole = _run("6")
        self.assertEqual(_order(sliced), _order(whole))
        s = _where(sliced, "result_req")
        # Two in forward 2's only hook, the other three right after that forward.
        self.assertEqual([s[f"1.{j}"] for j in range(5)], [(2, 6), (2, 6), (None, None), (None, None), (None, None)])
        # Every request of batch 1 was done before batch 3 was launched.
        launches = [a for a in sliced.actions if a[1] == "launch"]
        self.assertEqual([a[2] for a in launches], [1, 2, 3, 4])
        self.assertEqual(sliced.result_processes, {1: 1, 2: 1, 3: 1, 4: 1})
        # And the leftovers were polled right after forward 2 (with all slices done
        # in the hook, the loop does not poll there).
        i = sliced.timeline.index(("forward_end", 2, None))
        self.assertEqual(sliced.timeline[i + 1], ("inflight", None, None))
        j = whole.timeline.index(("forward_end", 2, None))
        self.assertNotEqual(whole.timeline[j + 1 : j + 2], [("inflight", None, None)])

    def test_slice_larger_than_batch_equals_whole(self):
        whole = _run("1,3,5")
        big = _run("1,3,5", slice_reqs=REQS_PER_BATCH)
        self.assertEqual(_order(big), _order(whole))
        self.assertEqual(_where(big, "result_req"), _where(whole, "result_req"))

    def test_early_sends_move_into_the_first_hook(self):
        before = _run("2,5")
        moved = _run("2,5", early_in_hook=True)
        self.assertEqual(hook_tests._launches(moved), hook_tests._launches(before))
        # The same sends with the same end index and wait event, in the same order.
        self.assertEqual(_order(moved, "send"), _order(before, "send"))
        early_before = {e[1]: (e[2], e[3]) for e in before.events if e[0] == "send" and e[4] != "last"}
        early_moved = {e[1]: (e[2], e[3]) for e in moved.events if e[0] == "send" and e[4] != "last"}
        # Today: before the launch. Moved: in the first hook of the batch's own forward.
        self.assertEqual(set(early_before.values()), {(None, None)})
        for rid, where in early_moved.items():
            self.assertEqual(where, (int(rid.split(".")[0]), 2), rid)
        # Within the first hook the early sends come before the previous batch's result.
        firsts = [e for e in moved.events if e[2] == 2 and e[3] == 2]
        kinds = [e[0] for e in firsts]
        self.assertEqual(kinds[:REQS_PER_BATCH], ["send"] * REQS_PER_BATCH)
        self.assertIn("result_req", kinds[REQS_PER_BATCH:])

    def test_early_sends_without_a_hook_go_right_after_the_forward(self):
        before = _run("9")
        moved = _run("9", early_in_hook=True)
        self.assertEqual(_order(moved), _order(before))
        sends = [e for e in moved.events if e[0] == "send" and e[4] != "last"]
        self.assertTrue(sends)
        self.assertEqual({(e[2], e[3]) for e in sends}, {(None, None)})
        # Batch 2's early sends came before batch 1's result, as run_batch had them.
        kinds = [(e[0], e[1].split(".")[0]) for e in moved.events]
        self.assertLess(kinds.index(("send", "2")), kinds.index(("result_req", "1")))

    def test_switches_off_or_unarmed_change_nothing(self):
        baseline = _run("")
        for kwargs in (
            {"slice_reqs": 2},
            {"early_in_hook": True},
            {"next_in_forward": True},
            {"slice_reqs": 2, "early_in_hook": True, "next_in_forward": True},
        ):
            with self.subTest(**kwargs):
                other = _run("", **kwargs)
                self.assertEqual(other.actions, baseline.actions)
                self.assertEqual(other.events, baseline.events)

    def test_next_batch_prep_runs_in_the_last_hook(self):
        before = _run("1,4")
        prepped = _run("1,4", next_in_forward=True)
        self.assertEqual(hook_tests._launches(prepped), hook_tests._launches(before))
        self.assertEqual(_order(prepped), _order(before))
        # The last hook (layer 4) of every forward drained the HiCache events, and the
        # loop still drained once more before building the next batch.
        in_hook = [d for d in prepped.hicache_drains if d[1] == 4]
        self.assertEqual(sorted(f for f, _ in in_hook), [1, 2, 3, 4])
        self.assertEqual(
            len([d for d in prepped.hicache_drains if d[0] is None]),
            len([d for d in before.hicache_drains if d[0] is None]),
        )

    def test_bootstrap_resolution_skips_requests_checked_in_the_last_hook(self):
        seen = []

        def fake_poll(pollers, cp_group, tp_group):
            seen.append(list(pollers))
            return [base.KVPoll.Bootstrapping] * len(pollers)

        scheduler = _StepScheduler([None], {}, {})
        old = [Req(f"w{j}", "", [1], SamplingParams()) for j in range(3)]
        for req in old:
            req.disagg_kv_sender = f"sender-{req.rid}"
        scheduler.waiting_queue = list(old)
        scheduler.forward_ct = 7
        with patch.object(prefill_mod, "poll_and_all_reduce_attn_cp_tp_group", fake_poll):
            # The last hook of forward 7: all three checked.
            scheduler._prepare_next_batch_midforward()
            # A request popped from the bootstrap queue after that hook.
            new = Req("n0", "", [1], SamplingParams())
            new.disagg_kv_sender = "sender-n0"
            scheduler.waiting_queue.append(new)
            prefill_mod.SchedulerDisaggregationPrefillMixin.get_next_disagg_prefill_batch_to_run(
                scheduler, running_batch=scheduler.running_batch, last_batch=None
            )
            # The next step (no hook ran in between): everyone again.
            scheduler.forward_ct = 8
            prefill_mod.SchedulerDisaggregationPrefillMixin.get_next_disagg_prefill_batch_to_run(
                scheduler, running_batch=scheduler.running_batch, last_batch=None
            )
        self.assertEqual(seen[0], [f"sender-w{j}" for j in range(3)])
        self.assertEqual(seen[1], ["sender-n0"])
        self.assertEqual(seen[2], [f"sender-w{j}" for j in range(3)] + ["sender-n0"])

    @unittest.skipUnless(dist.is_gloo_available(), "requires Gloo")
    def test_two_ranks_stay_in_lockstep_with_all_switches(self):
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
                proc.join(90)
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
        b2 = _batch(2)
        arrivals = [_batch(1), None, _batch(3), _batch(4)] + [None] * 6
        scheduler = _StepScheduler(
            arrivals,
            {name: _result() for name in (1, 2, 3, 4)},
            {name: 10**6 for name in (1, 2, 3, 4)},
            mid_arrivals={(0, 4): [base._input(b2)]},
        )
        scheduler.batches[2] = b2
        scheduler.tp_cpu_group = dist.group.WORLD
        with (
            envs.SGLANG_OPT_PREFILL_MIDFORWARD_LAYERS.override("1,3,6"),
            envs.SGLANG_OPT_PREFILL_MIDFORWARD_RESULT_SLICE.override(2),
            envs.SGLANG_OPT_PREFILL_EARLY_SEND_IN_HOOK.override(True),
            envs.SGLANG_OPT_PREFILL_NEXT_BATCH_IN_FORWARD.override(True),
            patch.object(prefill_mod.torch.cuda, "Event", _FakeCudaEvent),
        ):
            base._run_loop(
                scheduler, enabled=False, tp_size=2, ranks={"world_rank": rank}
            )
        assert hook_tests._results(scheduler) == [1, 2, 3, 4], scheduler.actions
        assert scheduler.result_processes == {1: 1, 2: 1, 3: 1, 4: 1}
        assert len([e for e in scheduler.events if e[0] == "result_req"]) == 20
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
