"""HiCache host pool released at checkpoint prepare and brought back at restore
(mem_cache/host_pool_checkpoint.py), and the L3 attach beside the PD transport
build (disaggregation/checkpoint_transport.py, overlap_l3).

GPU-free: a fake allocator stands in for the pages and the CUDA registration.
It keeps one residency bit per page and the set of registered (pointer, size)
ranges, and fails the way cudart does on a double register or unregister.
The real madvise/mincore/memfd behaviour is not covered by this suite; it
needs Linux.
"""

import contextlib
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation import checkpoint_transport as ct
from sglang.srt.disaggregation.utils import DisaggregationMode, TransferBackend
from sglang.srt.distributed.device_communicators import mooncake_transfer_engine as mte
from sglang.srt.managers.io_struct import (
    AttachHiCacheStorageReqOutput,
    CheckpointReqInput,
)
from sglang.srt.mem_cache import host_pool_checkpoint as hp
from sglang.srt.mem_cache.pool_host.common import (
    _CUDA_HOST_REGISTERED_RANGES_ATTR as ATTR,
)
from sglang.srt.mem_cache.pool_host.common import (
    make_kernel_ptr_table,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


# These tests build CPU tensors without naming the device. Another test in the
# same process may have left torch's default device set to cuda; pin it to CPU
# for this module and give the previous setting back afterwards.
_cpu_default_device = contextlib.ExitStack()


def setUpModule():
    _cpu_default_device.enter_context(torch.device("cpu"))


def tearDownModule():
    _cpu_default_device.close()


PAGE = hp.PAGE
_ENVS = (
    "SGLANG_CHECKPOINT_RELEASE_HOST_POOL",
    "SGLANG_CHECKPOINT_HOST_POOL_THREADS",
    "SGLANG_CHECKPOINT_RESTORE_OVERLAP_L3",
    "SGLANG_CHECKPOINT_DEFER_PD_TRANSPORT",
    "SGLANG_CHECKPOINT_DEFER_HICACHE_STORAGE",
    "SGLANG_HOST_IP",
    "MOONCAKE_LOCAL_HOSTNAME",
    "SGLANG_RUST_SERVER",
    "SGLANG_CHECKPOINT_RESTORE_KV_FILL",
    "SGLANG_CHECKPOINT_LOG_KV_REGIONS",
    "SGLANG_CHECKPOINT_KV_REGION_IMAGE",
)


class FakeOps:
    """Pages and CUDA registration of a fake allocator."""

    def __init__(self):
        self.lock = threading.Lock()
        self.resident = set()  # page addresses
        self.registered = {}  # ptr -> size
        self.kinds = {}  # (start, end) -> mapping kind
        self.calls = []
        self.fail_unregister = set()
        self.fail_register = set()
        self.fail_madvise = {}
        self.ignore_remove = False
        self.synced = 0
        self.devices = []

    def map(self, t: torch.Tensor, kind="shared"):
        lo = t.data_ptr() // PAGE * PAGE
        hi = (t.data_ptr() + t.nbytes + PAGE - 1) // PAGE * PAGE
        self.kinds[(lo, hi)] = kind
        for a in range(lo, hi, PAGE):
            self.resident.add(a)

    def register_like_boot(self, t: torch.Tensor, granularity=None):
        """What alloc_with_host_register leaves behind."""
        base, total = t.data_ptr(), t.nbytes
        step = total if granularity is None else granularity
        ranges = []
        for off in range(0, total, step):
            size = min(step, total - off)
            self.registered[base + off] = size
            ranges.append((base + off, size))
        setattr(t, ATTR, ranges)

    # -- HostOps interface
    def madvise(self, addr, length, advice):
        with self.lock:
            self.calls.append(("madvise", addr, length, advice))
            if advice in self.fail_madvise:
                return self.fail_madvise[advice]
            pages = range(addr, addr + length, PAGE)
            if advice in (hp.MADV_REMOVE, hp.MADV_DONTNEED):
                if not self.ignore_remove:
                    self.resident.difference_update(pages)
            elif advice == hp.MADV_POPULATE_WRITE:
                self.resident.update(pages)
            return 0

    def resident_fraction(self, addr, length):
        n = length // PAGE
        if n == 0:
            return 1.0
        with self.lock:
            return sum(a in self.resident for a in range(addr, addr + length, PAGE)) / n

    def mapping_kinds(self, addrs):
        out = []
        for addr in addrs:
            kind = "none"
            for (lo, hi), k in self.kinds.items():
                if lo <= addr < hi:
                    kind = k
            out.append(kind)
        return out

    def cuda_register(self, ptr, size):
        with self.lock:
            self.calls.append(("register", ptr, size))
            if ptr in self.fail_register:
                return 1
            if ptr in self.registered:
                return 712  # cudaErrorHostMemoryAlreadyRegistered
            # Pinning faults the pages in.
            self.resident.update(range(ptr // PAGE * PAGE, ptr + size, PAGE))
            self.registered[ptr] = size
            return 0

    def cuda_unregister(self, ptr):
        with self.lock:
            self.calls.append(("unregister", ptr))
            if ptr in self.fail_unregister:
                return 1
            if ptr not in self.registered:
                return 713  # cudaErrorHostMemoryNotRegistered
            del self.registered[ptr]
            return 0

    def cuda_error(self, rc):
        return f"err{rc}"

    def synchronize(self):
        self.synced += 1

    def current_device(self):
        return 5

    def set_device(self, device):
        with self.lock:
            self.devices.append(device)


def _page_aligned(nbytes: int) -> torch.Tensor:
    """A uint8 tensor that starts on a page boundary, nbytes long."""
    raw = torch.zeros(nbytes + PAGE, dtype=torch.uint8)
    off = (-raw.data_ptr()) % PAGE
    t = raw[off : off + nbytes]
    t._keep = raw  # the allocation outlives the view
    return t


class FakePool:
    """A host pool with the attributes the real ones carry."""

    def __init__(
        self, ops, name, layers, layer_bytes, *, layout="layer_first", kind="shared"
    ):
        self.pool_name = name
        self.pin_memory = True
        if layout == "layer_first":
            self.kv_buffer = [_page_aligned(layer_bytes) for _ in range(layers)]
            for t in self.kv_buffer:
                ops.map(t, kind)
                ops.register_like_boot(t)
            self.data_refs = [self.kv_buffer[i] for i in range(layers)]
            self.data_ptrs = make_kernel_ptr_table(
                self.data_refs, "cpu", host_memory_registered=True
            )
        else:
            self.kv_buffer = _page_aligned(layers * layer_bytes)
            ops.map(self.kv_buffer, kind)
            ops.register_like_boot(self.kv_buffer, granularity=2 * layer_bytes)
            self.data_refs = []
            self.data_ptrs = None


class FakeScheduler:
    def __init__(self, pools, *, enable_storage=False, flush_ok=True):
        group = SimpleNamespace(entries=[SimpleNamespace(host_pool=p) for p in pools])
        self.tree_cache = SimpleNamespace(
            cache_controller=SimpleNamespace(
                mem_pool_host=group, enable_storage=enable_storage
            )
        )
        self.flush_ok = flush_ok
        self.flushes = []

    def flush_cache(self, empty_cache=True):
        self.flushes.append(empty_cache)
        return self.flush_ok


class _Case(unittest.TestCase):
    def setUp(self):
        saved = {k: os.environ.get(k) for k in _ENVS}

        def restore():
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        self.addCleanup(restore)
        for k in _ENVS:
            os.environ.pop(k, None)
        os.environ["SGLANG_CHECKPOINT_RELEASE_HOST_POOL"] = "1"
        os.environ["SGLANG_CHECKPOINT_HOST_POOL_THREADS"] = "4"
        self.ops = FakeOps()
        hp.set_ops(self.ops)
        self.addCleanup(hp.set_ops, None)

    def pools(self):
        # Two layer_first pools (one aliased through data_refs) and one page_first
        # pool registered in two chunks, as the DSv4 pools are.
        return [
            FakePool(self.ops, "swa", 3, 4 * PAGE),
            FakePool(self.ops, "c4", 2, 8 * PAGE),
            FakePool(self.ops, "c4_state", 4, 2 * PAGE, layout="page_first"),
        ]


class TestReleaseRestoreCycle(_Case):
    def test_cycle_returns_pages_and_restores_the_same_registration(self):
        pools = self.pools()
        s = FakeScheduler(pools)
        boot = dict(self.ops.registered)
        boot_attr = [list(getattr(t, ATTR)) for t in hp.registered_host_tensors(pools)]
        self.assertEqual(len(boot_attr), 6)  # 3 + 2 + 1; data_refs not counted twice

        with self.assertLogs(hp.logger, level="INFO") as logs:
            released = hp.release_host_pool(s)
        self.assertEqual(self.ops.synced, 1)
        self.assertEqual(s.flushes, [False])
        self.assertEqual(self.ops.registered, {})
        self.assertEqual(len(released.buffers), 6)
        self.assertEqual(released.nbytes, (3 * 4 + 2 * 8 + 4 * 2) * PAGE)
        for b in released.buffers:
            self.assertEqual(self.ops.resident_fraction(b.start, b.nbytes), 0.0)
            self.assertEqual(getattr(b.tensor, ATTR), [])
        self.assertIn("Checkpoint prepare phase=host_pool_release", logs.output[-1])
        self.assertIn("max_resident_after=0.0000", logs.output[-1])
        # Unregistered before any page was returned.
        kinds = [c[0] for c in self.ops.calls]
        self.assertLess(
            max(i for i, k in enumerate(kinds) if k == "unregister"),
            min(i for i, k in enumerate(kinds) if k == "madvise"),
        )
        self.assertTrue(
            all(c[3] == hp.MADV_REMOVE for c in self.ops.calls if c[0] == "madvise")
        )
        self.assertIs(s.released_host_pool, released)

        self.ops.calls.clear()
        with self.assertLogs(hp.logger, level="INFO") as logs:
            hp.restore_host_pool(s)
        self.assertEqual(self.ops.registered, boot)
        self.assertEqual(
            [list(getattr(t, ATTR)) for t in hp.registered_host_tensors(pools)],
            boot_attr,
        )
        # Pages populated before the registration pins them; every task ran on
        # the scheduler's device.
        kinds = [c[0] for c in self.ops.calls]
        self.assertLess(
            max(i for i, k in enumerate(kinds) if k == "madvise"),
            min(i for i, k in enumerate(kinds) if k == "register"),
        )
        self.assertEqual(set(self.ops.devices), {5})
        lines = [l for l in logs.output if "phase=" in l]
        self.assertIn("phase=host_pool_populate", lines[0])
        self.assertIn("resident_before=0.0000", lines[0])
        self.assertIn("phase=host_pool_register", lines[1])
        self.assertIn("ptr_tables_checked=2", lines[1])
        self.assertIsNone(s.released_host_pool)

        # A second restore (redelivered request) changes nothing.
        self.ops.calls.clear()
        hp.restore_host_pool(s)
        self.assertEqual(self.ops.calls, [])

    def test_private_anonymous_buffers_use_dontneed(self):
        pool = FakePool(self.ops, "kv", 2, 4 * PAGE, kind="private_anon")
        hp.release_host_pool(FakeScheduler([pool]))
        advice = {c[3] for c in self.ops.calls if c[0] == "madvise"}
        self.assertEqual(advice, {hp.MADV_DONTNEED})

    def test_disabled_flag_touches_nothing(self):
        os.environ.pop("SGLANG_CHECKPOINT_RELEASE_HOST_POOL")
        pools = self.pools()
        s = FakeScheduler(pools)
        self.assertIsNone(hp.release_host_pool(s))
        self.assertEqual(s.flushes, [])
        self.assertEqual(self.ops.calls, [])
        self.assertEqual(len(self.ops.registered), 7)

    def test_two_pools_sharing_a_buffer_unregister_it_once(self):
        a = FakePool(self.ops, "a", 2, 4 * PAGE)
        b = SimpleNamespace(kv_buffer=a.kv_buffer[0], other=list(a.kv_buffer))
        released = hp.release_host_pool(FakeScheduler([a, b]))
        self.assertEqual(len(released.buffers), 2)
        self.assertEqual(sum(1 for c in self.ops.calls if c[0] == "unregister"), 2)


class TestReleaseRefusals(_Case):
    def assert_refused(self, s, pattern):
        with self.assertRaisesRegex(RuntimeError, pattern):
            hp.release_host_pool(s)

    def test_attached_l3_client_refuses(self):
        s = FakeScheduler(self.pools(), enable_storage=True)
        self.assert_refused(s, "L3 storage client is attached")
        self.assertEqual(s.flushes, [])
        self.assertEqual(len(self.ops.registered), 7)

    def test_flush_failure_refuses_before_anything_is_unregistered(self):
        s = FakeScheduler(self.pools(), flush_ok=False)
        self.assert_refused(s, "could not flush")
        self.assertEqual(len(self.ops.registered), 7)

    def test_file_backed_private_mapping_refuses_before_unregistering(self):
        pools = self.pools() + [
            FakePool(self.ops, "f", 1, 4 * PAGE, kind="private_file")
        ]
        self.assert_refused(FakeScheduler(pools), "private_file mapping")
        self.assertEqual(len(self.ops.registered), 8)
        self.assertNotIn("unregister", [c[0] for c in self.ops.calls])

    def test_unregister_failure_refuses(self):
        pools = self.pools()
        self.ops.fail_unregister.add(pools[1].kv_buffer[0].data_ptr())
        self.assert_refused(FakeScheduler(pools), "cudaHostUnregister")
        self.assertNotIn("madvise", [c[0] for c in self.ops.calls])

    def test_madvise_failure_refuses(self):
        self.ops.fail_madvise[hp.MADV_REMOVE] = 22
        self.assert_refused(FakeScheduler(self.pools()), "MADV_REMOVE")

    def test_pages_left_resident_refuse(self):
        self.ops.ignore_remove = True
        self.assert_refused(FakeScheduler(self.pools()), "still resident")


class TestRestoreFailures(_Case):
    def released(self):
        pools = self.pools()
        s = FakeScheduler(pools)
        hp.release_host_pool(s)
        self.ops.calls.clear()
        return pools, s

    def test_register_failure_fails_and_rolls_back_that_buffer(self):
        pools, s = self.released()
        bad = pools[2].kv_buffer  # two ranges; the second fails
        ranges = [
            r for b in s.released_host_pool.buffers if b.tensor is bad for r in b.ranges
        ]
        self.ops.fail_register.add(ranges[1][0])
        with self.assertRaisesRegex(
            RuntimeError, "re-registering the host pool failed"
        ):
            hp.restore_host_pool(s)
        self.assertNotIn(ranges[0][0], self.ops.registered)
        self.assertEqual(getattr(bad, ATTR), [])

    def test_changed_pointer_table_fails(self):
        pools, s = self.released()
        pools[0].data_ptrs.view(torch.int64)[1] += 4096
        with self.assertRaisesRegex(RuntimeError, "pointer table differs"):
            hp.restore_host_pool(s)

    def test_populated_from_outside_is_not_walked_again(self):
        pools, s = self.released()
        # What a restore helper that faulted the pages in from outside leaves.
        for b in s.released_host_pool.buffers:
            self.ops.resident.update(range(b.start, b.end, PAGE))
        with self.assertLogs(hp.logger, level="INFO") as logs:
            hp.restore_host_pool(s)
        self.assertNotIn("madvise", [c[0] for c in self.ops.calls])
        self.assertIn("skipped=1", logs.output[0])
        self.assertEqual(len(self.ops.registered), 7)

    def test_missing_populate_only_costs_speed(self):
        pools, s = self.released()
        self.ops.fail_madvise[hp.MADV_POPULATE_WRITE] = 22  # EINVAL: kernel < 5.14
        with self.assertLogs(hp.logger, level="INFO") as logs:
            hp.restore_host_pool(s)
        self.assertEqual(len(self.ops.registered), 7)
        self.assertIn("errno=Invalid argument", logs.output[0])


class TestValidation(_Case):
    def cfg(self, **kw):
        base = dict(
            enable_hierarchical_cache=True,
            checkpoint_prepare_on_startup=True,
            hicache_storage_backend="mooncake",
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def test_rules(self):
        with self.assertRaisesRegex(ValueError, "DEFER_HICACHE_STORAGE"):
            hp.validate_host_pool_release(self.cfg())
        os.environ["SGLANG_CHECKPOINT_DEFER_HICACHE_STORAGE"] = "1"
        hp.validate_host_pool_release(self.cfg())
        with self.assertRaisesRegex(ValueError, "prepare-on-startup"):
            hp.validate_host_pool_release(self.cfg(checkpoint_prepare_on_startup=False))
        # No host pool: nothing to release.
        hp.validate_host_pool_release(
            self.cfg(
                enable_hierarchical_cache=False, checkpoint_prepare_on_startup=False
            )
        )
        os.environ.pop("SGLANG_CHECKPOINT_RELEASE_HOST_POOL")
        os.environ.pop("SGLANG_CHECKPOINT_DEFER_HICACHE_STORAGE")
        hp.validate_host_pool_release(self.cfg())

    def test_deferral_validation_runs_it(self):
        cfg = self.cfg(
            disaggregation_mode="null",
            disaggregation_transfer_backend="mooncake",
            disaggregation_decode_enable_offload_kvcache=False,
        )
        with self.assertRaisesRegex(ValueError, "RELEASE_HOST_POOL"):
            ct.validate_checkpoint_deferral(cfg)


class TestCheckpointManagerWiring(_Case):
    def test_before_kv_release_captures_the_image_then_releases(self):
        order = []
        with (
            patch.object(
                ct,
                "capture_kv_region_image",
                side_effect=lambda s: order.append("image"),
            ),
            patch.object(
                hp, "release_host_pool", side_effect=lambda s: order.append("pool")
            ),
        ):
            ct.before_kv_release(object())
        self.assertEqual(order, ["image", "pool"])

    def test_request_field_defaults_to_none(self):
        self.assertIsNone(CheckpointReqInput().overlap_l3)

    def test_tokenizer_forwards_overlap_l3(self):
        import asyncio

        from sglang.srt.managers.io_struct import CheckpointReqType, CheckpointState
        from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin

        sent = []

        async def communicator(obj):
            sent.append(obj)
            return [
                SimpleNamespace(success=True, message="", state=CheckpointState.RUNNING)
            ]

        tokenizer = SimpleNamespace(
            auto_create_handle_loop=lambda: None,
            checkpoint_lock=asyncio.Lock(),
            checkpoint_communicator=communicator,
        )
        tokenizer._call_checkpoint = lambda req_type, **kw: (
            TokenizerControlMixin._call_checkpoint(tokenizer, req_type, **kw)
        )
        with patch(
            "sglang.srt.managers.io_struct.CheckpointReqOutput.merge",
            side_effect=lambda outs: outs[0],
        ):
            asyncio.run(
                TokenizerControlMixin.checkpoint_restore(
                    tokenizer, "10.2.0.7", None, True
                )
            )
        self.assertEqual(
            [(o.req_type, o.overlap_l3) for o in sent],
            [(CheckpointReqType.STATE, None), (CheckpointReqType.RESTORE, True)],
        )


class TestPrepareRefusedWhileServing(_Case):
    """A prepare the host pool release would refuse must be refused before
    generation is paused, with nothing changed: a released pool behind an
    attached L3 client (a second prepare after a restore, or an L3 client
    attached at runtime) would otherwise latch FAILED and leave a serving
    engine paused until it restarts.

    The checkpoint managers are built by Scheduler.init_checkpoint_manager on
    stand-in schedulers, under a tokenizer that records pause and continue."""

    def setUp(self):
        super().setUp()
        # The KV region image is covered elsewhere; this is the host pool.
        os.environ["SGLANG_CHECKPOINT_KV_REGION_IMAGE"] = "0"
        for p in (
            patch(
                "sglang.srt.managers.scheduler.get_exec",
                return_value=SimpleNamespace(
                    features=SimpleNamespace(enable_memory_saver=True)
                ),
            ),
            patch(
                "sglang.srt.managers.scheduler.get_parallel",
                return_value=SimpleNamespace(pp_size=1),
            ),
        ):
            p.start()
            self.addCleanup(p.stop)

    def server(self, storage, tp_size=1):
        """One DP rank per entry of `storage` (whether its L3 client is
        attached), `tp_size` TP ranks each, every rank with its own pools."""
        import asyncio

        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin

        calls = []
        gathered = []
        ranks = []
        for enable_storage in storage:
            for _ in range(tp_size):
                s = FakeScheduler(self.pools(), enable_storage=enable_storage)
                s._engine_paused = False
                s.is_fully_idle = lambda ignore_waiting=False: True
                s.weight_updater = SimpleNamespace(
                    offload_tags=set(),
                    release_memory_occupation=lambda req: calls.append("release"),
                    resume_memory_occupation=lambda req: calls.append("resume"),
                )
                s.dp_tp_group = SimpleNamespace(
                    all_gather_object=lambda out: gathered.append(out) or list(gathered)
                )
                Scheduler.init_checkpoint_manager(s)
                ranks.append(s)

        async def communicator(obj):
            replies = []
            for dp in range(len(storage)):
                gathered.clear()
                tp = ranks[dp * tp_size : (dp + 1) * tp_size]
                replies.append(
                    [r.checkpoint_manager.handle(obj) for r in reversed(tp)][-1]
                )
            return replies

        async def pause_generation(obj):
            calls.append("pause")
            tokenizer.is_pause = True

        async def continue_generation(obj):
            calls.append("continue")
            tokenizer.is_pause = False

        tokenizer = SimpleNamespace(
            auto_create_handle_loop=lambda: None,
            checkpoint_lock=asyncio.Lock(),
            is_pause=False,
            checkpoint_communicator=communicator,
            pause_generation=pause_generation,
            continue_generation=continue_generation,
        )
        tokenizer._call_checkpoint = lambda req_type, **kw: (
            TokenizerControlMixin._call_checkpoint(tokenizer, req_type, **kw)
        )

        def prepare():
            return asyncio.run(
                TokenizerControlMixin.checkpoint_prepare(tokenizer, "abort")
            )

        return tokenizer, ranks, calls, prepare

    def assert_untouched(self, tokenizer, ranks, calls):
        from sglang.srt.managers.io_struct import CheckpointState

        self.assertEqual(calls, [])  # never paused, so nothing to continue
        self.assertFalse(tokenizer.is_pause)
        for s in ranks:
            self.assertEqual(s.checkpoint_manager.state, CheckpointState.RUNNING)
            # A normal request reaches the scheduler and is not refused.
            self.assertFalse(ct.checkpoint_refuses_new_requests(s))
            self.assertIsNone(getattr(s, "released_host_pool", None))
            self.assertEqual(s.flushes, [])
        self.assertTrue(all(c[0] == "register" for c in self.ops.calls))

    def test_attached_l3_client_refuses_before_the_pause(self):
        from sglang.srt.managers.io_struct import CheckpointState

        tokenizer, ranks, calls, prepare = self.server([True])
        out = prepare()
        self.assertFalse(out.success)
        self.assertEqual(out.state, CheckpointState.RUNNING)
        self.assertIn("L3 storage client is attached", out.message)
        self.assert_untouched(tokenizer, ranks, calls)

        # Once the L3 client is gone the same engine prepares.
        ranks[0].tree_cache.cache_controller.enable_storage = False
        out = prepare()
        self.assertTrue(out.success)
        self.assertEqual(out.state, CheckpointState.PREPARED)
        self.assertEqual(calls, ["pause", "release"])
        self.assertIsNotNone(ranks[0].released_host_pool)

    def test_one_refusing_dp_rank_refuses_for_all(self):
        from sglang.srt.managers.io_struct import CheckpointState

        tokenizer, ranks, calls, prepare = self.server([False, True])
        out = prepare()
        self.assertFalse(out.success)
        self.assertEqual(out.state, CheckpointState.RUNNING)
        self.assert_untouched(tokenizer, ranks, calls)

    def test_one_refusing_tp_rank_refuses_for_its_dp_rank(self):
        from sglang.srt.managers.io_struct import CheckpointState

        tokenizer, ranks, calls, prepare = self.server([False], tp_size=2)
        # The rank that does not reply to the tokenizer refuses.
        ranks[1].tree_cache.cache_controller.enable_storage = True
        out = prepare()
        self.assertFalse(out.success)
        self.assertEqual(out.state, CheckpointState.RUNNING)
        self.assert_untouched(tokenizer, ranks, calls)

    def test_without_the_release_flag_an_attached_l3_client_does_not_refuse(self):
        from sglang.srt.managers.io_struct import CheckpointState

        os.environ.pop("SGLANG_CHECKPOINT_RELEASE_HOST_POOL")
        tokenizer, ranks, calls, prepare = self.server([True])
        out = prepare()
        self.assertTrue(out.success)
        self.assertEqual(out.state, CheckpointState.PREPARED)
        self.assertEqual(calls, ["pause", "release"])


# ---------------------------------------------------------------------------
# build_transport_on_restore: host pool before the L3 attach, and overlap_l3.
# ---------------------------------------------------------------------------


class _Engine:
    def __init__(self, hostname, gpu_id, ib_device):
        self.hostname, self.ib_device = hostname, ib_device

    def get_session_id(self):
        return f"{self.hostname}:15000"

    def get_ib_device(self):
        return self.ib_device


class TestRestoreOrder(_Case):
    def setUp(self):
        super().setUp()
        os.environ["SGLANG_CHECKPOINT_DEFER_HICACHE_STORAGE"] = "1"
        self.calls = []
        self.lock = threading.Lock()
        bag = SimpleNamespace(
            checkpoint_prepare_on_startup=True,
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="mooncake",
            disaggregation_ib_device="mlx5_0",
            enable_hierarchical_cache=True,
            hicache_storage_backend="mooncake",
            hicache_storage_backend_extra_config="{}",
            hicache_storage_prefetch_policy="timeout",
            hicache_write_policy="write_through",
            gpu_id=1,
            moe=SimpleNamespace(mooncake_ib_device=None),
        )
        for name in (
            "get_serving",
            "get_disagg",
            "get_memory",
            "get_device",
            "get_exec",
        ):
            p = patch.object(ct, name, return_value=bag)
            p.start()
            self.addCleanup(p.stop)
        self.parallel = SimpleNamespace(nnodes=1)
        p = patch("sglang.srt.runtime_context.get_parallel", return_value=self.parallel)
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(
            mte,
            "init_mooncake_transfer_engine",
            side_effect=lambda **kw: self.note("engine") or _Engine(**kw),
        )
        p.start()
        self.addCleanup(p.stop)

    def note(self, what):
        with self.lock:
            self.calls.append(what)

    def scheduler(self, *, pd_waits_for_l3=False):
        pools = self.pools()
        s = FakeScheduler(pools)
        hp.release_host_pool(s)
        self.l3_started = threading.Event()

        def init_kv_manager():
            if pd_waits_for_l3:
                # Only an overlapped attach can start while the PD build waits here.
                if not self.l3_started.wait(timeout=5):
                    raise AssertionError(
                        "the L3 attach did not run beside the PD build"
                    )
            self.note("kv_manager")
            return SimpleNamespace(
                buffers_registered=True,
                buffer_registration_seconds=0.1,
                num_registered_regions=4,
                local_ip=os.environ["SGLANG_HOST_IP"],
                rank_port=1,
                kv_args=None,
                bootstrap_registered=True,
                bootstrap_registration_seconds=0.01,
            )

        queue = SimpleNamespace(
            transfer_backend=TransferBackend.FAKE,
            kv_manager=SimpleNamespace(kv_args=None),
            _init_kv_manager=init_kv_manager,
        )

        def attach(req):
            self.note(("l3", len(self.ops.registered)))
            self.l3_started.set()
            return AttachHiCacheStorageReqOutput(success=True)

        s.disaggregation_mode = DisaggregationMode.PREFILL
        s.disagg_prefill_bootstrap_queue = queue
        s.disagg_decode_prealloc_queue = None
        s.pd_transport_deferred = True
        s.hicache_storage_deferred = True
        s.transfer_backend = TransferBackend.FAKE
        s.attach_hicache_storage_wrapped = attach
        return s

    def phases(self, logs):
        return [l.split("phase=")[1].split()[0] for l in logs.output if "phase=" in l]

    def test_sequential_registers_the_pool_before_the_l3_attach(self):
        s = self.scheduler()
        with self.assertLogs(level="INFO") as logs:
            ct.build_transport_on_restore(s, "10.2.0.7")
        self.assertEqual(self.calls, ["engine", "kv_manager", ("l3", 7)])
        phases = self.phases(logs)
        self.assertLess(
            phases.index("host_pool_register"), phases.index("transport_init")
        )
        self.assertIn("overlap_l3=0", logs.output[-1])

    def test_overlap_runs_the_attach_beside_the_pd_build(self):
        s = self.scheduler(pd_waits_for_l3=True)
        with self.assertLogs(level="INFO") as logs:
            ct.build_transport_on_restore(s, "10.2.0.7", overlap_l3=True)
        self.assertIn(("l3", 7), self.calls)  # registered again before its attach
        self.assertIn("kv_manager", self.calls)
        self.assertFalse(s.pd_transport_deferred)
        self.assertFalse(s.hicache_storage_deferred)
        phases = self.phases(logs)
        self.assertIn("overlap_l3", phases)
        self.assertIn("overlap_l3=1", logs.output[-1])

    def test_env_turns_it_on_and_the_request_turns_it_off(self):
        os.environ["SGLANG_CHECKPOINT_RESTORE_OVERLAP_L3"] = "1"
        s = self.scheduler(pd_waits_for_l3=True)
        ct.build_transport_on_restore(s, "10.2.0.7")
        self.ops = FakeOps()  # a fresh process: its own pages and registrations
        hp.set_ops(self.ops)
        s = self.scheduler()
        self.calls.clear()
        ct.build_transport_on_restore(s, "10.2.0.7", overlap_l3=False)
        self.assertEqual(self.calls, ["engine", "kv_manager", ("l3", 7)])

    def test_multi_node_falls_back_to_sequential(self):
        self.parallel.nnodes = 2
        s = self.scheduler()
        with self.assertLogs(level="WARNING") as logs:
            ct.build_transport_on_restore(s, "10.2.0.7", overlap_l3=True)
        self.assertIn("overlap_l3 not used: multi-node", logs.output[0])
        self.assertEqual(self.calls, ["engine", "kv_manager", ("l3", 7)])

    def test_overlap_failures_fail_the_restore(self):
        s = self.scheduler()
        s.attach_hicache_storage_wrapped = lambda req: AttachHiCacheStorageReqOutput(
            success=False, message="master unreachable"
        )
        with self.assertRaisesRegex(RuntimeError, "master unreachable"):
            ct.build_transport_on_restore(s, "10.2.0.7", overlap_l3=True)
        s = self.scheduler()
        s.disagg_prefill_bootstrap_queue._init_kv_manager = lambda: SimpleNamespace(
            buffers_registered=False, kv_args=None
        )
        with self.assertRaisesRegex(RuntimeError, "did not register"):
            ct.build_transport_on_restore(s, "10.2.0.7", overlap_l3=True)

    def test_in_place_restore_brings_the_pool_back(self):
        s = self.scheduler()
        s.pd_transport_deferred = s.hicache_storage_deferred = False
        s.disagg_prefill_bootstrap_queue = SimpleNamespace(
            kv_manager=SimpleNamespace(buffers_registered=True)
        )
        ct.build_transport_on_restore(s, None)
        self.assertEqual(len(self.ops.registered), 7)


if __name__ == "__main__":
    unittest.main()
