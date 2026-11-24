"""PD transport and HiCache L3 client deferred to /checkpoint/restore.

GPU-free: the TransferEngine, the KV managers and the L3 attach are stubs. What
is checked is the contract around them: what boots, what restore builds and in
which order, and that every registration failure fails the restore.
"""

import asyncio
import contextlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation import checkpoint_transport as ct
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    TransferBackend,
)
from sglang.srt.distributed.device_communicators import mooncake_transfer_engine as mte
from sglang.srt.managers.io_struct import (
    AttachHiCacheStorageReqOutput,
    CheckpointReqInput,
    CheckpointReqType,
    CheckpointState,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.checkpoint_manager import (
    SchedulerCheckpointManager,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# These tests build CPU tensors without naming the device. Another test in the
# same process may have left torch's default device set to cuda; pin it to CPU
# for this module and give the previous setting back afterwards.
_cpu_default_device = contextlib.ExitStack()


def setUpModule():
    _cpu_default_device.enter_context(torch.device("cpu"))


def tearDownModule():
    _cpu_default_device.close()


_ENVS = (
    "SGLANG_CHECKPOINT_DEFER_PD_TRANSPORT",
    "SGLANG_CHECKPOINT_DEFER_HICACHE_STORAGE",
    "SGLANG_HOST_IP",
    "MOONCAKE_LOCAL_HOSTNAME",
    "SGLANG_LOCAL_IP_NIC",
    "SGLANG_CHECKPOINT_RESTORE_KV_FILL",
    "SGLANG_CHECKPOINT_LOG_KV_REGIONS",
    # Read by the code under test too; unset so the runner's environment
    # cannot change the path a test takes.
    "SGLANG_CHECKPOINT_RESTORE_OVERLAP_L3",
    "SGLANG_CHECKPOINT_RELEASE_HOST_POOL",
    "SGLANG_CHECKPOINT_KV_REGION_IMAGE",
    "SGLANG_RUST_SERVER",
)


class _EnvCase(unittest.TestCase):
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

    def config(self, **overrides):
        """Patch the runtime-context accessors the module reads."""
        cfg = dict(
            checkpoint_prepare_on_startup=True,
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
            disaggregation_ib_device="mlx5_0,mlx5_2",
            enable_hierarchical_cache=False,
            hicache_storage_backend=None,
            hicache_storage_backend_extra_config='{"master_server_address": "m:50051"}',
            hicache_storage_prefetch_policy="timeout",
            hicache_write_policy="write_through",
            gpu_id=3,
        )
        cfg.update(overrides)
        bag = SimpleNamespace(**cfg, moe=SimpleNamespace(mooncake_ib_device=None))
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
        return bag


class TestPredicates(_EnvCase):
    def test_pd_transport_deferred_only_with_env_prepare_and_pd_mode(self):
        self.config()
        self.assertFalse(ct.defers_pd_transport())
        os.environ["SGLANG_CHECKPOINT_DEFER_PD_TRANSPORT"] = "1"
        self.assertTrue(ct.defers_pd_transport())
        self.config(disaggregation_mode="null")
        self.assertFalse(ct.defers_pd_transport())
        self.config(checkpoint_prepare_on_startup=False)
        self.assertFalse(ct.defers_pd_transport())

    def test_hicache_storage_deferred_only_with_an_l3_backend(self):
        os.environ["SGLANG_CHECKPOINT_DEFER_HICACHE_STORAGE"] = "1"
        self.config(enable_hierarchical_cache=True)
        self.assertFalse(ct.defers_hicache_storage())
        self.config(enable_hierarchical_cache=True, hicache_storage_backend="mooncake")
        self.assertTrue(ct.defers_hicache_storage())


def _cfg(**overrides):
    cfg = dict(
        checkpoint_prepare_on_startup=True,
        disaggregation_mode="decode",
        disaggregation_transfer_backend="mooncake",
        enable_pd_role_switch=False,
        enable_hierarchical_cache=False,
        hicache_storage_backend=None,
        elastic_ep_backend=None,
        enable_elastic_expert_backup=False,
        encoder_only=False,
        language_only=False,
        encoder_transfer_backend="zmq_to_scheduler",
        disaggregation_decode_enable_offload_kvcache=False,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


class TestValidation(_EnvCase):
    def test_flags_without_their_subject_are_no_ops(self):
        os.environ["SGLANG_CHECKPOINT_DEFER_PD_TRANSPORT"] = "1"
        os.environ["SGLANG_CHECKPOINT_DEFER_HICACHE_STORAGE"] = "1"
        ct.validate_checkpoint_deferral(
            _cfg(disaggregation_mode="null", checkpoint_prepare_on_startup=False)
        )

    def test_valid_decode_and_prefill_with_l3(self):
        os.environ["SGLANG_CHECKPOINT_DEFER_PD_TRANSPORT"] = "1"
        ct.validate_checkpoint_deferral(_cfg())
        os.environ["SGLANG_CHECKPOINT_DEFER_HICACHE_STORAGE"] = "1"
        ct.validate_checkpoint_deferral(
            _cfg(
                disaggregation_mode="prefill",
                enable_hierarchical_cache=True,
                hicache_storage_backend="mooncake",
            )
        )

    def test_refusals(self):
        os.environ["SGLANG_CHECKPOINT_DEFER_PD_TRANSPORT"] = "1"
        cases = {
            "checkpoint-prepare-on-startup": _cfg(checkpoint_prepare_on_startup=False),
            "transfer-backend mooncake": _cfg(disaggregation_transfer_backend="nixl"),
            "pd-role-switch": _cfg(enable_pd_role_switch=True),
            "elastic EP": _cfg(elastic_ep_backend="mooncake"),
            "encoder": _cfg(language_only=True, encoder_transfer_backend="mooncake"),
            # The L3 client would open RDMA devices at boot.
            "DEFER_HICACHE_STORAGE": _cfg(
                enable_hierarchical_cache=True, hicache_storage_backend="mooncake"
            ),
        }
        for needle, cfg in cases.items():
            with self.subTest(needle=needle):
                with self.assertRaisesRegex(ValueError, needle):
                    ct.validate_checkpoint_deferral(cfg)

    def test_l3_refusals(self):
        os.environ["SGLANG_CHECKPOINT_DEFER_HICACHE_STORAGE"] = "1"
        base = dict(
            disaggregation_mode="null",
            enable_hierarchical_cache=True,
            hicache_storage_backend="mooncake",
        )
        ct.validate_checkpoint_deferral(_cfg(**base))
        for needle, override in {
            "checkpoint-prepare-on-startup": dict(checkpoint_prepare_on_startup=False),
            "storage-backend mooncake": dict(hicache_storage_backend="file"),
            "offload-kvcache": dict(disaggregation_decode_enable_offload_kvcache=True),
        }.items():
            with self.subTest(needle=needle):
                with self.assertRaisesRegex(ValueError, needle):
                    ct.validate_checkpoint_deferral(_cfg(**{**base, **override}))


class TestHostIp(_EnvCase):
    def test_explicit_ip_wins_and_is_validated(self):
        self.assertEqual(ct.resolve_restore_host_ip("10.1.2.3"), "10.1.2.3")
        with self.assertRaises(ValueError):
            ct.resolve_restore_host_ip("not-an-ip")

    def test_detection_ignores_the_env_captured_at_checkpoint(self):
        # get_local_ip_auto() would return this stale value first.
        os.environ["SGLANG_HOST_IP"] = "10.0.0.1"
        with patch(
            "sglang.srt.utils.network.get_local_ip_by_remote", return_value="10.9.9.9"
        ):
            self.assertEqual(ct.resolve_restore_host_ip(None), "10.9.9.9")


class _Engine:
    def __init__(self, hostname, gpu_id, ib_device):
        self.hostname, self.gpu_id, self.ib_device = hostname, gpu_id, ib_device

    def get_session_id(self):
        return f"{self.hostname}:15000"

    def get_ib_device(self):
        return self.ib_device


def _kv_args(**overrides):
    fields = dict(
        kv_data_ptrs=[0x1000, 0x9000, 0x20000],
        kv_data_lens=[0x800, 0x800, 0x100],
        kv_item_lens=[8, 8, 4],
        num_draft_entries=1,
        aux_data_ptrs=[0x7F0000],
        aux_data_lens=[64],
        aux_item_lens=[16],
        state_types=[SimpleNamespace(value="swa")],
        state_data_ptrs=[[0x40000, 0x50000]],
        state_data_lens=[[0x200, 0]],
        state_item_lens=[[2, 2]],
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _queue(calls, **manager_fields):
    queue = SimpleNamespace(
        transfer_backend=TransferBackend.FAKE,
        kv_manager=SimpleNamespace(kv_args=_kv_args()),
    )

    def init_kv_manager():
        # The real manager reads the IP through get_local_ip_auto().
        calls.append(
            ("kv_manager", queue.transfer_backend, os.environ["SGLANG_HOST_IP"])
        )
        return SimpleNamespace(
            buffers_registered=True,
            buffer_registration_seconds=0.5,
            num_registered_regions=64,
            local_ip=os.environ["SGLANG_HOST_IP"],
            rank_port=4242,
            kv_args=_kv_args(),
            **manager_fields,
        )

    queue._init_kv_manager = init_kv_manager
    return queue


def _scheduler(mode, queue, *, defer_pd=True, defer_l3=False):
    s = SimpleNamespace(
        disaggregation_mode=mode,
        transfer_backend=TransferBackend.FAKE if defer_pd else TransferBackend.MOONCAKE,
        pd_transport_deferred=defer_pd,
        hicache_storage_deferred=defer_l3,
        disagg_decode_prealloc_queue=None,
        disagg_prefill_bootstrap_queue=None,
    )
    if mode == DisaggregationMode.DECODE:
        s.disagg_decode_prealloc_queue = queue
    elif mode == DisaggregationMode.PREFILL:
        s.disagg_prefill_bootstrap_queue = queue
    return s


class TestRestoreBuildsTransport(_EnvCase):
    def setUp(self):
        super().setUp()
        self.calls = []

        def init_engine(**kwargs):
            self.calls.append(("engine", kwargs["hostname"], kwargs["gpu_id"]))
            return _Engine(**kwargs)

        p = patch.object(mte, "init_mooncake_transfer_engine", side_effect=init_engine)
        p.start()
        self.addCleanup(p.stop)

    def test_decode_builds_engine_then_kv_manager_with_the_new_ip(self):
        self.config()
        os.environ["SGLANG_HOST_IP"] = "10.0.0.1"  # captured with the old pod
        queue = _queue(self.calls)
        s = _scheduler(DisaggregationMode.DECODE, queue)
        with self.assertLogs(ct.logger, level="INFO") as logs:
            ct.build_transport_on_restore(s, "10.2.0.7")
        self.assertEqual(
            self.calls,
            [
                ("engine", "10.2.0.7", 3),
                ("kv_manager", TransferBackend.MOONCAKE, "10.2.0.7"),
            ],
        )
        self.assertEqual(s.transfer_backend, TransferBackend.MOONCAKE)
        self.assertFalse(s.pd_transport_deferred)
        phases = [
            line.split("phase=")[1].split()[0]
            for line in logs.output
            if "phase=" in line
        ]
        self.assertEqual(
            phases,
            [
                "kv_reinit",
                "host_ip",
                "transport_init",
                "mr_registration",
                "kv_manager",
                "total",
            ],
        )

    def test_failed_memory_registration_fails_the_restore(self):
        self.config()
        queue = _queue(self.calls)
        init = queue._init_kv_manager
        queue._init_kv_manager = lambda: SimpleNamespace(
            **{**vars(init()), "buffers_registered": False}
        )
        s = _scheduler(DisaggregationMode.DECODE, queue)
        with self.assertRaisesRegex(RuntimeError, "did not register"):
            ct.build_transport_on_restore(s, "10.2.0.7")
        # Requests stay refused.
        self.assertTrue(s.pd_transport_deferred)

    def test_prefill_fails_on_a_failed_bootstrap_registration(self):
        self.config(disaggregation_mode="prefill")
        queue = _queue(
            self.calls, bootstrap_registered=False, bootstrap_registration_seconds=31.0
        )
        s = _scheduler(DisaggregationMode.PREFILL, queue)
        with self.assertRaisesRegex(RuntimeError, "bootstrap"):
            ct.build_transport_on_restore(s, "10.2.0.7")

    def test_prefill_attaches_l3_with_the_original_args_after_the_transport(self):
        bag = self.config(
            disaggregation_mode="prefill",
            enable_hierarchical_cache=True,
            hicache_storage_backend="mooncake",
        )
        os.environ["MOONCAKE_LOCAL_HOSTNAME"] = "10.0.0.1"
        queue = _queue(self.calls, bootstrap_registered=True)
        s = _scheduler(DisaggregationMode.PREFILL, queue, defer_l3=True)

        def attach(req):
            self.calls.append(("l3", req, os.environ["MOONCAKE_LOCAL_HOSTNAME"]))
            return AttachHiCacheStorageReqOutput(success=True)

        s.attach_hicache_storage_wrapped = attach
        ct.build_transport_on_restore(s, "10.2.0.7")
        self.assertEqual([c[0] for c in self.calls], ["engine", "kv_manager", "l3"])
        _, req, hostname = self.calls[-1]
        self.assertEqual(hostname, "10.2.0.7")
        self.assertEqual(req.hicache_storage_backend, "mooncake")
        self.assertEqual(
            req.hicache_storage_backend_extra_config_json,
            bag.hicache_storage_backend_extra_config,
        )
        self.assertEqual(req.hicache_write_policy, "write_through")
        self.assertEqual(req.hicache_storage_prefetch_policy, "timeout")
        self.assertFalse(s.hicache_storage_deferred)

    def test_failed_l3_attach_fails_the_restore(self):
        self.config(enable_hierarchical_cache=True, hicache_storage_backend="mooncake")
        s = _scheduler(DisaggregationMode.NULL, None, defer_pd=False, defer_l3=True)
        s.attach_hicache_storage_wrapped = lambda req: AttachHiCacheStorageReqOutput(
            success=False, message="master unreachable"
        )
        with self.assertRaisesRegex(RuntimeError, "master unreachable"):
            ct.build_transport_on_restore(s, "10.2.0.7")
        self.assertEqual(self.calls, [])  # no PD transport in a colocated engine

    def test_in_place_restore_only_checks_the_registration(self):
        """Without deferral the resume already re-registered the KV memory."""
        manager = SimpleNamespace(buffers_registered=True)
        queue = SimpleNamespace(kv_manager=manager)
        s = _scheduler(DisaggregationMode.DECODE, queue, defer_pd=False)
        ct.build_transport_on_restore(s, None)
        manager.buffers_registered = False
        with self.assertRaisesRegex(RuntimeError, "did not register"):
            ct.build_transport_on_restore(s, None)
        self.assertEqual(self.calls, [])


class TestKvDiagnostics(_EnvCase):
    def _scheduler(self, kv_args):
        s = _scheduler(
            DisaggregationMode.DECODE,
            SimpleNamespace(kv_manager=SimpleNamespace(kv_args=kv_args)),
            defer_pd=False,
        )
        s.req_to_token_pool = SimpleNamespace(req_to_token=MagicMock())
        return s

    def _fill(self, s, mode):
        calls = []
        with (
            patch.object(ct, "_cu_memset_d8", side_effect=lambda *a: calls.append(a)),
            patch("torch.cuda.synchronize"),
        ):
            ct.fill_kv_region(s, mode)
        return calls

    def test_fill_covers_kv_draft_and_state_and_zeroes_req_to_token(self):
        s = self._scheduler(_kv_args())
        calls = self._fill(s, "poison")
        # The zero-length state region is skipped; aux (host memory) is not filled.
        self.assertEqual(
            calls,
            [
                (0x1000, 0xFF, 0x800),
                (0x9000, 0xFF, 0x800),
                (0x20000, 0xFF, 0x100),
                (0x40000, 0xFF, 0x200),
            ],
        )
        s.req_to_token_pool.req_to_token.zero_.assert_called_once()
        self.assertEqual({c[1] for c in self._fill(s, "zero")}, {0})
        self.assertEqual(self._fill(s, "none"), [])
        with self.assertRaises(ValueError):
            self._fill(s, "garbage")

    def test_restore_fills_before_building_the_transport(self):
        self.config()
        order = []
        s = self._scheduler(_kv_args())
        with (
            patch.object(
                ct, "fill_kv_region", side_effect=lambda s, m: order.append(("fill", m))
            ),
            patch.object(
                ct, "check_pd_registration", side_effect=lambda s: order.append("check")
            ),
        ):
            ct.build_transport_on_restore(s, None, None)
            os.environ["SGLANG_CHECKPOINT_RESTORE_KV_FILL"] = "zero"
            ct.build_transport_on_restore(s, None, None)
            ct.build_transport_on_restore(s, None, "poison")
        self.assertEqual(
            order,
            [
                ("fill", "none"),
                "check",
                ("fill", "zero"),
                "check",
                ("fill", "poison"),
                "check",
            ],
        )

    def test_kv_args_differences(self):
        self.assertEqual(ct.kv_args_differences(_kv_args(), _kv_args()), [])
        self.assertEqual(
            ct.kv_args_differences(_kv_args(), _kv_args(kv_data_lens=[1, 2, 3])),
            ["kv_data_lens"],
        )

    def test_region_log_is_off_by_default_and_counts_split_allocations(self):
        with self.assertNoLogs(ct.logger, level="INFO"):
            ct.log_kv_regions("t", _kv_args())
        os.environ["SGLANG_CHECKPOINT_LOG_KV_REGIONS"] = "1"
        # Only the first KV region sits inside one allocation.
        alloc = {0x1000: (0x1000, 0x800)}
        with (
            patch.object(ct, "_cu_allocation_of", side_effect=lambda p: alloc.get(p)),
            self.assertLogs(ct.logger, level="INFO") as logs,
        ):
            ct.log_kv_regions("t", _kv_args())
        lines = {l.split("kind=")[1].split()[0]: l for l in logs.output}
        self.assertEqual(set(lines), {"kv", "draft_kv", "state0_swa", "aux_host"})
        self.assertIn("n=2 bytes=4096", lines["kv"])
        self.assertIn("not_one_alloc=1", lines["kv"])
        self.assertIn("not_one_alloc=-", lines["aux_host"])


class TestKvInitialState(_EnvCase):
    """A memory saver resume maps fresh pages under the KV pools. What the pools
    write once at construction must be written again on restore: the C4
    compressor's -1 padding row (score -inf) was garbage after a resume, and
    the first long PD request then decoded wrong."""

    def _state_pool(self, ratio, online=False, rows=6):
        import torch

        from sglang.srt.mem_cache.deepseek_v4_compress_state import (
            CompressStatePool,
            KVAndScore,
        )

        pool = object.__new__(CompressStatePool)
        pool.ratio, pool.online = ratio, online
        pool.kv_score_buffer = KVAndScore(torch.full((rows, 8), 7.0))
        return pool

    def test_reset_initial_rows_matches_construction(self):
        import torch

        c4 = self._state_pool(4)
        c4.reset_initial_rows()
        buf = c4.kv_score_buffer
        self.assertTrue(torch.all(buf[-1].kv == 0))
        self.assertTrue(torch.all(buf[-1].score == float("-inf")))
        self.assertTrue(torch.all(buf[:-1].kv_score == 7.0))  # live rows untouched
        pair = self._state_pool(2)
        pair.reset_initial_rows()
        self.assertTrue(torch.all(pair.kv_score_buffer.score == float("-inf")))
        online = self._state_pool(128, online=True)
        online.reset_initial_rows()
        self.assertTrue(torch.all(online.kv_score_buffer.kv_score == 7.0))

    def test_restore_resets_target_and_draft_pools_and_padding_row(self):
        import torch

        target_c4, draft_c4, idx_c4 = (self._state_pool(4) for _ in range(3))
        target = SimpleNamespace(
            compress_state_pools=[None, target_c4],
            indexer_compress_state_pools=[idx_c4],
        )
        draft = SimpleNamespace(compress_state_pools=[draft_c4])
        req_to_token = torch.full((3, 4), 9, dtype=torch.int32)
        s = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: target),
            draft_worker=SimpleNamespace(primary_draft_kv_pool=draft),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        )
        with self.assertLogs(ct.logger, level="INFO") as logs:
            ct.restore_kv_initial_state(s)
        for pool in (target_c4, draft_c4, idx_c4):
            self.assertTrue(torch.all(pool.kv_score_buffer[-1].score == float("-inf")))
        self.assertTrue(torch.all(req_to_token[0] == 0))
        self.assertTrue(torch.all(req_to_token[1:] == 9))
        self.assertIn("phase=kv_reinit", logs.output[-1])
        self.assertIn("compress_state_pools=3", logs.output[-1])

    def test_every_restore_runs_it_after_the_fill(self):
        order = []
        s = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.NULL,
            pd_transport_deferred=False,
            hicache_storage_deferred=False,
        )
        with (
            patch.object(
                ct, "fill_kv_region", side_effect=lambda *a: order.append("fill")
            ),
            patch.object(
                ct,
                "apply_kv_region_image",
                side_effect=lambda *a: order.append("image"),
            ),
            patch.object(
                ct,
                "restore_kv_initial_state",
                side_effect=lambda *a: order.append("reinit"),
            ),
        ):
            ct.build_transport_on_restore(s, None, None)
        self.assertEqual(order, ["fill", "image", "reinit"])

    def test_region_image_is_captured_once_and_applied_on_restore(self):
        import os

        import sglang.srt.mem_cache.kv_region_image as kri

        s = SimpleNamespace()
        image = SimpleNamespace(segments=[1, 2], nbytes=10)
        with patch.object(kri, "capture_kv_region", return_value=image) as cap:
            ct.capture_kv_region_image(s)
            ct.capture_kv_region_image(s)  # a later prepare keeps the startup image
        cap.assert_called_once()
        self.assertIs(s.kv_region_image, image)
        with (
            patch.object(kri, "apply_kv_region") as apply,
            self.assertLogs(ct.logger, level="INFO") as logs,
        ):
            ct.apply_kv_region_image(s)
        apply.assert_called_once_with("kv_cache", image)
        self.assertIn("phase=kv_region_image", logs.output[-1])
        os.environ["SGLANG_CHECKPOINT_KV_REGION_IMAGE"] = "0"  # _EnvCase restores it
        off = SimpleNamespace()
        with patch.object(kri, "capture_kv_region") as cap:
            ct.capture_kv_region_image(off)
        cap.assert_not_called()
        ct.apply_kv_region_image(off)  # nothing captured, nothing applied


class TestCheckpointManagerAfterRestore(unittest.TestCase):
    def _manager(self, after_restore):
        return SchedulerCheckpointManager(
            enable_memory_saver=True,
            pp_size=1,
            offload_tags=set(),
            is_engine_paused=lambda: False,
            is_fully_idle=lambda **kw: True,
            release_memory_occupation=lambda req: None,
            resume_memory_occupation=lambda req: None,
            all_gather_object=lambda out: [out],
            after_restore=after_restore,
        )

    def test_hook_failure_latches_failed(self):
        seen = []

        def hook(req):
            seen.append(req.host_ip)
            raise RuntimeError("registration failed")

        m = self._manager(hook)
        m.handle(CheckpointReqInput(req_type=CheckpointReqType.PREPARE))
        out = m.handle(
            CheckpointReqInput(req_type=CheckpointReqType.RESTORE, host_ip="10.2.0.7")
        )
        self.assertEqual(seen, ["10.2.0.7"])
        self.assertEqual(out.state, CheckpointState.FAILED)
        self.assertIn("registration failed", out.message)

    def test_before_release_runs_before_the_kv_release_and_can_fail_it(self):
        calls = []
        m = self._manager(None)
        m._before_release = lambda: calls.append("capture")
        m._release_memory_occupation = lambda req: calls.append("release")
        m.handle(CheckpointReqInput(req_type=CheckpointReqType.PREPARE))
        self.assertEqual(calls, ["capture", "release"])

        def boom():
            raise RuntimeError("image too big")

        m = self._manager(None)
        m._before_release = boom
        out = m.handle(CheckpointReqInput(req_type=CheckpointReqType.PREPARE))
        self.assertEqual(out.state, CheckpointState.FAILED)
        self.assertIn("image too big", out.message)

    def test_hook_runs_once_per_restore(self):
        hook = MagicMock()
        m = self._manager(hook)
        m.handle(CheckpointReqInput(req_type=CheckpointReqType.PREPARE))
        m.handle(CheckpointReqInput(req_type=CheckpointReqType.RESTORE))
        out = m.handle(CheckpointReqInput(req_type=CheckpointReqType.RESTORE))
        self.assertEqual(out.state, CheckpointState.RUNNING)
        hook.assert_called_once()

    def test_tokenizer_forwards_host_ip(self):
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
                TokenizerControlMixin.checkpoint_restore(tokenizer, "10.2.0.7", "zero")
            )
        self.assertEqual(
            [(o.req_type, o.host_ip, o.kv_fill) for o in sent],
            [
                (CheckpointReqType.STATE, None, None),
                (CheckpointReqType.RESTORE, "10.2.0.7", "zero"),
            ],
        )


class TestRegistrationOutcomes(unittest.TestCase):
    def _mooncake_manager(self, ret):
        m = object.__new__(MooncakeKVManager)
        m.kv_args = SimpleNamespace(
            kv_data_ptrs=[1000, 2000],
            kv_data_lens=[10, 10],
            aux_data_ptrs=[3000],
            aux_data_lens=[5],
            state_data_ptrs=[],
            state_data_lens=[],
        )
        m.engine = MagicMock()
        m.engine.batch_register.return_value = ret
        return m

    def test_mooncake_records_the_batch_register_result(self):
        ok = self._mooncake_manager(0)
        ok.register_buffer_to_engine()
        self.assertTrue(ok.buffers_registered)
        self.assertEqual(ok.num_registered_regions, 3)
        bad = self._mooncake_manager(-1)
        with self.assertLogs(level="ERROR"):
            bad.register_buffer_to_engine()
        self.assertFalse(bad.buffers_registered)
        ok.deregister_buffer_to_engine()
        self.assertFalse(ok.buffers_registered)

    def test_bootstrap_registration_outcome(self):
        m = object.__new__(CommonKVManager)
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.put",
            return_value=SimpleNamespace(status_code=200),
        ):
            m._register_topology_row("http://127.0.0.1:8998/route", {})
        self.assertTrue(m.bootstrap_registered)

        m = object.__new__(CommonKVManager)
        with (
            patch(
                "sglang.srt.disaggregation.common.conn.requests.put",
                side_effect=ConnectionError("refused"),
            ),
            patch("sglang.srt.disaggregation.common.conn.time.sleep"),
            self.assertLogs(level="ERROR"),
        ):
            m._register_topology_row("http://127.0.0.1:8998/route", {})
        self.assertFalse(m.bootstrap_registered)


class TestSharedEngineAtBoot(_EnvCase):
    def _run(self, **cfg):
        bag = SimpleNamespace(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
            disaggregation_ib_device=None,
            enable_hierarchical_cache=False,
            hicache_storage_backend=None,
            encoder_only=False,
            language_only=False,
            encoder_transfer_backend=None,
            moe=SimpleNamespace(
                enable_elastic_expert_backup=False,
                elastic_ep_backend=None,
                mooncake_ib_device=None,
            ),
            checkpoint_prepare_on_startup=True,
        )
        for k, v in cfg.items():
            setattr(bag, k, v)
        with (
            patch.object(mte, "init_mooncake_transfer_engine") as init,
            patch.object(mte, "get_local_ip_auto", return_value="10.0.0.1"),
        ):
            for name in ("get_disagg", "get_memory", "get_exec"):
                p = patch.object(mte, name, return_value=bag)
                p.start()
                self.addCleanup(p.stop)
            for name in ("get_disagg", "get_memory", "get_serving"):
                p = patch.object(ct, name, return_value=bag)
                p.start()
                self.addCleanup(p.stop)
            mte.maybe_init_shared_mooncake_transfer_engine(gpu_id=0)
        return init

    def test_no_engine_at_boot_when_deferred(self):
        self.assertTrue(self._run().called)
        os.environ["SGLANG_CHECKPOINT_DEFER_PD_TRANSPORT"] = "1"
        self.assertFalse(self._run().called)

    def test_l3_reuse_alone_is_deferred_too(self):
        os.environ["SGLANG_CHECKPOINT_DEFER_HICACHE_STORAGE"] = "1"
        init = self._run(
            disaggregation_mode="null",
            enable_hierarchical_cache=True,
            hicache_storage_backend="mooncake",
        )
        self.assertFalse(init.called)


class TestSchedulerBootsOnTheFakeBackend(unittest.TestCase):
    def _init_disaggregation(self, deferred):
        import sglang.srt.managers.scheduler as sched_mod

        s = Scheduler.__new__(Scheduler)
        s.draft_worker = None
        s.spec_algorithm = MagicMock()
        s.spec_algorithm.carries_draft_hidden_states.return_value = False
        s.model_config = MagicMock()
        s.req_to_token_pool = SimpleNamespace(size=8)
        s.server_args = SimpleNamespace(sampling_mask_max_tokens=0)
        s.token_to_kv_pool_allocator = MagicMock()
        s.tree_cache = MagicMock()
        s.max_total_num_tokens = 1024
        s.enable_unified_memory = False
        disagg = SimpleNamespace(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
            disaggregation_enable_kv_checksum=False,
            disaggregation_bootstrap_port=8998,
            num_reserved_decode_tokens=0,
            language_only=False,
            encoder_transfer_backend=None,
        )
        names = {
            "get_disagg": MagicMock(return_value=disagg),
            "get_device": MagicMock(return_value=SimpleNamespace(gpu_id=0)),
            "DecodePreallocQueue": MagicMock(),
            "DecodeTransferQueue": MagicMock(),
            "MetadataBuffers": MagicMock(),
            "ReqToMetadataIdxAllocator": MagicMock(),
            "get_dsa_seed_metadata_dim": MagicMock(return_value=0),
            "is_minimax_sparse": MagicMock(return_value=False),
        }
        with (
            patch.multiple(sched_mod, **names),
            patch.object(ct, "defers_pd_transport", return_value=deferred),
        ):
            Scheduler.init_disaggregation(s)
        kwargs = names["DecodePreallocQueue"].call_args.kwargs
        return s, kwargs["transfer_backend"]

    def test_deferred_decode_boots_on_fake(self):
        s, backend = self._init_disaggregation(deferred=True)
        self.assertTrue(s.pd_transport_deferred)
        self.assertEqual(backend, TransferBackend.FAKE)
        self.assertEqual(s.transfer_backend, TransferBackend.FAKE)

    def test_default_is_unchanged(self):
        s, backend = self._init_disaggregation(deferred=False)
        self.assertFalse(s.pd_transport_deferred)
        self.assertEqual(backend, TransferBackend.MOONCAKE)


class TestSchedulerRefusesRealRequestsWhileDeferred(unittest.TestCase):
    def _scheduler(self):
        s = Scheduler.__new__(Scheduler)
        s.pd_transport_deferred = True
        s.disaggregation_mode = DisaggregationMode.DECODE
        s._set_or_validate_priority = lambda req: True
        s._release_aborted_request = MagicMock()
        s._reject_while_pd_transport_deferred = MagicMock()
        s.disagg_decode_prealloc_queue = MagicMock()
        return s

    def test_real_request_is_rejected_and_warmup_passes(self):
        s = self._scheduler()
        real = SimpleNamespace(bootstrap_host="10.0.0.5", time_stats=MagicMock())
        Scheduler._add_request_to_queue(s, real)
        s._reject_while_pd_transport_deferred.assert_called_once_with(real)
        s.disagg_decode_prealloc_queue.add.assert_not_called()

        warmup = SimpleNamespace(
            bootstrap_host=FAKE_BOOTSTRAP_HOST, time_stats=MagicMock()
        )
        Scheduler._add_request_to_queue(s, warmup)
        s.disagg_decode_prealloc_queue.add.assert_called_once()


class TestStartupPrepareWaitsForEveryTokenizerWorker(unittest.TestCase):
    """EP8 decoder, --tokenizer-worker-num=4: each worker warms up on its own.
    The first worker's startup prepare released the KV cache while a later
    worker's PD warmup still ran batches, and the schedulers crashed."""

    def _tokenizer(self, workers=4):
        calls = []
        manager = SchedulerCheckpointManager(
            enable_memory_saver=True,
            pp_size=1,
            offload_tags=set(),
            is_engine_paused=lambda: False,
            is_fully_idle=lambda **kw: True,
            release_memory_occupation=lambda req: calls.append("release"),
            resume_memory_occupation=lambda req: None,
            all_gather_object=lambda out: [out],
        )

        async def communicator(obj):
            return [manager.handle(obj)]

        async def pause_generation(obj):
            calls.append("pause")

        tokenizer = SimpleNamespace(
            auto_create_handle_loop=lambda: None,
            checkpoint_lock=asyncio.Lock(),
            is_pause=False,
            checkpoint_communicator=communicator,
            pause_generation=pause_generation,
        )
        tokenizer._call_checkpoint = lambda req_type, **kw: (
            TokenizerControlMixin._call_checkpoint(tokenizer, req_type, **kw)
        )
        serving = SimpleNamespace(tokenizer_worker_num=workers)
        p = patch(
            "sglang.srt.managers.tokenizer_control_mixin.get_serving",
            return_value=serving,
        )
        p.start()
        self.addCleanup(p.stop)
        return tokenizer, manager, calls

    def _startup(self, tokenizer, pid):
        return asyncio.run(
            TokenizerControlMixin.checkpoint_prepare(tokenizer, "abort", pid)
        )

    def test_only_the_last_workers_report_prepares(self):
        tokenizer, manager, calls = self._tokenizer(workers=4)
        for pid in (101, 102, 102, 103):  # a retried report counts once
            out = self._startup(tokenizer, pid)
            self.assertEqual(out.state, CheckpointState.RUNNING)
            self.assertIn("/4", out.message)
        self.assertEqual(calls, [])  # nothing paused, nothing released
        out = self._startup(tokenizer, 104)
        self.assertEqual(out.state, CheckpointState.PREPARED)
        self.assertEqual(calls, ["pause", "release"])
        self.assertEqual(manager.state, CheckpointState.PREPARED)

    def test_one_worker_prepares_at_once(self):
        tokenizer, _, calls = self._tokenizer(workers=1)
        self.assertEqual(self._startup(tokenizer, 7).state, CheckpointState.PREPARED)
        self.assertEqual(calls, ["pause", "release"])

    def test_merge_reports_the_fewest_ready_workers(self):
        from sglang.srt.managers.io_struct import CheckpointReqOutput

        outs = [
            CheckpointReqOutput(
                success=True, state=CheckpointState.RUNNING, ready_workers=n
            )
            for n in (3, 2, 3)
        ]
        self.assertEqual(CheckpointReqOutput.merge(outs).ready_workers, 2)


class TestHttpStartupPrepare(unittest.TestCase):
    def _run(self, post, gets):
        import sglang.srt.entrypoints.http_server as hs

        server_args = SimpleNamespace(url=lambda: "http://127.0.0.1:30000")
        resp = lambda code, body: SimpleNamespace(  # noqa: E731
            status_code=code, json=lambda: body, text=str(body)
        )
        get_iter = iter(resp(*g) for g in gets)
        with (
            patch.object(hs.requests, "post", return_value=resp(*post)) as p,
            patch.object(
                hs.requests, "get", side_effect=lambda *a, **k: next(get_iter)
            ),
            patch.object(hs.time, "sleep"),
            patch.object(hs, "ssl_verify_of", return_value=True),
            patch.object(
                hs,
                "get_serving",
                return_value=SimpleNamespace(admin_api_key=None, api_key=None),
            ),
        ):
            ok = hs._prepare_on_startup(server_args)
        self.assertIn("startup_worker=", p.call_args.args[0])
        return ok

    def test_last_worker_returns_when_the_prepare_starts(self):
        self.assertTrue(self._run((200, {"state": "preparing"}), []))

    def test_earlier_worker_waits_for_prepared(self):
        self.assertTrue(
            self._run(
                (200, {"state": "running", "message": "1/4"}), [(503, {}), (200, {})]
            )
        )
        self.assertFalse(
            self._run(
                (200, {"state": "running", "message": "1/4"}),
                [(500, {"state": "failed"})],
            )
        )

    def test_refusal_fails_the_boot(self):
        self.assertFalse(self._run((400, {"state": "running"}), []))


class TestSchedulerRefusesRequestsDuringAbortPrepare(unittest.TestCase):
    def test_refusal_rule(self):
        for state, paused, refuse in (
            (CheckpointState.RUNNING, False, False),
            (CheckpointState.PREPARING, False, True),
            (CheckpointState.PREPARED, False, True),
            (CheckpointState.PREPARED, True, False),  # retract: held, not run
        ):
            s = SimpleNamespace(
                checkpoint_manager=SimpleNamespace(state=state), _engine_paused=paused
            )
            self.assertEqual(ct.checkpoint_refuses_new_requests(s), refuse, state)
        self.assertFalse(ct.checkpoint_refuses_new_requests(SimpleNamespace()))

    def test_warmup_request_after_prepare_is_not_queued(self):
        s = Scheduler.__new__(Scheduler)
        s.checkpoint_manager = SimpleNamespace(state=CheckpointState.PREPARED)
        s._engine_paused = False
        s.disaggregation_mode = DisaggregationMode.DECODE
        s._set_or_validate_priority = lambda req: True
        s._release_aborted_request = MagicMock()
        s._reject_while_pd_transport_deferred = MagicMock()
        s.disagg_decode_prealloc_queue = MagicMock()
        warmup = SimpleNamespace(
            bootstrap_host=FAKE_BOOTSTRAP_HOST, time_stats=MagicMock()
        )
        Scheduler._add_request_to_queue(s, warmup)
        s._reject_while_pd_transport_deferred.assert_called_once()
        self.assertIn(
            "process checkpoint",
            s._reject_while_pd_transport_deferred.call_args.args[1],
        )
        s.disagg_decode_prealloc_queue.add.assert_not_called()


if __name__ == "__main__":
    unittest.main()
