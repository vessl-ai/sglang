"""CPU-only tests for the INF-834 prefill boot levers (no server, no GPU):

- SGLANG_OPT_INF834_EXPERT_BULK_LOAD: the blockwise expert loader and the regex
  expert route must leave exactly the bytes FusedMoE.weight_loader leaves,
  including through DeepseekV4ForCausalLM.load_weights itself.
- SGLANG_OPT_INF834_ENGRAM_PARALLEL_COPY: the threaded row copy equals copy_.
- SGLANG_OPT_INF834_HOST_POOL_PREALLOC: take/finalize/record bookkeeping.
- SGLANG_OPT_INF834_DEEPGEMM_SWEEP_STAMP: a sweep runs once per stamp.
"""

import os
import tempfile
import threading
import types
import unittest
from unittest import mock

import torch
from torch import nn

from sglang.srt.layers.moe.fused_moe_triton.layer import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)
from sglang.srt.model_loader import inf834_expert_bulk
from sglang.srt.model_loader.inf834_expert_bulk import (
    Inf834ExpertBulkLoader,
    make_expert_route,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

E8M0 = torch.float8_e8m0fnu
# Small stand-ins for the DeepSeek-V4.1 expert shapes (2304 x 2560 / 5120 x 1152).
INTER, HID_BYTES, HID, INTER_BYTES = 16, 12, 20, 8
SCALE_COLS_13, SCALE_COLS_2 = 3, 2


class Mxfp4MarlinMoEMethod:  # the class name is what _weight_loader_impl looks at
    pass


def make_experts(num_experts, tp_size, tp_rank, method=Mxfp4MarlinMoEMethod):
    """A real FusedMoE (no __init__) with host parameters shaped as
    Mxfp4MarlinMoEMethod.create_weights shapes them for one TP rank."""
    e = object.__new__(FusedMoE)
    nn.Module.__init__(e)
    e.layer_id = 0
    e.uses_expert_location = True
    e.num_fused_shared_experts = 0
    e._has_fused_shared = False
    e._num_global_routed = num_experts
    e._expert_storage_rank = 0
    e._num_local_routed = num_experts
    e.num_local_experts = num_experts
    e.moe_tp_size, e.moe_tp_rank = tp_size, tp_rank
    e.use_triton_kernels = False
    e.use_flashinfer_trtllm_moe = False
    e.use_presharded_weights = False
    e.__dict__["use_padded_loading"] = False  # cached_property: True on a CPU host
    e.quant_method = method()
    e.quant_config = None
    e.moe_runner_config = types.SimpleNamespace(is_gated=True)
    inter = INTER // tp_size
    params = {
        "w13_weight": torch.zeros(num_experts, 2 * inter, HID_BYTES, dtype=torch.int8),
        "w2_weight": torch.zeros(
            num_experts, HID, INTER_BYTES // tp_size, dtype=torch.int8
        ),
        "w13_weight_scale_inv": torch.zeros(
            num_experts, 2 * inter, SCALE_COLS_13, dtype=torch.uint8
        ).view(E8M0),
        "w2_weight_scale_inv": torch.zeros(
            num_experts, HID, SCALE_COLS_2 * 4 // tp_size, dtype=torch.uint8
        ).view(E8M0),
    }
    for name, value in params.items():
        p = nn.Parameter(value, requires_grad=False)
        p.weight_loader = e.weight_loader
        if "scale" in name:
            p.quant_method = FusedMoeWeightScaleSupported.BLOCK.value
        e.register_parameter(name, p)
    return e


def checkpoint_expert(gen):
    """One expert's six checkpoint tensors (weights I8, scales E8M0), full width."""

    def rnd(*shape):
        return torch.randint(0, 255, shape, generator=gen, dtype=torch.uint8)

    return {
        ("w1", False): rnd(INTER, HID_BYTES).view(torch.int8),
        ("w3", False): rnd(INTER, HID_BYTES).view(torch.int8),
        ("w2", False): rnd(HID, INTER_BYTES).view(torch.int8),
        ("w1", True): rnd(INTER, SCALE_COLS_13).view(E8M0),
        ("w3", True): rnd(INTER, SCALE_COLS_13).view(E8M0),
        ("w2", True): rnd(HID, SCALE_COLS_2 * 4).view(E8M0),
    }


def param_for(experts, shard_id, is_scale):
    stem = "w2" if shard_id == "w2" else "w13"
    name = f"{stem}_weight_scale_inv" if is_scale else f"{stem}_weight"
    return getattr(experts, name), f"model.layers.0.mlp.experts.{name}"


def lexicographic(n):
    return sorted(range(n), key=str)  # the order sorted checkpoint names arrive in


def state(experts):
    return {k: v.data.view(torch.uint8).clone() for k, v in experts.named_parameters()}


class TestExpertBulkLoader(CustomTestCase):
    def _load(self, num_experts, tp_size, tp_rank, block, skip=(), bulk=True):
        gen = torch.Generator().manual_seed(1234)
        ckpt = {e: checkpoint_expert(gen) for e in range(num_experts)}
        experts = make_experts(num_experts, tp_size, tp_rank)
        pending = []
        loader = None
        if bulk:
            with mock.patch.dict(
                os.environ, {"SGLANG_OPT_INF834_EXPERT_BULK_BLOCK": str(block)}
            ):
                loader = Inf834ExpertBulkLoader(
                    lambda func, *args: pending.append((func, args))
                )
        for e in lexicographic(num_experts):
            for (shard_id, is_scale), tensor in ckpt[e].items():
                if (e, shard_id, is_scale) in skip:
                    continue
                param, name = param_for(experts, shard_id, is_scale)
                if loader is None or not loader.offer(param, tensor, name, shard_id, e):
                    experts.weight_loader(param, tensor, name, shard_id, e)
        if loader is not None:
            loader.finish()
            for func, args in pending:
                func(*args)
            loader.close("test")
        return experts, loader

    def test_same_bytes_as_weight_loader(self):
        for tp_size, tp_rank in ((1, 0), (4, 0), (4, 3), (2, 1)):
            for block in (1, 4, 5, 64):
                with self.subTest(tp_size=tp_size, tp_rank=tp_rank, block=block):
                    want, _ = self._load(11, tp_size, tp_rank, block, bulk=False)
                    got, loader = self._load(11, tp_size, tp_rank, block)
                    self.assertEqual(loader.tensors_bulk, 11 * 6)
                    self.assertEqual(loader.tensors_fallback, 0)
                    for (k, a), b in zip(state(want).items(), state(got).values()):
                        self.assertTrue(torch.equal(a, b), k)
                    self.assertGreater(int(state(got)["w13_weight"].sum()), 0)

    def test_incomplete_block_falls_back_to_the_same_bytes(self):
        skip = {(7, "w3", False), (2, "w2", True)}
        want, _ = self._load(11, 4, 1, 4, skip=skip, bulk=False)
        got, loader = self._load(11, 4, 1, 4, skip=skip)
        self.assertGreater(loader.tensors_fallback, 0)
        self.assertGreater(loader.tensors_bulk, 0)
        self.assertEqual(loader.tensors_bulk + loader.tensors_fallback, 11 * 6 - 2)
        for (k, a), b in zip(state(want).items(), state(got).values()):
            self.assertTrue(torch.equal(a, b), k)

    def test_not_offered_outside_the_plain_path(self):
        gen = torch.Generator().manual_seed(1)
        tensors = checkpoint_expert(gen)

        def offered(experts, shard_id="w1", is_scale=False, tensor=None):
            loader = Inf834ExpertBulkLoader(lambda func, *args: None)
            param, name = param_for(experts, shard_id, is_scale)
            tensor = tensors[(shard_id, is_scale)] if tensor is None else tensor
            return loader.offer(param, tensor, name, shard_id, 0)

        self.assertTrue(offered(make_experts(4, 4, 0)))
        other = make_experts(4, 4, 0, method=type("Fp8MoEMethod", (), {}))
        self.assertFalse(offered(other))
        for attr, value in (
            ("use_presharded_weights", True),
            ("use_triton_kernels", True),
            ("use_flashinfer_trtllm_moe", True),
            ("_has_fused_shared", True),
            ("scheme", object()),
        ):
            experts = make_experts(4, 4, 0)
            setattr(experts, attr, value)
            self.assertFalse(offered(experts), attr)
        experts = make_experts(4, 4, 0)
        experts.__dict__["use_padded_loading"] = True
        self.assertFalse(offered(experts))
        experts = make_experts(4, 4, 0)
        experts.w13_weight._sglang_require_global_experts = True
        self.assertFalse(offered(experts))
        # A checkpoint tensor narrower than the parameter (padded buffer) is not ours.
        self.assertFalse(
            offered(make_experts(4, 4, 0), tensor=tensors[("w1", False)][:, :-4])
        )
        # Another dtype would be converted by copy_; leave that to the loader.
        self.assertFalse(
            offered(make_experts(4, 4, 0), tensor=tensors[("w1", False)].float())
        )
        with mock.patch(
            "sglang.srt.eplb.expert_location.get_global_expert_location_metadata",
            return_value=object(),
        ):
            self.assertFalse(offered(make_experts(4, 4, 0)))

    def test_expert_without_local_slot_is_dropped_like_weight_loader(self):
        experts = make_experts(4, 1, 0)
        experts._expert_storage_rank = 1  # this rank stores global experts 4..7
        loader = Inf834ExpertBulkLoader(lambda func, *args: None)
        gen = torch.Generator().manual_seed(1)
        param, name = param_for(experts, "w1", False)
        before = state(experts)
        self.assertTrue(
            loader.offer(param, checkpoint_expert(gen)[("w1", False)], name, "w1", 0)
        )
        self.assertEqual(loader.tensors_noop, 1)
        for (k, a), b in zip(before.items(), state(experts).values()):
            self.assertTrue(torch.equal(a, b), k)


class TestExpertRoute(CustomTestCase):
    def setUp(self):
        self.mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=384,
        )

    @staticmethod
    def scan(mapping, name):
        return tuple(m for m in mapping if m[1] in name)

    def test_route_matches_the_scan(self):
        route = make_expert_route(self.mapping)
        names = [
            f"model.layers.{layer}.mlp.experts.{e}.{proj}.{leaf}"
            for layer in (0, 39)
            for e in (0, 1, 7, 17, 70, 100, 383, 384, 1000)
            for proj in ("gate_proj", "down_proj", "up_proj", "side_proj")
            for leaf in ("weight", "weight_scale_inv")
        ] + [
            "model.layers.3.mlp.shared_experts.gate_proj.weight",
            "model.layers.3.self_attn.wq_b.weight",
            "model.layers.3.mlp.experts.007.gate_proj.weight",
            "model.layers.3.mlp.experts.7.gate_proj",
            "model.layers.3.mlp.experts.7.gate_proj.experts.9.up_proj.weight",
            "experts.5.up_proj.",
            "",
        ]
        for name in names:
            # What the loop does with the entries: only those whose weight_name is a
            # substring matter, in mapping order.
            got = tuple(m for m in route(name) if m[1] in name)
            self.assertEqual(got, self.scan(self.mapping, name), name)

    def test_input_scale_entries_keep_the_scan_semantics(self):
        mapping = self.mapping + FusedMoE.make_expert_input_scale_params_mapping(
            num_experts=384
        )
        route = make_expert_route(mapping)
        self.assertIsNotNone(route)
        for name in (
            "model.layers.3.mlp.experts.7.w1.input_scale",
            "model.layers.3.mlp.experts.7.gate_proj.weight",
        ):
            got = tuple(m for m in route(name) if m[1] in name)
            self.assertEqual(got, self.scan(mapping, name), name)

    def test_other_mapping_shapes_keep_the_scan(self):
        self.assertIsNone(
            make_expert_route([("experts.w13_", "experts.gate.", 0, "w1")])
        )
        self.assertIsNone(make_expert_route([]))


class TestLoadWeightsIntegration(CustomTestCase):
    """DeepseekV4ForCausalLM.load_weights with the switch on and off, same stream."""

    LAYERS, EXPERTS = 2, 9

    def _model(self, tp_size, tp_rank):
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        m = object.__new__(DeepseekV4ForCausalLM)
        nn.Module.__init__(m)
        m.model = nn.Module()
        layers = []
        for _ in range(self.LAYERS):
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.experts = make_experts(self.EXPERTS, tp_size, tp_rank)
            layer.dense = nn.Parameter(torch.zeros(5, 3), requires_grad=False)
            layers.append(layer)
        m.model.layers = nn.ModuleList(layers)
        m.model.start_layer, m.model.end_layer = 0, self.LAYERS
        m.config = types.SimpleNamespace(
            n_routed_experts=self.EXPERTS,
            num_hidden_layers=self.LAYERS,
            model_type="deepseek_v41",
        )
        m.num_fused_shared_experts = 0
        m.wo_a_fp8 = True  # skip the streaming dequant wrapper
        m.quant_config = None
        m.vision = None
        m.pp_group = types.SimpleNamespace(is_first_rank=True, is_last_rank=True)
        m.post_load_weights = lambda **kw: None
        m._prewarm_mhc_kernels = lambda: None
        return m

    def _stream(self):
        gen = torch.Generator().manual_seed(99)
        out = {}
        for layer in range(self.LAYERS):
            for e in range(self.EXPERTS):
                for (shard_id, is_scale), tensor in checkpoint_expert(gen).items():
                    leaf = "scale" if is_scale else "weight"
                    out[f"layers.{layer}.ffn.experts.{e}.{shard_id}.{leaf}"] = tensor
            out[f"layers.{layer}.dense"] = torch.randn(5, 3, generator=gen)
        out["mtp.0.ffn.experts.0.w1.weight"] = torch.zeros(INTER, HID_BYTES).to(
            torch.int8
        )
        return [(k, out[k]) for k in sorted(out)]

    def _run(self, env, tp_size=4, tp_rank=2):
        model = self._model(tp_size, tp_rank)
        with mock.patch.dict(os.environ, env):
            with (
                self.assertLogs("sglang", level="INFO") if env else mock.MagicMock()
            ) as logs:
                model.load_weights(iter(self._stream()))
        text = "\n".join(logs.output) if env else ""
        return {k: v.data.clone() for k, v in model.named_parameters()}, text

    def test_bulk_switch_leaves_the_same_parameters(self):
        want, _ = self._run({})
        got, text = self._run(
            {
                "SGLANG_OPT_INF834_EXPERT_BULK_LOAD": "1",
                "SGLANG_OPT_INF834_EXPERT_BULK_BLOCK": "4",
                "SGLANG_OPT_INF834_LOAD_PROFILE": "1",
            }
        )
        self.assertEqual(sorted(want), sorted(got))
        for k in want:
            self.assertTrue(
                torch.equal(want[k].view(torch.uint8), got[k].view(torch.uint8)), k
            )
        self.assertGreater(
            int(got["model.layers.1.mlp.experts.w2_weight"].abs().sum()), 0
        )
        n = self.LAYERS * self.EXPERTS * 6
        self.assertIn(f"INF834 expert bulk load [target]: {n} tensors in", text)
        self.assertIn("; 0 left to the per-tensor loader", text)
        self.assertIn("INF834 load profile [target]", text)

    def test_block_larger_than_the_layer(self):
        want, _ = self._run({}, tp_size=1, tp_rank=0)
        got, _ = self._run(
            {"SGLANG_OPT_INF834_EXPERT_BULK_LOAD": "1"}, tp_size=1, tp_rank=0
        )
        for k in want:
            self.assertTrue(
                torch.equal(want[k].view(torch.uint8), got[k].view(torch.uint8)), k
            )


class TestEngramParallelCopy(CustomTestCase):
    def test_same_bytes_as_copy(self):
        from sglang.srt.layers.engram import inf834_copy_rows_parallel

        gen = torch.Generator().manual_seed(7)
        rows, dim = 3 * (72 << 20) // 1024 + 5, 1024  # a little over 3 x 64 MiB
        src = torch.randint(0, 255, (rows + 9, dim), generator=gen, dtype=torch.uint8)
        src = src.view(torch.float8_e4m3fn)[4 : 4 + rows]
        want = torch.empty(rows, dim, dtype=torch.float8_e4m3fn)
        want.copy_(src)
        got = torch.zeros(rows, dim, dtype=torch.uint8).view(torch.float8_e4m3fn)
        with mock.patch.dict(
            os.environ, {"SGLANG_OPT_INF834_ENGRAM_COPY_THREADS": "3"}
        ), self.assertLogs("sglang", level="INFO") as logs:
            inf834_copy_rows_parallel(got, src)
        self.assertTrue(torch.equal(want.view(torch.uint8), got.view(torch.uint8)))
        self.assertIn("INF834 engram parallel copy", "\n".join(logs.output))
        self.assertIn("on 3 threads", "\n".join(logs.output))

    def test_small_or_mismatched_falls_back_to_copy(self):
        from sglang.srt.layers.engram import inf834_copy_rows_parallel

        src = torch.arange(12, dtype=torch.uint8).view(3, 4)
        dst = torch.zeros(3, 4, dtype=torch.uint8)
        inf834_copy_rows_parallel(dst, src)
        self.assertTrue(torch.equal(dst, src))
        dst = torch.zeros(3, 4, dtype=torch.uint8)
        inf834_copy_rows_parallel(dst, src[:1])  # broadcast, as copy_ does
        self.assertTrue(torch.equal(dst, src[:1].expand(3, 4)))


class TestHostPoolPrealloc(CustomTestCase):
    def setUp(self):
        from sglang.srt.mem_cache import inf834_host_prealloc as hp

        self.hp = hp
        hp._reset_for_tests()
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(
            os.environ,
            {
                "SGLANG_OPT_INF834_HOST_POOL_PREALLOC": "1",
                "SGLANG_CACHE_DIR": self.tmp.name,
            },
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.hp._reset_for_tests()
        self.tmp.cleanup()

    def test_take_finalize_and_record(self):
        hp = self.hp
        allocated, freed = [], []
        gate = threading.Event()

        def alloc(size):
            gate.wait(5)
            allocated.append(size)
            return 1000 + len(allocated)

        hp.start([100, 200, 300, 200], alloc, freed.append)
        got = {}
        t = threading.Thread(target=lambda: got.setdefault("p", hp.take(100)))
        t.start()
        gate.set()  # the foreground waits for the segment that is being made
        t.join(5)
        self.assertEqual(got["p"], 1001)
        self.assertIsNone(hp.take(999))  # a size nobody recorded
        self.assertEqual({hp.take(200), hp.take(200)}, {1002, 1004})
        self.assertIsNone(hp.take(200))  # the record had only two of these
        self.assertEqual(hp.unclaimed_bytes(), 300)
        with self.assertLogs("sglang", level="INFO") as logs:
            hp.finalize()
        self.assertEqual(freed, [1003])
        self.assertEqual(hp.unclaimed_bytes(), 0)
        self.assertEqual(hp._read_record(), [100, 999, 200, 200, 200])
        self.assertIn("took 3 of 5 requested segments", "\n".join(logs.output))
        self.assertIsNone(hp.take(100))  # after finalize everything is foreground

    def test_first_boot_only_records(self):
        hp = self.hp
        self.assertEqual(hp._read_record(), [])
        self.assertIsNone(hp.take(4096))
        self.assertIsNone(hp.take(8192))
        hp.finalize()
        self.assertEqual(hp._read_record(), [4096, 8192])

    def test_off_switch_does_nothing(self):
        hp = self.hp
        with mock.patch.dict(os.environ, {"SGLANG_OPT_INF834_HOST_POOL_PREALLOC": "0"}):
            hp.maybe_start_host_pool_prealloc()
            hp.finalize()
        self.assertFalse(hp._started)
        self.assertEqual(hp._read_record(), [])


class TestDeepGemmSweepStamp(CustomTestCase):
    def setUp(self):
        try:
            from sglang.srt.layers.deep_gemm_wrapper import compile_utils as cu
        except Exception as e:  # deep_gemm needs CUDA libraries to import
            self.skipTest(f"deep_gemm wrapper unavailable: {e}")
        self.cu = cu
        self.tmp = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(self.tmp.name, "cache", "kernel.a.1"))
        os.makedirs(os.path.join(self.tmp.name, "cache", "kernel.b.2"))
        self.identity = {
            "kernel_type": "GEMM_NT_F8F8BF16",
            "n": 8,
            "k": 16,
            "num_groups": 1,
        }
        self.patches = [
            mock.patch.dict(os.environ, {"DG_JIT_CACHE_DIR": self.tmp.name}),
            mock.patch.object(cu, "_ENABLE_JIT_DEEPGEMM_PRECOMPILE", True),
            mock.patch.object(cu, "_DO_COMPILE_ALL", True),
            mock.patch.object(cu, "_INITIALIZATION_DICT", {}),
            mock.patch.object(cu, "_BUILTIN_M_LIST", [1, 2, 3]),
            mock.patch.object(
                cu, "_inf834_sweep_identity", lambda *a: dict(self.identity)
            ),
        ]
        for p in self.patches:
            p.start()
        self.sweeps = []
        self.swept_all = True
        self.sweep_patch = mock.patch.object(
            cu,
            "_compile_deep_gemm_one_type_all",
            lambda **kw: self.sweeps.append(kw) or self.swept_all,
        )
        self.sweep_patch.start()

    def tearDown(self):
        if hasattr(self, "cu"):
            self.sweep_patch.stop()
            for p in reversed(self.patches):
                p.stop()
            self.tmp.cleanup()

    def _call(self):
        self.cu._INITIALIZATION_DICT.clear()  # a new process
        self.cu._maybe_compile_deep_gemm_one_type_all(
            self.cu.DeepGemmKernelType.GEMM_NT_F8F8BF16, 8, 16, 1
        )

    def test_switch_off_always_sweeps_and_writes_nothing(self):
        self._call()
        self._call()
        self.assertEqual(len(self.sweeps), 2)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "inf834_sweeps")))

    def test_sweep_once_per_stamp(self):
        with mock.patch.dict(
            os.environ, {"SGLANG_OPT_INF834_DEEPGEMM_SWEEP_STAMP": "1"}
        ):
            self._call()
            self.assertEqual(len(self.sweeps), 1)
            self.assertEqual(
                len(os.listdir(os.path.join(self.tmp.name, "inf834_sweeps"))), 1
            )
            with self.assertLogs("sglang", level="INFO") as logs:
                self._call()
            self.assertEqual(len(self.sweeps), 1)
            self.assertIn("INF834 deepgemm sweep: skipped", "\n".join(logs.output))
            # Another device, library or M list is another identity.
            self.identity["n"] = 9
            self._call()
            self.assertEqual(len(self.sweeps), 2)
            # Kernels gone from the cache: the stamp no longer vouches for it.
            self.identity["n"] = 8
            os.rmdir(os.path.join(self.tmp.name, "cache", "kernel.b.2"))
            self._call()
            self.assertEqual(len(self.sweeps), 3)

    def test_cut_sweep_leaves_no_stamp(self):
        self.swept_all = False
        with mock.patch.dict(
            os.environ, {"SGLANG_OPT_INF834_DEEPGEMM_SWEEP_STAMP": "1"}
        ):
            self._call()
            self._call()
        self.assertEqual(len(self.sweeps), 2)



# ------------------------------------------------------------------ direct read
_ST_DTYPE = {
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.float32: "F32",
    E8M0: "F8_E8M0",
}


def write_safetensors(path, tensors):
    """A safetensors file written by hand (no dependency on the writer's dtype list)."""
    import json
    import struct

    header, blobs, pos = {}, [], 0
    for name, t in tensors.items():
        raw = t.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
        header[name] = {
            "dtype": _ST_DTYPE[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [pos, pos + len(raw)],
        }
        blobs.append(raw)
        pos += len(raw)
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        for b in blobs:
            f.write(b)


def file_views(tmp, named, shards=2, direct=True):
    """The checkpoint as SELECTIVE_READ hands it out: views over file mappings."""
    from sglang.srt.model_loader import inf834_selective_read as sr

    names = sorted(named)
    paths = []
    for i in range(shards):
        part = {n: named[n] for n in names[i::shards]}
        path = os.path.join(tmp, f"model-{i:05d}.safetensors")
        write_safetensors(path, part)
        paths.append(path)
    env = {
        "SGLANG_OPT_INF834_EXPERT_BULK_LOAD": "1",
        "SGLANG_OPT_INF834_DIRECT_READ": "1" if direct else "0",
    }
    stats = sr.new_stats()
    flt = sr.Inf834ReadFilter(label="test", keep=lambda n: True)
    with mock.patch.dict(os.environ, env):
        out = {}
        for path in paths:
            out.update(sr.load_file_filtered(path, flt, stats))
    return out, stats


class TestDirectRead(CustomTestCase):
    def _ckpt(self, num_experts, seed=4321):
        gen = torch.Generator().manual_seed(seed)
        named = {}
        for e in range(num_experts):
            for (shard_id, is_scale), t in checkpoint_expert(gen).items():
                leaf = "scale" if is_scale else "weight"
                named[(e, shard_id, is_scale)] = (
                    f"layers.0.ffn.experts.{e}.{shard_id}.{leaf}",
                    t,
                )
        return named

    def _load(self, views, named, num_experts, tp_size, tp_rank, block, direct):
        experts = make_experts(num_experts, tp_size, tp_rank)
        pending = []
        env = {
            "SGLANG_OPT_INF834_EXPERT_BULK_LOAD": "1",
            "SGLANG_OPT_INF834_EXPERT_BULK_BLOCK": str(block),
            "SGLANG_OPT_INF834_DIRECT_READ": "1" if direct else "0",
        }
        with mock.patch.dict(os.environ, env):
            loader = Inf834ExpertBulkLoader(
                lambda func, *args: pending.append((func, args))
            )
            for e in lexicographic(num_experts):
                for (ee, shard_id, is_scale), (name, _) in named.items():
                    if ee != e:
                        continue
                    param, pname = param_for(experts, shard_id, is_scale)
                    t = views[name]
                    if not loader.offer(param, t, pname, shard_id, e):
                        experts.weight_loader(param, t, pname, shard_id, e)
            loader.finish()
            for func, args in pending:
                func(*args)
            loader.close("test")
        return experts, loader

    def _reference(self, named, num_experts, tp_size, tp_rank):
        experts = make_experts(num_experts, tp_size, tp_rank)
        for e in lexicographic(num_experts):
            for (ee, shard_id, is_scale), (_, t) in named.items():
                if ee == e:
                    param, pname = param_for(experts, shard_id, is_scale)
                    experts.weight_loader(param, t, pname, shard_id, e)
        return experts

    def test_same_bytes_as_weight_loader(self):
        n = 11
        named = self._ckpt(n)
        with tempfile.TemporaryDirectory() as tmp:
            views, stats = file_views(tmp, {k: t for k, t in named.values()})
            self.assertGreater(stats.bytes_direct, 0)
            self.assertEqual(stats.bytes_prefetched, 0)  # experts are all it holds
            for tp_size, tp_rank in ((1, 0), (2, 1), (4, 0), (4, 3)):
                for block in (1, 4, 64):
                    with self.subTest(tp_size=tp_size, tp_rank=tp_rank, block=block):
                        want = self._reference(named, n, tp_size, tp_rank)
                        got, loader = self._load(
                            views, named, n, tp_size, tp_rank, block, direct=True
                        )
                        self.assertEqual(loader.tensors_fallback, 0)
                        self.assertEqual(loader.blocks_direct, loader.blocks_done)
                        self.assertGreater(loader.bytes_direct_read, 0)
                        for (k, a), b in zip(
                            state(want).items(), state(got).values()
                        ):
                            self.assertTrue(torch.equal(a, b), k)

    def test_untagged_tensors_take_the_mapping_path(self):
        n = 5
        named = self._ckpt(n)
        with tempfile.TemporaryDirectory() as tmp:
            views, stats = file_views(
                tmp, {k: t for k, t in named.values()}, direct=False
            )
            self.assertEqual(stats.bytes_direct, 0)
            self.assertGreater(stats.bytes_prefetched, 0)
            self.assertFalse(
                any(hasattr(v, "_inf834_file_src") for v in views.values())
            )
            want = self._reference(named, n, 4, 2)
            got, loader = self._load(views, named, n, 4, 2, 4, direct=True)
            self.assertEqual(loader.blocks_direct, 0)
            self.assertGreater(loader.blocks_done, 0)
            for (k, a), b in zip(state(want).items(), state(got).values()):
                self.assertTrue(torch.equal(a, b), k)

    def test_switch_needs_bulk_load(self):
        from sglang.srt.model_loader import inf834_selective_read as sr

        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_OPT_INF834_DIRECT_READ": "1",
                "SGLANG_OPT_INF834_EXPERT_BULK_LOAD": "0",
            },
        ):
            self.assertFalse(sr.direct_read_enabled())

    def test_load_weights_from_files(self):
        """DeepseekV4ForCausalLM.load_weights over file-backed views: switch on and
        off leave the same parameters, and every expert block is read directly."""
        case = TestLoadWeightsIntegration()
        stream = case._stream()
        with tempfile.TemporaryDirectory() as tmp:
            views, _ = file_views(tmp, dict(stream))
            ordered = [(k, views[k]) for k, _ in stream]

            def run(env):
                model = case._model(4, 1)
                with mock.patch.dict(os.environ, env):
                    with self.assertLogs("sglang", level="INFO") as logs:
                        model.load_weights(iter(ordered))
                params = {k: v.data.clone() for k, v in model.named_parameters()}
                return params, "\n".join(logs.output)

            want, _ = run({"SGLANG_OPT_INF834_LOAD_PROFILE": "1"})
            got, text = run(
                {
                    "SGLANG_OPT_INF834_EXPERT_BULK_LOAD": "1",
                    "SGLANG_OPT_INF834_EXPERT_BULK_BLOCK": "4",
                    "SGLANG_OPT_INF834_DIRECT_READ": "1",
                }
            )
        for k in want:
            self.assertTrue(
                torch.equal(want[k].view(torch.uint8), got[k].view(torch.uint8)), k
            )
        import re

        m = re.search(r"INF834 direct read \[target\]: (\d+) of (\d+) blocks", text)
        self.assertIsNotNone(m, text)
        self.assertGreater(int(m.group(2)), 0)
        self.assertEqual(m.group(1), m.group(2))


if __name__ == "__main__":
    unittest.main()
