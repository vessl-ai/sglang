"""CPU-only tests for SGLANG_OPT_INF834_WEIGHT_SNAPSHOT (model_loader/inf834_weight_snapshot.py).

- byte-exact round trip over a model built from a real decoder rank's tensor list (names, dtypes, shapes from
  a weight digest of an EP8 decoder boot when INF834_DIGEST_JSON exists; large tensors are cut on dim 0 so
  the test writes about 1 GB), buffered and O_DIRECT reads
- identity mismatch -> fallback before any parameter is touched
- truncated or uncommitted snapshot -> fallback / ignored
- checksum failure after the parameters were filled -> fallback reported as touched (caller reloads)
- two processes saving the same rank at once: one writes, the other skips; the result is valid
- phase "repacked" through the real EP decode repack (Mxfp4MarlinMoEMethod, v2w pack on CPU): save, restore
  without the pack (packed bytes equal), fallback after a checksum failure in a packed region (normal load +
  real pack), the phase in the identity, an unconsumed staged layer fails the boot, an uncovered layer
  abandons the save
- Engram host table pre-fault deferral: no pre-fault / registration at init, registered after the restore
  without pre-fault, pre-faulted first when nothing is restored
- per-chunk IO policy: resident chunks buffered, the others O_DIRECT, then rewarm
"""

import json
import multiprocessing as mp
import os
import re
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from sglang.srt.model_loader import inf834_weight_snapshot as ws
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

DIGEST = os.environ.get("INF834_DIGEST_JSON", "/floor-dec/digest/r3dgn/DeepseekV4ForCausalLM-rank0.json")
CAP = 1 << 20  # bytes per tensor in the test (the real list is 254 GB); larger ones are cut on dim 0
BIG = (("big.multichunk_a", torch.int8, (150, 1 << 20)), ("big.multichunk_b", torch.bfloat16, (3, 70 << 20)))
WORK = os.environ.get("INF834_SNAP_TEST_DIR")  # a real disk for O_DIRECT (tmpfs refuses it)

_DT = {
    "torch.float32": torch.float32, "torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16,
    "torch.int8": torch.int8, "torch.uint8": torch.uint8, "torch.int32": torch.int32, "torch.int64": torch.int64,
    "torch.float8_e4m3fn": torch.float8_e4m3fn, "torch.float8_e8m0fnu": torch.float8_e8m0fnu, "torch.bool": torch.bool,
}


def real_tensor_list(limit=None):
    """(name, dtype, shape) of every param/buffer of a real decoder rank, or a synthetic stand-in."""
    out = []
    if os.path.exists(DIGEST):
        d = json.load(open(DIGEST))
        for k, v in d.items():
            if not (k.startswith("param:") or k.startswith("buffer:")) or "._sm90_packed." in k:
                continue
            dt, shape = v.split("|")[0], v.split("|")[1]
            if dt not in _DT:
                continue
            out.append((k.split(":", 1)[1], _DT[dt], tuple(int(x) for x in re.findall(r"\d+", shape))))
    if not out:
        for i in range(40):
            out.append((f"model.layers.{i}.mlp.experts.w13_weight", torch.int8, (52, 4608, 1024)))
            out.append((f"model.layers.{i}.attn.wq.weight", torch.bfloat16, (1024, 4096)))
            out.append((f"model.layers.{i}.attn.wq.weight_scale_inv", torch.float32, (8, 32)))
    out = out[:limit] if limit else out
    return out + list(BIG) if not limit else out


class Fake(nn.Module):
    pass


def make_model(tensors, seed, cls_name="DeepseekV4ForCausalLM"):
    cls = type(cls_name, (nn.Module,), {})
    m = cls()
    gen = torch.Generator().manual_seed(seed)
    for i, (name, dtype, shape) in enumerate(tensors):
        shape = shape or (1,)
        elem = torch.empty(0, dtype=dtype).element_size()
        n = 1
        for s in shape:
            n *= s
        if n * elem > CAP and len(shape) > 0 and not name.startswith("big."):
            row = max(1, n // shape[0]) * elem
            shape = (max(1, CAP // row),) + tuple(shape[1:])
        raw = torch.randint(0, 255, (int(torch.tensor(shape).prod()) * elem,), generator=gen, dtype=torch.uint8)
        t = raw.view(dtype).view(shape) if dtype != torch.bool else (raw % 2).bool().view(shape)
        mod = m
        parts = name.split(".")
        safe = "_".join(parts)  # flat names are enough: entries() walks the registry
        if i % 7 == 0:
            m.register_buffer(f"b{i}_{safe[-40:]}", t)
        else:
            m.register_parameter(f"p{i}_{safe[-40:]}", nn.Parameter(t, requires_grad=False))
    return m


def state(m):
    return {k: v.detach().clone().view(-1).view(torch.uint8) for k, v in list(m.named_parameters()) + list(m.named_buffers())}


def zero(m):
    with torch.no_grad():
        for _, v in list(m.named_parameters()) + list(m.named_buffers()):
            v.view(-1).view(torch.uint8).zero_()


class Base(CustomTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=WORK)
        self.root = os.path.join(self.tmp.name, "snap")
        self.ckpt = os.path.join(self.tmp.name, "ckpt")
        os.makedirs(self.ckpt)
        json.dump({"weight_map": {}}, open(os.path.join(self.ckpt, "model.safetensors.index.json"), "w"))
        open(os.path.join(self.ckpt, "model-00001-of-00001.safetensors"), "wb").write(b"x" * 16)
        self.env = mock.patch.dict(os.environ, {
            "SGLANG_OPT_INF834_WEIGHT_SNAPSHOT": "auto",
            "SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_DIR": self.root,
            "SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_THREADS": "4",
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def snap(self, model):
        return ws.WeightSnapshot(model, "target", model_path=self.ckpt)


class TestRoundTrip(Base):
    def _round_trip(self, direct):
        tensors = real_tensor_list()
        src = make_model(tensors, seed=1)
        s = self.snap(src)
        self.assertFalse(s.begin())  # nothing saved yet
        s.end()
        self.assertTrue(os.path.exists(s.commit))
        dst = make_model(tensors, seed=1)
        zero(dst)
        with mock.patch.dict(os.environ, {"SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_DIRECT_IO": "1" if direct else "0"}):
            d = self.snap(dst)
            self.assertTrue(d.begin())
        a, b = state(src), state(dst)
        self.assertEqual(list(a), list(b))
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)
        meta = json.load(open(d.commit))
        return len(tensors), meta["size"], d

    def test_buffered(self):
        n, size, _ = self._round_trip(direct=False)
        print(f"round trip (buffered): {n} tensors from {'the real digest' if os.path.exists(DIGEST) else 'a synthetic list'}, {size / 1e9:.2f} GB")

    @unittest.skipUnless(WORK, "O_DIRECT needs a real disk (set INF834_SNAP_TEST_DIR)")
    def test_o_direct(self):
        n, size, _ = self._round_trip(direct=True)
        print(f"round trip (O_DIRECT): {n} tensors, {size / 1e9:.2f} GB")


class TestFallbacks(Base):
    def _saved(self):
        tensors = real_tensor_list(limit=60)
        src = make_model(tensors, seed=2)
        s = self.snap(src)
        s.end()
        return tensors, src, s

    def _sentinel(self, tensors):
        m = make_model(tensors, seed=2)
        with torch.no_grad():
            for _, v in list(m.named_parameters()) + list(m.named_buffers()):
                v.view(-1).view(torch.uint8).fill_(0xAB)
        return m

    def _untouched(self, m):
        for k, v in state(m).items():
            self.assertTrue(bool((v == 0xAB).all()), k)

    def test_identity_mismatch_falls_back_untouched(self):
        tensors, _, _ = self._saved()
        m = self._sentinel(tensors)
        with mock.patch.dict(os.environ, {"SGLANG_SOME_LOAD_SWITCH": "1"}):
            s = self.snap(m)
            self.assertFalse(s.begin())  # different directory: nothing committed there
        # Same directory key but a different stored identity (e.g. a mtime the key hashes the same way)
        s = self.snap(m)
        meta = json.load(open(s.commit))
        meta["identity"]["gpu"] = "another GPU"
        json.dump(meta, open(s.commit, "w"))
        with self.assertLogs("sglang", level="WARNING") as logs:
            self.assertFalse(s.begin())
        self.assertIn(ws.FALLBACK, "\n".join(logs.output))
        self.assertIn("no parameter was touched", "\n".join(logs.output))
        self._untouched(m)

    def test_entry_table_mismatch_falls_back_untouched(self):
        tensors, _, _ = self._saved()
        m = self._sentinel(tensors[:-1])
        with self.assertLogs("sglang", level="WARNING"):
            self.assertFalse(self.snap(m).begin())
        self._untouched(m)

    def test_truncated_file_falls_back_untouched(self):
        tensors, _, s = self._saved()
        with open(s.bin, "r+b") as f:
            f.truncate(os.path.getsize(s.bin) - 4096)
        m = self._sentinel(tensors)
        with self.assertLogs("sglang", level="WARNING") as logs:
            self.assertFalse(self.snap(m).begin())
        self.assertIn("file size", "\n".join(logs.output))
        self._untouched(m)

    def test_uncommitted_is_ignored(self):
        tensors, _, s = self._saved()
        os.remove(s.commit)
        m = self._sentinel(tensors)
        self.assertFalse(self.snap(m).begin())
        self._untouched(m)

    def test_checksum_failure_reported_touched(self):
        tensors, src, s = self._saved()
        size = os.path.getsize(s.bin)
        with open(s.bin, "r+b") as f:  # flip a byte inside the first entry's data
            f.seek(17)
            b = f.read(1)
            f.seek(17)
            f.write(bytes([b[0] ^ 0xFF]))
        self.assertEqual(os.path.getsize(s.bin), size)
        m = self._sentinel(tensors)
        with self.assertLogs("sglang", level="WARNING") as logs:
            self.assertFalse(self.snap(m).begin())
        text = "\n".join(logs.output)
        self.assertIn(ws.FALLBACK, text)
        self.assertIn("checksum", text)
        self.assertIn("overwrites every parameter", text)  # the caller then runs the normal load


class TestIdentityArgs(Base):
    """Server args on the explicit denylist (addresses, ports, logging, metrics, timeouts, ...) do not
    change the identity; any other argument does."""

    ARGS = {
        "model_path": "/m", "tp_size": 8, "quantization": None, "mem_fraction_static": 0.85,
        "host": "0.0.0.0", "port": 30000, "log_level": "info", "watchdog_timeout": 3600,
        "hicache_storage_backend_extra_config": '{"master_server_address": "10.0.0.1:50051"}',
        "served_model_name": "a", "_raw_internal_flag": True,
    }

    def _with_args(self, args, fn):
        import types

        sa = types.SimpleNamespace(_raw_input=dict(args))
        with mock.patch("sglang.srt.runtime_context.get_server_args", lambda: sa):
            return fn()

    def test_denylisted_change_hits_other_change_misses(self):
        tensors = real_tensor_list(limit=30)
        src = make_model(tensors, seed=5)
        self._with_args(self.ARGS, lambda: self.snap(src).end())
        ops = dict(self.ARGS, host="10.1.2.3", port=31000, log_level="debug", watchdog_timeout=60,
                   hicache_storage_backend_extra_config='{"master_server_address": "10.9.9.9:50051"}',
                   served_model_name="b", _raw_internal_flag=False)
        dst = make_model(tensors, seed=5)
        zero(dst)
        self.assertTrue(self._with_args(ops, lambda: self.snap(dst).begin()))
        for (k, a), b in zip(state(src).items(), state(dst).values()):
            self.assertTrue(torch.equal(a, b), k)
        for changed in ({"mem_fraction_static": 0.80}, {"quantization": "fp8"}, {"some_new_arg": 1}):
            m = make_model(tensors, seed=5)
            with self.assertLogs("sglang", level="INFO") as logs:
                self.assertFalse(self._with_args(dict(self.ARGS, **changed), lambda: self.snap(m).begin()))
            self.assertIn("none at", "\n".join(logs.output), changed)

    def test_foundry_save_prepare_and_load_cell_hit(self):
        """The prepare boot runs --cuda-graph-persistence save, the cell boot load (each with its own TOML):
        both must resolve to the same snapshot; a weight-relevant change next to them still misses."""
        m = make_model(real_tensor_list(limit=5), seed=8)
        prepare = dict(self.ARGS, cuda_graph_persistence="save", cuda_graph_persistence_config="/prep/foundry.toml")
        cell = dict(self.ARGS, cuda_graph_persistence="load", cuda_graph_persistence_config="/opt/foundry/cell.toml")
        native = dict(self.ARGS)  # no persistence at all
        dirs = [self._with_args(a, lambda: self.snap(m).dir) for a in (prepare, cell, native)]
        self.assertEqual(dirs[0], dirs[1])
        self.assertEqual(dirs[0], dirs[2])
        self.assertNotEqual(dirs[1], self._with_args(dict(cell, mem_fraction_static=0.80), lambda: self.snap(m).dir))
        # and a snapshot saved by the prepare boot is restored by the cell boot
        src = make_model(real_tensor_list(limit=5), seed=8)
        self._with_args(prepare, lambda: self.snap(src).end())
        dst = make_model(real_tensor_list(limit=5), seed=8)
        zero(dst)
        self.assertTrue(self._with_args(cell, lambda: self.snap(dst).begin()))
        for (k, a), b in zip(state(src).items(), state(dst).values()):
            self.assertTrue(torch.equal(a, b), k)

    def test_symm_mem_and_deepgemm_precompile_hit(self):
        """--enable-symm-mem (collective buffers) and SGLANG_JIT_DEEPGEMM_PRECOMPILE (kernel JIT) do not change
        weight bytes: toggling either finds the same snapshot."""
        m = make_model(real_tensor_list(limit=5), seed=9)
        on = dict(self.ARGS, enable_symm_mem=True)
        base = self._with_args(on, lambda: self.snap(m).dir)
        self.assertEqual(base, self._with_args(dict(self.ARGS, enable_symm_mem=False), lambda: self.snap(m).dir))
        with mock.patch.dict(os.environ, {"SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0"}):
            self.assertEqual(base, self._with_args(on, lambda: self.snap(m).dir))
        with mock.patch.dict(os.environ, {"SGLANG_SOME_LOAD_SWITCH": "1"}):  # any other SGLANG_* env still misses
            self.assertNotEqual(base, self._with_args(on, lambda: self.snap(m).dir))

    def test_two_pods_of_one_cell_hit(self):
        """Two prefill pods of the cell draft (runs/img/manifests/cell-s02, e2e mkcell): the only per-pod
        values in the engine args are $(POD_IP) inside --hicache-storage-backend-extra-config (and, when
        set, addresses such as dist_init_addr). Their identities must be equal."""
        def pod_args(ip):
            extra = ('{"master_server_address": "deepseek-v41-flash-1017-ep8-mooncake-master.inference.svc:50051", '
                     '"protocol": "rdma", "metadata_server": "http://deepseek-v41-flash-1017-ep8-mooncake-master.'
                     'inference.svc:8080/metadata", "local_hostname": "%s", "global_segment_size": "0"}' % ip)
            return {
                "model_path": "/opt/lv/data/models/DeepSeek-V4.1-Flash-1017", "tp_size": 4, "ep_size": 1,
                "mem_fraction_static": 0.82, "chunked_prefill_size": 65280, "enable_hierarchical_cache": True,
                "hicache_ratio": 1.2, "hicache_storage_backend": "mooncake", "disaggregation_mode": "prefill",
                "enable_prefill_cp": True, "host": "0.0.0.0", "port": 30000, "disaggregation_bootstrap_port": 8998,
                "hicache_storage_backend_extra_config": extra, "dist_init_addr": f"{ip}:5000",
            }

        m = make_model(real_tensor_list(limit=5), seed=7)
        ids = []
        for ip in ("10.42.3.17", "10.42.9.201"):
            snap = self._with_args(pod_args(ip), lambda: self.snap(m))
            ids.append((snap.dir, json.dumps(snap.ident, sort_keys=True, default=str)))
        self.assertEqual(ids[0], ids[1])


def _save_proc(root, ckpt, seed, go):
    os.environ.update({"SGLANG_OPT_INF834_WEIGHT_SNAPSHOT": "save", "SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_DIR": root})
    m = make_model(real_tensor_list(limit=40), seed=seed)
    s = ws.WeightSnapshot(m, "target", model_path=ckpt)
    go.wait(60)  # start both saves together
    s.end()


class TestConcurrentSave(Base):
    def test_lock_and_two_writers(self):
        import fcntl

        # Hold the lock in this process first: a concurrent saver must skip, not write.
        m = make_model(real_tensor_list(limit=40), seed=3)
        s = self.snap(m)
        os.makedirs(s.dir, exist_ok=True)
        holder = open(os.path.join(s.dir, f"rank{s.rank}.lock"), "w")
        fcntl.flock(holder, fcntl.LOCK_EX)
        with self.assertLogs("sglang", level="INFO") as logs:
            s.end()
        self.assertIn("another process is saving", "\n".join(logs.output))
        self.assertFalse(os.path.exists(s.commit))
        fcntl.flock(holder, fcntl.LOCK_UN)
        holder.close()
        # Two real processes at once (fork: CPU only): one commit or one skip, never a broken file.
        ctx = mp.get_context("fork")
        go = ctx.Event()
        procs = [ctx.Process(target=_save_proc, args=(self.root, self.ckpt, 3, go)) for _ in range(2)]
        for p in procs:
            p.start()
        go.set()
        for p in procs:
            p.join(180)
            self.assertEqual(p.exitcode, 0)
        self.assertTrue(os.path.exists(s.commit))
        self.assertEqual([f for f in os.listdir(s.dir) if f.endswith(".commit")], [f"rank{s.rank}.commit"])
        dst = make_model(real_tensor_list(limit=40), seed=3)
        zero(dst)
        self.assertTrue(self.snap(dst).begin())
        for (k, a), b in zip(state(m).items(), state(dst).values()):
            self.assertTrue(torch.equal(a, b), k)


# ------------------------------------------------------------------------------------- phase "repacked"

HID, INTER = 5120, 2304  # the v2w shape (DSv4.1-Flash EP8 decoder: 52 local experts; 1 here)


def _moe_method(prefix, activation="silu"):
    import types

    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    qm = object.__new__(Mxfp4MarlinMoEMethod)
    qm.prefix = prefix
    qm._use_w4a8 = False
    qm._sm90_decode_w4a8 = qm._sm90_decode_w4a8_v2 = qm._sm90_decode_w4a8_ep = True
    qm.hidden_pad = 0
    qm.runner = types.SimpleNamespace(config=types.SimpleNamespace(
        activation=activation, is_gated=True, gemm1_alpha=None, swiglu_limit=7.0))
    return qm


def make_moe_model(seed, layers=2, experts=1, activation=None):
    cls = type("DeepseekV4ForCausalLM", (nn.Module,), {})
    m = cls()
    m.model = nn.Module()
    m.model.layers = nn.ModuleList()
    gen = torch.Generator().manual_seed(seed)

    def rnd(shape, dtype, hi=255):
        n = int(torch.tensor(shape).prod()) * torch.empty(0, dtype=dtype).element_size()
        return torch.randint(0, hi, (n,), generator=gen, dtype=torch.uint8).view(dtype).view(shape)

    for i in range(layers):
        layer = nn.Module()
        layer.attn = nn.Module()
        layer.attn.weight = nn.Parameter(rnd((64, 128), torch.bfloat16), requires_grad=False)
        layer.mlp = nn.Module()
        ex = layer.mlp.experts = nn.Module()
        ex.w13_weight = nn.Parameter(rnd((experts, 2 * INTER, HID // 2), torch.int8), requires_grad=False)
        ex.w2_weight = nn.Parameter(rnd((experts, HID, INTER // 2), torch.int8), requires_grad=False)
        # e8m0 exponents in a realistic band (the pack refuses > MAX_E8M0)
        ex.w13_weight_scale_inv = nn.Parameter(
            (torch.randint(110, 130, (experts, 2 * INTER, HID // 32), generator=gen, dtype=torch.uint8)
             .view(torch.float8_e8m0fnu)), requires_grad=False)
        ex.w2_weight_scale_inv = nn.Parameter(
            (torch.randint(110, 130, (experts, HID, INTER // 32), generator=gen, dtype=torch.uint8)
             .view(torch.float8_e8m0fnu)), requires_grad=False)
        ex.quant_method = _moe_method(f"model.layers.{i}.mlp.experts",
                                      activation=(activation or {}).get(i, "silu"))
        m.model.layers.append(layer)
    m.model.norm = nn.Parameter(rnd((HID,), torch.bfloat16), requires_grad=False)
    return m


def _repack_all(m):
    for layer in m.model.layers:
        ex = layer.mlp.experts
        ex.quant_method._prepare_sm90_decode_ep_weights(ex)


def _packed_bytes(m):
    out = {}
    for i, layer in enumerate(m.model.layers):
        qm = layer.mlp.experts.quant_method
        p = getattr(qm, "_sm90_packed", None)
        if p is not None:
            out[i] = ([f.detach().clone().view(-1).view(torch.uint8) for f in p if isinstance(f, torch.Tensor)],
                      [f for f in p if not isinstance(f, torch.Tensor)], qm._ep_split)
    return out


class TestRepacked(Base):
    def setUp(self):
        import types

        super().setUp()
        from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ep as ek

        self.patches = [
            mock.patch.dict(os.environ, {"SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_PHASE": "repacked"}),
            mock.patch("sglang.srt.runtime_context.get_parallel", lambda: types.SimpleNamespace(moe_ep_size=8)),
            mock.patch.object(ek, "kernel_mode", lambda: "v2w"),
            mock.patch.object(ek, "prepare_ep", lambda *a, **k: None),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        super().tearDown()

    def _save(self, seed=11, **kw):
        m = make_moe_model(seed, **kw)
        s = self.snap(m)
        self.assertFalse(s.begin())
        s.end()
        self.assertFalse(os.path.exists(s.commit))  # waits for the repack
        # post_load_weights registers aliases (DeepseekV4: layers[i]._next_layer): named_modules() then names
        # layer 1 "model.layers.0._next_layer"; the save must still find it
        m.model.layers[0]._next_layer = m.model.layers[1]
        self.assertIn("model.layers.0._next_layer.mlp.experts", dict(m.named_modules()))
        _repack_all(m)
        ws.finish_model(m)
        return m, s

    def test_save_restore_skips_the_pack(self):
        src, s = self._save()
        self.assertTrue(os.path.exists(s.commit))
        meta = json.load(open(s.commit))
        self.assertEqual(meta["identity"]["phase"], "repacked")
        self.assertEqual([r["layer"] for r in meta["repacked"]], ["model.layers.0.mlp.experts", "model.layers.1.mlp.experts"])
        ref = _packed_bytes(src)
        self.assertEqual(len(ref), 2)
        # the packed bytes fill the raw expert parameters exactly on the DSv4 shapes
        rec = meta["repacked"][0]
        room = sum(r["nbytes"] for r in meta["table"] if r["name"].startswith("param:model.layers.0.mlp.experts."))
        self.assertEqual(ws._record_nbytes(rec), room)

        dst = make_moe_model(11)
        zero(dst)
        d = self.snap(dst)
        from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ep as ek

        with mock.patch.object(ek, "pack_experts_ep", side_effect=AssertionError("pack must not run")):
            self.assertTrue(d.begin())
            self.assertEqual(d.staged, ["model.layers.0.mlp.experts", "model.layers.1.mlp.experts"])
            dst.model.layers[0]._next_layer = dst.model.layers[1]  # as post_load_weights does
            _repack_all(dst)
        ws.finish_model(dst)
        got = _packed_bytes(dst)
        for i in ref:
            for a, b in zip(ref[i][0], got[i][0]):
                self.assertTrue(torch.equal(a, b), i)
            self.assertEqual(ref[i][1:], got[i][1:])
            ex = dst.model.layers[i].mlp.experts
            self.assertFalse(hasattr(ex, "w13_weight"))
            self.assertNotIn("_inf834_snapshot_packed", ex.__dict__)
            self.assertEqual(type(dst.model.layers[i].mlp.experts.quant_method._sm90_packed).__name__, "PackedExperts")
        for (k, a), (k2, b) in zip(sorted(state(src).items()), sorted(state(dst).items())):
            self.assertEqual(k, k2)
            self.assertTrue(torch.equal(a, b), k)  # attention weights, norm (experts are gone on both)

    def test_checksum_failure_in_packed_region_reloads_and_repacks(self):
        src, s = self._save()
        ref = _packed_bytes(src)
        meta = json.load(open(s.commit))
        row = next(r for r in meta["table"] if r["name"] == "param:model.layers.1.mlp.experts.w2_weight")
        with open(s.bin, "r+b") as f:
            f.seek(row["offset"] + 123)
            b = f.read(1)
            f.seek(row["offset"] + 123)
            f.write(bytes([b[0] ^ 0x5A]))
        dst = make_moe_model(11)
        zero(dst)
        with self.assertLogs("sglang", level="WARNING") as logs:
            self.assertFalse(self.snap(dst).begin())
        text = "\n".join(logs.output)
        self.assertIn(ws.FALLBACK, text)
        self.assertIn("the repack runs", text)
        for layer in dst.model.layers:
            self.assertNotIn("_inf834_snapshot_packed", layer.mlp.experts.__dict__)
        # the normal load: the checkpoint stream writes every parameter again, then the real pack runs
        stream = make_moe_model(11)
        with torch.no_grad():
            for (k, a), (_, b) in zip(dst.named_parameters(), stream.named_parameters()):
                a.copy_(b)
        _repack_all(dst)
        ws.finish_model(dst)
        got = _packed_bytes(dst)
        for i in ref:
            for a, b in zip(ref[i][0], got[i][0]):
                self.assertTrue(torch.equal(a, b), i)

    def test_phase_is_in_the_identity(self):
        _, s = self._save()
        with mock.patch.dict(os.environ, {"SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_PHASE": "stream_end"}):
            m = make_moe_model(11)
            t = self.snap(m)
            self.assertNotEqual(t.dir, s.dir)
            with self.assertLogs("sglang", level="INFO") as logs:
                self.assertFalse(t.begin())
            self.assertIn("none at", "\n".join(logs.output))

    def test_unconsumed_staged_layer_fails_the_boot(self):
        self._save()
        dst = make_moe_model(11)
        self.assertTrue(self.snap(dst).begin())
        ex = dst.model.layers[0].mlp.experts
        ex.quant_method._prepare_sm90_decode_ep_weights(ex)  # layer 1 never repacked
        with self.assertRaisesRegex(RuntimeError, "did not consume"):
            ws.finish_model(dst)

    def test_staged_layer_on_an_uncovered_path_raises(self):
        self._save()
        dst = make_moe_model(11)
        self.assertTrue(self.snap(dst).begin())
        ex = dst.model.layers[0].mlp.experts
        ex.quant_method.runner.config.activation = "gelu"
        with self.assertRaisesRegex(RuntimeError, "holds restored packed experts"):
            ex.quant_method._prepare_sm90_decode_ep_weights(ex)

    def test_layer_not_repacked_abandons_the_save(self):
        m = make_moe_model(12, activation={1: "gelu"})  # layer 1: shape check fails, no pack
        s = self.snap(m)
        self.assertFalse(s.begin())
        s.end()
        _repack_all(m)
        with self.assertLogs("sglang", level="WARNING") as logs:
            ws.finish_model(m)
        self.assertIn("save abandoned", "\n".join(logs.output))
        self.assertFalse(os.path.exists(s.commit))
        self.assertEqual([f for f in os.listdir(os.path.join(s.dir, ".tmp"))], [])


# ------------------------------------------------------------------------------ Engram pre-fault deferral

class _Registrar:
    def __init__(self):
        self.calls = []

    def cudaHostRegister(self, ptr, n, flags):
        self.calls.append((ptr, n))
        return 0


class TestEngramDeferral(Base):
    N = 6 << 20

    def _table_model(self, reg, seed=None):
        from sglang.srt.layers import engram

        with mock.patch.dict(os.environ, {"SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_DEFER_PREFAULT": "1",
                                          "SGLANG_OPT_INF834_KEEP_PAGE_CACHE": "1"}), \
                mock.patch.object(engram, "_graphs_persisted", lambda: False), \
                mock.patch("torch.cuda.cudart", lambda: reg):
            ht = engram._HostTable("per_rank", self.N, "t0", None)
        cls = type("DeepseekV4ForCausalLM", (nn.Module,), {})
        m = cls()
        m.embed = nn.Module()
        m.embed.host_table = ht
        m.embed.weight = nn.Parameter(ht.bytes.view(torch.float8_e4m3fn), requires_grad=False)
        m.other = nn.Parameter(torch.arange(4096, dtype=torch.float32), requires_grad=False)
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)
            ht.bytes.copy_(torch.randint(0, 255, (self.N,), generator=gen, dtype=torch.uint8))
        return m, ht

    def test_restore_registers_without_prefault(self):
        reg = _Registrar()
        with mock.patch("torch.cuda.cudart", lambda: reg):
            src, sht = self._table_model(reg)
            self.assertTrue(sht.deferred)
            self.assertEqual(reg.calls, [])
            s = self.snap(src)
            self.assertFalse(s.begin())  # nothing saved: pre-fault + register, then the stream writes
            self.assertFalse(sht.deferred)
            self.assertEqual(len(reg.calls), 1)
            gen = torch.Generator().manual_seed(3)
            sht.bytes.copy_(torch.randint(0, 255, (self.N,), generator=gen, dtype=torch.uint8))
            s.end()
            dst, dht = self._table_model(reg)
            self.assertTrue(dht.deferred)
            with mock.patch.object(type(dht), "_prefault", side_effect=AssertionError("pre-fault on restore")):
                self.assertTrue(self.snap(dst).begin())
            self.assertFalse(dht.deferred)
            self.assertEqual(reg.calls[-1], (dht.bytes.data_ptr(), self.N))
            self.assertEqual(dht.device_ptr, dht.bytes.data_ptr())
            self.assertTrue(torch.equal(sht.bytes, dht.bytes))
            dht.finish_load("layer 0")  # no second registration
            self.assertEqual(len(reg.calls), 2)

    def test_no_snapshot_prefaults_then_registers(self):
        reg = _Registrar()
        with mock.patch("torch.cuda.cudart", lambda: reg):
            m, ht = self._table_model(reg)
            with mock.patch.object(type(ht), "_prefault", autospec=True, side_effect=lambda self_: None) as pf:
                self.assertFalse(self.snap(m).begin())
            pf.assert_called_once()
            self.assertEqual(len(reg.calls), 1)
            self.assertFalse(ht.deferred)

    def test_off_without_the_switch(self):
        reg = _Registrar()
        from sglang.srt.layers import engram

        with mock.patch.dict(os.environ, {"SGLANG_OPT_INF834_KEEP_PAGE_CACHE": "1"}), \
                mock.patch.object(engram, "_graphs_persisted", lambda: False), \
                mock.patch("torch.cuda.cudart", lambda: reg):
            ht = engram._HostTable("per_rank", self.N, "t1", None)
        self.assertFalse(ht.deferred)
        self.assertEqual(len(reg.calls), 1)


# -------------------------------------------------------------------------------------- per-chunk IO

@unittest.skipUnless(WORK, "O_DIRECT needs a real disk (set INF834_SNAP_TEST_DIR)")
class TestChunkPolicy(Base):
    def test_resident_buffered_rest_direct_then_rewarm(self):
        tensors = [("big.a", torch.int8, (256, 1 << 20)), ("big.b", torch.int8, (256, 1 << 20))]
        src = make_model(tensors, seed=4)
        s = self.snap(src)
        s.end()  # O_DIRECT writes: nothing of the file is in the page cache
        size = os.path.getsize(s.bin)
        fd = os.open(s.bin, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            half = size // 2 // ws.CHUNK * ws.CHUNK
            off = 0
            while off < half:  # make the first half resident
                off += len(os.pread(fd, min(ws.CHUNK, half - off), off))
        finally:
            os.close(fd)
        dst = make_model(tensors, seed=4)
        zero(dst)
        d = self.snap(dst)
        with self.assertLogs("sglang", level="INFO") as logs:
            self.assertTrue(d.begin())
        text = "\n".join(logs.output)
        m = re.search(r"policy chunk: ([0-9.]+) GB buffered \+ ([0-9.]+) GB O_DIRECT", text)
        self.assertIsNotNone(m, text)
        self.assertAlmostEqual(float(m.group(1)), half / 1e9, delta=0.07)
        self.assertGreater(float(m.group(2)), 0.2)
        for (k, a), b in zip(state(src).items(), state(dst).values()):
            self.assertTrue(torch.equal(a, b), k)
        d._rewarm_thread.join(30)
        import time as _t

        for _ in range(50):
            if ws._resident_fraction(s.bin) > 0.95:
                break
            _t.sleep(0.2)
        self.assertGreater(ws._resident_fraction(s.bin), 0.95)


if __name__ == "__main__":
    unittest.main()
