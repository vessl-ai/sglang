"""CPU-only tests for SGLANG_OPT_INF834_WEIGHT_SNAPSHOT (model_loader/inf834_weight_snapshot.py).

- byte-exact round trip over a model built from a real decoder rank's tensor list (names, dtypes, shapes from
  a weight digest of an EP8 decoder boot when INF834_DIGEST_JSON exists; large tensors are cut on dim 0 so
  the test writes about 1 GB), buffered and O_DIRECT reads
- identity mismatch -> fallback before any parameter is touched
- truncated or uncommitted snapshot -> fallback / ignored
- checksum failure after the parameters were filled -> fallback reported as touched (caller reloads)
- two processes saving the same rank at once: one writes, the other skips; the result is valid
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


if __name__ == "__main__":
    unittest.main()
