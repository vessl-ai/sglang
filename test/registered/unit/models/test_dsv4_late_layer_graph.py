"""SGLANG_OPT_DSV4_LATE_LAYER_GRAPH: the late-layer runner's mechanics on a toy layer
stack. The stack has the real runner's shape -- an mHC state in, row-local ops captured,
a window store and an "attention" that read the live batch through graph breaks -- so a
bucket captured on one batch must replay any later batch of that bucket exactly as the
eager stack computes it, write the same pool rows, and leave the rows past the live
count out of every break."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.models.deepseek_common import late_layer_graph as lg
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

H, HC, L, POOL_ROWS = 256, 4, 6, 8192
DSPARK_LAYER = 3

_pool = {}


def _store(x, fb, n):
    _pool["kv"][fb.write_loc[:n]] = x[:n]


def _attend(x, out, fb, n):
    # Reads pool rows written this step and earlier ones (the window), by live index.
    out[:n] = _pool["kv"][fb.read_idx[:n]] * 0.5 + x[:n] * fb.scale[:n, None]
    out[n:] = 0


@eager_on_graph(capture_stub=lg._noop)
def _store_break(x):
    step = lg._live()
    _store(x, step.forward_batch, step.rows)


@eager_on_graph(capture_stub=lg._noop)
def _attend_break(x, out):
    step = lg._live()
    _attend(x, out, step.forward_batch, step.rows)


class _ToyModel:
    late_layer_start, end_layer = 0, L

    def __init__(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        self.w = torch.rand((L, H), generator=g, device="cuda").to(torch.bfloat16) + 0.5
        self.calls = 0

    def _forward_late_layers(
        self, *, state, positions, input_ids, input_ids_global, forward_batch,
        capture_dspark,
    ):  # fmt: skip
        self.calls += 1
        aux = []
        r, pre = state.residual, state.pre
        for layer in range(L):
            h = (r.float() * pre[..., None]).sum(1).to(torch.bfloat16) * self.w[layer]
            h = h + (positions[:, None] % 7).to(torch.bfloat16) * 0.01
            if lg.active():
                _store_break(h)
            else:
                _store(h, forward_batch, h.shape[0])
            out = torch.empty_like(h)
            if lg.active():
                _attend_break(h, out)
            else:
                _attend(h, out, forward_batch, h.shape[0])
            r = r + out[:, None, :] * 0.25
            if capture_dspark and layer == DSPARK_LAYER:
                aux.append(r.mean(dim=1))
        return mhc.HcState(r, pre), aux


def _batch(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    write_loc = torch.randperm(POOL_ROWS - 1, generator=g, device="cuda")[:rows] + 1
    read_idx = torch.randint(0, POOL_ROWS, (rows,), generator=g, device="cuda")
    # Some reads hit this step's own writes.
    read_idx[::3] = write_loc[::3]
    fb = SimpleNamespace(
        write_loc=write_loc,
        read_idx=read_idx,
        scale=torch.rand((rows,), generator=g, device="cuda").to(torch.bfloat16),
    )
    residual = torch.randn((rows, HC, H), generator=g, device="cuda").to(
        torch.bfloat16
    )
    pre = torch.rand((rows, HC), generator=g, device="cuda")
    positions = torch.randint(0, 75000, (rows,), generator=g, device="cuda")
    return fb, residual, pre, positions


@contextmanager
def _graph_capture(stream=None):
    with torch.cuda.stream(stream):
        yield SimpleNamespace(stream=stream)


def _eager(model, fb, residual, pre, positions, dspark):
    state, aux = model._forward_late_layers(
        state=mhc.HcState(residual.clone(), pre.clone()),
        positions=positions,
        input_ids=None,
        input_ids_global=None,
        forward_batch=fb,
        capture_dspark=dspark,
    )
    return state.residual, state.pre, aux


def _run(runner, fb, residual, pre, positions, dspark):
    aux = []
    state = runner.run(
        state=mhc.HcState(residual.clone(), pre.clone()),
        positions=positions,
        input_ids=None,
        input_ids_global=None,
        forward_batch=fb,
        capture_dspark=dspark,
        aux_out=aux,
    )
    if state is None:
        return None
    return state.residual, state.pre, aux


@pytest.mark.parametrize("dspark", [False, True])
def test_replay_matches_eager_on_new_batches(dspark, monkeypatch):
    monkeypatch.setenv("SGLANG_OPT_DSV4_LATE_LAYER_GRAPH_BUCKETS", "32,64,128")
    _pool["kv"] = torch.randn((POOL_ROWS, H), device="cuda").to(torch.bfloat16)
    model = _ToyModel()
    with mock.patch(
        "sglang.srt.distributed.parallel_state.graph_capture", _graph_capture
    ):
        runner = lg.LateLayerGraphRunner(model)
        # 50 rows: warm-up + capture of bucket 64; 37 and 64 rows replay it; 100 rows
        # capture 128; 20 rows capture 32; 37 rows replay 64 again after the others.
        for rows, seed in [(50, 1), (37, 2), (64, 3), (100, 4), (20, 5), (37, 6)]:
            fb, residual, pre, positions = _batch(rows, seed)
            before = _pool["kv"].clone()
            want = _eager(model, fb, residual, pre, positions, dspark)
            want_pool = _pool["kv"].clone()
            _pool["kv"].copy_(before)
            got = _run(runner, fb, residual, pre, positions, dspark)
            assert got is not None
            for a, b in zip((got[0], got[1], *got[2]), (want[0], want[1], *want[2])):
                assert a.shape == b.shape
                assert torch.equal(a, b), (rows, (a.float() - b.float()).abs().max())
            assert len(got[2]) == (1 if dspark else 0)
            # Same pool rows written, nothing past the live rows.
            assert torch.equal(_pool["kv"], want_pool), rows
    assert runner.stats["capture"] == 3
    assert runner.stats["replay"] == 3
    assert sorted(e.bucket for e in runner._entries.values()) == [32, 64, 128]


def test_too_many_rows_run_eagerly(monkeypatch):
    monkeypatch.setenv("SGLANG_OPT_DSV4_LATE_LAYER_GRAPH_BUCKETS", "32,64")
    _pool["kv"] = torch.zeros((POOL_ROWS, H), device="cuda", dtype=torch.bfloat16)
    runner = lg.LateLayerGraphRunner(_ToyModel())
    fb, residual, pre, positions = _batch(65, 7)
    assert _run(runner, fb, residual, pre, positions, False) is None
    assert runner.stats == {"replay": 0, "capture": 0, "eager": 1}


def test_capture_failure_disables_the_runner(monkeypatch):
    monkeypatch.setenv("SGLANG_OPT_DSV4_LATE_LAYER_GRAPH_BUCKETS", "64")
    _pool["kv"] = torch.zeros((POOL_ROWS, H), device="cuda", dtype=torch.bfloat16)
    model = _ToyModel()
    runner = lg.LateLayerGraphRunner(model)
    fb, residual, pre, positions = _batch(40, 8)
    want = _eager(model, fb, residual, pre, positions, False)

    @contextmanager
    def _broken(stream=None):
        raise RuntimeError("capture not allowed")
        yield

    with mock.patch("sglang.srt.distributed.parallel_state.graph_capture", _broken):
        got = _run(runner, fb, residual, pre, positions, False)
    # The warm-up result is still this step's result; later steps run eagerly.
    assert torch.equal(got[0], want[0])
    assert runner._disabled is not None
    assert _run(runner, fb, residual, pre, positions, False) is None


def test_buckets():
    assert lg.parse_buckets("") == lg._DEFAULT_BUCKETS
    assert lg.parse_buckets("64, 32,64") == (32, 64)
    assert lg.select_bucket((32, 64), 33) == 64
    assert lg.select_bucket((32, 64), 65) is None
    with pytest.raises(ValueError):
        lg.parse_buckets("0")


# ---------------------------------------------------------------------------
# Prefill CP on, but off for a batch of fewer extend tokens than CP ranks (a 1-token
# /health_generate step, a fully cached prompt): that batch's late layers take the
# non-CP code. A graph captured on a CP batch must not replay it (the CP window-KV
# break asserted is_cp_active inside cp_materialize_global_token_order, INF-681 i3
# PD cell 2026-10-29 16:50Z), and a non-CP capture must not serve CP batches.
# ---------------------------------------------------------------------------

from sglang.srt.layers.cp import utils as cp_utils  # noqa: E402

CP_SIZE = 4


class _FakeAttn:
    """The two window-KV stores of MqaAttention, writing the toy pool."""

    def _cp_store_window_kv(self, kv, forward_batch, attn_backend):
        # The real gather: asserts the batch runs prefill CP.
        kv = cp_utils.cp_materialize_global_token_order(kv.contiguous(), forward_batch)
        _pool["kv"][forward_batch.write_loc[: kv.shape[0]]] = kv

    def _compute_kv_to_cache(
        self, x, positions, forward_batch, attn_backend, qkv_a=None, q_rope=None
    ):
        _pool["kv"][forward_batch.write_loc[: x.shape[0]]] = x


class _ToyCPModel(_ToyModel):
    def __init__(self):
        super().__init__()
        self.attn = _FakeAttn()

    def _forward_late_layers(
        self, *, state, positions, input_ids, input_ids_global, forward_batch,
        capture_dspark,
    ):  # fmt: skip
        cp = cp_utils.is_cp_active(forward_batch)
        r, pre = state.residual, state.pre
        for layer in range(L):
            h = (r.float() * pre[..., None]).sum(1).to(torch.bfloat16) * self.w[layer]
            if cp:
                h = h * 2  # stands for the CP-only ops of a layer (gathers)
            if lg.active():
                if cp:
                    lg.cp_store_window_kv(self.attn, h)
                else:
                    lg.store_window_kv(self.attn, h, positions, None, None)
            elif cp:
                self.attn._cp_store_window_kv(h, forward_batch, None)
            else:
                self.attn._compute_kv_to_cache(h, positions, forward_batch, None)
            out = torch.empty_like(h)
            if lg.active():
                _attend_break(h, out)
            else:
                _attend(h, out, forward_batch, h.shape[0])
            r = r + out[:, None, :] * 0.25
        return mhc.HcState(r, pre), []


@pytest.fixture
def _prefill_cp4(monkeypatch):
    """A CP4 server: prefill CP is active iff the batch says so; the CP gather of
    this single process is the identity."""
    monkeypatch.setattr(cp_utils, "is_cp_active", lambda fb: fb.cp_active)
    monkeypatch.setattr(
        cp_utils,
        "get_cp_strategy",
        lambda: SimpleNamespace(gather_kv_cache=lambda x, fb, stream=None: x),
    )
    monkeypatch.setattr(
        "sglang.srt.runtime_context.get_parallel",
        lambda: SimpleNamespace(attn_cp_size=CP_SIZE),
    )
    monkeypatch.setattr(
        "sglang.srt.model_executor.forward_context.get_attn_backend",
        lambda: SimpleNamespace(forward_metadata=None),
    )
    monkeypatch.setenv("SGLANG_OPT_DSV4_LATE_LAYER_GRAPH_BUCKETS", "32,64")


def test_cp_inactive_batches_run_eagerly_between_cp_batches(_prefill_cp4):
    _pool["kv"] = torch.randn((POOL_ROWS, H), device="cuda").to(torch.bfloat16)
    model = _ToyCPModel()
    with mock.patch(
        "sglang.srt.distributed.parallel_state.graph_capture", _graph_capture
    ):
        runner = lg.LateLayerGraphRunner(model)
        # 4-token PD warm-up (CP on, 1 tail row per rank: captures bucket 32), a
        # 1-token health_generate step (CP off), CP steps, a 2-token step, CP again.
        steps = [(1, True), (1, False), (8, True), (2, False), (20, True), (3, False)]
        for i, (rows, cp_active) in enumerate(steps):
            fb, residual, pre, positions = _batch(rows, 100 + i)
            fb.cp_active = cp_active
            before = _pool["kv"].clone()
            want = _eager(model, fb, residual, pre, positions, False)
            want_pool = _pool["kv"].clone()
            _pool["kv"].copy_(before)
            got = _run(runner, fb, residual, pre, positions, False)
            if not cp_active:
                # The caller runs the late layers eagerly.
                assert got is None, (rows, cp_active)
                got = _eager(model, fb, residual, pre, positions, False)
            assert torch.equal(got[0], want[0]), (rows, cp_active)
            assert torch.equal(_pool["kv"], want_pool), (rows, cp_active)
    # Every CP step after the first replays the one CP capture of bucket 32.
    assert runner.stats["capture"] == 1
    assert runner.stats["replay"] == 2
    assert [k[1][-1] for k in runner._entries] == [True]
