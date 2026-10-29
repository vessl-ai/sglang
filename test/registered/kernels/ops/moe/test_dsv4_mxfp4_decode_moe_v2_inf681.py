"""INF-681 SGLANG_OPT_DSV4_SM90_DECODE_MOE_W4A8_V2: one persistent decode MoE kernel (gate/up, h row
quantization, down, dynamically handed-out items) plus a one-launch prologue (dsv4_mxfp4_decode_moe_v2).

Its output must be bit-identical to the W4A8 kernels it replaces (dsv4_mxfp4_decode_moe_ws, whose accuracy
test_dsv4_mxfp4_decode_moe_w4a8_inf681.py checks against an fp32 W4A8 reference): on target and draft shapes
(DSpark gamma 5 and 3), with masked slots and dp-attention pad rows, experts that need several 32-row passes,
repeated calls (no ordering race), CUDA graph replay with fresh inputs, capture as the first call after
prepare() (also in a fresh process with JIT module builds forbidden during capture), batches above the prologue's
limit, and through the engine's Mxfp4MarlinMoEMethod switch."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.utils.common import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm90_supported(),
    reason="SM90 wgmma W4A8 decode MoE",
)

HIDDEN, INTERMEDIATE = 5120, 576
SWIGLU_LIMIT = 10.0
# (experts, top_k, rows per request): gamma 5 and gamma 3 (verify = gamma + 1 rows, draft = gamma rows)
SHAPES = {
    "target": (384, 6, 6),
    "draft": (128, 3, 5),
    "target_g3": (384, 6, 4),
    "draft_g3": (128, 3, 3),
}


def _weights(num_experts, seed=0):
    g = torch.Generator().manual_seed(seed)
    h, i = HIDDEN, INTERMEDIATE

    def nibbles(*shape):
        return torch.randint(-128, 128, shape, generator=g, dtype=torch.int8)

    def scales(*shape):
        return torch.randint(119, 123, shape, generator=g, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )

    return {
        "w13_weight": nibbles(num_experts, 2 * i, h // 2),
        "w2_weight": nibbles(num_experts, h, i // 2),
        "w13_weight_scale_inv": scales(num_experts, 2 * i, h // 32),
        "w2_weight_scale_inv": scales(num_experts, h, i // 32),
    }


def _packed(num_experts):
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

    w = _weights(num_experts)
    return ws.pack_experts(
        w["w13_weight"].cuda(),
        w["w13_weight_scale_inv"].cuda(),
        w["w2_weight"].cuda(),
        w["w2_weight_scale_inv"].cuda(),
    )


def _routing(bs, rows_per_req, num_experts, top_k, seed, masked_rows=0):
    """Rows of one request share most experts (as DSpark verify rows do), so hot experts get several
    32-row passes."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = bs * rows_per_req
    req = torch.randn(bs, num_experts, device="cuda", generator=g).repeat_interleave(
        rows_per_req, 0
    )
    logits = req + 0.5 * torch.randn(rows, num_experts, device="cuda", generator=g)
    w, ids = torch.topk(torch.softmax(logits, -1), top_k, dim=-1)
    ids = ids.int()
    if masked_rows:
        ids[-masked_rows:] = -1
    x = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)
    return x, ids, (w / w.sum(-1, keepdim=True)).float()


@pytest.mark.parametrize("shape", list(SHAPES))
def test_v2_bit_identical_to_w4a8(shape):
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2 as v2
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

    num_experts, top_k, rows_per_req = SHAPES[shape]
    packed = _packed(num_experts)
    # 1 request up to the DP4 operating point (4 ranks x 80 requests); hot experts need 2-4 passes
    for bs in (1, 7, 48, 125, 240, 320):
        x, ids, wts = _routing(
            bs, rows_per_req, num_experts, top_k, bs, masked_rows=min(3, bs - 1)
        )
        ref = ws.decode_moe(x, ids, wts, packed, SWIGLU_LIMIT)
        for _ in range(3):  # dynamic scheduling: every call must give the same bits
            out = v2.decode_moe(x, ids, wts, packed, SWIGLU_LIMIT)
            assert torch.equal(out, ref), bs


@pytest.mark.parametrize("shape", ["target_g3", "draft_g3"])
def test_v2_dp_pad_rows(shape):
    """dp-attention MAX_LEN pad rows reach the MoE with topk_ids -1 (SGLANG_OPT_MASK_DP_PAD_MOE): their
    output is 0 and they change no other row."""
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2 as v2
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

    num_experts, top_k, rows_per_req = SHAPES[shape]
    packed = _packed(num_experts)
    for bs in (40, 170, 240):
        x, ids, wts = _routing(bs, rows_per_req, num_experts, top_k, 3 * bs)
        pad = max(1, len(ids) // 5)
        xp = torch.cat([x, torch.randn(pad, HIDDEN, device="cuda", dtype=x.dtype)])
        idsp = torch.cat(
            [ids, torch.full((pad, top_k), -1, dtype=ids.dtype, device="cuda")]
        )
        wtsp = torch.cat([wts, torch.rand(pad, top_k, device="cuda")])
        out = v2.decode_moe(xp, idsp, wtsp, packed, SWIGLU_LIMIT)
        assert torch.equal(out[len(ids) :], torch.zeros_like(out[len(ids) :])), bs
        assert torch.equal(out[: len(ids)], ws.decode_moe(x, ids, wts, packed, SWIGLU_LIMIT)), bs
    # every slot masked: zeros, no expert work
    x, ids, wts = _routing(4, rows_per_req, num_experts, top_k, 5)
    out = v2.decode_moe(x, torch.full_like(ids, -1), wts, packed, SWIGLU_LIMIT)
    assert torch.equal(out, torch.zeros_like(out))


def test_v2_prologue_matches_w4a8_prologue():
    """The one-launch prologue gives #104's sort (counts, stable order, tokens, weights) and _row_quant."""
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2 as v2
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

    mod = v2._module()
    for shape, bs, masked in (
        ("target_g3", 1, 0),
        ("target_g3", 64, 9),
        ("target_g3", 336, 50),
        ("draft_g3", 252, 30),
        ("target", 450, 0),
    ):
        num_experts, top_k, rows_per_req = SHAPES[shape]
        x, ids, wts = _routing(bs, rows_per_req, num_experts, top_k, bs, masked_rows=masked)
        if bs > 1:
            x[0] = 0  # an all-zero row (amax clamps to 1e-4)
        counts, slots, toks, sw = ws._sort_slots(ids, wts, num_experts)
        x8, xs = ws._row_quant(x)
        n = ids.numel()
        c2 = torch.empty_like(counts)
        s2, t2 = torch.empty(n, dtype=torch.int32, device="cuda"), torch.empty(
            n, dtype=torch.int32, device="cuda"
        )
        w2 = torch.empty(n, dtype=torch.float32, device="cuda")
        x82 = torch.empty(x.shape, dtype=torch.uint8, device="cuda")
        xs2 = torch.empty(x.shape[0], dtype=torch.float32, device="cuda")
        work = torch.full((v2.WORKSPACE_BASE + 2 * num_experts,), 7, dtype=torch.int32, device="cuda")
        mod.prep(ids, wts, x, c2, s2, t2, w2, x82, xs2, work)
        k = int(counts.sum())
        assert torch.equal(c2, counts), (shape, bs)
        assert torch.equal(s2[:k], slots[:k]) and torch.equal(t2[:k], toks[:k]), (shape, bs)
        assert torch.equal(w2[:k], sw[:k]), (shape, bs)
        assert torch.equal(x82, x8.view(torch.uint8)) and torch.equal(xs2, xs), (shape, bs)
        assert torch.equal(work, torch.zeros_like(work)), (shape, bs)


@pytest.mark.parametrize("shape", list(SHAPES))
def test_v2_cuda_graph_replay(shape):
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2 as v2
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

    num_experts, top_k, rows_per_req = SHAPES[shape]
    packed = _packed(num_experts)
    for bs in (1, 16, 64, 256):
        x, ids, wts = _routing(bs, rows_per_req, num_experts, top_k, 1000 + bs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            v2.decode_moe(x, ids, wts, packed, SWIGLU_LIMIT)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = v2.decode_moe(x, ids, wts, packed, SWIGLU_LIMIT)
        for rep in range(2):  # the workspace is re-zeroed inside the graph on every replay
            x2, ids2, wts2 = _routing(
                bs, rows_per_req, num_experts, top_k, 2000 + 7 * rep + bs, masked_rows=1
            )
            x.copy_(x2)
            ids.copy_(ids2)
            wts.copy_(wts2)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, ws.decode_moe(x2, ids2, wts2, packed, SWIGLU_LIMIT)), (bs, rep)


def test_v2_large_batch_fallback():
    """Above the prologue's 16,384 routed slots, #104's prologue runs; same bits."""
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2 as v2
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

    num_experts, top_k, rows_per_req = SHAPES["target_g3"]
    packed = _packed(num_experts)
    x, ids, wts = _routing(700, rows_per_req, num_experts, top_k, 11, masked_rows=5)
    assert ids.numel() > v2.PREP_MAX_SLOTS
    assert torch.equal(
        v2.decode_moe(x, ids, wts, packed, SWIGLU_LIMIT),
        ws.decode_moe(x, ids, wts, packed, SWIGLU_LIMIT),
    )


def test_v2_prepare_then_capture():
    """After prepare(), the first v2 call of a process may be CUDA graph capture itself (no eager warmup);
    replay with fresh inputs matches the W4A8 kernels."""
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2 as v2
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

    v2.prepare(torch.cuda.current_device())
    v2.prepare(torch.cuda.current_device())  # once per process and device
    num_experts, top_k, rows_per_req = SHAPES["target_g3"]
    packed = _packed(num_experts)
    x, ids, wts = _routing(64, rows_per_req, num_experts, top_k, 21, masked_rows=6)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = v2.decode_moe(x, ids, wts, packed, SWIGLU_LIMIT)
    for rep in range(2):
        x2, ids2, wts2 = _routing(64, rows_per_req, num_experts, top_k, 30 + rep, masked_rows=6)
        x.copy_(x2)
        ids.copy_(ids2)
        wts.copy_(wts2)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ws.decode_moe(x2, ids2, wts2, packed, SWIGLU_LIMIT)), rep


_FRESH = r"""
import sys, torch
sys.path.insert(0, {testdir!r})
import test_dsv4_mxfp4_decode_moe_v2_inf681 as t
from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2 as v2
from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws
v2.prepare(torch.cuda.current_device())
packed = t._packed(384)
x, ids, wts = t._routing(48, 4, 384, 6, 5, masked_rows=4)

def no_build(*args, **kwargs):
    raise RuntimeError("JIT build during capture")

saved = (v2.load_jit, ws.load_jit)
v2.load_jit = ws.load_jit = no_build  # a JIT module build inside capture raises
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):  # the first v2 call of this process
    out = v2.decode_moe(x, ids, wts, packed, t.SWIGLU_LIMIT)
v2.load_jit, ws.load_jit = saved
x2, ids2, wts2 = t._routing(48, 4, 384, 6, 6, masked_rows=4)
x.copy_(x2); ids.copy_(ids2); wts.copy_(wts2)
graph.replay()
torch.cuda.synchronize()
assert torch.equal(out, ws.decode_moe(x2, ids2, wts2, packed, t.SWIGLU_LIMIT))
print("FRESH_OK")
"""


def test_v2_capture_first_call_in_fresh_process():
    """A fresh process: prepare(), then capture is the first v2 call, with JIT module builds forbidden during
    capture (the v2 call path inside capture has no Triton kernel); replay matches the W4A8 kernels."""
    import os
    import subprocess

    code = _FRESH.format(testdir=os.path.dirname(os.path.abspath(__file__)))
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=900
    )
    assert run.returncode == 0 and "FRESH_OK" in run.stdout, run.stderr[-4000:]


def test_v2_switch_wiring(monkeypatch):
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2 as v2
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    monkeypatch.setenv("SGLANG_OPT_DSV4_SM90_DECODE_MOE_W4A8", "1")
    monkeypatch.setenv("SGLANG_OPT_DSV4_SM90_DECODE_MOE_W4A8_V2", "1")
    num_experts, top_k, rows_per_req = SHAPES["target_g3"]
    w = _weights(num_experts)
    method = Mxfp4MarlinMoEMethod(
        SimpleNamespace(process_weights_after_loading=lambda layer: None), "test"
    )
    method.hidden_pad = 0
    layer = torch.nn.Module()
    layer.hidden_size = HIDDEN
    layer.intermediate_size_per_partition = INTERMEDIATE
    layer.dispatcher = SimpleNamespace()
    for name, tensor in w.items():
        layer.register_parameter(
            name, torch.nn.Parameter(tensor.cuda().clone(), requires_grad=False)
        )
    config = MoeRunnerConfig(
        num_experts=num_experts,
        num_local_experts=num_experts,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTERMEDIATE,
        top_k=top_k,
        params_dtype=torch.bfloat16,
        activation="silu",
        is_gated=True,
        swiglu_limit=SWIGLU_LIMIT,
        gate_up_interleaved=False,
        inplace=False,
        layer=layer,
    )
    layer.moe_runner_config = config
    method.create_moe_runner(layer, config)
    prepared = []
    real_prepare = v2.prepare
    monkeypatch.setattr(v2, "prepare", lambda dev: (prepared.append(dev), real_prepare(dev)))
    method.process_weights_after_loading(layer)
    assert layer._dsv4_mxfp4_backend == "sm90_decode_w4a8"
    assert prepared == [torch.cuda.current_device()]  # the v2 kernels were built at weight processing
    calls = []
    real = v2.decode_moe

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(v2, "decode_moe", spy)
    x, ids, wts = _routing(240, rows_per_req, num_experts, top_k, 7, masked_rows=4)
    tk = StandardTopKOutput(topk_weights=wts, topk_ids=ids, router_logits=None)
    out = method.apply(
        layer,
        StandardDispatchOutput(hidden_states=x, hidden_states_scale=None, topk_output=tk),
    )
    assert calls == [1]  # the v2 kernels ran
    assert torch.equal(
        out.hidden_states, ws.decode_moe(x, ids, wts, method._sm90_packed, SWIGLU_LIMIT)
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
