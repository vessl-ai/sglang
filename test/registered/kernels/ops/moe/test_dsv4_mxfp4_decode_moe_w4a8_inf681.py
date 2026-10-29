"""INF-681 SGLANG_OPT_DSV4_SM90_DECODE_MOE_W4A8: persistent warp-specialized W4A8 wgmma decode MoE
(dsv4_mxfp4_decode_moe_ws) for the DeepSeek-V4 TP4 MXFP4 experts.

Checks against an fp32 W4A8 reference (activations of both GEMMs quantized to e4m3 with a power-of-2
row scale, exact MXFP4 weights, the engine's bf16 rounding points) on target and draft shapes (DSpark
gamma 5: 6 / 5 rows per request, gamma 3: 4 / 3), with masked slots and experts that need several
32-row passes; dp-attention pad rows (topk_ids -1) are dropped by every sort path; CUDA graph
capture/replay with fresh inputs; and the switch wiring in Mxfp4MarlinMoEMethod."""

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
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


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


def _dequant(weight, scale):
    lut = torch.tensor(E2M1, device="cuda")
    w = weight.cuda().view(torch.uint8).long()
    values = torch.stack([lut[w & 0xF], lut[w >> 4]], -1).flatten(-2)
    s = torch.exp2(scale.cuda().view(torch.uint8).float() - 127.0)
    return values * s.repeat_interleave(32, -1)


def _q_rows(v):
    vf = v.float()
    amax = vf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    sc = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (vf / sc).to(torch.float8_e4m3fn).float() * sc


def _bf(t):
    return t.to(torch.bfloat16).float()


def _reference(x, ids, wts, w13f, w2f):
    out = torch.zeros(x.shape[0], HIDDEN, device=x.device)
    xq = _q_rows(x)
    for e in torch.unique(ids.clamp_min(0)).tolist():
        tok, kk = torch.nonzero(ids.clamp_min(0) == e, as_tuple=True)
        gu = _bf(xq[tok] @ w13f[e].T)
        g = gu[:, :INTERMEDIATE].clamp(max=SWIGLU_LIMIT)
        u = gu[:, INTERMEDIATE:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = _bf(_bf(g / (1 + torch.exp(-g))) * u)
        w = torch.where(ids[tok, kk] < 0, 0.0, wts[tok, kk]).unsqueeze(1)
        out.index_add_(0, tok, _bf((_q_rows(h) @ w2f[e].T) * w))
    return _bf(out)


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


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("shape", list(SHAPES))
def test_w4a8_matches_reference(shape):
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

    num_experts, top_k, rows_per_req = SHAPES[shape]
    w = _weights(num_experts)
    w13, s13 = w["w13_weight"].cuda(), w["w13_weight_scale_inv"].cuda()
    w2, s2 = w["w2_weight"].cuda(), w["w2_weight_scale_inv"].cuda()
    packed = ws.pack_experts(w13, s13, w2, s2)
    assert packed.w13.numel() == w13.numel() + s13.numel()  # raw MXFP4 size
    w13f = torch.stack([_dequant(w13[e], s13[e]) for e in range(num_experts)])
    w2f = torch.stack([_dequant(w2[e], s2[e]) for e in range(num_experts)])
    for bs in (1, 48, 125, 300):
        # mask the last rows, but never all of them (the relative error needs a nonzero reference)
        masked = min(3, bs * rows_per_req - 1)
        x, ids, wts = _routing(
            bs, rows_per_req, num_experts, top_k, bs, masked_rows=masked
        )
        out = ws.decode_moe(x, ids, wts, packed, SWIGLU_LIMIT)
        assert not torch.isnan(out).any()
        err = _rel(out, _reference(x, ids, wts, w13f, w2f))
        # FP8 tensor-core accumulation inside each 32-k block; measured 0.1-0.26%.
        assert err < 5e-3, (bs, err)


@pytest.mark.parametrize("shape", ["target_g3", "draft_g3"])
def test_w4a8_dp_pad_rows(shape):
    """dp-attention MAX_LEN pad rows reach the MoE with topk_ids -1 (SGLANG_OPT_MASK_DP_PAD_MOE):
    their output is 0, they change no other row, and no sort path counts them for any expert."""
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

    num_experts, top_k, rows_per_req = SHAPES[shape]
    w = _weights(num_experts)
    packed = ws.pack_experts(
        w["w13_weight"].cuda(),
        w["w13_weight_scale_inv"].cuda(),
        w["w2_weight"].cuda(),
        w["w2_weight_scale_inv"].cuda(),
    )
    # bs 40 / 170 / 700: the triton sort, the one-CTA CUDA sort and the torch fallback
    for bs in (40, 170, 700):
        x, ids, wts = _routing(bs, rows_per_req, num_experts, top_k, 3 * bs)
        pad = max(1, len(ids) // 5)
        xp = torch.cat([x, torch.randn(pad, HIDDEN, device="cuda", dtype=x.dtype)])
        idsp = torch.cat(
            [ids, torch.full((pad, top_k), -1, dtype=ids.dtype, device="cuda")]
        )
        wtsp = torch.cat([wts, torch.rand(pad, top_k, device="cuda")])
        counts = ws._sort_slots(idsp, wtsp, num_experts)[0]
        assert int(counts.sum()) == ids.numel(), bs
        out = ws.decode_moe(xp, idsp, wtsp, packed, SWIGLU_LIMIT)
        assert torch.equal(out[len(ids) :], torch.zeros_like(out[len(ids) :])), bs
        ref = ws.decode_moe(x, ids, wts, packed, SWIGLU_LIMIT)
        assert torch.equal(out[: len(ids)], ref), bs


@pytest.mark.parametrize("shape", list(SHAPES))
def test_w4a8_cuda_graph_replay(shape):
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

    num_experts, top_k, rows_per_req = SHAPES[shape]
    w = _weights(num_experts)
    packed = ws.pack_experts(
        w["w13_weight"].cuda(),
        w["w13_weight_scale_inv"].cuda(),
        w["w2_weight"].cuda(),
        w["w2_weight_scale_inv"].cuda(),
    )
    for bs in (1, 16, 64, 128):
        x, ids, wts = _routing(bs, rows_per_req, num_experts, top_k, 1000 + bs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            ws.decode_moe(x, ids, wts, packed, SWIGLU_LIMIT)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = ws.decode_moe(x, ids, wts, packed, SWIGLU_LIMIT)
        x2, ids2, wts2 = _routing(
            bs, rows_per_req, num_experts, top_k, 2000 + bs, masked_rows=1
        )
        x.copy_(x2)
        ids.copy_(ids2)
        wts.copy_(wts2)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ws.decode_moe(x2, ids2, wts2, packed, SWIGLU_LIMIT)), bs


def test_w4a8_switch_wiring(monkeypatch):
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    monkeypatch.setenv("SGLANG_OPT_DSV4_SM90_DECODE_MOE_W4A8", "1")
    num_experts, top_k, rows_per_req = SHAPES["target"]
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
    method.process_weights_after_loading(layer)
    assert layer._dsv4_mxfp4_backend == "sm90_decode_w4a8"
    x, ids, wts = _routing(32, rows_per_req, num_experts, top_k, 7)
    tk = StandardTopKOutput(topk_weights=wts, topk_ids=ids, router_logits=None)
    out = method.apply(
        layer,
        StandardDispatchOutput(
            hidden_states=x, hidden_states_scale=None, topk_output=tk
        ),
    )
    packed = ws.pack_experts(
        w["w13_weight"].cuda(),
        w["w13_weight_scale_inv"].cuda(),
        w["w2_weight"].cuda(),
        w["w2_weight_scale_inv"].cuda(),
    )
    assert torch.equal(
        out.hidden_states, ws.decode_moe(x, ids, wts, packed, SWIGLU_LIMIT)
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
