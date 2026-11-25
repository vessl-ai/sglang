"""INF-874 dsv4_mxfp4_decode_moe_k1 through the engine seam (dsv4_mxfp4_decode_moe_ep with
SGLANG_OPT_DSV4_EP_DECODE_MOE_KERNEL=k1): whole 2304-wide experts in expert parallel.

- accurate against an fp32 W4A8 reference (x and h row-quantized), close to v2w on the same inputs;
- batch invariance: a token's output does not depend on the other tokens of the call (whole call == token chunks);
- the engine's received form: local top-k ids with -1 (other ranks) and pad rows (all -1) -> every row written, zero
  where a token has no local slot (as combine_ep);
- a pair list with a device-side count gives the same bits as the top-k form;
- hot experts (rows above one 128-row tile), token chunks above MAX_VSLOTS (eager prefill);
- CUDA graph replay with fresh routing in the same buffers == eager, bitwise;
- a layer whose block exponents span more than 11 inside a channel falls back to v2w packing.
"""

import pytest
import torch

from sglang.srt.utils.common import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm90_supported(),
    reason="SM90 wgmma W4A8 decode MoE",
)

HIDDEN = 5120
INTER = 2304
LIMIT = 10.0
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


@pytest.fixture(autouse=True)
def _k1_mode(monkeypatch):
    monkeypatch.setenv("SGLANG_OPT_DSV4_EP_DECODE_MOE_KERNEL", "k1")


def _experts(num_experts, seed=0, scale_lo=119, scale_hi=123):
    """Engine layout: w13 [E, 2I, H/2] (gate rows then up rows), s13, w2 [E, H, I/2], s2."""
    g = torch.Generator().manual_seed(seed)
    nib = lambda *s: torch.randint(-128, 128, s, generator=g, dtype=torch.int8).cuda()  # noqa: E731
    sc = lambda *s: (  # noqa: E731
        torch.randint(scale_lo, scale_hi, s, generator=g, dtype=torch.uint8).view(torch.float8_e8m0fnu).cuda()
    )
    return (
        nib(num_experts, 2 * INTER, HIDDEN // 2),
        sc(num_experts, 2 * INTER, HIDDEN // 32),
        nib(num_experts, HIDDEN, INTER // 2),
        sc(num_experts, HIDDEN, INTER // 32),
    )


def _dequant(weight, scale):
    lut = torch.tensor(E2M1, device="cuda")
    w = weight.view(torch.uint8).long()
    values = torch.stack([lut[w & 0xF], lut[w >> 4]], -1).flatten(-2)
    return values * torch.exp2(scale.view(torch.uint8).float() - 127.0).repeat_interleave(32, -1)


def _q_rows(v):
    vf = v.float()
    amax = vf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    sc = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (vf / sc).to(torch.float8_e4m3fn).float() * sc


def _bf(t):
    return t.to(torch.bfloat16).float()


def _reference(x, ids, wts, experts):
    w13, s13, w2, s2 = experts
    out = torch.zeros(x.shape[0], HIDDEN, device="cuda")
    xq = _q_rows(x)
    for e in torch.unique(ids[ids >= 0]).tolist():
        tok, kk = torch.nonzero(ids == e, as_tuple=True)
        wgu = _dequant(w13[e], s13[e])
        g = _bf(xq[tok] @ wgu[:INTER].T).clamp(max=LIMIT)
        u = _bf(xq[tok] @ wgu[INTER:].T).clamp(-LIMIT, LIMIT)
        h = _bf(_bf(g / (1 + torch.exp(-g))) * u)
        out.index_add_(0, tok, _bf((_q_rows(h) @ _dequant(w2[e], s2[e]).T) * wts[tok, kk].unsqueeze(1)))
    return _bf(out)


def _routing(tokens, num_experts, local, seed, top_k=6, pad=0, hot=0.0):
    """ids [tokens + pad, top_k]: a slot is local with probability `local` (else -1); pad rows all -1. hot > 0 sends that
    share of the local slots to expert 0."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(0, num_experts, (tokens, top_k), device="cuda", generator=g)
    if hot > 0:
        ids = torch.where(torch.rand(ids.shape, device="cuda", generator=g) < hot, torch.zeros_like(ids), ids)
    ids = torch.where(torch.rand(ids.shape, device="cuda", generator=g) < local, ids, torch.full_like(ids, -1))
    ids = torch.cat([ids, torch.full((pad, top_k), -1, device="cuda", dtype=ids.dtype)]).int()
    x = torch.randn(tokens + pad, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.rand(tokens + pad, top_k, device="cuda", generator=g) * 0.3
    return x, ids, w


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _pack(experts, mode):
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ep as ek

    packed, split = ek.pack_experts_ep(*experts, mode=mode)
    assert split == 1
    return packed


def _call(x, ids, w, packed, out=None):
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ep as ek

    if out is None:
        out = torch.full((x.shape[0], HIDDEN), float("nan"), dtype=torch.bfloat16, device="cuda")
    ek.decode_moe_ep(x, ids, w, packed, LIMIT, 1, output=out)
    return out


@pytest.mark.parametrize("tokens,local,hot", [(3, 0.5, 0.0), (300, 0.3, 0.0), (1500, 0.2, 0.0), (800, 0.3, 0.5)])
def test_k1_matches_reference_and_v2w(tokens, local, hot):
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_k1 as K1

    experts = _experts(16)
    pk = _pack(experts, "k1")
    assert isinstance(pk, K1.PackedK1)
    pv = _pack(experts, "v2w")
    x, ids, w = _routing(tokens, 16, local, tokens, pad=5, hot=hot)
    out = _call(x, ids, w, pk)
    ref = _reference(x, ids, w, experts)
    assert not torch.isnan(out).any()
    has = (ids >= 0).any(1)
    assert torch.count_nonzero(out[~has]) == 0  # tokens without a local slot and pad rows: zero rows
    out_v = _call(x, ids, w, pv)
    err, err_v = _rel(out[has], ref[has]), _rel(out_v[has], ref[has])
    # K1 accumulates all of K in the tensor core (v2w promotes every 32-k block in fp32): on these random weights its
    # error against this reference is ~5x v2w's (~1.1% vs ~0.2%); on the checkpoint's weights ~1.0x
    assert err < 2e-2 and err < 8 * err_v, (tokens, local, hot, err, err_v)
    assert _rel(out[has], out_v[has]) < 2e-2


def test_k1_pair_list_equals_topk_form():
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_k1 as K1
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as WS

    experts = _experts(16, seed=3)
    pk = _pack(experts, "k1")
    x, ids, w = _routing(400, 16, 0.3, 9)
    out = _call(x, ids, w, pk)
    # the same slots as a pair list (token order, k ascending) with a fixed capacity and a device-side count
    tok, kk = torch.nonzero(ids >= 0, as_tuple=True)
    P, cap = tok.numel(), ids.numel()
    ptok = torch.full((cap,), x.shape[0], dtype=torch.int32, device="cuda")
    peid = torch.full((cap,), -1, dtype=torch.int32, device="cuda")
    pw = torch.zeros(cap, device="cuda")
    ptok[:P], peid[:P], pw[:P] = tok.int(), ids[tok, kk], w[tok, kk]
    x8, xs = WS._row_quant(x)
    out_p = torch.zeros_like(out)
    T = x.shape[0]
    sc = K1.x_scratch(cap, T, 16, x.device)
    y = torch.empty(cap, HIDDEN, dtype=torch.bfloat16, device="cuda")
    K1.decode_moe(x8.view(torch.uint8), xs, peid, pw, K1.PackedW13(pk.w13, pk.cs13, 0), K1.PackedW2(pk.w2, pk.cs2, 0),
                  LIMIT, out_p, y, sc, tok=ptok, num_valid=torch.tensor([P], dtype=torch.int32, device="cuda"))
    has = (ids >= 0).any(1)
    assert torch.equal(out_p[has], out[has])


def test_k1_token_chunks(monkeypatch):
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ep as ek

    experts = _experts(8, seed=5)
    pk, pv = _pack(experts, "k1"), _pack(experts, "v2w")
    x, ids, w = _routing(1000, 8, 0.4, 11)
    whole = _call(x, ids, w, pk)
    monkeypatch.setattr(ek, "max_virtual_slots", lambda: 6 * 300)  # chunks of 300 tokens (eager)
    chunked = _call(x, ids, w, pk)
    assert torch.equal(whole, chunked)
    assert _rel(chunked, _call(x, ids, w, pv)) < 2e-2


def test_k1_cuda_graph_replay():
    experts = _experts(16, seed=7)
    pk = _pack(experts, "k1")
    T = 512
    x, ids, w = _routing(T, 16, 0.3, 20)
    out = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device="cuda")
    _call(x, ids, w, pk, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _call(x, ids, w, pk, out)
    for seed in (21, 22, 23):
        x2, ids2, w2 = _routing(T, 16, 0.3, seed, hot=0.3 if seed == 23 else 0.0)
        x.copy_(x2), ids.copy_(ids2), w.copy_(w2)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, _call(x2, ids2, w2, pk)), seed


def test_k1_falls_back_to_v2w_when_not_exact():
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ep as ek
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2w as V2W

    w13, s13, w2, s2 = _experts(4, seed=9)
    s13 = s13.view(torch.uint8).clone()
    s13[1, 7, 3] = 100  # a block 21 below its channel's largest exponent (with nonzero values)
    packed, split = ek.pack_experts_ep(w13, s13.view(torch.float8_e8m0fnu), w2, s2, mode="k1")
    assert isinstance(packed, V2W.PackedExperts) and split == 1
