"""INF-730 dsv4_mxfp4_decode_moe_v2w: the v2 decode MoE kernel with the per-rank intermediate width a compile-time
constant (576 = TP4 as v2, 2304 = whole experts for expert parallel, 320 = a TP8 slice of 288 zero-padded).

- 576: bit-identical to v2 (same packed bytes, same code).
- 2304: accurate against an fp32 W4A8 reference that row-quantizes h over the whole 2304-wide expert row; the
  64-row passes and the grid-wide h quantization (its defaults) are bit-identical to 32-row passes with per-CTA
  quantization; (token, expert) pair rows with top_k 1 equal token rows with a local top-k; hot experts; repeated
  calls; CUDA graph replay with fresh inputs.
- 320: a zero-padded TP8 slice equals the 288-wide fp32 W4A8 reference.
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
LIMIT = 10.0
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def _experts(num_experts, inter, seed=0):
    g = torch.Generator().manual_seed(seed)
    nib = lambda *s: torch.randint(-128, 128, s, generator=g, dtype=torch.int8).cuda()  # noqa: E731
    sc = lambda *s: (
        torch.randint(119, 123, s, generator=g, dtype=torch.uint8)
        .view(torch.float8_e8m0fnu)
        .cuda()
    )  # noqa: E731
    return (
        nib(num_experts, inter, HIDDEN // 2),
        sc(num_experts, inter, HIDDEN // 32),
        nib(num_experts, inter, HIDDEN // 2),
        sc(num_experts, inter, HIDDEN // 32),
        nib(num_experts, HIDDEN, inter // 2),
        sc(num_experts, HIDDEN, inter // 32),
    )


def _dequant(weight, scale):
    lut = torch.tensor(E2M1, device="cuda")
    w = weight.view(torch.uint8).long()
    values = torch.stack([lut[w & 0xF], lut[w >> 4]], -1).flatten(-2)
    return values * torch.exp2(
        scale.view(torch.uint8).float() - 127.0
    ).repeat_interleave(32, -1)


def _q_rows(v):
    vf = v.float()
    amax = vf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    sc = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    return (vf / sc).to(torch.float8_e4m3fn).float() * sc


def _bf(t):
    return t.to(torch.bfloat16).float()


def _reference(x, ids, wts, experts, width=None):
    """fp32 W4A8: x and h row-quantized (h over its first `width` channels), bf16 h and per-expert outputs."""
    w1, s1, w3, s3, w2, s2 = experts
    out = torch.zeros(x.shape[0], HIDDEN, device="cuda")
    xq = _q_rows(x)
    for e in torch.unique(ids[ids >= 0]).tolist():
        tok, kk = torch.nonzero(ids == e, as_tuple=True)
        g = _bf(xq[tok] @ _dequant(w1[e], s1[e]).T).clamp(max=LIMIT)
        u = _bf(xq[tok] @ _dequant(w3[e], s3[e]).T).clamp(-LIMIT, LIMIT)
        h = _bf(_bf(g / (1 + torch.exp(-g))) * u)
        d2 = _dequant(w2[e], s2[e])
        if width is not None:
            h, d2 = h[:, :width], d2[:, :width]
        out.index_add_(0, tok, _bf((_q_rows(h) @ d2.T) * wts[tok, kk].unsqueeze(1)))
    return _bf(out)


def _routing(tokens, num_experts, top_k, seed, s=0.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pop = (
        1.0 / torch.arange(1, num_experts + 1, device="cuda", dtype=torch.float64)
    ) ** s
    ids = torch.multinomial(
        pop.expand(tokens, num_experts), top_k, replacement=False, generator=g
    ).int()
    x = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.softmax(
        torch.randn(tokens, top_k, device="cuda", generator=g), -1
    ).float()
    return x, ids, w


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_v2w_576_bit_identical_to_v2():
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2 as V2
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2w as V2W
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as WS

    w1, s1, w3, s3, w2, s2 = _experts(384, 576)
    u8 = torch.uint8
    p = WS.pack_experts(
        torch.cat([w1, w3], 1),
        torch.cat([s1.view(u8), s3.view(u8)], 1).view(s1.dtype),
        w2,
        s2,
    )
    pw = V2W.PackedExperts(p.w13, p.w2, 576)
    for tokens, s in ((4, 0.0), (300, 0.0), (1100, 1.2)):
        x, ids, w = _routing(tokens, 384, 6, tokens, s)
        ids[-1] = -1  # a masked (dp pad) row
        assert torch.equal(
            V2W.decode_moe(x, ids, w, pw, LIMIT), V2.decode_moe(x, ids, w, p, LIMIT)
        )


@pytest.mark.parametrize("s", [0.0, 1.2])
def test_v2w_2304_matches_reference(s):
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2w as V2W

    experts = _experts(48, 2304)
    packed = V2W.pack_experts(*experts)
    for tokens in (3, 200, 1500):
        x, ids, w = _routing(tokens, 48, 2, 7 + tokens, s)
        out = V2W.decode_moe(x, ids, w, packed, LIMIT)
        assert not torch.isnan(out).any()
        err = _rel(out, _reference(x, ids, w, experts))
        assert err < 5e-3, (tokens, s, err)


def test_v2w_2304_build_switches_bit_identical():
    """64-row passes and the grid-wide h quantization (2304 defaults) against 32-row passes, per-CTA quantization;
    repeated calls; pair rows (top_k 1, output in place) against token rows."""
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2w as V2W

    packed = V2W.pack_experts(*_experts(48, 2304))
    x, ids, w = _routing(
        3000, 48, 1, 11, 1.2
    )  # hot experts: hundreds of rows, several 64-row passes
    base = V2W.decode_moe(
        x, ids, w, packed, LIMIT, defines=("DSV4_PASS_ROWS=32", "DSV4_GLOBAL_QUANT=0")
    ).clone()
    for _ in range(3):
        assert torch.equal(V2W.decode_moe(x, ids, w, packed, LIMIT), base)
    o = torch.empty_like(x)
    assert torch.equal(V2W.decode_moe(x, ids, w, packed, LIMIT, output=o), base)
    # token rows with a local top-3 (two masked) give the same per-token sums up to the bf16 combine
    ids3 = torch.cat([ids, torch.full_like(ids, -1), torch.full_like(ids, -1)], 1)
    w3 = torch.cat([w, w, w], 1)
    assert _rel(V2W.decode_moe(x, ids3, w3, packed, LIMIT), base) < 1e-6


def test_v2w_2304_cuda_graph_replay():
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2w as V2W

    packed = V2W.pack_experts(*_experts(48, 2304))
    V2W.prepare(torch.cuda.current_device(), 2304)
    x, ids, w = _routing(512, 48, 1, 3)
    out = torch.empty_like(x)
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        V2W.decode_moe(x, ids, w, packed, LIMIT, output=out)
    torch.cuda.current_stream().wait_stream(st)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        V2W.decode_moe(x, ids, w, packed, LIMIT, output=out)
    for seed in (5, 6):
        x2, ids2, w2 = _routing(512, 48, 1, seed, 1.0)
        x.copy_(x2), ids.copy_(ids2), w.copy_(w2)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, V2W.decode_moe(x2, ids2, w2, packed, LIMIT))


def test_v2w_tp8_padded_slice_matches_288_reference():
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2w as V2W

    experts = _experts(16, 2304, seed=3)
    rank = 5
    packed = V2W.pack_tp_slice(*experts, rank=rank, tp=8, pad_to=320)
    sl = slice(288 * rank, 288 * (rank + 1))
    w1, s1, w3, s3, w2, s2 = experts
    ref_experts = (
        w1[:, sl],
        s1[:, sl],
        w3[:, sl],
        s3[:, sl],
        w2[:, :, 144 * rank : 144 * (rank + 1)],
        s2[:, :, 9 * rank : 9 * (rank + 1)],
    )
    x, ids, w = _routing(400, 16, 3, 9)
    err = _rel(
        V2W.decode_moe(x, ids, w, packed, LIMIT), _reference(x, ids, w, ref_experts)
    )
    assert err < 5e-3, err


def _ep_receive(tokens, num_local, top_k, seed, s=1.0, local_share=0.4):
    """A rank's receive side: token rows (e4m3 + row scale, as the dispatch sends them), their LOCAL top-k ids
    (-1 = expert of another rank) and weights; some rows have no local expert (pad rows)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x, ids, w = _routing(tokens, num_local, top_k, seed, s)
    keep = torch.rand(tokens, top_k, device="cuda", generator=g) < local_share
    ids = torch.where(keep, ids, torch.full_like(ids, -1))
    return x, ids, w


def test_v2w_gather_modes_bit_identical():
    """Token rows + local top-k (decode_moe_tokens) and token rows + a compact pair list (decode_moe_pairs, also at a
    fixed capacity with a device-side count): each slot's output equals the pair-row call, each token row equals the
    fp32 k-ascending sum of its slots; rows without a local slot are not written."""
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2w as V2W

    packed = V2W.pack_experts(*_experts(48, 2304))
    T, K = 700, 6
    x, ids, w = _ep_receive(T, 48, K, 21)
    x8 = torch.empty(T, HIDDEN, dtype=torch.uint8, device="cuda")
    xs = torch.empty(T, dtype=torch.float32, device="cuda")
    sentinel = torch.full((T, HIDDEN), 7.0, dtype=torch.bfloat16, device="cuda")
    out_t = sentinel.clone()
    V2W.decode_moe_tokens(
        x8, xs, ids, w, packed, LIMIT, out_t, x=x
    )  # also quantizes x into x8 / xs
    t2, k2 = torch.nonzero(ids >= 0, as_tuple=True)
    pid, pw = ids[t2, k2].unsqueeze(1).contiguous(), w[t2, k2].unsqueeze(1).contiguous()
    pair_rows = V2W.decode_moe(
        x[t2].contiguous(),
        pid,
        pw,
        packed,
        LIMIT,
        output=torch.empty(t2.numel(), HIDDEN, dtype=torch.bfloat16, device="cuda"),
    )
    # per pair: bit-identical to the pair-row call
    _, per_pair = V2W.decode_moe_pairs(
        x8, xs, t2.int(), pid.flatten(), pw.flatten(), packed, LIMIT
    )
    assert torch.equal(per_pair, pair_rows)
    # per token row: fp32 k-ascending sum of its pairs, one bf16 round
    Y = torch.zeros(T, K, HIDDEN, device="cuda")
    Y[t2, k2] = pair_rows.float()
    ref = torch.zeros(T, HIDDEN, device="cuda")
    for k in range(K):
        ref += Y[:, k]
    has = (ids >= 0).any(1)
    assert torch.equal(out_t[has], ref.to(torch.bfloat16)[has])
    assert torch.equal(out_t[~has], sentinel[~has])  # not written
    out_p = sentinel.clone()
    V2W.decode_moe_pairs(
        x8, xs, t2.int(), pid.flatten(), pw.flatten(), packed, LIMIT, out_tokens=out_p
    )
    assert torch.equal(out_p, out_t)
    # fixed capacity (graph shapes): unused entries at the end, device-side count
    cap = T * K
    tok_c = torch.full((cap,), T, dtype=torch.int32, device="cuda")
    eid_c = torch.full((cap,), -1, dtype=torch.int32, device="cuda")
    w_c = torch.zeros(cap, device="cuda")
    tok_c[: t2.numel()], eid_c[: t2.numel()], w_c[: t2.numel()] = (
        t2.int(),
        pid.flatten(),
        pw.flatten(),
    )
    out_c = sentinel.clone()
    V2W.decode_moe_pairs(
        x8,
        xs,
        tok_c,
        eid_c,
        w_c,
        packed,
        LIMIT,
        out_tokens=out_c,
        num_valid=torch.tensor([t2.numel()], dtype=torch.int32, device="cuda"),
    )
    assert torch.equal(out_c, out_t)


def test_v2w_gather_pairs_cuda_graph():
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2w as V2W

    packed = V2W.pack_experts(*_experts(48, 2304))
    V2W.prepare(torch.cuda.current_device(), 2304)
    T, K = 512, 6
    cap = T * K
    x8 = torch.empty(T, HIDDEN, dtype=torch.uint8, device="cuda")
    xs = torch.empty(T, dtype=torch.float32, device="cuda")
    tok_c = torch.full((cap,), T, dtype=torch.int32, device="cuda")
    eid_c = torch.full((cap,), -1, dtype=torch.int32, device="cuda")
    w_c = torch.zeros(cap, device="cuda")
    nv = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.zeros(T, HIDDEN, dtype=torch.bfloat16, device="cuda")

    def load(seed):
        x, ids, w = _ep_receive(T, 48, K, seed)
        V2W.decode_moe_tokens(
            x8, xs, torch.full_like(ids, -1), w, packed, LIMIT, out, x=x
        )  # quantize only
        t2, k2 = torch.nonzero(ids >= 0, as_tuple=True)
        tok_c.fill_(T), eid_c.fill_(-1), w_c.zero_()
        tok_c[: t2.numel()], eid_c[: t2.numel()], w_c[: t2.numel()] = (
            t2.int(),
            ids[t2, k2],
            w[t2, k2],
        )
        nv.fill_(t2.numel())
        ref = torch.zeros_like(out)
        V2W.decode_moe_tokens(x8, xs, ids, w, packed, LIMIT, ref)
        return ref, (ids >= 0).any(1)

    load(1)
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        V2W.decode_moe_pairs(
            x8, xs, tok_c, eid_c, w_c, packed, LIMIT, out_tokens=out, num_valid=nv
        )
    torch.cuda.current_stream().wait_stream(st)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        V2W.decode_moe_pairs(
            x8, xs, tok_c, eid_c, w_c, packed, LIMIT, out_tokens=out, num_valid=nv
        )
    for seed in (2, 3):
        ref, has = load(seed)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out[has], ref[has])


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
