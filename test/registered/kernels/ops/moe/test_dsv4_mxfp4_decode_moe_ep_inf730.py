"""INF-730 SGLANG_OPT_DSV4_SM90_DECODE_MOE_W4A8_EP: the v2 W4A8 decode MoE on expert-parallel layers, each local
whole expert (intermediate 2304) run as four virtual 576-channel experts (dsv4_mxfp4_decode_moe_ep).

Virtual expert (e, s) is the slice TP rank s of a TP4 layer holds of expert e, so every virtual slot's output row
must equal, bit for bit, the row the v2 kernel computes for that slot on the TP4 slice s; the EP combine then sums
a token's (k, s) rows in fp32. Checked on the target (48 local experts, top 6) and DSpark draft (16 local experts,
top 3) shapes with rows routed to other ranks (-1), masked pad rows, hot experts, repeated calls (the slot order
inside an expert changes from call to call), CUDA graph replay with fresh inputs and token chunking. The weight
re-layout is also checked on CPU against a float64 reference of the whole expert."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")

HIDDEN, INTER, FULL = 5120, 576, 2304
SPLIT = FULL // INTER
SWIGLU_LIMIT = 10.0
EP = 8


def _sm90() -> bool:
    if not torch.cuda.is_available():
        return False
    from sglang.srt.utils.common import is_sm90_supported

    return is_sm90_supported()


needs_sm90 = pytest.mark.skipif(not _sm90(), reason="SM90 wgmma W4A8 decode MoE")


def _weights(num_experts, inter=FULL, seed=0, hidden=HIDDEN):
    g = torch.Generator().manual_seed(seed)

    def nibbles(*shape):
        return torch.randint(-128, 128, shape, generator=g, dtype=torch.int8)

    def scales(*shape):
        return torch.randint(119, 123, shape, generator=g, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )

    return (
        nibbles(num_experts, 2 * inter, hidden // 2),
        scales(num_experts, 2 * inter, hidden // 32),
        nibbles(num_experts, hidden, inter // 2),
        scales(num_experts, hidden, inter // 32),
    )


# ------------------------------------------------------------------------------------------------- CPU


_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float64,
)


def _dequant(w, s):
    """[.., N, K/2] int8 (low nibble = even k) + [.., N, K/32] e8m0 -> [.., N, K] float64."""
    b = w.view(torch.uint8).to(torch.int64)
    vals = torch.stack([_E2M1[b & 0xF], _E2M1[b >> 4]], dim=-1).flatten(-2)
    exp = s.view(torch.uint8).to(torch.float64) - 127.0
    return vals * torch.repeat_interleave(torch.exp2(exp), 32, dim=-1)


def _expert_ref(x, w13, s13, w2, s2, limit=SWIGLU_LIMIT):
    """float64 expert: down(clamp-SwiGLU(gate, up)), [w1; w3] layout."""
    gu = x @ _dequant(w13, s13).T
    inter = gu.shape[-1] // 2
    g = gu[..., :inter].clamp(max=limit)
    u = gu[..., inter:].clamp(-limit, limit)
    return (g * torch.sigmoid(g) * u) @ _dequant(w2, s2).T


def test_virtual_experts_slice_whole_experts_cpu():
    from sglang.kernels.ops.moe.dsv4_mxfp4_decode_moe_ep import virtual_expert_weights

    hidden = 256  # small hidden for a fast float64 reference; the slicing is along the intermediate dim
    w13, s13, w2, s2 = _weights(3, hidden=hidden, seed=1)
    v13, vs13, v2w, vs2 = virtual_expert_weights(
        w13.view(torch.uint8), s13.view(torch.uint8), w2.view(torch.uint8), s2.view(torch.uint8)
    )
    assert v13.shape == (3 * SPLIT, 2 * INTER, hidden // 2)
    assert v2w.shape == (3 * SPLIT, hidden, INTER // 2)
    for e in range(3):
        for s in range(SPLIT):
            v = e * SPLIT + s
            gate = slice(s * INTER, (s + 1) * INTER)
            up = slice(FULL + s * INTER, FULL + (s + 1) * INTER)
            assert torch.equal(v13[v, :INTER], w13[e, gate].view(torch.uint8))
            assert torch.equal(v13[v, INTER:], w13[e, up].view(torch.uint8))
            assert torch.equal(vs13[v, :INTER], s13[e, gate].view(torch.uint8))
            assert torch.equal(vs13[v, INTER:], s13[e, up].view(torch.uint8))
            cols = slice(s * INTER // 2, (s + 1) * INTER // 2)
            assert torch.equal(v2w[v], w2[e, :, cols].view(torch.uint8))
            scols = slice(s * INTER // 32, (s + 1) * INTER // 32)
            assert torch.equal(vs2[v], s2[e, :, scols].view(torch.uint8))
    # The whole expert is the sum of its virtual experts (float64, no quantization).
    x = torch.randn(5, hidden, dtype=torch.float64) * 0.05
    for e in range(3):
        whole = _expert_ref(x, w13[e], s13[e], w2[e], s2[e])
        parts = sum(
            _expert_ref(x, v13[e * SPLIT + s], vs13[e * SPLIT + s], v2w[e * SPLIT + s], vs2[e * SPLIT + s])
            for s in range(SPLIT)
        )
        torch.testing.assert_close(parts, whole, rtol=1e-9, atol=1e-9)


def test_split_and_support_rules():
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ep as ek

    assert ek.split_of(2304) == 4 and ek.split_of(1152) == 2 and ek.split_of(576) == 1
    assert ek.split_of(288) == 0 and ek.split_of(0) == 0
    assert ek.supports_ep(5120, 2304, 48) and ek.supports_ep(5120, 2304, 16)
    assert ek.supports_ep(5120, 2304, 128) and not ek.supports_ep(5120, 2304, 129)
    assert not ek.supports_ep(4096, 2304, 48) and not ek.supports_ep(5120, 288, 48)
    assert ek.supports_ep(5120, 2304, 48, "v2w") and not ek.supports_ep(5120, 1152, 48, "v2w")


# ------------------------------------------------------------------------------------------------- GPU


def _global_routing(tokens, num_global, top_k, seed, skew=0.0, rows_per_req=4):
    """Global top-k ids: rows of one request share most experts (DSpark verify rows); ``skew`` > 0 gives a
    zipf popularity (hot experts with several 32-row passes)."""
    g = torch.Generator().manual_seed(seed)
    reqs = (tokens + rows_per_req - 1) // rows_per_req
    pop = torch.zeros(num_global)
    if skew > 0:
        pop = torch.log((1.0 / torch.arange(1, num_global + 1, dtype=torch.float64)) ** skew).float()
        pop = pop[torch.randperm(num_global, generator=g)]
    req = torch.randn(reqs, num_global, generator=g).repeat_interleave(rows_per_req, 0)[:tokens]
    logits = pop + req + 0.5 * torch.randn(tokens, num_global, generator=g)
    w, ids = torch.topk(torch.softmax(logits, -1), top_k, dim=-1)
    return ids.int(), (w / w.sum(-1, keepdim=True)).float()


def _local(ids_global, rank, num_local):
    lo = rank * num_local
    local = ids_global - lo
    return torch.where((ids_global >= lo) & (ids_global < lo + num_local), local, -1).int()


def _slice_y(x, ids_local, wts, packed_slice):
    """The v2 kernel's per-slot output rows y [T*K, 5120] (original-slot order) for one TP slice."""
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2 as v2
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

    T, K = ids_local.shape
    E = packed_slice.w13.shape[0]
    counts, sorted_slots, sorted_tokens, sorted_weights = ws._sort_slots(ids_local, wts, E)
    x8, xs = ws._row_quant(x.contiguous())
    slots = T * K
    work = torch.zeros(v2.WORKSPACE_BASE + 2 * (E + slots // 32 + 1), dtype=torch.int32, device="cuda")
    h = x.new_empty(slots, INTER)
    h8 = torch.empty(slots, INTER, dtype=torch.uint8, device="cuda")
    hs = torch.empty(slots, dtype=torch.float32, device="cuda")
    y = torch.zeros(slots, HIDDEN, dtype=torch.bfloat16, device="cuda")
    v2._module().experts(
        x8.view(torch.uint8), xs, packed_slice.w13, packed_slice.w2, counts, sorted_tokens,
        sorted_slots, sorted_weights, h, h8, hs, y, work, SWIGLU_LIMIT,
    )
    return y


def _reference(x, ids_local, wts, slices):
    """EP output from the TP slices' rows: per token, sum over k then s of y_s[t, k] in fp32 (skipping slots of
    other ranks), rounded once to bf16 -- the order the EP combine uses."""
    T, K = ids_local.shape
    ys = [_slice_y(x, ids_local, wts, p).view(T, K, HIDDEN).float() for p in slices]
    use = (ids_local >= 0).cuda()
    acc = torch.zeros(T, HIDDEN, dtype=torch.float32, device="cuda")
    for k in range(K):
        for s in range(SPLIT):
            acc = torch.where(use[:, k : k + 1], acc + ys[s][:, k], acc)
    return acc.to(torch.bfloat16)


def _packed_ep_and_slices(num_local, seed=0):
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ep as ek
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

    # byte views: torch has no cat for e8m0
    w13, s13, w2, s2 = (t.view(torch.uint8).cuda() for t in _weights(num_local, seed=seed))
    packed, split = ek.pack_experts_ep(w13, s13, w2, s2)
    assert split == SPLIT and packed.w13.shape[0] == num_local * SPLIT
    slices = []
    for s in range(SPLIT):
        g = slice(s * INTER, (s + 1) * INTER)
        u = slice(FULL + s * INTER, FULL + (s + 1) * INTER)
        slices.append(
            ws.pack_experts(
                torch.cat([w13[:, g], w13[:, u]], 1).contiguous(),
                torch.cat([s13[:, g], s13[:, u]], 1).contiguous(),
                w2[:, :, s * INTER // 2 : (s + 1) * INTER // 2].contiguous(),
                s2[:, :, s * INTER // 32 : (s + 1) * INTER // 32].contiguous(),
            )
        )
    return packed, slices


# (local experts, top_k): target (384 / EP8) and DSpark draft (128 / EP8)
SHAPES = {"target": (48, 6), "draft": (16, 3)}


@needs_sm90
@pytest.mark.parametrize("shape", list(SHAPES))
def test_ep_bit_identical_to_tp_slices(shape):
    from sglang.kernels.ops.moe.dsv4_mxfp4_decode_moe_ep import decode_moe_ep

    num_local, top_k = SHAPES[shape]
    packed, slices = _packed_ep_and_slices(num_local)
    rank = 3
    # 1 token up to DP8 x 64 requests x 4 rows; skewed routing gives hot experts with several passes
    for tokens, skew in ((1, 0.0), (7, 0.0), (64, 0.0), (520, 0.0), (520, 1.2), (2048, 1.0)):
        ids_g, wts = _global_routing(tokens, num_local * EP, top_k, seed=tokens)
        if skew:
            ids_g, wts = _global_routing(tokens, num_local * EP, top_k, seed=tokens, skew=skew)
        ids = _local(ids_g, rank, num_local).cuda()
        if tokens > 8:
            ids[-3:] = -1  # dp-attention pad rows (SGLANG_OPT_MASK_DP_PAD_MOE)
        wts = wts.cuda()
        x = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
        ref = _reference(x, ids, wts, slices)
        for _ in range(3):  # the slot order inside an expert changes from call to call
            out = decode_moe_ep(x, ids, wts, packed, SWIGLU_LIMIT, SPLIT)
            assert torch.equal(out, ref), (shape, tokens, skew)


@needs_sm90
def test_ep_no_local_slot_gives_zero():
    from sglang.kernels.ops.moe.dsv4_mxfp4_decode_moe_ep import decode_moe_ep

    packed, _ = _packed_ep_and_slices(48)
    x = torch.randn(33, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ids = torch.full((33, 6), -1, dtype=torch.int32, device="cuda")
    out = torch.full((33, HIDDEN), 7.0, dtype=torch.bfloat16, device="cuda")
    decode_moe_ep(x, ids, torch.ones(33, 6, device="cuda"), packed, SWIGLU_LIMIT, SPLIT, output=out)
    assert torch.count_nonzero(out) == 0


@needs_sm90
def test_ep_row_quant_input_matches():
    """The pre-quantized input path (a dispatch that sends e4m3 rows + row scales) gives the same bits."""
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws
    from sglang.kernels.ops.moe.dsv4_mxfp4_decode_moe_ep import decode_moe_ep

    packed, _ = _packed_ep_and_slices(48)
    ids_g, wts = _global_routing(300, 384, 6, seed=5)
    ids, wts = _local(ids_g, 0, 48).cuda(), wts.cuda()
    x = torch.randn(300, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ref = decode_moe_ep(x, ids, wts, packed, SWIGLU_LIMIT, SPLIT)
    x8, xs = ws._row_quant(x)
    out = decode_moe_ep(None, ids, wts, packed, SWIGLU_LIMIT, SPLIT, row_quant=(x8, xs))
    assert torch.equal(out, ref)


@needs_sm90
def test_ep_cuda_graph_replay():
    from sglang.kernels.ops.moe.dsv4_mxfp4_decode_moe_ep import decode_moe_ep, prepare_ep

    packed, _ = _packed_ep_and_slices(48)
    prepare_ep(torch.cuda.current_device(), 48 * SPLIT, SPLIT, 6)
    tokens = 1024
    x = torch.zeros(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ids = torch.full((tokens, 6), -1, dtype=torch.int32, device="cuda")
    wts = torch.zeros(tokens, 6, device="cuda")
    out = torch.empty(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    decode_moe_ep(x, ids, wts, packed, SWIGLU_LIMIT, SPLIT, output=out)  # warm the allocator
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_moe_ep(x, ids, wts, packed, SWIGLU_LIMIT, SPLIT, output=out)
    for seed, skew in ((1, 0.0), (2, 1.2), (3, 0.6)):
        ids_g, w = _global_routing(tokens, 384, 6, seed=seed, skew=skew)
        ids.copy_(_local(ids_g, 5, 48))
        wts.copy_(w)
        x.copy_(torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, decode_moe_ep(x, ids, wts, packed, SWIGLU_LIMIT, SPLIT)), seed


@needs_sm90
def test_ep_token_chunks(monkeypatch):
    """Calls above SGLANG_OPT_DSV4_EP_DECODE_MOE_MAX_VSLOTS run in token chunks with the same output."""
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ep as ek

    packed, _ = _packed_ep_and_slices(48)
    ids_g, wts = _global_routing(1000, 384, 6, seed=9)
    ids, wts = _local(ids_g, 7, 48).cuda(), wts.cuda()
    x = torch.randn(1000, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ref = ek.decode_moe_ep(x, ids, wts, packed, SWIGLU_LIMIT, SPLIT)
    monkeypatch.setattr(ek, "max_virtual_slots", lambda: 7 * 6 * SPLIT * 16)  # 112 tokens per call
    assert torch.equal(ek.decode_moe_ep(x, ids, wts, packed, SWIGLU_LIMIT, SPLIT), ref)


# ------------------------------------------------------------------------------------- v2w (whole experts)


def _packed_v2w(num_local, seed=0):
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ep as ek
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2w as v2w

    w13, s13, w2, s2 = (t.view(torch.uint8).cuda() for t in _weights(num_local, seed=seed))
    packed, split = ek.pack_experts_ep(w13, s13, w2, s2, mode="v2w")
    assert split == 1 and packed.inter == FULL and packed.w13.shape[0] == num_local
    whole = v2w.pack_w13_w2(w13, s13, w2, s2)  # a-moe's own packing of the same experts, in one piece
    assert torch.equal(packed.w13, whole.w13) and torch.equal(packed.w2, whole.w2)
    return packed


@needs_sm90
@pytest.mark.parametrize("shape", list(SHAPES))
def test_ep_v2w_bit_identical_to_v2w_wrapper(shape):
    """SGLANG_OPT_DSV4_EP_DECODE_MOE_KERNEL=v2w: this module's prologue + combine around the v2w experts kernel give
    the bits of a-moe's own wrapper (its prep + the v2 top-k combine), on token rows with local top-k ids."""
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2w as v2w
    from sglang.kernels.ops.moe.dsv4_mxfp4_decode_moe_ep import decode_moe_ep

    num_local, top_k = SHAPES[shape]
    packed = _packed_v2w(num_local)
    for tokens, skew in ((1, 0.0), (7, 0.0), (520, 0.0), (520, 1.2), (2048, 1.0), (4096, 0.6)):
        ids_g, wts = _global_routing(tokens, num_local * EP, top_k, seed=tokens, skew=skew)
        ids = _local(ids_g, 2, num_local).cuda()
        if tokens > 8:
            ids[-3:] = -1
        wts = wts.cuda()
        x = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
        ref = v2w.decode_moe(x, ids, wts, packed, SWIGLU_LIMIT)
        for _ in range(3):
            assert torch.equal(decode_moe_ep(x, ids, wts, packed, SWIGLU_LIMIT, 1), ref), (shape, tokens, skew)
        from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

        out = decode_moe_ep(None, ids, wts, packed, SWIGLU_LIMIT, 1, row_quant=ws._row_quant(x))
        assert torch.equal(out, ref), (shape, tokens, "row_quant")


@needs_sm90
def test_ep_v2w_cuda_graph_replay():
    from sglang.kernels.ops.moe.dsv4_mxfp4_decode_moe_ep import decode_moe_ep, prepare_ep

    packed = _packed_v2w(48)
    prepare_ep(torch.cuda.current_device(), 48, 1, 6, FULL)
    tokens = 2048
    x = torch.zeros(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ids = torch.full((tokens, 6), -1, dtype=torch.int32, device="cuda")
    wts = torch.zeros(tokens, 6, device="cuda")
    out = torch.empty(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    decode_moe_ep(x, ids, wts, packed, SWIGLU_LIMIT, 1, output=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode_moe_ep(x, ids, wts, packed, SWIGLU_LIMIT, 1, output=out)
    for seed, skew in ((1, 0.0), (2, 1.2)):
        ids_g, w = _global_routing(tokens, 384, 6, seed=seed, skew=skew)
        ids.copy_(_local(ids_g, 6, 48))
        wts.copy_(w)
        x.copy_(torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, decode_moe_ep(x, ids, wts, packed, SWIGLU_LIMIT, 1)), seed


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
