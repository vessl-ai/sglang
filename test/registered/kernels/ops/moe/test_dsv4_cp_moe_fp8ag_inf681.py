"""INF-681 k-pfp8: the prefill-CP MoE routes and quantizes each CP shard before the
all-gather (SGLANG_OPT_DSV4_CP_MOE_LOCAL_ROUTER / _FP8_ALLGATHER,
layers/quantization/dsv4_cp_moe_fp8ag.py). Same bits as routing and quantizing the
gathered rows needs every per-row function to give the same bits for a row whatever the
row count of the call; this checks that on the GPU, and the whole path with the
all-gather simulated by a rank-major concatenation (NCCL's all-gather copies bytes)."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 only",
)

CP = 4
# Rows per CP4 shard at the verdict load: 1 request (6,912 rows), 3 and 10 requests
# (the full 65,280-row step), an odd count, and the low end that k-pfp8 still takes
# (SGLANG_OPT_DSV4_CP_MOE_LOCAL_MIN_ROWS = 1024; on H200 some shards of <= 792 rows give
# the router GEMM other bits than the gathered call, k-pfp8 sweep 2026-11-01).
SHARD_ROWS = [1728, 5184, 16320, 4099, 1024, 1025, 1100, 2047, 3001]


def _gathered(shards):
    return torch.cat(shards)  # rank-major, as attn_cp_interleave_gather gives them


def _router(logits, bias, num_token_non_padded):
    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

    # DeepSeek-V4.1: 384 experts, top 6, sqrtsoftplus via log1p, renormalized, scaled.
    return moe_fused_gate(
        logits,
        bias,
        6,
        scoring_func="sqrtsoftplus",
        renormalize=True,
        routed_scaling_factor=1.5,
        apply_routed_scaling_factor_on_output=True,
        num_token_non_padded=num_token_non_padded,
        sqrtsoftplus_log1p=True,
    )


@pytest.mark.parametrize("rows", SHARD_ROWS)
def test_router_gemm_and_topk_rows_do_not_depend_on_the_row_count(rows):
    """The gate GEMM (cuBLAS bf16 x bf16 -> fp32, MoEGate's branch above its tiny-GEMM
    limit) and the fused top-k at the DeepSeek-V4.1 shapes."""
    from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32

    g = torch.Generator(device="cuda").manual_seed(rows)
    hidden, experts = 5120, 384
    shards = [
        torch.randn(rows, hidden, device="cuda", generator=g).bfloat16()
        for _ in range(CP)
    ]
    weight = (
        torch.randn(experts, hidden, device="cuda", generator=g) * 0.02
    ).bfloat16()
    bias = torch.randn(experts, device="cuda", generator=g) * 0.1
    full = linear_bf16_fp32(_gathered(shards), weight)
    total = torch.tensor(rows * CP, device="cuda", dtype=torch.int32)
    w_full, ids_full = _router(full, bias, total)
    for r, shard in enumerate(shards):
        part = linear_bf16_fp32(shard, weight)
        assert torch.equal(part, full[r * rows : (r + 1) * rows]), f"logits rank {r}"
        local = torch.tensor(rows, device="cuda", dtype=torch.int32)
        w, ids = _router(part, bias, local)
        assert torch.equal(ids, ids_full[r * rows : (r + 1) * rows])
        assert torch.equal(w, w_full[r * rows : (r + 1) * rows])


@pytest.mark.parametrize("rows", SHARD_ROWS)
def test_row_quant_rows_do_not_depend_on_the_row_count(rows):
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import _row_quant

    g = torch.Generator(device="cuda").manual_seed(rows + 1)
    shards = [
        (torch.randn(rows, 5120, device="cuda", generator=g) * 3).bfloat16()
        for _ in range(CP)
    ]
    q_full, s_full = _row_quant(_gathered(shards))
    q = torch.cat([_row_quant(x)[0] for x in shards])
    s = torch.cat([_row_quant(x)[1] for x in shards])
    assert torch.equal(q.view(torch.uint8), q_full.view(torch.uint8))
    assert torch.equal(s, s_full)


@pytest.mark.parametrize("with_scale", [True, False])
@pytest.mark.parametrize("rows", [0, 1, 255, 257, 16320])
def test_pack_unpack_roundtrip(with_scale, rows):
    from sglang.srt.layers.quantization.dsv4_cp_moe_fp8ag import pack_meta, unpack_meta

    g = torch.Generator(device="cuda").manual_seed(7)
    ids = torch.randint(
        0, 384, (rows, 6), device="cuda", generator=g, dtype=torch.int32
    )
    w = torch.rand(rows, 6, device="cuda", generator=g)
    w[:1] = torch.tensor([float("nan"), -0.0, 1e-38, 3.0, -2.5, 0.0])[: w[:1].shape[1]]
    s = torch.exp2(torch.randint(-8, 8, (rows,), device="cuda").float())
    meta = pack_meta(s if with_scale else None, ids, w)
    assert meta.shape == (rows, int(with_scale) + 12) and meta.dtype == torch.int32
    s2, ids2, w2 = unpack_meta(meta, 6, with_scale)
    assert torch.equal(ids2, ids)
    assert torch.equal(w2.view(torch.int32), w.view(torch.int32))  # bits, NaN included
    if with_scale:
        assert torch.equal(s2, s)
    else:
        assert s2 is None


# --- the whole path on small expert shapes, all-gather simulated -----------------------

E, H, I, TOPK = 16, 1024, 192, 6
LIMIT = 10.0


def _weights(seed, spread=4):
    g = torch.Generator().manual_seed(seed)

    def one(n, k):
        w = torch.randint(-128, 128, (E, n, k // 2), generator=g, dtype=torch.int8)
        s = torch.randint(
            122 - spread, 123, (E, n, k // 32), generator=g, dtype=torch.uint8
        )
        return w.cuda(), s.view(torch.float8_e8m0fnu).cuda()

    return (*one(2 * I, H), *one(H, I))


def _experts(w, **kw):
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import (
        Mxfp4W4A8PrefillExperts,
    )

    return Mxfp4W4A8PrefillExperts(*w, LIMIT, prefetch=False, sorted_sum=True, **kw)


# The i6xs prefill switches of the W4A8 experts.
I6XS = dict(fused_swiglu=True, counting_route=True, w2_two_cta_min_rows=16384)


@pytest.mark.parametrize("rows", [3, 1728, 4099])
@pytest.mark.parametrize("fp8", [True, False])
def test_shard_routed_path_equals_gathered_path(rows, fp8):
    pytest.importorskip("triton_kernels")
    from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32
    from sglang.srt.layers.quantization.dsv4_cp_moe_fp8ag import pack_meta, unpack_meta
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import (
        W4A8RowQuant,
        _row_quant,
    )

    g = torch.Generator(device="cuda").manual_seed(rows)
    shards = [
        torch.randn(rows, H, device="cuda", generator=g).bfloat16() for _ in range(CP)
    ]
    gate = (torch.randn(E, H, device="cuda", generator=g) * 0.05).bfloat16()
    bias = torch.randn(E, device="cuda", generator=g) * 0.1
    total = rows * CP

    def route(x):
        n = torch.tensor(x.shape[0], device="cuda", dtype=torch.int32)
        from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

        return moe_fused_gate(
            linear_bf16_fp32(x, gate),
            bias,
            TOPK,
            scoring_func="sqrtsoftplus",
            renormalize=True,
            routed_scaling_factor=1.5,
            apply_routed_scaling_factor_on_output=True,
            num_token_non_padded=n,
            sqrtsoftplus_log1p=True,
        )

    experts = _experts(_weights(5), **I6XS)
    # Reference: the BF16 rows of all ranks, routed and quantized together.
    x_all = _gathered(shards)
    w_ref, ids_ref = route(x_all)
    out_ref = experts(x_all, ids_ref, w_ref)

    # k-pfp8: each rank routes (and quantizes) its shard and packs the results.
    metas, qs = [], []
    for x in shards:
        w, ids = route(x)
        q, s = _row_quant(x) if fp8 else (None, None)
        metas.append(pack_meta(s, ids, w))
        qs.append(q)
    s_all, ids_all, w_all = unpack_meta(_gathered(metas), TOPK, fp8)
    assert torch.equal(ids_all, ids_ref) and torch.equal(w_all, w_ref)
    if fp8:
        carrier = torch.empty(1, H, device="cuda", dtype=torch.bfloat16).expand(
            total, H
        )
        row_quant = W4A8RowQuant(_gathered(qs), s_all)
        out = experts(carrier, ids_all, w_all, row_quant=row_quant)
    else:
        out = experts(x_all, ids_all, w_all)
    assert out.shape == out_ref.shape and out.is_contiguous()
    assert torch.equal(out, out_ref)


def test_row_quant_needs_the_fp8_copy_path():
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import (
        W4A8RowQuant,
        _row_quant,
    )

    experts = _experts(_weights(6), small_step_rows=64)
    x = torch.randn(32, H, device="cuda").bfloat16()
    ids = torch.randint(0, E, (32, TOPK), device="cuda", dtype=torch.int32)
    w = torch.rand(32, TOPK, device="cuda")
    with pytest.raises(RuntimeError, match="FP8-copy path"):
        experts(x, ids, w, row_quant=W4A8RowQuant(*_row_quant(x)))


def test_plan_is_off_by_default(monkeypatch):
    from sglang.srt.layers.quantization import dsv4_cp_moe_fp8ag as m

    monkeypatch.delenv("SGLANG_OPT_DSV4_CP_MOE_LOCAL_ROUTER", raising=False)
    monkeypatch.delenv("SGLANG_OPT_DSV4_CP_MOE_FP8_ALLGATHER", raising=False)
    x = torch.empty(4096, 5120, device="cuda", dtype=torch.bfloat16)
    assert m.plan(object(), x, shared_local=True) is None


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
