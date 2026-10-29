"""INF-681 k-qcp research kernels (kernels/ops/layernorm/mhc_mega_boundary_sm90.py), bit for bit
against the path they would replace:

* mhc_mega_boundary_sm90 == mhc_post_combine_norm_prefill (R', normed x) followed by the Triton
  bf16x3 stats + sinkhorn on R' (partials and the next triplet), at CP4 rank row counts from 4096
  (16,302 = the 65,208-token step); at 1,728 rows (a 1-request step) R' and x are identical and the
  partials equal the Triton kernel's BLOCK_M 128 configuration (the path itself runs BLOCK_M 64 there);
* the mega kernel replays bit-identically from a CUDA graph (no host sync, fixed per-shape launch);
* mhc_stats_bf16x3_sm90 == the Triton stats kernel's BLOCK_M 128 configuration at any row count.
Neither kernel is used by the model: both are slower than today's path on H200 (see their headers).
"""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 wgmma",
)

H, MULT = 5120, 4


def _hc(seed):
    from sglang.kernels.ops.layernorm import mhc as K

    g = torch.Generator(device="cuda").manual_seed(seed)
    fn = torch.randn(24, MULT * H, device="cuda", generator=g) * 0.02
    scale = torch.rand(3, device="cuda", generator=g) + 0.5
    base = torch.randn(24, device="cuda", generator=g) * 0.1
    return K.split_bf16_hc_weight(fn.contiguous()), scale, base


def _inputs(m, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    residual = torch.randn(m, MULT, H, device="cuda", generator=g).bfloat16()
    residual[: m // 5] *= 1e-3
    y = (torch.randn(m, H, device="cuda", generator=g) * 0.5).bfloat16()
    weight = (1 + 0.1 * torch.randn(H, device="cuda", generator=g)).bfloat16()
    return residual, y, weight


def _triton_partials(x, parts, block_m):
    import triton

    from sglang.kernels.ops.layernorm import mhc as K

    m = x.shape[0]
    pm = torch.empty((16, m, 24), device="cuda")
    sq = torch.empty((16, m), device="cuda")
    K._hc_mix_stats_bf16x3_kernel[(triton.cdiv(m, block_m), 16)](
        x, *parts, pm, sq, m, K=MULT * H, K_PER_SLICE=MULT * H // 16, MIX_COLS=24, MIX_PAD=32,
        BLOCK_K=64, BLOCK_M=block_m, num_warps=4, num_stages=3,
    )
    return pm, sq


@pytest.mark.parametrize("m", [4096, 16302, 1728, 65, 1])
def test_mega_boundary_matches_fused_post_and_triton_stats(m):
    from sglang.kernels.ops.layernorm import mhc as K
    from sglang.kernels.ops.layernorm.mhc_mega_boundary_sm90 import (
        SLICES,
        _mhc_mega_boundary_sm90,
        mhc_mega_boundary_sm90,
    )
    from sglang.kernels.ops.layernorm.mhc_post_combine_norm_prefill import (
        mhc_post_combine_norm_prefill,
    )

    this_parts, this_scale, this_base = _hc(1)
    next_parts, next_scale, next_base = _hc(2)
    residual, y, weight = _inputs(m, m)
    if m >= 32:
        pre, post, comb = K.hc_mix_stats_sinkhorn_bf16x3(
            residual.flatten(1), this_parts, this_scale, this_base, 20, 1e-20, 1e-6
        )
    else:
        g = torch.Generator(device="cuda").manual_seed(3)
        pre = torch.rand(m, MULT, device="cuda", generator=g)
        post = torch.rand(m, MULT, device="cuda", generator=g) * 2
        comb = torch.softmax(torch.randn(m, MULT, MULT, device="cuda", generator=g), -1)
    ref_r, ref_x = mhc_post_combine_norm_prefill(y, residual, post, comb, pre, weight, 1e-20)
    r, x, triplet = mhc_mega_boundary_sm90(
        y, residual, pre, post, comb, next_parts, next_scale, next_base, weight, 1e-20, 1e-6, 1e-20, 20
    )
    assert torch.equal(r, ref_r) and torch.equal(x, ref_x)
    pm = torch.empty((SLICES, m, 24), device="cuda")
    sq = torch.empty((SLICES, m), device="cuda")
    _mhc_mega_boundary_sm90(
        y, residual, post, comb, pre, *next_parts, weight, torch.empty_like(r), torch.empty_like(x), pm, sq, 1e-20
    )
    ref_pm, ref_sq = _triton_partials(ref_r.flatten(1), next_parts, 128)
    assert torch.equal(pm, ref_pm) and torch.equal(sq, ref_sq)
    if m >= 4096:  # the path's own configuration: the next triplet is identical too
        ref = K.hc_mix_stats_sinkhorn_bf16x3(
            ref_r.flatten(1), next_parts, next_scale, next_base, 20, 1e-20, 1e-6
        )
        assert all(torch.equal(a, b) for a, b in zip(triplet, ref))


def test_mega_boundary_cuda_graph_replay():
    from sglang.kernels.ops.layernorm.mhc_mega_boundary_sm90 import mhc_mega_boundary_sm90

    m = 4096
    next_parts, next_scale, next_base = _hc(4)
    residual, y, weight = _inputs(m, 5)
    pre = torch.rand(m, MULT, device="cuda")
    post = torch.rand(m, MULT, device="cuda") * 2
    comb = torch.softmax(torch.randn(m, MULT, MULT, device="cuda"), -1)

    def run():
        return mhc_mega_boundary_sm90(
            y, residual, pre, post, comb, next_parts, next_scale, next_base, weight, 1e-20, 1e-6, 1e-20, 20
        )

    eager = run()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()  # warm up on the capture stream
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured[0], eager[0]) and torch.equal(captured[1], eager[1])
    assert all(torch.equal(a, b) for a, b in zip(captured[2], eager[2]))


@pytest.mark.parametrize("m", [1, 33, 128, 1728, 4096, 16302])
def test_cuda_stats_match_triton_block_m_128(m):
    from sglang.kernels.ops.layernorm.mhc_mega_boundary_sm90 import mhc_stats_bf16x3_sm90

    parts, _, _ = _hc(6)
    g = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn(m, MULT * H, device="cuda", generator=g).bfloat16()
    x[: m // 3] *= 0.01
    pm, sq = mhc_stats_bf16x3_sm90(x, parts)
    ref_pm, ref_sq = _triton_partials(x, parts, 128)
    assert torch.equal(pm, ref_pm) and torch.equal(sq, ref_sq)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
