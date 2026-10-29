"""INF-681 k-qcp research kernels (not used by the model; see the .cuh headers):

* `mhc_mega_boundary_sm90`: this hyper-connection's post, the next one's combine + RMSNorm and the
  next one's mix stats in one pass over the residual. Bit-identical to mhc_post_combine_norm_prefill
  followed by hc_mix_stats_sinkhorn_bf16x3 (from 4096 rows, where the Triton stats run BLOCK_M 128),
  but slower than that pair on H200 (0.75 vs 0.66 ms at 16,302 rows).
* `mhc_stats_bf16x3_sm90`: the Triton _hc_mix_stats_bf16x3_kernel's partials in CUDA, bit-identical
  to its BLOCK_M 128 configuration at any row count (0.29 vs 0.22 ms at 16,302 rows).
"""

from typing import Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

HIDDEN, MULT, MIX, SLICES = 5120, 4, 24, 16


@cache_once
def _stats_module():
    return load_jit(
        "mhc_stats_bf16x3_sm90",
        cuda_files=["deepseek_v4/mhc_stats_bf16x3_sm90.cuh"],
        cuda_wrappers=[("run", "sglang::mhc_stats::MhcStatsBf16x3::run")],
    )


def mhc_stats_bf16x3_sm90(x_flat: torch.Tensor, bf16_parts):
    """(part_mix [16, M, 24], part_sq [16, M]) of x_flat [M, 20480] bf16, as the Triton kernel."""
    m = x_flat.shape[0]
    part_mix = torch.empty((SLICES, m, MIX), device=x_flat.device, dtype=torch.float32)
    part_sq = torch.empty((SLICES, m), device=x_flat.device, dtype=torch.float32)
    if m > 0:
        _stats_module().run(x_flat, *bf16_parts, part_mix, part_sq)
    return part_mix, part_sq


@cache_once
def _module():
    return load_jit(
        "mhc_mega_boundary_sm90",
        cuda_files=["deepseek_v4/mhc_mega_boundary_sm90.cuh"],
        cuda_wrappers=[("run", "sglang::mhc_mega::MhcMegaBoundary::run")],
    )


@register_custom_op(
    mutates_args=["updated", "normalized", "part_mix", "part_sq"]
)
def _mhc_mega_boundary_sm90(
    y: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    w_hi: torch.Tensor,
    w_mid: torch.Tensor,
    w_lo: torch.Tensor,
    norm_weight: torch.Tensor,
    updated: torch.Tensor,
    normalized: torch.Tensor,
    part_mix: torch.Tensor,
    part_sq: torch.Tensor,
    eps: float,
) -> None:
    _module().run(
        y,
        residual,
        post,
        comb,
        pre,
        w_hi,
        w_mid,
        w_lo,
        norm_weight,
        updated,
        normalized,
        part_mix,
        part_sq,
        eps,
    )


def mhc_mega_boundary_sm90(
    y: torch.Tensor,
    residual: torch.Tensor,
    pre: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    next_bf16_parts,
    next_scale: torch.Tensor,
    next_base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    norm_eps: float,
    sinkhorn_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """(R', the next sublayer's normed input, the next triplet (pre, post, comb))."""
    from sglang.kernels.ops.layernorm.mhc import _hc_mix_reduce_sinkhorn_kernel

    m = y.shape[0]
    dev = y.device
    updated = torch.empty_like(residual)
    normalized = torch.empty_like(y)
    part_mix = torch.empty((SLICES, m, MIX), device=dev, dtype=torch.float32)
    part_sq = torch.empty((SLICES, m), device=dev, dtype=torch.float32)
    n_pre = torch.empty((m, MULT), device=dev, dtype=torch.float32)
    n_post = torch.empty_like(n_pre)
    n_comb = torch.empty((m, MULT, MULT), device=dev, dtype=torch.float32)
    if m == 0:
        return updated, normalized, (n_pre, n_post, n_comb)
    w_hi, w_mid, w_lo = next_bf16_parts
    _mhc_mega_boundary_sm90(
        y,
        residual,
        post,
        comb,
        pre,
        w_hi,
        w_mid,
        w_lo,
        norm_weight,
        updated,
        normalized,
        part_mix,
        part_sq,
        norm_eps,
    )
    # The unchanged reduction + sinkhorn of hc_mix_stats_sinkhorn_bf16x3.
    _hc_mix_reduce_sinkhorn_kernel[(m,)](
        part_mix,
        part_sq,
        next_scale,
        next_base,
        n_pre,
        n_post,
        n_comb,
        m,
        1.0 / (MULT * HIDDEN),
        rms_eps,
        MIX=MIX,
        HC=MULT,
        NUM_SLICES=SLICES,
        ITERS=sinkhorn_iters,
        EPS=hc_eps,
        num_warps=1,
    )
    return updated, normalized, (n_pre, n_post, n_comb)
