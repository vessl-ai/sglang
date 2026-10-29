"""Per-token-group FP8 (e4m3) quantization with power-of-two (UE8M0) scales stored as fp32.

The Hopper DeepSeek-V4 dense path quantizes every DeepGEMM activation this way
(row-major fp32 scales; DeepGEMM transposes them for TMA). The CUDA kernel that
serves it (``per_token_group_quant_8bit_v2``) runs at ~1/4 of HBM bandwidth at
prefill sizes; this Triton kernel computes the same values, bit for bit:

    amax  = max(1e-10, max |x| over the group)
    e     = ceil(log2(amax * (1/448)))        (exponent bits, exact)
    scale = 2^e  (stored),   q = sat_e4m3(rn(x * 2^-e))
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _ue8m0_group_quant_kernel(
    X,
    Q,
    S,
    M,
    stride_xm,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = rows < M
    mask = row_mask[:, None] & (cols[None, :] < K)
    x = tl.load(
        X + rows[:, None].to(tl.int64) * stride_xm + cols[None, :], mask, other=0.0
    ).to(tl.float32)
    xg = tl.reshape(x, (BLOCK_M, BLOCK_K // GROUP, GROUP))
    amax = tl.maximum(tl.max(tl.abs(xg), axis=2), 1e-10)
    # ceil(log2(amax / 448)) from the exponent and mantissa bits, as the CUDA kernel.
    bits = (amax * (1.0 / 448.0)).to(tl.uint32, bitcast=True)
    exp = ((bits >> 23) & 0xFF).to(tl.int32) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale_inv = ((exp + 127) << 23).to(tl.float32, bitcast=True)
    scale = ((127 - exp) << 23).to(tl.float32, bitcast=True)
    q = xg * scale[:, :, None]
    q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    q = tl.reshape(q, (BLOCK_M, BLOCK_K)).to(Q.dtype.element_ty)
    tl.store(Q + rows[:, None].to(tl.int64) * K + cols[None, :], q, mask)
    groups = pid_k * (BLOCK_K // GROUP) + tl.arange(0, BLOCK_K // GROUP)
    tl.store(
        S + rows[:, None].to(tl.int64) * (K // GROUP) + groups[None, :],
        scale_inv,
        row_mask[:, None] & (groups[None, :] < K // GROUP),
    )


def ue8m0_group_quant_fp8(
    x: torch.Tensor, group_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(q [M, K] e4m3, s [M, K // group_size] fp32 powers of two)``, row-major,
    equal to ``sglang_per_token_group_quant_fp8(x, group_size, scale_ue8m0=True)``."""
    assert x.dim() == 2 and x.stride(1) == 1, "rows must be contiguous"
    m, k = x.shape
    assert k % group_size == 0
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((m, k // group_size), dtype=torch.float32, device=x.device)
    if m == 0:
        return q, s
    block_k = min(512, triton.next_power_of_2(k))
    while k % block_k:
        block_k //= 2
    assert block_k % group_size == 0
    block_m = 16
    grid = (triton.cdiv(m, block_m), k // block_k)
    _ue8m0_group_quant_kernel[grid](
        x, q, s, m, x.stride(0), k, group_size, block_m, block_k, num_warps=4
    )
    return q, s
