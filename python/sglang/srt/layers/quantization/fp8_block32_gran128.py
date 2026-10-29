"""Hopper (SM90) dense FP8 GEMMs for checkpoints quantized with 32x32 UE8M0 blocks.

DeepSeek-V4.1-Flash ships its dense weights as FP8 e4m3 with 32x32 power-of-two
(UE8M0) block scales, and the engine quantizes activations per 1x32 group with
power-of-two scales. On SM90 DeepGEMM serves that recipe with SF_GRAN=32: it
promotes the FP32 accumulators after every K=32 WGMMA, four times as often as
the 128-granularity kernel, and runs at ~38-40% of the FP8 peak where the
128-granularity kernel reaches 65-72% at the same shapes.

With power-of-two scales the group size barely matters numerically: an element
quantized against 2^e and against 2^(e + d) gives the same dequantized value as
long as it stays a normal e4m3 number (|x| > 2^-6 of its scale), because scaling
by a power of two commutes with rounding. So this module

* requantizes each eligible weight once, at load, to 128x128 UE8M0 blocks
  (`requant_block32_to_block128`; on the V4.1-Flash checkpoint 1e-6..5e-4 of the
  elements move, relative weight error 1e-8..1e-6), and
* quantizes activations per 1x128 group (`ue8m0_group_quant_fp8_mn`, the same
  formula as the 1x32 Triton/CUDA quantizers), writing the scales directly in the
  MN-major TMA-aligned layout DeepGEMM consumes, so DeepGEMM skips its own scale
  transpose.

Enabled by SGLANG_OPT_DSV4_DENSE_GEMM_GRAN128 (default off). The layer is marked
with ``dense_gran128 = True`` and Fp8LinearMethod.apply routes it here.
"""

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_GROUP = 128
_FP8_MAX = 448.0


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _ue8m0_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(amax / 448)) computed from the fp32 bits (the activation quantizers' rule)."""
    bits = (amax.float().clamp(min=1e-10) * (1.0 / _FP8_MAX)).view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return torch.exp2(exp.float())


def is_power_of_two_scale(scale: torch.Tensor) -> bool:
    finite = torch.isfinite(scale) & (scale > 0)
    return bool((finite & (torch.frexp(scale)[0] == 0.5)).all().item())


@torch.no_grad()
def requant_block32_to_block128(
    weight: torch.Tensor, scale32: torch.Tensor, rows_per_chunk: int = 2048
) -> Tuple[torch.Tensor, torch.Tensor]:
    """[N, K] e4m3 weight with [ceil(N/32), ceil(K/32)] fp32 power-of-two scales ->
    the same weight with [ceil(N/128), ceil(K/128)] UE8M0 block scales.

    The dequantized value w = q * s32 is exact in fp32; it is divided by the
    128x128 block's power-of-two scale (exact) and rounded to e4m3 (round to
    nearest even). Elements whose 32x32 sub-block already had the block's
    exponent keep their value bit for bit; the others shift by a power of two
    and only change if they land in the e4m3 subnormal range.
    Works in row chunks so the fp32 temporary stays small (2048 x K x 4 bytes).
    """
    assert weight.dim() == 2 and weight.dtype == torch.float8_e4m3fn
    n, k = weight.shape
    assert scale32.shape == (_ceil_div(n, 32), _ceil_div(k, 32)), (
        f"expected 32x32 block scales for {tuple(weight.shape)}, got {tuple(scale32.shape)}"
    )
    assert rows_per_chunk % _GROUP == 0
    nb, kb = _ceil_div(n, _GROUP), _ceil_div(k, _GROUP)
    out_q = torch.empty_like(weight)
    out_s = torch.empty(nb, kb, dtype=torch.float32, device=weight.device)
    s32 = scale32.float()
    for r0 in range(0, n, rows_per_chunk):
        r1 = min(n, r0 + rows_per_chunk)
        rows = r1 - r0
        s = s32[r0 // 32 : _ceil_div(r1, 32)]
        s = s.repeat_interleave(32, 0)[:rows].repeat_interleave(32, 1)[:, :k]
        w = weight[r0:r1].float() * s
        bn = _ceil_div(rows, _GROUP)
        wp = torch.zeros(bn * _GROUP, kb * _GROUP, device=w.device)
        wp[:rows, :k] = w
        amax = wp.view(bn, _GROUP, kb, _GROUP).abs().amax(dim=(1, 3))
        s128 = _ue8m0_scale_from_amax(amax)
        out_s[r0 // _GROUP : r0 // _GROUP + bn] = s128
        inv = s128.repeat_interleave(_GROUP, 0)[:rows].repeat_interleave(_GROUP, 1)[
            :, :k
        ]
        out_q[r0:r1] = (w / inv).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return out_q, out_s


@triton.jit
def _ue8m0_quant_mn_kernel(
    X,
    Q,
    S,
    M,
    K,
    stride_xm,
    stride_sk,
    NUM_GROUPS,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUPS_PER_PROG: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    BLOCK_K: tl.constexpr = GROUP * GROUPS_PER_PROG
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = rows < M
    mask = row_mask[:, None] & (cols[None, :] < K)
    x = tl.load(
        X + rows[:, None].to(tl.int64) * stride_xm + cols[None, :], mask, other=0.0
    ).to(tl.float32)
    xg = tl.reshape(x, (BLOCK_M, GROUPS_PER_PROG, GROUP))
    amax = tl.maximum(tl.max(tl.abs(xg), axis=2), 1e-10)
    # ceil(log2(amax / 448)) from the exponent and mantissa bits, as the 1x32 quantizers.
    bits = (amax * (1.0 / 448.0)).to(tl.uint32, bitcast=True)
    exp = (
        ((bits >> 23) & 0xFF).to(tl.int32) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    )
    scale_inv = ((exp + 127) << 23).to(tl.float32, bitcast=True)
    scale = ((127 - exp) << 23).to(tl.float32, bitcast=True)
    q = xg * scale[:, :, None]
    q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    q = tl.reshape(q, (BLOCK_M, BLOCK_K)).to(Q.dtype.element_ty)
    tl.store(Q + rows[:, None].to(tl.int64) * K + cols[None, :], q, mask)
    groups = pid_k * GROUPS_PER_PROG + tl.arange(0, GROUPS_PER_PROG)
    # MN-major: scale[g, m] at S + g * stride_sk + m (stride_sk = M aligned to 4 floats)
    tl.store(
        S + groups[None, :].to(tl.int64) * stride_sk + rows[:, None],
        scale_inv,
        row_mask[:, None] & (groups[None, :] < NUM_GROUPS),
    )


def _tma_aligned_rows(m: int) -> int:
    # DeepGEMM's MN-major SF layout pads the M stride to 16 bytes (4 fp32).
    return _ceil_div(m, 4) * 4


def ue8m0_group_quant_fp8_mn(
    x: torch.Tensor, group: int = _GROUP
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(q [M, K] e4m3 row-major, s [M, ceil(K/group)] fp32 powers of two, MN-major with the
    M stride aligned to 4) for a 2-D bf16/fp16/fp32 input, group 32 or 128. K need not be a
    multiple of the group: the last group covers the remaining columns. Same values as
    ``sglang_per_token_group_quant_fp8(x, group, scale_ue8m0=True)`` where that applies."""
    assert x.dim() == 2 and x.stride(1) == 1, "rows must be contiguous"
    assert group in (32, 128)
    m, k = x.shape
    num_groups = _ceil_div(k, group)
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    m_al = _tma_aligned_rows(m)
    s_buf = torch.empty((num_groups, m_al), dtype=torch.float32, device=x.device)
    s = s_buf[:, :m].t()
    if m == 0:
        return q, s
    # 512 columns per program for both group sizes
    groups_per_prog = min(512 // group, triton.next_power_of_2(num_groups))
    block_m = 16
    grid = (triton.cdiv(m, block_m), triton.cdiv(num_groups, groups_per_prog))
    _ue8m0_quant_mn_kernel[grid](
        x,
        q,
        s_buf,
        m,
        k,
        x.stride(0),
        m_al,
        num_groups,
        GROUP=group,
        BLOCK_M=block_m,
        GROUPS_PER_PROG=groups_per_prog,
        num_warps=4,
    )
    return q, s


def ue8m0_group_quant_fp8_g128(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return ue8m0_group_quant_fp8_mn(x, _GROUP)


# DeepGEMM 0.2.2's SM90 heuristic picks, for a few (N, K) and small M, a one-wave
# BLOCK_M=256 x BLOCK_N=128 config with 3 stages that runs ~2x slower than its
# neighbours (e.g. N=5120 K=2048: 17.4 us at M=640, 36.6 us at M=704..768, 21.4 us at
# M=832 on H200). Inside these measured M windows the same requantized weights run
# faster through the SF_GRAN 32 kernel with the 128x128 scales expanded to the 32 grid
# (numerically the same weights; 1x32 activation groups give the same values as 1x128
# outside the e4m3 subnormal range). Inclusive M ranges, measured on H200 (132 SMs)
# with sgl-deep-gemm 0.2.2 (INF-681 runs/k-gemm/results/sweep_full.log, refine.log; > 10%
# faster in the window, 1.4-1.6x at the worst M); applied
# only on a 132-SM device.
_GRAN32_WINDOWS = {
    # (N, K): [(first M, last M)]; the edges sit on 128-row boundaries: the last
    # good and first slow measured M are 8..16 rows apart.
    (5120, 2048): [(641, 768)],  # wo_b
    (5120, 576): [(641, 768)],  # shared experts w2
    (5120, 15360): [(641, 768)],  # DSpark main_proj
    (8192, 1280): [(385, 512)],  # wq_b
    (4096, 1280): [(897, 1024)],  # indexer wq_b
    (1792, 5120): [(2177, 2304)],  # wq_a + wkv
    (1152, 5120): [(3073, 3328), (3457, 3584)],  # shared experts w1 + w3
    (1536, 5120): [(2305, 2816)],
}
_WINDOW_NUM_SMS = 132


def prefer_gran32_kernel(m: int, n: int, k: int) -> bool:
    windows = _GRAN32_WINDOWS.get((n, k))
    if not windows:
        return False
    import deep_gemm

    if deep_gemm.get_num_sms() != _WINDOW_NUM_SMS:
        return False
    return any(lo <= m <= hi for lo, hi in windows)


def block128_fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    weight_scale_32x: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """y = x @ dequant(weight).T for a weight requantized to 128x128 UE8M0 blocks.
    x is bf16 [..., K]; returns bf16 [..., N]. Runs DeepGEMM SF_GRAN 128 with 1x128
    activation groups, except inside the measured slow-heuristic M windows where it runs
    SF_GRAN 32 on the same weight with `weight_scale_32x` (the 128x128 scales expanded)."""
    from sglang.kernels.ops.gemm.fp8_kernel import deep_gemm_fp8_fp8_bf16_nt

    k = x.shape[-1]
    n = weight.shape[0]
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if m > 0:
        if weight_scale_32x is not None and prefer_gran32_kernel(m, n, k):
            q, s = ue8m0_group_quant_fp8_mn(x2, 32)
            deep_gemm_fp8_fp8_bf16_nt(q, s, weight, weight_scale_32x, out, 32, 32)
        else:
            q, s = ue8m0_group_quant_fp8_mn(x2, _GROUP)
            deep_gemm_fp8_fp8_bf16_nt(q, s, weight, weight_scale, out, _GROUP, _GROUP)
    if bias is not None:
        out += bias
    return out.view(*x.shape[:-1], n)


def expand_scale_128_to_32(scale128: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """[ceil(N/128), ceil(K/128)] -> [ceil(N/32), ceil(K/32)] (each 128x128 scale repeated)."""
    s = scale128.repeat_interleave(4, 0).repeat_interleave(4, 1)
    return s[: _ceil_div(n, 32), : _ceil_div(k, 32)].contiguous()


def layer_is_eligible(method, layer, use_deepgemm_runner: bool) -> bool:
    """Static eligibility of an Fp8LinearMethod layer for the 128-granularity path."""
    from sglang.srt.runtime_context import get_platform
    from sglang.srt.utils import is_cuda

    weight = getattr(layer, "weight", None)
    scale = getattr(layer, "weight_scale_inv", None)
    if not (is_cuda() and get_platform().is_sm90 and use_deepgemm_runner):
        return False
    if method.weight_block_size != [32, 32]:
        return False
    if getattr(method.quant_config, "scale_fmt", None) != "ue8m0":
        return False
    if method.use_marlin or getattr(method, "block_fp8_as_mxfp8", False):
        return False
    if getattr(layer, "keep_plain_weight_layout", False):
        return False
    if getattr(layer, "_block_fp8_bf16_weight", None) is not None:
        return False
    if weight is None or scale is None or weight.dim() != 2 or scale.dim() != 2:
        return False
    if weight.dtype != torch.float8_e4m3fn or not weight.is_cuda:
        return False
    n, k = weight.shape
    # D stores and B loads need 16-byte rows; DeepGEMM's own [128,128] gate wants N % 64.
    if n % 64 != 0 or k % 16 != 0 or k < _GROUP:
        return False
    if tuple(scale.shape) != (_ceil_div(n, 32), _ceil_div(k, 32)):
        return False
    return is_power_of_two_scale(scale.data)


def convert_layer(layer) -> None:
    q, s = requant_block32_to_block128(layer.weight.data, layer.weight_scale_inv.data)
    layer.weight.data = q
    layer.weight_scale_inv.data = s
    n, k = q.shape
    if (n, k) in _GRAN32_WINDOWS:
        layer.register_buffer(
            "weight_scale_inv_32x", expand_scale_128_to_32(s, n, k), persistent=False
        )
    layer.dense_gran128 = True
    layer._derived_weight_cache_error = (
        "Online weight updates are not supported while "
        "SGLANG_OPT_DSV4_DENSE_GEMM_GRAN128 is on: the dense FP8 weights were "
        "requantized from 32x32 to 128x128 blocks at load. Restart with the switch off."
    )


def apply(layer, x, bias: Optional[torch.Tensor]) -> torch.Tensor:
    if isinstance(x, tuple):
        raise RuntimeError(
            "SGLANG_OPT_DSV4_DENSE_GEMM_GRAN128: this layer was requantized to 128x128 "
            "blocks and cannot take a pre-quantized 1x32 activation; turn the switch off."
        )
    return block128_fp8_linear(
        x,
        layer.weight,
        layer.weight_scale_inv,
        bias,
        getattr(layer, "weight_scale_inv_32x", None),
    )
