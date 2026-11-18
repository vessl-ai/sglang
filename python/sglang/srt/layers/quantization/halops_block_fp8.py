# SPDX-License-Identifier: Apache-2.0

"""HALOps MXFP8 dense GEMM for block-FP8 linears served as MXFP8.

A 32-wide-K ue8m0 block-FP8 weight is already an MXFP8 operand: the
``block_fp8_as_mxfp8`` path keeps the fp8 weight row-major and builds UE8M0
bytes ``[N, K // 32]`` from the block scales. HALOps' ``dense_gemm`` takes the
same weight and those bytes in the CUTLASS SfAtom 128x4 layout, which is the
layout FlashInfer's ``block_scale_interleave`` writes; with ``N % 128 == 0`` and
``K % 128 == 0`` neither pads, so the two scales are the same bytes, and
``halops_weight_scale`` checks that per layer at load.

Used only when ``SGLANG_HQ_HALOPS_BLOCK_FP8`` is set (``Fp8LinearMethod``).
The HALOps helpers follow upstream sgl-project/sglang e2ad447d00 (#47853),
which wires HALOps into the ``use_mxfp8`` path only.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Callable, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

HALOPS_MODE = "w8a8-mxfp8"

# HALOps w8a8 needs K % 128 == 0.
HALOPS_K_ALIGN = 128

# Row alignment of the CUTLASS SfAtom 128x4 scale layout. halops 0.1.0's
# dense_gemm writes output rows up to the next multiple of 128 when M is not
# one, so halops_linear gives every call whole 128-row atoms.
HALOPS_SFATOM_ROW_ALIGN = 128

# The bias epilogue has no per-row store predicate: M must be a multiple of 256.
HALOPS_BIAS_M_ALIGN = 256

LOG_PREFIX = "HQ HALOps block-FP8"

_used_shapes = set()
_selfcheck_left: Optional[dict[bool, int]] = None


def enabled() -> bool:
    return envs.SGLANG_HQ_HALOPS_BLOCK_FP8.get()


@lru_cache(maxsize=1)
def load_halops_gemm():
    """Return HALOps' ``(dense_gemm, group_wise_quant)``."""
    try:
        from halops import dense_gemm, group_wise_quant
    except ImportError as exc:
        raise RuntimeError(
            "SGLANG_HQ_HALOPS_BLOCK_FP8 is set, but the `halops` package is not "
            "installed (pip install halops==0.1.0 "
            "-f https://www.paddlepaddle.org.cn/packages/stable/cu132/halops/)"
        ) from exc
    return dense_gemm, group_wise_quant


def weight_supported(*, n: int, k: int) -> bool:
    return k % HALOPS_K_ALIGN == 0 and n % HALOPS_SFATOM_ROW_ALIGN == 0


def swizzle_scale_to_sfatom(scale_u8: torch.Tensor) -> torch.Tensor:
    """Row-major UE8M0 ``[M, K // 32]`` -> the CUTLASS SfAtom 128x4 layout.

    Needs ``M % 128 == 0`` and ``K % 128 == 0``.
    """
    m, k_groups = scale_u8.shape
    return (
        scale_u8.contiguous()
        .view(m // 128, 128, k_groups // 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(m // 128, k_groups // 4, 4, 32, 4)
        .permute(0, 1, 3, 2, 4)
        .reshape(m, k_groups)
        .contiguous()
    )


def halops_weight_scale(
    weight: torch.Tensor,
    scale_u8: torch.Tensor,
    flashinfer_scale: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """The SfAtom weight scale ``dense_gemm`` takes for this layer, or None when
    the layer stays on FlashInfer.

    ``scale_u8`` is the row-major UE8M0 ``[N, K // 32]`` the layer's FlashInfer
    scale was built from, ``flashinfer_scale`` that FlashInfer scale. A layer
    whose two scales differ in any byte stays on FlashInfer.
    """
    load_halops_gemm()
    n, k = weight.shape
    if not (weight_supported(n=n, k=k) and weight.is_contiguous()):
        return None
    sfatom = swizzle_scale_to_sfatom(scale_u8.contiguous().view(torch.uint8))
    if flashinfer_scale is not None:
        flat = flashinfer_scale.contiguous().view(torch.uint8).reshape(-1)
        if flat.numel() != sfatom.numel() or not torch.equal(flat, sfatom.reshape(-1)):
            logger.warning(
                "%s: N=%d K=%d stays on FlashInfer: its scale differs from the "
                "SfAtom scale",
                LOG_PREFIX,
                n,
                k,
            )
            return None
    return sfatom


def _capturing(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _selfcheck(
    out: torch.Tensor, fallback: Callable[[], torch.Tensor], m: int, n: int, k: int
) -> None:
    """Recompute this call on the original kernel and log the difference, for
    the first SGLANG_HQ_HALOPS_BLOCK_FP8_SELFCHECK calls with a nonzero
    reference outside CUDA graph capture, counted separately for M a multiple
    of HALOPS_SFATOM_ROW_ALIGN and M not one."""
    global _selfcheck_left
    if _selfcheck_left is None:
        count = envs.SGLANG_HQ_HALOPS_BLOCK_FP8_SELFCHECK.get()
        _selfcheck_left = {True: count, False: count}
    aligned = m % HALOPS_SFATOM_ROW_ALIGN == 0
    if _selfcheck_left[aligned] <= 0 or _capturing(out.device):
        return
    ref = fallback().float()
    ref_absmax = ref.abs().max().item()
    if ref_absmax == 0:
        return
    diff = out.float() - ref
    max_abs = diff.abs().max().item()
    rel_l2 = (diff.norm() / ref.norm()).item()
    _selfcheck_left[aligned] -= 1
    logger.info(
        "%s self-check: M=%d N=%d K=%d max_abs_diff=%.6g ref_absmax=%.6g rel_l2=%.6g",
        LOG_PREFIX,
        m,
        n,
        k,
        max_abs,
        ref_absmax,
        rel_l2,
    )


def halops_linear(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    fallback: Callable[[], torch.Tensor],
) -> torch.Tensor:
    """``input @ weight^T + bias`` on HALOps ``dense_gemm``.

    ``input`` is bf16/fp16 ``[..., K]``, ``weight`` fp8 ``[N, K]``, and
    ``weight_scale`` its SfAtom scale (``halops_weight_scale``). ``fallback``
    runs the layer's original kernel; it serves what no HALOps tile takes and
    is the self-check's reference.
    """
    n, k = weight.shape
    m = input.numel() // input.shape[-1]
    if (
        m == 0
        or input.dtype not in (torch.bfloat16, torch.float16)
        or (bias is not None and m % HALOPS_BIAS_M_ALIGN != 0)
    ):
        return fallback()

    dense_gemm, group_wise_quant = load_halops_gemm()
    # dense_gemm and group_wise_quant read raw row-major pointers.
    input_2d = input.reshape(-1, k).contiguous()
    q_input, sfa = group_wise_quant(input_2d, mode=HALOPS_MODE, use_swizzle=True)
    if bias is not None:
        # dense_gemm types the bias as its (always bf16) output element type.
        bias = bias.to(torch.bfloat16)
    output = torch.empty((m, n), dtype=torch.bfloat16, device=input_2d.device)

    def gemm(a: torch.Tensor, sfa_rows: torch.Tensor, out: torch.Tensor) -> None:
        dense_gemm(
            input=a,
            weight=weight,
            sfa=sfa_rows,
            sfb=weight_scale,
            output=out,
            bias=bias,
            alpha=None,
            sm_count=-1,
            mode=HALOPS_MODE,
        )

    body = m // HALOPS_SFATOM_ROW_ALIGN * HALOPS_SFATOM_ROW_ALIGN
    rest = m - body
    if body > 0:
        gemm(q_input[:body], sfa[:body], output[:body])
    if rest > 0:
        tail_input = q_input.new_zeros((HALOPS_SFATOM_ROW_ALIGN, k))
        tail_input[:rest].copy_(q_input[body:])
        tail_output = torch.empty(
            (HALOPS_SFATOM_ROW_ALIGN, n), dtype=torch.bfloat16, device=output.device
        )
        gemm(tail_input, sfa[body:], tail_output)
        output[body:].copy_(tail_output[:rest])
    output = output.to(input.dtype).reshape(*input.shape[:-1], n)

    if (n, k) not in _used_shapes:
        _used_shapes.add((n, k))
        logger.info("%s dense GEMM in use: N=%d K=%d first M=%d", LOG_PREFIX, n, k, m)
    _selfcheck(output, fallback, m, n, k)
    return output
