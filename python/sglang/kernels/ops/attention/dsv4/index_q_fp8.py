"""SM90 prefill index queries straight to e4m3, in one Triton launch.

The DeepGEMM prefill indexer on SM90 prepares its queries in two launches:
``index_q_rope_pack_weights`` (JIT CUDA, one warp per (token, head) row: RoPE, the
fp4 fake-quant, the fp4 pack, and the head weights) and ``fp4_index_rows_to_fp8``
(widen the packed rows to e4m3 with one power-of-two scale per row), then folds the
row scale into the head weight with a torch multiply. This kernel does all of it for
a block of rows per program, with the same operations and roundings:

- RoPE as the CUDA kernel compiles it: ``fma(x_re, f_re, -(x_im * f_im))`` and
  ``fma(x_re, f_im, x_im * f_re)``, rounded to bf16;
- the fp4 fake-quant (``block_scale`` / ``fake_quant_x2``) and the packer
  (``index_pack_exponent``, IEEE ``amax / 6``, nearest-even e2m1 codes, no ``-0``);
- the widening of ``_fp4_rows_to_fp8`` (block at ``2 ** (exponent - row max + 6)``,
  row scale ``2 ** (row max - 133)``), taking the e2m1 value straight from the
  rounding instead of packing and unpacking the code; the weight is
  ``float(bf16(raw * weight_scale)) * row_scale`` in fp32.

Every reduction is a maximum, so the row layout does not change any bit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .fp4_indexer import _ceil_ue8m0_exp
from .torch_quant import FP4_AMAX_FLOOR


@triton.jit
def _e2m1_grid_rne(x):
    """The e2m1 value nearest-even to x (|x| <= 6 up to an ulp, saturating at 6):
    ``cvt.rn.satfinite.e2m1`` then widen. The sign is kept only for a nonzero value,
    as the packer drops ``-0``."""
    mag = tl.abs(x)
    step = tl.where(mag < 2.0, 0.5, tl.where(mag < 4.0, 1.0, 2.0))
    # 1 / step is a power of two: the product is the exact quotient.
    inv_step = tl.where(mag < 2.0, 2.0, tl.where(mag < 4.0, 1.0, 0.5))
    q = tl.minimum(libdevice.rint(mag * inv_step) * step, 6.0)
    return tl.where((x < 0) & (q != 0.0), -q, q)


@triton.jit
def _index_q_fp8_kernel(
    X,
    F,
    f_stride,
    Pos,
    HW,
    Q8,
    WOut,
    num_rows,
    HEADS: tl.constexpr,
    RD: tl.constexpr,
    WEIGHT_SCALE: tl.constexpr,
    AMAX_FLOOR: tl.constexpr,
    BR: tl.constexpr,
):
    # [rows, 4 fp4 blocks, 32] throughout, so no reduction needs a layout change.
    r = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    live = r < num_rows
    pos = tl.load(Pos + r // HEADS, mask=live, other=0).to(tl.int64)
    col = (tl.arange(0, 4)[:, None] * 32 + tl.arange(0, 32)[None, :])[None, :, :]
    r3 = r[:, None, None]
    m3 = live[:, None, None] & (col < 128)
    row = X + r3 * 128
    x = tl.load(row + col, mask=m3, other=0.0).to(tl.float32)

    # RoPE over the tail half, rounded to bf16 (rope_tail ends in .to(bf16)), in the
    # CUDA kernel's form: fma(re, f_re, -(im * f_im)) and fma(re, f_im, im * f_re).
    tail = col >= 128 - RD
    pair = tl.where(tail, (col - (128 - RD)) // 2, 0)
    imaginary = (col % 2) == 1
    mt = m3 & tail
    re = tl.load(row + 128 - RD + 2 * pair, mask=mt, other=0.0).to(tl.float32)
    im = tl.load(row + 128 - RD + 2 * pair + 1, mask=mt, other=0.0).to(tl.float32)
    fbase = F + pos[:, None, None] * f_stride + 2 * pair
    fr = tl.load(fbase, mask=mt, other=1.0)
    fi = tl.load(fbase + 1, mask=mt, other=0.0)
    rot_re = tl.fma(re, fr, -(libdevice.mul_rn(im, fi)))
    rot_im = tl.fma(re, fi, libdevice.mul_rn(im, fr))
    rot = tl.where(imaginary, rot_im, rot_re)
    v = tl.where(tail, rot.to(tl.bfloat16).to(tl.float32), x)

    # fp4 fake-quant (block_scale / fake_quant_x2): ue8m0 scale of
    # max(amax, floor) * (1 / 6), exact power-of-two scaling, nearest-even e2m1.
    amax = tl.max(tl.abs(v), axis=2)
    bits = (tl.maximum(amax, AMAX_FLOOR) * (1.0 / 6.0)).to(tl.int32, bitcast=True)
    expo = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (expo << 23).to(tl.float32, bitcast=True)
    inv = ((254 - expo) << 23).to(tl.float32, bitcast=True)
    fq = _e2m1_grid_rne(v * inv[:, :, None]) * scale[:, :, None]

    # Packer (index_pack_exponent: IEEE amax / 6, floor 1e-4, clamped) and the e2m1
    # value of its code; _fp4_rows_to_fp8 then puts each block at
    # 2 ** (exponent - row max + 6) and the row scale at 2 ** (row max - 133).
    pamax = tl.max(tl.abs(fq), axis=2)
    pexp = _ceil_ue8m0_exp(tl.maximum(tl.div_rn(pamax, 6.0), 1.0e-4))
    pinv = ((254 - pexp) << 23).to(tl.float32, bitcast=True)
    grid = _e2m1_grid_rne(fq * pinv[:, :, None])
    row_exp = tl.max(pexp, axis=1)
    factor = tl.exp2((pexp - row_exp[:, None] + 6).to(tl.float32))
    tl.store(Q8 + r3 * 128 + col, (grid * factor[:, :, None]).to(tl.float8e4nv), mask=m3)
    row_scale = tl.exp2((row_exp - 127 - 6).to(tl.float32))
    hw = tl.load(HW + r, mask=live, other=0.0).to(tl.float32)
    w = (hw * WEIGHT_SCALE).to(tl.bfloat16).to(tl.float32)
    tl.store(WOut + r, w * row_scale, mask=live)


def index_q_rope_fp8_weights(
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    head_weights: torch.Tensor,
    weight_scale: float,
    *,
    block_rows: int = 16,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``index_q_rope_pack_weights`` + ``fp4_index_rows_to_fp8`` + the weight fold.

    :param q: ``[tokens, heads, 128]`` bf16, ``wq_b(q_lora)``.
    :param freqs_cis: ``[max_pos, 64]`` fp32, real/imag interleaved.
    :param positions: ``[tokens]`` int32 or int64.
    :param head_weights: ``[tokens, heads]`` bf16, the raw ``weights_proj(x)``.
    :return: (``[tokens, heads, 128]`` e4m3 queries, ``[tokens, heads]`` fp32 weights
        with the row scale folded in), as the SM90 DeepGEMM prefill indexer takes them.
    """
    tokens, heads, d = q.shape
    assert d == 128 and q.dtype == torch.bfloat16 and q.is_contiguous()
    assert freqs_cis.dtype == torch.float32 and freqs_cis.shape[-1] == 64
    assert head_weights.shape == (tokens, heads) and head_weights.is_contiguous()
    rows = tokens * heads
    out = torch.empty((tokens, heads, 128), dtype=torch.float8_e4m3fn, device=q.device)
    weights = torch.empty((tokens, heads), dtype=torch.float32, device=q.device)
    if rows:
        br = block_rows
        _index_q_fp8_kernel[(triton.cdiv(rows, br),)](
            q,
            freqs_cis,
            freqs_cis.stride(0),
            positions,
            head_weights,
            out,
            weights,
            rows,
            HEADS=heads,
            RD=64,
            WEIGHT_SCALE=float(weight_scale),
            AMAX_FLOOR=FP4_AMAX_FLOOR,
            BR=br,
            num_warps=num_warps,
        )
    return out, weights
