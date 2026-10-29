"""W4A8 MXFP4 MoE experts for SM90 prefill (INF-674).

Hopper has no FP4 tensor cores, so the W4A16 paths (Marlin, triton_kernels) upcast the
weights to BF16 inside the GEMM and run at BF16 tensor-core rate. This path instead:

1. converts the layer's MXFP4 experts to FP8 e4m3 just in time, into one scratch buffer
   shared by all layers (a bandwidth-bound pass, ~1.3 ms per layer on H200 at the TP4
   DeepSeek-V4.1-Flash expert shapes);
2. quantizes the activations to FP8 with one power-of-2 scale per row;
3. runs both expert GEMMs as triton_kernels FP8 x FP8 ragged matmuls (FP8 tensor cores),
   with the row scale and a per-output-channel weight scale applied in the epilogue.

Numerics:
- Weights are exact. An e2m1 value times 2^d is exact in e4m3 for d in [-5, 6] with the
  integer conversion below, so with d = s - max_k(s) + 6 the per-32 e8m0 scale folds into
  one per-channel scale 2^(max_k(s) - 133) as long as the block exponents of an output
  channel span at most 11 binades (the DeepSeek-V4.1-Flash checkpoint spans at most 4).
  Layers that do not satisfy this raise at load time.
- Activations follow the model's own reference inference code, which quantizes every
  expert GEMM input to FP8 with a power-of-2 (UE8M0) scale per 1x32 block: a power-of-2
  scale per row rounds every value that stays in the e4m3 normal range identically.
  Against a BF16-activation computation the MoE output differs by ~4.5-5% (relative L2),
  which is the model's native FP8 activation quantization, not an error of this kernel.

Prefetch (SGLANG_DSV4_W4A8_PREFETCH, on by default): the conversion does not depend on the
activations, so right after a layer's experts finish, the next layer's FP8 weights are
converted on a side stream and overlap that layer's attention instead of sitting on the
critical path. The next layer is the one observed to follow last time (construction order
before that); a wrong guess only costs an in-line conversion. Two scratch buffers (w13, w2;
3.4 GB per GPU at the TP4 shapes) are shared by all layers.

Use for prefill only (no CUDA graph support: the routing sort and ragged metadata are
computed eagerly). Enabled by SGLANG_OPT_DSV4_W4A8_PREFILL_MOE=1 on SM90.
"""

from __future__ import annotations

import contextlib
from typing import NamedTuple

import torch
import triton
import triton.language as tl

_LUT: dict[torch.device, torch.Tensor] = {}

# A FP4 nibble scaled by 2^d with d in [_MIN_SHIFT, _MAX_SHIFT] is a normal e4m3 number.
_MAX_SHIFT = 6
_MIN_SHIFT = -5
# Conversion tile: scale blocks (16 packed bytes each) per program.
_CONVERT_BLOCK = 256
# An overlapped prefetch (SGLANG_DSV4_W4A8_PREFETCH_CHUNKS > 0) is launched only after a
# layer with at least this many rows, whose MoE all-reduce is long enough to cover the
# conversion. A smaller layer keeps the single launch before its top-k sum.
_OVERLAP_MIN_ROWS = 8192


class _Scratch:
    """FP8 weight buffers shared by every layer on one device, plus the stream state that
    says which layer they currently hold. Never freed: the size is fixed by the model.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.w13 = torch.empty(0, dtype=torch.float8_e4m3fn, device=device)
        self.w2 = torch.empty(0, dtype=torch.float8_e4m3fn, device=device)
        # The experts whose FP8 weights are (being) written into the buffers.
        self.holder = None
        self.stream = None
        self.ready = None  # recorded after the last conversion into the buffers
        self.free = None  # recorded after the last GEMM that read the buffers
        self.last_called = None

    def reserve(self, n13: int, n2: int):
        if self.w13.numel() < n13:
            self.w13 = torch.empty(n13, dtype=torch.float8_e4m3fn, device=self.device)
        if self.w2.numel() < n2:
            self.w2 = torch.empty(n2, dtype=torch.float8_e4m3fn, device=self.device)


_SCRATCH: dict[torch.device, _Scratch] = {}
# Construction order of the experts; the first guess for "next layer".
_INSTANCES: list = []


def _scratch(device: torch.device) -> _Scratch:
    sc = _SCRATCH.get(device)
    if sc is None:
        sc = _SCRATCH[device] = _Scratch(device)
    return sc


def _lut(device: torch.device) -> torch.Tensor:
    """int32 per packed byte. Low 16 bits: the two e4m3 bytes of its nibbles at d = 0
    (low nibble = even k in the low byte). High 16 bits: 0x08 in each byte whose nibble is
    nonzero, so `lut16 + d * mask16` adds d to both exponent fields and keeps zeros zero.
    """
    lut = _LUT.get(device)
    if lut is None:
        b = torch.arange(256, dtype=torch.int32)

        def nibble(c):
            mag = c & 7
            exp = torch.where(mag >= 2, (mag >> 1) + 6, torch.full_like(mag, 6))
            bits = (exp << 3) | (((mag & 1) << 2) * (mag >= 2))
            bits = torch.where(mag == 0, torch.zeros_like(bits), bits)
            return bits | ((c & 8) << 4), (mag != 0).int() * 8

        lo, lo_mask = nibble(b & 0xF)
        hi, hi_mask = nibble(b >> 4)
        lut = ((lo | (hi << 8)) | ((lo_mask | (hi_mask << 8)) << 16)).to(device)
        _LUT[device] = lut
    return lut


@triton.jit
def _mxfp4_to_fp8_kernel(
    W,
    SCALE,
    SMAX,
    LUT,
    OUT,
    first_block,
    n_blocks,
    BLOCKS_PER_ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # W viewed as [n_blocks, 16] bytes: one 32-element scale block per row of 16 bytes.
    # This launch converts scale blocks [first_block, n_blocks).
    pid = tl.program_id(0).to(tl.int64)
    blk = first_block + pid * BLOCK + tl.arange(0, BLOCK)
    mask = blk < n_blocks
    j = tl.arange(0, 16)
    b = tl.load(W + blk[:, None] * 16 + j[None, :], mask=mask[:, None], other=0).to(
        tl.int32
    )
    s = tl.load(SCALE + blk, mask=mask, other=0).to(tl.int32)
    smax = tl.load(SMAX + blk // BLOCKS_PER_ROW, mask=mask, other=0).to(tl.int32)
    d = (s - smax + 6)[:, None]
    lut = tl.load(LUT + b)
    v = (lut & 0xFFFF) + d * (lut >> 16)
    tl.store(OUT + blk[:, None] * 16 + j[None, :], v.to(tl.int16), mask=mask[:, None])


# e4m3 bytes of the e2m1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 at exponent shift 0, as
# the two source registers of PTX prmt (byte i of the pair = magnitude code i).
_PRMT_LUT_LO = tl.constexpr(0x3C383000)
_PRMT_LUT_HI = tl.constexpr(0x4C484440)
_WORDS_BLOCK = 1024


@triton.jit
def _prmt(a, b, c):
    return tl.inline_asm_elementwise(
        "prmt.b32 $0, $1, $2, $3;",
        "=r,r,r,r",
        [a, b, c],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _nibbles_to_fp8(x16, dd, lut_lo, lut_hi):
    """4 e2m1 nibbles (bits 0..15, nibble i = element i) -> 4 e4m3 bytes (byte i).
    dd = 8 * d + 128 with d the exponent shift of the 32-element block."""
    mag = _prmt(lut_lo, lut_hi, x16 & 0x7777)
    # 0x01 in each byte whose magnitude is nonzero (bytes are <= 0x4C: no carries).
    nz = ((mag + 0x7F7F7F7F) >> 7) & 0x01010101
    # Add 8d to each nonzero exponent field: +128 keeps every byte positive and below 256,
    # so bytes do not carry into each other; the xor removes the 128 again.
    v = (mag + nz * dd) ^ (nz << 7)
    # Sign: nibble bit 3 -> byte bit 7.
    t = (x16 & 0xFF) | ((x16 & 0xFF00) << 8)
    spread = (t & 0x000F000F) | ((t & 0x00F000F0) << 4)
    return v | ((spread << 4) & -2139062144)  # 0x80808080 as int32


@triton.jit
def _mxfp4_to_fp8_words_kernel(
    W,
    SCALE,
    SMAX,
    OUT,
    first_word,
    n_words,
    BLOCKS_PER_ROW: tl.constexpr,
    BLOCK: tl.constexpr,
    INTERLEAVE_HALF: tl.constexpr = 0,
):
    # Same result as _mxfp4_to_fp8_kernel, bit for bit, but on 32-bit words (8 nibbles,
    # 4 words per scale block) with the nibble table in registers (prmt) instead of a
    # per-byte table gather: HBM-bound (~4.1 TB/s on H200) instead of SM-bound (~3.9),
    # so it keeps its speed beside other kernels on fewer SMs.
    # INTERLEAVE_HALF = h > 0: each expert has 2h rows [gate; up]; output row r goes to
    # 2r (gate) or 2(r - h) + 1 (up), the layout of a fused SwiGLU epilogue.
    pid = tl.program_id(0).to(tl.int64)
    wi = first_word + pid * BLOCK + tl.arange(0, BLOCK)
    m = wi < n_words
    w = tl.load(W + wi, mask=m, other=0)
    blk = wi // 4
    s = tl.load(SCALE + blk, mask=m, other=0).to(tl.int32)
    smax = tl.load(SMAX + blk // BLOCKS_PER_ROW, mask=m, other=0)
    dd = (s - smax + 6) * 8 + 128
    lut_lo = tl.full([BLOCK], _PRMT_LUT_LO, tl.int32)
    lut_hi = tl.full([BLOCK], _PRMT_LUT_HI, tl.int32)
    lo = _nibbles_to_fp8(w & 0xFFFF, dd, lut_lo, lut_hi)
    hi = _nibbles_to_fp8((w >> 16) & 0xFFFF, dd, lut_lo, lut_hi)
    if INTERLEAVE_HALF > 0:
        # 4 words per scale block, BLOCKS_PER_ROW blocks per row.
        words_per_row = BLOCKS_PER_ROW * 4
        row = wi // words_per_row
        e = row // (2 * INTERLEAVE_HALF)
        r = row % (2 * INTERLEAVE_HALF)
        r2 = tl.where(r < INTERLEAVE_HALF, 2 * r, 2 * (r - INTERLEAVE_HALF) + 1)
        wo = (e * (2 * INTERLEAVE_HALF) + r2) * words_per_row + wi % words_per_row
    else:
        wo = wi
    tl.store(OUT + wo, (hi.to(tl.int64) << 32) | (lo.to(tl.int64) & 0xFFFFFFFF), mask=m)


def _mxfp4_to_fp8(
    w: torch.Tensor,
    scale: torch.Tensor,
    smax: torch.Tensor,
    out,
    num_launches: int = 0,
    interleave_half: int = 0,
):
    """[E, N, K/2] packed e2m1 + [E, N, K/32] e8m0 -> [E, N, K] e4m3 (scaled by 2^(133 - smax)).

    num_launches = 0: the INF-674 byte kernel in one launch. num_launches >= 1: the word
    kernel (same bits) split into that many back-to-back launches on the current stream.
    One launch takes every SM until all its programs have been dispatched, so a kernel
    another stream launches meanwhile (e.g. an all-reduce) waits for the whole conversion.
    Between two launches the hardware can dispatch such a kernel, so it waits for at most
    one slice and then runs beside the remaining slices.
    interleave_half = h > 0 (word kernel only): store the 2h rows of each expert as
    gate/up pairs (row r -> 2r for r < h, 2(r - h) + 1 otherwise).
    """
    if interleave_half > 0:
        num_launches = max(num_launches, 1)
    if num_launches <= 0:
        n_blocks = w.numel() // 16
        _mxfp4_to_fp8_kernel[(triton.cdiv(n_blocks, _CONVERT_BLOCK),)](
            w.view(torch.uint8),
            scale.view(torch.uint8),
            smax,
            _lut(w.device),
            out.view(torch.int16),
            0,
            n_blocks,
            scale.shape[-1],
            _CONVERT_BLOCK,
            num_warps=4,
        )
        return out
    n_words = w.numel() // 4
    n_programs = triton.cdiv(n_words, _WORDS_BLOCK)
    per_launch = triton.cdiv(n_programs, num_launches)
    for first in range(0, n_programs, per_launch):
        _mxfp4_to_fp8_words_kernel[(min(per_launch, n_programs - first),)](
            w.view(torch.int32),
            scale.view(torch.uint8),
            smax,
            out.view(torch.int64),
            first * _WORDS_BLOCK,
            n_words,
            scale.shape[-1],
            _WORDS_BLOCK,
            interleave_half,
            num_warps=8,
        )
    return out


@triton.jit
def _row_quant_kernel(X, Y, S, N: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    o = tl.arange(0, BLOCK)
    x = tl.load(X + r * N + o, mask=o < N, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), 1e-4)
    s = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
    tl.store(Y + r * N + o, (x / s).to(tl.float8e4nv), mask=o < N)
    tl.store(S + r, s)


def _row_quant(x: torch.Tensor):
    M, N = x.shape
    y = torch.empty(M, N, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(M, device=x.device, dtype=torch.float32)
    _row_quant_kernel[(M,)](x, y, s, N, triton.next_power_of_2(N), num_warps=8)
    return y, s


@triton.jit
def _swiglu_quant_kernel(
    GU, Y, S, I: tl.constexpr, LIMIT: tl.constexpr, BLOCK: tl.constexpr
):
    r = tl.program_id(0).to(tl.int64)
    o = tl.arange(0, BLOCK)
    m = o < I
    g = tl.load(GU + r * (2 * I) + o, mask=m, other=0.0).to(tl.float32)
    u = tl.load(GU + r * (2 * I) + I + o, mask=m, other=0.0).to(tl.float32)
    g = tl.minimum(g, LIMIT)
    u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
    a = (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(a)), 1e-4)
    s = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
    tl.store(Y + r * I + o, (a / s).to(tl.float8e4nv), mask=m)
    tl.store(S + r, s)


def _swiglu_quant(gate_up: torch.Tensor, limit: float):
    R, N2 = gate_up.shape
    inter = N2 // 2
    y = torch.empty(R, inter, device=gate_up.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(R, device=gate_up.device, dtype=torch.float32)
    _swiglu_quant_kernel[(R,)](
        gate_up, y, s, inter, limit, triton.next_power_of_2(inter), num_warps=4
    )
    return y, s


@triton.jit
def _swiglu_epilogue_fn(acc, limit):
    """triton_kernels fused activation (reduction_n 2) over a W13 tile whose columns are
    gate/up pairs: the same arithmetic as _swiglu_quant_kernel on the bf16 GEMM output
    (the tile is rounded to bf16 first), returned in fp32; the GEMM stores bf16."""
    g, u = tl.split(tl.reshape(acc, (acc.shape[0], acc.shape[1] // 2, 2)))
    g = g.to(tl.bfloat16).to(tl.float32)
    u = u.to(tl.bfloat16).to(tl.float32)
    g = tl.minimum(g, limit)
    u = tl.minimum(tl.maximum(u, -limit), limit)
    return g / (1.0 + tl.exp(-g)) * u


_SWIGLU_ACTIVATIONS: dict = {}


def _swiglu_activation(limit: float):
    act = _SWIGLU_ACTIVATIONS.get(limit)
    if act is None:
        from triton_kernels.matmul import FusedActivation
        from triton_kernels.specialize import FnSpecs

        act = FusedActivation(
            FnSpecs("dsv4_swiglu", _swiglu_epilogue_fn, ("limit",), reduction_n=2),
            (limit,),
        )
        _SWIGLU_ACTIVATIONS[limit] = act
    return act


@triton.jit
def _rows_quant_kernel(
    A, Y, S, R, I: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr
):
    # The quantization half of _swiglu_quant_kernel on rows that already hold the bf16
    # SwiGLU output, ROWS rows per program.
    r = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    o = tl.arange(0, BLOCK)
    m = (r < R)[:, None] & (o < I)[None, :]
    a = tl.load(A + r[:, None] * I + o[None, :], mask=m, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(a), axis=1), 1e-4)
    s = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
    tl.store(
        Y + r[:, None] * I + o[None, :], (a / s[:, None]).to(tl.float8e4nv), mask=m
    )
    tl.store(S + r, s, mask=r < R)


def _rows_quant(a: torch.Tensor):
    R, inter = a.shape
    y = torch.empty(R, inter, device=a.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(R, device=a.device, dtype=torch.float32)
    rows = 8
    _rows_quant_kernel[(triton.cdiv(R, rows),)](
        a, y, s, R, inter, rows, triton.next_power_of_2(inter), num_warps=8
    )
    return y, s


@triton.jit
def _gather_topk_sum_kernel(
    Y,
    POS,
    OUT,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ADD_TO_OUT: tl.constexpr = False,
):
    # OUT[t] = sum_k Y[POS[t * TOPK + k]] with fp32 accumulation in k order, the same
    # arithmetic as moe_topk_sum over token-major rows.
    # ADD_TO_OUT: OUT[t] = bf16(OUT[t] + bf16(sum)) in fp32, i.e. what
    # shared.add_(routed) computes after the plain sum (in place: one program per row block).
    t = tl.program_id(0).to(tl.int64)
    o = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    m = o < H
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        r = tl.load(POS + t * TOPK + k).to(tl.int64)
        # r < 0: a masked slot the direct routing dropped (contributes nothing).
        acc += tl.load(Y + r * H + o, mask=m & (r >= 0), other=0.0).to(tl.float32)
    if ADD_TO_OUT:
        prev = tl.load(OUT + t * H + o, mask=m, other=0.0).to(tl.float32)
        acc = prev + acc.to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + t * H + o, acc.to(tl.bfloat16), mask=m)


def _topk_sum_by_pos(down_sorted, pos, out, top_k: int, add_to_out: bool = False):
    """out[t] = sum_k down_sorted[pos[t * top_k + k]], skipping pos < 0
    (add_to_out: out[t] += that sum, rounded like a separate bf16 add)."""
    num_tokens, hidden = out.shape
    block_h = 1024
    _gather_topk_sum_kernel[(num_tokens, triton.cdiv(hidden, block_h))](
        down_sorted, pos, out, hidden, top_k, block_h, add_to_out, num_warps=4
    )
    return out


def _gather_topk_sum(
    down_sorted, combine_indx, out, top_k: int, add_to_out: bool = False
):
    """out[t] = sum over the top-k slots of token t of down_sorted[sorted position of slot],
    where combine_indx[i] is the slot (token * top_k + k) at sorted position i."""
    n_slots = combine_indx.numel()
    pos = torch.empty_like(combine_indx)
    pos[combine_indx.long()] = torch.arange(
        n_slots, dtype=combine_indx.dtype, device=combine_indx.device
    )
    return _topk_sum_by_pos(down_sorted, pos, out, top_k, add_to_out)


# --- Small steps: W4A8 GEMMs that read the MXFP4 weights directly (no FP8 copy) ---------
#
# A converted FP8 copy of a layer's experts costs ~1.3 ms per layer whatever the row count
# (5.2 GB of HBM traffic). At a few thousand rows per layer that is about as much as the two
# GEMMs, so for small steps these kernels expand the e2m1 nibbles to e4m3 in registers (the
# same exact shift as the conversion) and feed them to the FP8 tensor cores as the A operand
# (weights x activations^T, so a block of up to BR rows of one expert reads each weight byte
# once). The activations are quantized like the W4A8 path (power-of-2 scale per row) but
# written with each BK-wide chunk of columns permuted (even k first, then odd k), so the low
# and high nibbles of a weight byte meet two contiguous activation half-chunks.
# Everything is computed on the device from the routing (no host sync, fixed grid for a
# fixed row count), so these calls can be captured in a CUDA graph.

# (d = 0) e4m3 byte of the e2m1 magnitudes; same table as _PRMT_LUT_LO/_HI.
_DQ_LO_ASM = tl.constexpr(
    """
{
.reg .b32 v, t, sel, mag, nz, a, lo, hi;
mov.b32 lo, 0x3C383000;
mov.b32 hi, 0x4C484440;
and.b32 v, $1, 0x07070707;
shr.b32 t, v, 4;
or.b32 v, v, t;
prmt.b32 sel, v, 0, 0x4420;
prmt.b32 mag, lo, hi, sel;
add.u32 nz, mag, 0x7F7F7F7F;
shr.b32 nz, nz, 7;
and.b32 nz, nz, 0x01010101;
mul.lo.u32 a, nz, 255;
and.b32 a, a, $2;
add.u32 v, mag, a;
shl.b32 t, nz, 7;
xor.b32 v, v, t;
shl.b32 t, $1, 4;
and.b32 t, t, 0x80808080;
or.b32 $0, v, t;
}
"""
)
_DQ_HI_ASM = tl.constexpr(
    """
{
.reg .b32 v, t, sel, mag, nz, a, lo, hi;
mov.b32 lo, 0x3C383000;
mov.b32 hi, 0x4C484440;
shr.b32 v, $1, 4;
and.b32 v, v, 0x07070707;
shr.b32 t, v, 4;
or.b32 v, v, t;
prmt.b32 sel, v, 0, 0x4420;
prmt.b32 mag, lo, hi, sel;
add.u32 nz, mag, 0x7F7F7F7F;
shr.b32 nz, nz, 7;
and.b32 nz, nz, 0x01010101;
mul.lo.u32 a, nz, 255;
and.b32 a, a, $2;
add.u32 v, mag, a;
shl.b32 t, nz, 7;
xor.b32 v, v, t;
and.b32 t, $1, 0x80808080;
or.b32 $0, v, t;
}
"""
)


@triton.jit
def _dequant_nibbles(wb, dd, HIGH: tl.constexpr):
    """Packed e2m1 bytes -> e4m3 of their low (HIGH=False) or high nibbles, scaled by 2^d.
    dd = 8 d + 128 per byte (uint8), with d in [-5, 6]: exact, as in _nibbles_to_fp8."""
    if HIGH:
        return tl.inline_asm_elementwise(
            _DQ_HI_ASM,
            "=r,r,r",
            [wb, dd],
            dtype=tl.float8e4nv,
            is_pure=True,
            pack=4,
        )
    return tl.inline_asm_elementwise(
        _DQ_LO_ASM, "=r,r,r", [wb, dd], dtype=tl.float8e4nv, is_pure=True, pack=4
    )


@triton.jit
def _perm_col(o, BK: tl.constexpr):
    # Column of k in the activation layout: inside each BK chunk, even k first.
    return (o // BK) * BK + (o % 2) * (BK // 2) + (o % BK) // 2


@triton.jit
def _row_quant_perm_kernel(
    X, Y, S, N: tl.constexpr, BK: tl.constexpr, BLOCK: tl.constexpr
):
    r = tl.program_id(0).to(tl.int64)
    o = tl.arange(0, BLOCK)
    m = o < N
    x = tl.load(X + r * N + o, mask=m, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), 1e-4)
    s = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
    tl.store(Y + r * N + _perm_col(o, BK), (x / s).to(tl.float8e4nv), mask=m)
    tl.store(S + r, s)


def _row_quant_perm(x: torch.Tensor, bk: int):
    """_row_quant with each bk-wide chunk of columns stored even-k first."""
    M, N = x.shape
    y = torch.empty(M, N, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(M, device=x.device, dtype=torch.float32)
    _row_quant_perm_kernel[(M,)](
        x, y, s, N, bk, triton.next_power_of_2(N), num_warps=8
    )
    return y, s


@triton.jit
def _swiglu_quant_perm_kernel(
    GU, Y, S, I: tl.constexpr, LIMIT: tl.constexpr, BK: tl.constexpr, BLOCK: tl.constexpr
):
    r = tl.program_id(0).to(tl.int64)
    o = tl.arange(0, BLOCK)
    m = o < I
    g = tl.load(GU + r * (2 * I) + o, mask=m, other=0.0).to(tl.float32)
    u = tl.load(GU + r * (2 * I) + I + o, mask=m, other=0.0).to(tl.float32)
    g = tl.minimum(g, LIMIT)
    u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
    a = (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(a)), 1e-4)
    s = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
    tl.store(Y + r * I + _perm_col(o, BK), (a / s).to(tl.float8e4nv), mask=m)
    tl.store(S + r, s)


def _swiglu_quant_perm(gate_up: torch.Tensor, limit: float, bk: int):
    """_swiglu_quant with each bk-wide chunk of columns stored even-k first."""
    R, N2 = gate_up.shape
    inter = N2 // 2
    y = torch.empty(R, inter, device=gate_up.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(R, device=gate_up.device, dtype=torch.float32)
    _swiglu_quant_perm_kernel[(R,)](
        gate_up, y, s, inter, limit, bk, triton.next_power_of_2(inter), num_warps=4
    )
    return y, s


@triton.jit
def _direct_hist_kernel(IDS, COUNTS, n_slots, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ids = tl.load(IDS + o, mask=o < n_slots, other=-1)
    tl.atomic_add(COUNTS + ids, 1, mask=ids >= 0, sem="relaxed")


@triton.jit
def _direct_items(SCHED, c, ex, em, e, max_items, BR: tl.constexpr, BLOCK_J: tl.constexpr):
    # Expert e's row blocks of BR rows go to SCHED[first block of e + j] = e << 16 | j;
    # program 0 also writes -1 from the last block to max_items.
    blocks = (c + BR - 1) // BR
    first = tl.sum(tl.where(em & (ex < e), blocks, 0), 0)
    nb = tl.sum(tl.where(ex == e, blocks, 0), 0)
    for j0 in range(0, nb, BLOCK_J):
        j = j0 + tl.arange(0, BLOCK_J)
        tl.store(SCHED + first + j, (e << 16) | j, mask=j < nb)
    if e == 0:
        total = tl.sum(tl.where(em, blocks, 0), 0)
        for i0 in range(total, max_items, BLOCK_J):
            i = i0 + tl.arange(0, BLOCK_J)
            tl.store(SCHED + i, tl.full([BLOCK_J], -1, tl.int32), mask=i < max_items)


@triton.jit
def _direct_scan_kernel(
    COUNTS,
    OFFS,
    CURSOR,
    SCHED1,
    SCHED2,
    E,
    max_items1,
    max_items2,
    BR1: tl.constexpr,
    BR2: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    # One program per expert e: its first row (OFFS[e], also the scatter cursor) and its
    # row-block items for both GEMMs (BR1 and BR2 rows per block). OFFS[E] = valid slots.
    e = tl.program_id(0)
    ex = tl.arange(0, BLOCK_E)
    em = ex < E
    c = tl.load(COUNTS + ex, mask=em, other=0)
    first_row = tl.sum(tl.where(ex < e, c, 0), 0)
    tl.store(OFFS + e, first_row)
    tl.store(CURSOR + e, first_row)
    if e == 0:
        tl.store(OFFS + E, tl.sum(c, 0))
    _direct_items(SCHED1, c, ex, em, e, max_items1, BR1, BLOCK_J)
    _direct_items(SCHED2, c, ex, em, e, max_items2, BR2, BLOCK_J)


@triton.jit
def _direct_scatter_kernel(
    IDS,
    WTS,
    CURSOR,
    GATHER,
    SORTED_W,
    POS,
    n_slots,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Counting-sort scatter: slot -> next free row of its expert. The order of rows within
    # an expert is not deterministic, but every row is computed on its own and the top-k
    # sum reads rows by position, so the output does not depend on it.
    slot = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = slot < n_slots
    ids = tl.load(IDS + slot, mask=m, other=-1)
    valid = ids >= 0
    p = tl.atomic_add(CURSOR + ids, 1, mask=valid, sem="relaxed")
    tl.store(GATHER + p, slot // TOPK, mask=valid)
    tl.store(SORTED_W + p, tl.load(WTS + slot, mask=valid, other=0.0), mask=valid)
    tl.store(POS + slot, tl.where(valid, p, -1), mask=m)


@triton.jit
def _count_scan_kernel(COUNTS, CURSOR, E, BLOCK_E: tl.constexpr):
    # One program per expert e: CURSOR[e] = first sorted row of e (exclusive prefix sum).
    e = tl.program_id(0)
    ex = tl.arange(0, BLOCK_E)
    c = tl.load(COUNTS + ex, mask=ex < E, other=0)
    tl.store(CURSOR + e, tl.sum(tl.where(ex < e, c, 0), 0))


class _CountRouting(NamedTuple):
    expert_counts: torch.Tensor  # [E] int32
    gather_indx: torch.Tensor  # [slots] int32 token of each sorted row
    sorted_weights: torch.Tensor  # [slots] fp32 top-k weight of each sorted row
    pos: torch.Tensor  # [slots] int32 sorted row of each slot, -1 for masked slots


def _count_route(topk_ids, topk_weights, num_experts: int) -> _CountRouting:
    """Counting sort of the valid top-k slots by expert for the FP8-copy path (histogram,
    per-expert offsets, atomic-cursor scatter: 3 launches + a memset, no host sync). The
    order of rows inside an expert is not deterministic; every row is computed on its own
    and the top-k sum reads rows by position, so the MoE output does not depend on it.
    Masked slots (id -1) get no row (pos -1), so valid rows come first."""
    device = topk_ids.device
    n_slots = topk_ids.numel()
    ids = topk_ids.reshape(-1)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    block = 1024
    _direct_hist_kernel[(triton.cdiv(n_slots, block),)](
        ids, counts, n_slots, block, num_warps=4
    )
    cursor = torch.empty(num_experts, dtype=torch.int32, device=device)
    _count_scan_kernel[(num_experts,)](
        counts, cursor, num_experts, triton.next_power_of_2(num_experts), num_warps=4
    )
    gather_indx = torch.empty(n_slots, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(n_slots, dtype=torch.float32, device=device)
    pos = torch.empty(n_slots, dtype=torch.int32, device=device)
    _direct_scatter_kernel[(triton.cdiv(n_slots, block),)](
        ids,
        topk_weights.reshape(-1).float(),
        cursor,
        gather_indx,
        sorted_weights,
        pos,
        n_slots,
        topk_ids.shape[1],
        block,
        num_warps=4,
    )
    return _CountRouting(counts, gather_indx, sorted_weights, pos)


class _DirectRouting(NamedTuple):
    offs: torch.Tensor  # [E + 1] int32 first row of each expert; offs[E] = valid slots
    sched13: torch.Tensor  # int32 expert << 16 | row block for W13, -1 = no item
    sched2: torch.Tensor  # the same for W2's row blocks
    gather_indx: torch.Tensor  # [slots] int32 token of each sorted row
    sorted_weights: torch.Tensor  # [slots] fp32 top-k weight of each sorted row
    pos: torch.Tensor  # [slots] int32 sorted row of each slot, -1 for masked slots


def _direct_route(topk_ids, topk_weights, num_experts: int, br13: int, br2: int):
    """Sort the valid top-k slots by expert (counting sort, 4 launches, no host sync, sizes
    fixed by the slot count) and list the row blocks of the two direct GEMMs."""
    device = topk_ids.device
    n_slots = topk_ids.numel()
    ids = topk_ids.reshape(-1)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    block = 1024
    _direct_hist_kernel[(triton.cdiv(n_slots, block),)](
        ids, counts, n_slots, block, num_warps=4
    )
    items13 = triton.cdiv(n_slots, br13) + num_experts
    items2 = triton.cdiv(n_slots, br2) + num_experts
    offs = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    cursor = torch.empty(num_experts, dtype=torch.int32, device=device)
    sched13 = torch.empty(items13, dtype=torch.int32, device=device)
    sched2 = torch.empty(items2, dtype=torch.int32, device=device)
    _direct_scan_kernel[(num_experts,)](
        counts,
        offs,
        cursor,
        sched13,
        sched2,
        num_experts,
        items13,
        items2,
        br13,
        br2,
        triton.next_power_of_2(num_experts),
        128,
        num_warps=4,
    )
    gather_indx = torch.empty(n_slots, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(n_slots, dtype=torch.float32, device=device)
    pos = torch.empty(n_slots, dtype=torch.int32, device=device)
    _direct_scatter_kernel[(triton.cdiv(n_slots, block),)](
        ids,
        topk_weights.reshape(-1).float(),
        cursor,
        gather_indx,
        sorted_weights,
        pos,
        n_slots,
        topk_ids.shape[1],
        block,
        num_warps=4,
    )
    return _DirectRouting(offs, sched13, sched2, gather_indx, sorted_weights, pos)


@triton.jit
def _w4a8_direct_kernel(
    X,
    XS,
    GATHER,
    W,
    SC,
    SMAX,
    CS,
    GAMMA,
    OUT,
    OFFS,
    SCHED,
    N: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
    HAS_GATHER: tl.constexpr,
    HAS_GAMMA: tl.constexpr,
    SWAP: tl.constexpr,
    DD_PER_BLOCK: tl.constexpr = True,
):
    # OUT[r, n] = XS[src] * CS[e, n] * (GAMMA[r]) * sum_k X[src, k] * W[e, n, k] over the
    # rows r of one expert e (sorted order), src = GATHER[r] or r.
    # SWAP: acc is [BN, BR] (dequantized weights = A operand, activations = B operand);
    # else acc is [BR, BN] (activations = A, dequantized weights = B in shared memory).
    item = tl.load(SCHED + tl.program_id(1))
    if item >= 0:
        e = item >> 16
        r0 = tl.load(OFFS + e) + (item & 0xFFFF) * BR
        end = tl.load(OFFS + e + 1)
        rows = r0 + tl.arange(0, BR)
        rmask = rows < end
        if HAS_GATHER:
            src = tl.load(GATHER + rows, mask=rmask, other=0).to(tl.int64)
        else:
            src = rows.to(tl.int64)
        n = tl.program_id(0) * BN + tl.arange(0, BN)
        nmask = n < N
        wrow = (e * N + n).to(tl.int64)
        smax = tl.load(SMAX + wrow, mask=nmask, other=0)
        jb = tl.arange(0, BK // 2)
        if SWAP:
            acc = tl.zeros([BN, BR], dtype=tl.float32)
        else:
            acc = tl.zeros([BR, BN], dtype=tl.float32)
        for k0 in range(0, K, BK):
            wb = tl.load(
                W + wrow[:, None] * (K // 2) + (k0 // 2 + jb)[None, :],
                mask=nmask[:, None],
                other=0,
            )
            if DD_PER_BLOCK:
                # One shift per 32-element block, broadcast to its 16 packed bytes.
                jblk = tl.arange(0, BK // 32)
                s = tl.load(
                    SC + wrow[:, None] * (K // 32) + (k0 // 32 + jblk)[None, :],
                    mask=nmask[:, None],
                    other=0,
                ).to(tl.int32)
                dd = ((s - smax[:, None] + 6) * 8 + 128).to(tl.uint8)
                dd = tl.reshape(
                    tl.broadcast_to(dd[:, :, None], [BN, BK // 32, 16]), [BN, BK // 2]
                )
            else:
                s = tl.load(
                    SC + wrow[:, None] * (K // 32) + (k0 // 32 + jb // 16)[None, :],
                    mask=nmask[:, None],
                    other=0,
                ).to(tl.int32)
                dd = ((s - smax[:, None] + 6) * 8 + 128).to(tl.uint8)
            x_lo = tl.load(
                X + src[:, None] * K + (k0 + jb)[None, :], mask=rmask[:, None], other=0.0
            )
            x_hi = tl.load(
                X + src[:, None] * K + (k0 + BK // 2 + jb)[None, :],
                mask=rmask[:, None],
                other=0.0,
            )
            w_lo = _dequant_nibbles(wb, dd, False)
            w_hi = _dequant_nibbles(wb, dd, True)
            if SWAP:
                acc = tl.dot(w_lo, tl.trans(x_lo), acc)
                acc = tl.dot(w_hi, tl.trans(x_hi), acc)
            else:
                acc = tl.dot(x_lo, tl.trans(w_lo), acc)
                acc = tl.dot(x_hi, tl.trans(w_hi), acc)
        cs = tl.load(CS + wrow, mask=nmask, other=0.0)
        xs = tl.load(XS + src, mask=rmask, other=0.0)
        if HAS_GAMMA:
            xs = xs * tl.load(GAMMA + rows, mask=rmask, other=0.0)
        if SWAP:
            tl.store(
                OUT + rows.to(tl.int64)[None, :] * N + n[:, None],
                (acc * cs[:, None] * xs[None, :]).to(tl.bfloat16),
                mask=nmask[:, None] & rmask[None, :],
            )
        else:
            tl.store(
                OUT + rows.to(tl.int64)[:, None] * N + n[None, :],
                (acc * xs[:, None] * cs[None, :]).to(tl.bfloat16),
                mask=rmask[:, None] & nmask[None, :],
            )


def _w4a8_direct_matmul(
    x_perm,
    x_scale,
    w,
    scale,
    smax,
    channel_scale,
    routing: _DirectRouting,
    sched: torch.Tensor,
    bk: int,
    tiles,
    gather: bool,
    gamma: bool,
):
    """[slots, N] bf16 in expert-sorted row order (see _w4a8_direct_kernel); rows past
    routing.offs[E] (masked slots) are left unwritten."""
    E, N, KH = w.shape
    bn, br, warps, stages, swap = tiles
    out = torch.empty(routing.pos.numel(), N, device=w.device, dtype=torch.bfloat16)
    _w4a8_direct_kernel[(triton.cdiv(N, bn), sched.numel())](
        x_perm,
        x_scale,
        routing.gather_indx,
        w.view(torch.uint8),
        scale.view(torch.uint8),
        smax,
        channel_scale,
        routing.sorted_weights,
        out,
        routing.offs,
        sched,
        N,
        KH * 2,
        bk,
        bn,
        br,
        gather,
        gamma,
        swap,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def _fold(scale: torch.Tensor, name: str):
    """Per-output-channel max exponent and the matching fp32 channel scale."""
    s = scale.view(torch.uint8).to(torch.int16)
    smax = s.amax(-1)
    spread = int((smax - s.amin(-1)).max())
    if spread > _MAX_SHIFT - _MIN_SHIFT:
        raise ValueError(
            f"SGLANG_OPT_DSV4_W4A8_PREFILL_MOE: {name} block exponents span {spread} "
            f"binades in one output channel (max {_MAX_SHIFT - _MIN_SHIFT}); FP8 weights "
            "would not be exact. Unset the flag for this model."
        )
    channel_scale = torch.exp2(smax.float() - 127.0 - _MAX_SHIFT).contiguous()
    return smax.to(torch.int32).contiguous(), channel_scale


# Direct-kernel tiles (BN, BR, warps, stages, swap) for W13 (K 5120, BK 128) and W2 (K 576,
# BK 64), by routed rows per expert on average. Best of a sweep on H200 at the TP4
# DeepSeek-V4.1-Flash shapes, 6,912 / 13,824 / 27,648 rows; the W13 switch from 128- to
# 256-row blocks measured between 7,680 and 8,192 rows (~123 rows per expert).
_DIRECT_BK13, _DIRECT_BK2 = 128, 64
# FP8 tensor-core accumulation is the hardware's reduced-precision one over the whole K loop,
# like the triton_kernels path (promoting every 32 products: 1.28% -> 0.41% vs the fp32 W4A8
# reference at 2,048 rows, but 1.4-2.6x slower).
_DIRECT_TILES = (
    # (max rows per expert, W13 tiles, W2 tiles)
    (120, (128, 128, 8, 3, False), (128, 128, 4, 3, True)),
    (None, (128, 256, 8, 3, True), (128, 128, 4, 3, True)),
)


def _direct_tiles(rows_per_expert: float):
    for limit, t13, t2 in _DIRECT_TILES:
        if limit is None or rows_per_expert <= limit:
            return t13, t2


class Mxfp4W4A8PrefillExperts:
    """Keeps the checkpoint's MXFP4 experts ([E, N, K/2] int8 + [E, N, K/32] e8m0) and runs
    them as FP8 x FP8 GEMMs."""

    def __init__(
        self,
        w13,
        w13_scale,
        w2,
        w2_scale,
        swiglu_limit: float,
        prefetch: bool,
        prefetch_chunks: int = 0,
        sorted_sum: bool = False,
        small_step_rows: int = 0,
        fused_swiglu: bool = False,
        fused_shared_add: bool = False,
        counting_route: bool = False,
        w2_two_cta_min_rows: int = 0,
    ):
        self.w13, self.w13_scale = w13, w13_scale
        self.w2, self.w2_scale = w2, w2_scale
        self.w13_smax, self.w13_channel_scale = _fold(w13_scale, "w13")
        self.w2_smax, self.w2_channel_scale = _fold(w2_scale, "w2")
        self.num_experts = w13.shape[0]
        self.swiglu_limit = swiglu_limit
        self.prefetch = prefetch
        # > 0: after a large layer, prefetch in this many launches per weight, issued
        # after the layer's top-k sum so the all-reduce does not queue behind it.
        self.prefetch_chunks = prefetch_chunks
        # W2 writes its rows in expert order (no epilogue scatter) and the top-k sum
        # gathers them by position: same sums, bit for bit, ~12% less W2 time at 32k rows.
        self.sorted_sum = sorted_sum
        # Calls with at most this many rows read the MXFP4 weights directly (no FP8 copy,
        # see _w4a8_direct_kernel). 0 = never.
        self.small_step_rows = small_step_rows
        # INF-681 k-pmoe large-step switches (see environ.py); all bit-identical.
        # fused_swiglu: the FP8 copy of w13 holds gate/up rows interleaved and the gate/up
        # GEMM applies SwiGLU in its epilogue; the channel scale follows the row order.
        self.fused_swiglu = fused_swiglu
        if fused_swiglu:
            half = w13.shape[1] // 2
            cs = self.w13_channel_scale
            self.w13_channel_scale_il = (
                torch.stack([cs[:, :half], cs[:, half:]], -1)
                .reshape(cs.shape)
                .contiguous()
            )
        # The top-k sum adds into the shared expert's output (the position-gather sums:
        # sorted-sum or counting-route large steps, and the small-step direct path).
        self.fused_shared_add = fused_shared_add
        # Counting-sort routing (sorted-sum path only: it needs the per-slot positions).
        self.counting_route = counting_route and sorted_sum
        # Calls with at least this many rows run W2 with 128x128 tiles, two CTAs per SM.
        self.w2_two_cta_min_rows = w2_two_cta_min_rows
        self.successor = None  # the experts observed to run after these ones
        self.scratch = _scratch(w13.device)
        # Reserve the buffers now so KV-cache sizing sees them.
        self.scratch.reserve(w13.numel() * 2, w2.numel() * 2)
        _lut(w13.device)
        _INSTANCES.append(self)

    def _convert_into_scratch(self, num_launches: int = 0):
        sc = self.scratch
        half13 = self.w13.shape[1] // 2 if self.fused_swiglu else 0
        for w, scale, smax, buf, half in (
            (self.w13, self.w13_scale, self.w13_smax, sc.w13, half13),
            (self.w2, self.w2_scale, self.w2_smax, sc.w2, 0),
        ):
            E, N, KH = w.shape
            _mxfp4_to_fp8(
                w,
                scale,
                smax,
                buf[: E * N * KH * 2].view(E, N, KH * 2),
                num_launches,
                interleave_half=half,
            )

    def _direct(
        self, hidden_states, topk_ids, topk_weights, add_to=None
    ) -> torch.Tensor:
        """The experts with _w4a8_direct_kernel: same quantization as the FP8-copy path
        (exact FP8 weights, FP8 activations with a power-of-2 row scale), weights read once
        per block of BR rows of an expert instead of converted for the whole layer."""
        num_tokens, top_k = topk_ids.shape
        t13, t2 = _direct_tiles(num_tokens * top_k / self.num_experts)
        routing = _direct_route(topk_ids, topk_weights, self.num_experts, t13[1], t2[1])
        x_perm, x_scale = _row_quant_perm(hidden_states.contiguous(), _DIRECT_BK13)
        gate_up = _w4a8_direct_matmul(
            x_perm,
            x_scale,
            self.w13,
            self.w13_scale,
            self.w13_smax,
            self.w13_channel_scale,
            routing,
            routing.sched13,
            _DIRECT_BK13,
            t13,
            gather=True,
            gamma=False,
        )
        a_perm, a_scale = _swiglu_quant_perm(gate_up, self.swiglu_limit, _DIRECT_BK2)
        del gate_up
        down = _w4a8_direct_matmul(
            a_perm,
            a_scale,
            self.w2,
            self.w2_scale,
            self.w2_smax,
            self.w2_channel_scale,
            routing,
            routing.sched2,
            _DIRECT_BK2,
            t2,
            gather=False,
            gamma=True,
        )
        if add_to is not None:  # add_to += sum, rounded like a separate bf16 add
            return _topk_sum_by_pos(down, routing.pos, add_to, top_k, True)
        return _topk_sum_by_pos(down, routing.pos, torch.empty_like(hidden_states), top_k)

    def _single_launch(self) -> int:
        """num_launches for a conversion that is not sliced: the word kernel when the
        overlapped prefetch is on (same bits, faster), else the INF-674 byte kernel."""
        return 1 if self.prefetch_chunks > 0 else 0

    def _fp8_view(self, w, buf):
        E, N, KH = w.shape
        # triton_kernels takes an FP8 rhs as [E, K, N] column-major.
        return buf[: E * N * KH * 2].view(E, N, KH * 2).transpose(-1, -2)

    def _acquire(self):
        """Make the scratch hold this layer's FP8 weights, in stream order."""
        sc = self.scratch
        current = torch.cuda.current_stream(sc.device)
        if sc.ready is not None:
            # A prefetch may still be writing the buffers.
            current.wait_event(sc.ready)
        if sc.holder is not self:
            self._convert_into_scratch(self._single_launch())
            sc.holder = self
        if sc.last_called is not None and sc.last_called is not self:
            sc.last_called.successor = self
        sc.last_called = self

    def _prefetch_next(self, num_launches: int = 0):
        sc = self.scratch
        nxt = self.successor
        if nxt is None:
            i = _INSTANCES.index(self)
            nxt = _INSTANCES[(i + 1) % len(_INSTANCES)]
        if nxt is self or nxt.scratch is not sc:
            return
        if sc.stream is None:
            sc.stream = torch.cuda.Stream(sc.device)
            sc.ready = torch.cuda.Event()
            sc.free = torch.cuda.Event()
        sc.free.record(torch.cuda.current_stream(sc.device))
        with torch.cuda.stream(sc.stream):
            sc.stream.wait_event(sc.free)
            nxt._convert_into_scratch(num_launches)
            sc.ready.record(sc.stream)
        sc.holder = nxt

    def __call__(
        self, hidden_states, topk_ids, topk_weights, add_to: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Routed experts' output [tokens, hidden] bf16. add_to (the shared expert's
        output, same shape): return add_to after add_to += routed output, the same bits
        as a separate shared.add_(routed); fused into the top-k sum when
        fused_shared_add is on and the call takes the sorted-sum path."""
        import triton_kernels.matmul_details.opt_flags as opt_flags
        from triton_kernels.matmul import PrecisionConfig, matmul
        from triton_kernels.tensor import make_ragged_tensor_metadata

        from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum
        from sglang.srt.layers.quantization.mxfp4_marlin_moe import (
            route_slots_by_expert,
        )
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        num_tokens = topk_ids.shape[0]
        if 0 < num_tokens <= self.small_step_rows:
            # No scratch, no events, no host sync: safe inside a CUDA graph.
            if add_to is not None and self.fused_shared_add:
                return self._direct(hidden_states, topk_ids, topk_weights, add_to)
            out = self._direct(hidden_states, topk_ids, topk_weights)
            return out if add_to is None else add_to.add_(out)
        # The Python state of the shared scratch (which layer it holds, the prefetch
        # events) is not replayed by a CUDA graph: a captured call would freeze it and
        # could read another layer's FP8 weights, or fail capture on the events.
        if torch.cuda.is_current_stream_capturing() or get_is_capture_mode():
            raise RuntimeError(
                "SGLANG_OPT_DSV4_W4A8_PREFILL_MOE does not support CUDA graphs; enable "
                "it only on a PD prefill server whose MoE runs eagerly."
            )
        num_tokens, top_k = topk_ids.shape
        if num_tokens == 0:
            return torch.empty_like(hidden_states) if add_to is None else add_to
        fuse_add = (
            add_to is not None
            and self.fused_shared_add
            and (self.sorted_sum or self.counting_route)
        )
        output = add_to if fuse_add else torch.empty_like(hidden_states)
        pos = None
        if self.counting_route:
            routing = _count_route(topk_ids, topk_weights, self.num_experts)
            pos = routing.pos
        else:
            routing = route_slots_by_expert(topk_ids, topk_weights, self.num_experts)
        ragged_metadata = make_ragged_tensor_metadata(
            routing.expert_counts, num_tokens * top_k
        )

        x_fp8, x_scale = _row_quant(hidden_states.contiguous())
        self._acquire()
        # Use triton_kernels' own tile heuristics with no constraint at all: constraints
        # set by others (the W4A16 path's num_stages pin, the MXFP4 swizzle's split_k=1)
        # are tuned for BF16 x MXFP4 kernels. scoped_opt_flags_constraints merges into
        # them, so replace the ContextVar directly (triton_kernels 3.8 internals).
        token = opt_flags._opt_flags_constraints.set({})
        try:
            if self.fused_swiglu:
                act = matmul(
                    x_fp8,
                    self._fp8_view(self.w13, self.scratch.w13),
                    None,
                    a_ragged_metadata=ragged_metadata,
                    gather_indx=routing.gather_indx,
                    precision_config=PrecisionConfig(
                        a_mx_tensor_scale=x_scale,
                        b_mx_tensor_scale=self.w13_channel_scale_il,
                        out_dtype=torch.bfloat16,
                    ),
                    fused_activation=_swiglu_activation(self.swiglu_limit),
                )
                a_fp8, a_scale = _rows_quant(act.view(-1, act.shape[-1]))
                del act
            else:
                gate_up = matmul(
                    x_fp8,
                    self._fp8_view(self.w13, self.scratch.w13),
                    None,
                    a_ragged_metadata=ragged_metadata,
                    gather_indx=routing.gather_indx,
                    precision_config=PrecisionConfig(
                        a_mx_tensor_scale=x_scale,
                        b_mx_tensor_scale=self.w13_channel_scale,
                        out_dtype=torch.bfloat16,
                    ),
                )
                a_fp8, a_scale = _swiglu_quant(gate_up, self.swiglu_limit)
                del gate_up
            w2_flags = None
            # Only where triton_kernels' own choice is also 128-row persistent wgmma tiles
            # (>= 128 rows per expert on average): those accumulate every output element
            # in the same order, so the two tilings give the same bits (smaller row counts
            # get other kernels with other accumulation).
            if (
                0 < self.w2_two_cta_min_rows <= num_tokens
                and num_tokens * top_k >= 128 * self.num_experts
            ):
                w2_flags = _w2_two_cta_flags()
            with (
                opt_flags.scoped_opt_flags(w2_flags)
                if w2_flags is not None
                else contextlib.nullcontext()
            ):
                down = matmul(
                    a_fp8,
                    self._fp8_view(self.w2, self.scratch.w2),
                    None,
                    a_ragged_metadata=ragged_metadata,
                    scatter_indx=None if self.sorted_sum else routing.combine_indx,
                    precision_config=PrecisionConfig(
                        a_mx_tensor_scale=a_scale,
                        b_mx_tensor_scale=self.w2_channel_scale,
                        out_dtype=torch.bfloat16,
                    ),
                    gammas=routing.sorted_weights,
                )
        finally:
            opt_flags._opt_flags_constraints.reset(token)
        overlapped = self.prefetch_chunks > 0 and num_tokens >= _OVERLAP_MIN_ROWS
        if self.prefetch and not overlapped:
            self._prefetch_next(self._single_launch())
        if pos is not None:
            _topk_sum_by_pos(down, pos, output, top_k, fuse_add)
        elif self.sorted_sum:
            _gather_topk_sum(down, routing.combine_indx, output, top_k, fuse_add)
        else:
            moe_topk_sum(down.view(num_tokens, top_k, -1), output)
        if self.prefetch and overlapped:
            # Launched before the top-k sum in one piece, the conversion takes every SM
            # and the main stream's next kernels (shared add, all-reduce) queue behind it
            # (equal stream priority): p1 measured 0.8 of 57 ms overlapped. Issued after
            # the top-k sum in slices, the all-reduce gets its SMs after one slice and the
            # rest of the conversion runs beside it.
            self._prefetch_next(self.prefetch_chunks)
        if add_to is not None and not fuse_add:
            return add_to.add_(output)
        return output


_W2_TWO_CTA_FLAGS = None


def _w2_two_cta_flags():
    """triton_kernels flags for the K = 576 down-projection at large row counts: 128x128
    tiles, 4 warps, 3 stages, epilogue in 2 subtiles, 2 CTAs per SM (~112 KB shared
    memory each), so one CTA's epilogue (the 4 GB bf16 output) runs beside the other's
    main loop. H200, 65,208 rows, real TP4 weights: 2.55 vs 2.69 ms, same bits."""
    global _W2_TWO_CTA_FLAGS
    if _W2_TWO_CTA_FLAGS is None:
        from triton_kernels.matmul_details.opt_flags import OptFlags

        _W2_TWO_CTA_FLAGS = OptFlags(
            block_m=128,
            block_n=128,
            block_k=128,
            num_warps=4,
            num_stages=3,
            group_m=8,
            xcd_swizzle=1,
            w_cache_modifier=None,
            split_k=1,
            is_persistent=True,
            idle_sms=0,
            epilogue_subtile=2,
            arch=None,
            occupancy_target=2,
            target_kernel_kwargs={"maxnreg": 256, "FLATTEN_LOOPS": True},
        )
    return _W2_TWO_CTA_FLAGS
