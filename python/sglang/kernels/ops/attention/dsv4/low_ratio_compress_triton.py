"""Fused ratio-1/2 compressor for the prefill-CP path, any CUDA arch (SM90 included).

The prefill-CP branch of the V4.1 backend writes every row of the prompt's
compressed state through ``DeepseekV4AttnBackend._low_ratio_compress_torch``: about
twenty torch launches over all T rows (partner selection by ``roll`` / ``where``, the
ring ``index_put``, ``stack``, ``softmax``, ``mul``, ``sum``, RMSNorm, RoPE + fp4
fake-quant, the FlashMLA store, the index-K fake-quant and pack). The fused SM100
kernels (``low_ratio_compress.py``) do not run on SM90 and pool the pair with a
different formula, so they are not bitwise the torch path.

These kernels are bitwise the torch path on CUDA. Each piece repeats the operation
and rounding of the launch it replaces:

- softmax over the pair: ``exp(x - max) / sum`` with libdevice ``expf`` and an IEEE
  divide, the pooled value as two rounded products and one add (no FMA);
- ``finish``: bf16 rounding, then ``rmsnorm_fp32``'s Triton reduction, reloaded from
  the bf16 row so the reduction sees the same layout;
- ``_rope_fq4(..., compressed_kv=True)``: the same Triton row function;
- the V4 FlashMLA store (``fused_store_flashmla_cache``): per-64 ue8m0 scale, e4m3
  RNE with clipping, bf16 RoPE tail;
- the index-K branch: ``k_norm`` as ``rmsnorm_fp32``, then ``index_k_rope_pack``'s
  RoPE, fake-quant and RNE pack (the dsv41 low-ratio pool rounds to nearest even).

Rows that publish nothing (an even position at ratio 2, a padded row) are skipped;
the torch path writes them to the reserved slot 0, which no reader uses.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .fp4_indexer import INDEX_K_SLOT_BYTES, _ceil_ue8m0_exp, _fp4_e2m1_code_rne
from .fp4_rope_fake_quant import rope_tail_fake_quant_fp4_row
from .torch_quant import FP4_AMAX_FLOOR

# V4 FlashMLA page format: 448 e4m3 + 64 bf16 data bytes, then 7 ue8m0 scales + 1 pad.
_V4_DATA_BYTES = 576
_V4_SCALE_BYTES = 8
_V4_PAGE_ALIGN = 576


def v4_page_bytes(page_size: int) -> int:
    raw = page_size * (_V4_DATA_BYTES + _V4_SCALE_BYTES)
    return (raw + _V4_PAGE_ALIGN - 1) // _V4_PAGE_ALIGN * _V4_PAGE_ALIGN


@triton.jit
def _store_v4_row(Cache, loc, q, PAGE_SIZE: tl.constexpr, PAGE_BYTES: tl.constexpr):
    """``fused_store_flashmla_cache`` for one 512-wide row ``q`` (fp32 holding the
    bf16 value the torch path passes to the store)."""
    c = tl.reshape(q.to(tl.bfloat16).to(tl.float32), (8, 64))
    amax = tl.max(tl.abs(c), axis=1)
    scale_raw = tl.div_rn(tl.maximum(amax, 1e-4), 448.0)
    bits = scale_raw.to(tl.int32, bitcast=True)
    expo = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    inv_scale = ((254 - expo) << 23).to(tl.float32, bitcast=True)
    y = tl.minimum(tl.maximum(c * inv_scale[:, None], -448.0), 448.0)

    page = loc // PAGE_SIZE
    slot = loc - page * PAGE_SIZE
    data = Cache + page * PAGE_BYTES + slot * 576
    tile = tl.arange(0, 8)[:, None]
    col = tl.arange(0, 64)[None, :]
    tl.store(
        data.to(tl.pointer_type(tl.float8e4nv)) + tile * 64 + col,
        y.to(tl.float8e4nv),
        mask=tile < 7,
    )
    # The RoPE tile stays bf16 at byte 448: bf16 element 224 + col.
    tl.store(
        data.to(tl.pointer_type(tl.bfloat16)) + 224 + col + tile * 0,
        c.to(tl.bfloat16),
        mask=tile == 7,
    )
    scale_ptr = Cache + page * PAGE_BYTES + PAGE_SIZE * 576 + slot * 8
    t1 = tl.arange(0, 8)
    tl.store(scale_ptr + t1, expo.to(tl.uint8), mask=t1 < 7)


@triton.jit
def _low_ratio_compress_kernel(
    KV,
    Score,
    kv_stride,
    score_stride,
    State,
    state_stride,
    Req,
    Pos,
    RawLoc,
    OutLoc,
    NormW,
    Freqs,
    f_stride,
    Latent,
    Cache,
    ring_size,
    RATIO: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    EPS: tl.constexpr,
    AMAX_FLOOR: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
):
    """One program per row; rows with ``out_loc <= 0`` publish nothing."""
    i = tl.program_id(0).to(tl.int64)
    out_loc = tl.load(OutLoc + i).to(tl.int64)
    if out_loc > 0:
        offs = tl.arange(0, D)
        pos = tl.load(Pos + i).to(tl.int64)
        if RATIO == 2:
            req = tl.load(Req + i).to(tl.int64)
            ip = tl.maximum(i - 1, 0)
            # torch: in_batch = same request, consecutive position, neither row padded
            # (this row is not padded: a padded row has out_loc 0).
            prev_ok = (
                (i > 0)
                & (tl.load(Req + ip).to(tl.int64) == req)
                & (tl.load(Pos + ip).to(tl.int64) == pos - 1)
                & (tl.load(RawLoc + ip) != 0)
            )
            # Otherwise the carried ring row of position pos - 1 (>= 0: pos is odd).
            srow = req * ring_size + (pos - 1) % ring_size
            k_new = tl.load(KV + i * kv_stride + offs)
            s_new = tl.load(Score + i * score_stride + offs)
            k_prev = tl.load(KV + ip * kv_stride + offs, mask=(offs < D) & prev_ok)
            s_prev = tl.load(Score + ip * score_stride + offs, mask=(offs < D) & prev_ok)
            from_ring = prev_ok == 0
            k_ring = tl.load(State + srow * state_stride + offs, mask=(offs < D) & from_ring)
            s_ring = tl.load(
                State + srow * state_stride + D + offs, mask=(offs < D) & from_ring
            )
            k_old = tl.where(prev_ok, k_prev, k_ring)
            s_old = tl.where(prev_ok, s_prev, s_ring)
            # torch softmax over dim 1 of [n, 2, D]: exp(x - max) / sum, IEEE divide.
            m = tl.maximum(s_old, s_new)
            e_old = libdevice.exp(s_old - m)
            e_new = libdevice.exp(s_new - m)
            denom = e_old + e_new
            p_old = tl.div_rn(e_old, denom)
            p_new = tl.div_rn(e_new, denom)
            # (kv2 * p).sum(dim=1): two rounded products, one add, never an FMA.
            pooled = libdevice.add_rn(
                libdevice.mul_rn(k_old, p_old), libdevice.mul_rn(k_new, p_new)
            )
            rope_pos = pos - 1
        else:
            # Ratio 1 projects straight to bf16; finish() is the norm alone.
            pooled = tl.load(KV + i * kv_stride + offs).to(tl.float32)
            rope_pos = pos
        row = Latent + i * D
        # finish(): kv.to(bf16), then rmsnorm_fp32 reading that bf16 row.
        tl.store(row + offs, pooled.to(tl.bfloat16))
        tl.debug_barrier()
        col = tl.arange(0, D)
        x = tl.load(row + col, col < D, 0).to(tl.float32)
        weight = tl.load(NormW + col, col < D, 0).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(libdevice.mul_rn(x, x), axis=0) / D + EPS)
        normalized = x * inv_rms
        tl.debug_barrier()
        tl.store(row + col, (normalized * weight).to(tl.bfloat16), col < D)
        tl.debug_barrier()
        # _rope_fq4(latent, freqs[group_pos], rope_dim, compressed_kv=True)
        q = rope_tail_fake_quant_fp4_row(
            row,
            Freqs + rope_pos * f_stride,
            D=D,
            RD=RD,
            BLK=16,
            AMAX_FLOOR=AMAX_FLOOR,
            INVERSE=False,
            COMPRESSED_KV=True,
        )
        _store_v4_row(Cache, out_loc, q, PAGE_SIZE=PAGE_SIZE, PAGE_BYTES=PAGE_BYTES)


@triton.jit
def _c2_ring_write_kernel(
    KV,
    Score,
    kv_stride,
    score_stride,
    State,
    state_stride,
    state_rows,
    Req,
    Pos,
    RawLoc,
    num_tokens,
    ring_size,
    D: tl.constexpr,
    ROWS: tl.constexpr,
):
    """``_low_ratio_pair_partners``' ring write: each request's last ``ring_size``
    unpadded rows park ``| kv | score |`` at ``req * ring_size + pos % ring_size``.
    A separate launch: the pooling kernel reads ring rows this one overwrites.
    ``ROWS`` rows per program; only a request's last rows copy anything."""
    pid = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, D)
    if pid == 0:
        # set_state_by_state_loc clears the sentinel row -1 after its scatter.
        last = State + (state_rows - 1) * state_stride
        tl.store(last + offs, tl.zeros([D], dtype=tl.float32))
        tl.store(last + D + offs, tl.full([D], float("-inf"), dtype=tl.float32))
    rows = pid * ROWS + tl.arange(0, ROWS)
    live = rows < num_tokens
    raw = tl.load(RawLoc + rows, mask=live, other=0)
    req = tl.load(Req + rows, mask=live, other=0).to(tl.int64)
    later = rows + ring_size < num_tokens
    j = tl.where(later, rows + ring_size, 0)
    req_j = tl.load(Req + j, mask=later, other=0).to(tl.int64)
    raw_j = tl.load(RawLoc + j, mask=later, other=0)
    # torch: keep = ~pad; keep[:-ring] &= (req[:-ring] != req[ring:]) | pad[ring:]
    keep = live & (raw != 0) & ((later == 0) | (req_j != req) | (raw_j == 0))
    # Almost every block holds no request tail: skip the copy's address math there.
    if tl.max(keep.to(tl.int32), axis=0) > 0:
        pos = tl.load(Pos + rows, mask=keep, other=0).to(tl.int64)
        dst = (
            State
            + ((req * ring_size + pos % ring_size) * state_stride)[:, None]
            + offs[None, :]
        )
        m = keep[:, None] & (offs[None, :] < D)
        src = (rows * kv_stride)[:, None] + offs[None, :]
        tl.store(dst, tl.load(KV + src, mask=m), mask=m)
        src = (rows * score_stride)[:, None] + offs[None, :]
        tl.store(dst + D, tl.load(Score + src, mask=m), mask=m)


@triton.jit
def _index_k_norm_rope_pack_kernel(
    K,
    k_stride,
    NormW,
    Normed,
    Freqs,
    f_stride,
    Pos,
    OutLoc,
    Cache,
    POS_OFFSET: tl.constexpr,
    EPS: tl.constexpr,
    RD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    AMAX_FLOOR: tl.constexpr,
):
    """``k_norm`` (``rmsnorm_fp32``), then ``index_k_rope_pack``'s RoPE, fp4
    fake-quant, RNE pack and the index-K cache store, for rows with ``out_loc > 0``."""
    i = tl.program_id(0).to(tl.int64)
    out_loc = tl.load(OutLoc + i).to(tl.int64)
    if out_loc > 0:
        col = tl.arange(0, 128)
        x = tl.load(K + i * k_stride + col, col < 128, 0).to(tl.float32)
        weight = tl.load(NormW + col, col < 128, 0).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(libdevice.mul_rn(x, x), axis=0) / 128 + EPS)
        normalized = x * inv_rms
        X = Normed + i * 128
        tl.store(X + col, (normalized * weight).to(tl.bfloat16), col < 128)
        tl.debug_barrier()

        frow = tl.load(Pos + i).to(tl.int64) - POS_OFFSET
        offsets = tl.arange(0, 128)
        value = tl.load(X + offsets).to(tl.float32)
        tail = offsets >= 128 - RD
        pair = (offsets - (128 - RD)) // 2
        imaginary = (offsets % 2) == 1
        real = tl.load(X + 128 - RD + 2 * pair, tail, 0).to(tl.float32)
        imag = tl.load(X + 128 - RD + 2 * pair + 1, tail, 0).to(tl.float32)
        fr = tl.load(Freqs + frow * f_stride + 2 * pair, tail, 1.0)
        fi = tl.load(Freqs + frow * f_stride + 2 * pair + 1, tail, 0.0)
        rotated = tl.where(imaginary, real * fi + imag * fr, real * fr - imag * fi)
        value = tl.where(tail, rotated.to(tl.bfloat16).to(tl.float32), value)

        blocks = tl.reshape(value, (4, 32))
        amax = tl.maximum(tl.max(tl.abs(blocks), 1), AMAX_FLOOR) * (1.0 / 6.0)
        bits = amax.to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        fake_scale = (exponent << 23).to(tl.float32, bitcast=True)
        scaled = tl.minimum(tl.maximum(blocks / fake_scale[:, None], -6.0), 6.0)
        magnitude = tl.abs(scaled)
        step = tl.where(magnitude < 2.0, 0.5, tl.where(magnitude < 4.0, 1.0, 2.0))
        sign = tl.where(scaled > 0, 1.0, tl.where(scaled < 0, -1.0, 0.0))
        rounded = libdevice.rint(magnitude / step) * step * sign
        dequantized = (rounded * fake_scale[:, None]).to(tl.bfloat16).to(tl.float32)
        pack_amax = tl.max(tl.abs(dequantized), 1)
        pack_exponent = _ceil_ue8m0_exp(tl.maximum(pack_amax / 6.0, 1.0e-4))
        pack_scale = (pack_exponent << 23).to(tl.float32, bitcast=True)
        codes = _fp4_e2m1_code_rne(dequantized / pack_scale[:, None])
        low, high = tl.split(tl.reshape(codes, (64, 2)))
        payload = low | (high << 4)
        sf = tl.sum(pack_exponent.to(tl.uint32) << (tl.arange(0, 4) * 8), 0)
        page = out_loc // PAGE_SIZE
        slot = out_loc % PAGE_SIZE
        byte_offsets = tl.arange(0, 64)
        tl.store(Cache + page * CACHE_STRIDE + slot * 64 + byte_offsets, payload)
        scale_bytes = (sf >> (tl.arange(0, 4) * 8)) & 0xFF
        tl.store(
            Cache + page * CACHE_STRIDE + PAGE_SIZE * 64 + slot * 4 + tl.arange(0, 4),
            scale_bytes,
        )


def low_ratio_compress_store(
    *,
    ratio: int,
    kv: torch.Tensor,
    score: Optional[torch.Tensor],
    req: torch.Tensor,
    pos: torch.Tensor,
    raw_out_loc: torch.Tensor,
    out_loc: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    rope_dim: int,
    kv_cache: torch.Tensor,
    kv_page_size: int,
    state: Optional[torch.Tensor] = None,
    ring_size: int = 0,
    latent: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pair-pool (ratio 2), RMSNorm, RoPE + fp4 fake-quant and the V4 FlashMLA store.

    :param kv: ``[T, 512]`` fp32 at ratio 2 (``linear_bf16_fp32``), bf16 at ratio 1.
    :param score: ``[T, 512]`` fp32 at ratio 2, None at ratio 1.
    :param req, pos, raw_out_loc: ``[T]`` int, global token order.
    :param out_loc: ``[T]`` int64 ``c{ratio}_out_loc`` (-1: nothing to publish).
    :param freqs_cis: ``[max_pos, rope_dim // 2]`` complex64 layer table.
    :param kv_cache: the layer's compressed V4 pool buffer ``[pages, page_bytes]`` uint8.
    :param state: ratio 2: the ring ``kv_score`` buffer ``[rows, 1024]`` fp32. The ring
        write runs after the pooling launch, as in the torch path.
    :return: ``[T, 512]`` bf16, the pre-RoPE latent. Only publishing rows are written.
    """
    num_tokens, d = kv.shape
    assert d == 512 and rope_dim == 64 and kv.stride(1) == 1
    assert kv_cache.dtype == torch.uint8 and kv_cache.is_contiguous()
    page_bytes = v4_page_bytes(kv_page_size)
    assert kv_cache.shape[1] == page_bytes, (kv_cache.shape, page_bytes)
    if latent is None:
        latent = torch.empty((num_tokens, d), dtype=torch.bfloat16, device=kv.device)
    if num_tokens == 0:
        return latent
    freqs = torch.view_as_real(freqs_cis).flatten(-2)
    assert freqs.is_contiguous()
    if ratio == 2:
        assert score is not None and score.stride(1) == 1 and state is not None
        assert state.dtype == torch.float32 and state.shape[1] == 2 * d
        assert kv.dtype == torch.float32 and score.dtype == torch.float32
    else:
        assert ratio == 1 and kv.dtype == torch.bfloat16
        score, state, ring_size = kv, kv, 1
    _low_ratio_compress_kernel[(num_tokens,)](
        kv,
        score,
        kv.stride(0),
        score.stride(0),
        state,
        state.stride(0),
        req,
        pos,
        raw_out_loc,
        out_loc,
        norm_weight,
        freqs,
        freqs.stride(0),
        latent,
        kv_cache,
        ring_size,
        RATIO=ratio,
        D=d,
        RD=rope_dim,
        EPS=eps,
        AMAX_FLOOR=FP4_AMAX_FLOOR,
        PAGE_SIZE=kv_page_size,
        PAGE_BYTES=page_bytes,
        num_warps=4,
    )
    if ratio == 2:
        rows = 16
        _c2_ring_write_kernel[(triton.cdiv(num_tokens, rows),)](
            kv,
            score,
            kv.stride(0),
            score.stride(0),
            state,
            state.stride(0),
            state.shape[0],
            req,
            pos,
            raw_out_loc,
            num_tokens,
            ring_size,
            D=d,
            ROWS=rows,
            num_warps=4,
        )
    return latent


def index_k_norm_rope_pack_store_triton(
    *,
    k: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    rope_dim: int,
    pos: torch.Tensor,
    pos_offset: int,
    out_loc: torch.Tensor,
    cache: torch.Tensor,
) -> None:
    """``k_norm`` + ``index_keys`` + ``set_index_k_fp4`` (rne) for the rows with
    ``out_loc > 0``. ``k`` is ``wk(latent)`` ``[T, 128]`` bf16; the RoPE position is
    ``pos - pos_offset`` (the group position: 1 at ratio 2, 0 at ratio 1)."""
    num_tokens = k.shape[0]
    assert k.shape[1] == 128 and k.dtype == torch.bfloat16 and k.stride(1) == 1
    assert cache.ndim == 2 and cache.shape[1] % INDEX_K_SLOT_BYTES == 0
    if num_tokens == 0:
        return
    freqs = torch.view_as_real(freqs_cis).flatten(-2)
    normed = torch.empty((num_tokens, 128), dtype=torch.bfloat16, device=k.device)
    _index_k_norm_rope_pack_kernel[(num_tokens,)](
        k,
        k.stride(0),
        norm_weight,
        normed,
        freqs,
        freqs.stride(0),
        pos,
        out_loc,
        cache,
        POS_OFFSET=pos_offset,
        EPS=eps,
        RD=rope_dim,
        PAGE_SIZE=cache.shape[1] // INDEX_K_SLOT_BYTES,
        CACHE_STRIDE=cache.stride(0),
        AMAX_FLOOR=FP4_AMAX_FLOOR,
        num_warps=4,
    )
