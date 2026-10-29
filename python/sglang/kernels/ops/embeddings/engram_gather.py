"""Triton gather of DeepSeek-V4.1 engram rows: fp8 e4m3 payload, e8m0 block scales.

The table pointers arrive as raw addresses so one kernel serves a device table, a
pinned host table, or (on Grace-Blackwell, through ATS) a plain host mapping. The
output is bf16 computed as fp32(row) * 2**(exp - 127) then rounded once, which is
the arithmetic of the torch lookup it replaces.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

# e8m0 has no zero: the exponent byte 0 encodes 2**-127.
_E8M0_ZERO = 2.0**-127


@triton.jit
def _engram_gather_kernel(
    w_ptr,
    s_ptr,
    ids_ptr,
    out_ptr,
    row_lo,
    row_hi,
    DIM: tl.constexpr,
    BLK: tl.constexpr,
    E8M0_ZERO: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    # The table holds rows [row_lo, row_hi); an id outside it is not read and
    # comes out as zeros, which is what the sharded all-reduce sums.
    owned = (idx >= row_lo) & (idx < row_hi)
    local = tl.where(owned, idx - row_lo, 0)
    w = w_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    s = s_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    offs = tl.arange(0, DIM)
    vals = tl.load(w + local * DIM + offs, mask=owned, other=0.0).to(tl.float32)
    exps = tl.load(s + local * (DIM // BLK) + offs // BLK, mask=owned, other=0).to(
        tl.int32
    )
    # 2**(e - 127) from the exponent bits: exact, no exp2 rounding or denormal flush.
    scale = (exps << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exps == 0, E8M0_ZERO, scale)
    out = tl.where(owned, vals * scale, 0.0)
    tl.store(out_ptr + row * DIM + offs, out.to(tl.bfloat16))


def engram_gather(
    weight_ptr: int,
    scale_ptr: int,
    ids: torch.Tensor,
    out: torch.Tensor,
    dim: int,
    block_size: int,
    row_lo: int = 0,
    row_hi: int = 2**62,
) -> torch.Tensor:
    """Gather rows ``ids`` ([N] int) into ``out`` ([N, dim] bf16, contiguous).

    ``weight_ptr`` addresses [rows, dim] fp8 e4m3 bytes and ``scale_ptr``
    [rows, dim // block_size] e8m0 bytes for global rows [row_lo, row_hi); both
    may live in device or host memory. Ids outside the range produce zero rows.
    """
    assert dim > 0 and dim & (dim - 1) == 0 and block_size > 0, (dim, block_size)
    assert dim % block_size == 0, (dim, block_size)
    assert (
        ids.is_cuda and ids.is_contiguous() and ids.dtype in (torch.int32, torch.int64)
    )
    assert out.is_contiguous() and out.dtype == torch.bfloat16
    assert out.device == ids.device and out.shape == (ids.numel(), dim), out.shape
    n = ids.numel()
    if n:
        _engram_gather_kernel[(n,)](
            weight_ptr,
            scale_ptr,
            ids,
            out,
            row_lo,
            row_hi,
            DIM=dim,
            BLK=block_size,
            E8M0_ZERO=_E8M0_ZERO,
        )
    return out


# Raw gather + separate dequant (SGLANG_OPT_DSV4_ENGRAM_PREFETCH). The raw buffer is
# laid out in `chunks`, one per receiving rank; chunk k holds lookups
# [k * rows_per_chunk, (k + 1) * rows_per_chunk) as a payload region
# [rows_per_chunk, dim] fp8 bytes followed by a scale region
# [rows_per_chunk, dim // block_size] e8m0 bytes. Unowned lookups are zero bytes,
# so an integer sum over the shards (all-reduce or reduce-scatter) is exact and
# hands every rank the owner's bytes for its own chunk.


def engram_raw_chunk_bytes(rows_per_chunk: int, dim: int, block_size: int) -> int:
    return rows_per_chunk * (dim + dim // block_size)


@triton.jit
def _engram_gather_raw_kernel(
    w_ptr,
    s_ptr,
    ids_ptr,
    order_ptr,
    out_ptr,
    n_rows,
    rows_per_chunk,
    row_lo,
    row_hi,
    DIM_WORDS: tl.constexpr,
    SCALE_WORDS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    ORDERED: tl.constexpr,
):
    # Persistent: a small grid leaves SMs to the stream it overlaps. Each program
    # copies BLOCK_R rows per iteration as 4-byte words (16-byte vectors). With
    # ORDERED the rows are visited in `order` (ids sorted), so the programs in
    # flight touch few table pages at a time; each row still lands at its own slot.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    w = w_ptr.to(tl.int64).to(tl.pointer_type(tl.int32))
    s = s_ptr.to(tl.int64).to(tl.pointer_type(tl.int32))
    words = tl.arange(0, DIM_WORDS)
    swords = tl.arange(0, SCALE_WORDS)
    n_blocks = tl.cdiv(n_rows, BLOCK_R)
    for blk in range(pid, n_blocks, nprog):
        r = blk.to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
        valid = r < n_rows
        if ORDERED:
            r = tl.load(order_ptr + r, mask=valid, other=0).to(tl.int64)
        idx = tl.load(ids_ptr + r, mask=valid, other=0).to(tl.int64)
        owned = valid & (idx >= row_lo) & (idx < row_hi)
        local = tl.where(owned, idx - row_lo, 0)
        vals = tl.load(
            w + local[:, None] * DIM_WORDS + words[None, :],
            mask=owned[:, None],
            other=0,
        )
        svals = tl.load(
            s + local[:, None] * SCALE_WORDS + swords[None, :],
            mask=owned[:, None],
            other=0,
        )
        # int64 from r onwards: a TP4 65k step is ~1.6M rows of 66 words.
        chunk = r // rows_per_chunk
        j = r - chunk * rows_per_chunk
        base = chunk * rows_per_chunk * (DIM_WORDS + SCALE_WORDS)
        tl.store(
            out_ptr + (base + j * DIM_WORDS)[:, None] + words[None, :],
            vals,
            mask=valid[:, None],
        )
        sbase = base + j * SCALE_WORDS + rows_per_chunk * DIM_WORDS
        tl.store(
            out_ptr + sbase[:, None] + swords[None, :], svals, mask=valid[:, None]
        )


def engram_gather_raw(
    weight_ptr: int,
    scale_ptr: int,
    ids: torch.Tensor,
    out: torch.Tensor,
    dim: int,
    block_size: int,
    rows_per_chunk: int,
    row_lo: int = 0,
    row_hi: int = 2**62,
    num_programs: int = 0,
    block_rows: int = 16,
    order: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Copy the fp8 payload and e8m0 scale bytes of rows ``ids`` ([N] int) into the
    chunked raw buffer ``out`` (uint8, ``N // rows_per_chunk`` chunks of
    ``engram_raw_chunk_bytes`` each, contiguous). Ids outside [row_lo, row_hi) give
    zero bytes. ``num_programs`` 0 = one program per SM. ``order`` ([N], a
    permutation, e.g. argsort of ``ids``) is the visiting order only; the output is
    the same with or without it."""
    assert dim % block_size == 0, (dim, block_size)
    scale_bytes = dim // block_size
    assert dim % 4 == 0 and scale_bytes % 4 == 0, (dim, scale_bytes)
    dim_words, scale_words = dim // 4, scale_bytes // 4
    assert dim_words & (dim_words - 1) == 0, dim
    assert scale_words & (scale_words - 1) == 0, scale_bytes
    assert weight_ptr % 16 == 0 and scale_ptr % 4 == 0, (weight_ptr, scale_ptr)
    assert (
        ids.is_cuda and ids.is_contiguous() and ids.dtype in (torch.int32, torch.int64)
    )
    n = ids.numel()
    assert rows_per_chunk > 0 and n % rows_per_chunk == 0, (n, rows_per_chunk)
    assert out.is_contiguous() and out.dtype == torch.uint8 and out.device == ids.device
    assert out.numel() == (n // rows_per_chunk) * engram_raw_chunk_bytes(
        rows_per_chunk, dim, block_size
    ), (out.shape, n, rows_per_chunk)
    if n:
        if num_programs <= 0:
            num_programs = torch.cuda.get_device_properties(
                ids.device
            ).multi_processor_count
        grid = (max(1, min(num_programs, triton.cdiv(n, block_rows))),)
        if order is not None:
            assert order.is_cuda and order.is_contiguous() and order.numel() == n
            assert order.dtype in (torch.int32, torch.int64)
        _engram_gather_raw_kernel[grid](
            weight_ptr,
            scale_ptr,
            ids,
            ids if order is None else order,
            out.view(torch.int32),
            n,
            rows_per_chunk,
            row_lo,
            row_hi,
            DIM_WORDS=dim_words,
            SCALE_WORDS=scale_words,
            BLOCK_R=block_rows,
            ORDERED=order is not None,
            num_warps=4,
        )
    return out


@triton.jit
def _engram_dequant_raw_kernel(
    raw_ptr,
    out_ptr,
    n_rows,
    scale_off,
    DIM: tl.constexpr,
    BLK: tl.constexpr,
    E8M0_ZERO: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    valid = r < n_rows
    offs = tl.arange(0, DIM)
    payload = raw_ptr.to(tl.pointer_type(tl.float8e4nv))
    vals = tl.load(
        payload + r[:, None] * DIM + offs[None, :], mask=valid[:, None], other=0.0
    ).to(tl.float32)
    exps = tl.load(
        raw_ptr + (r[:, None] * (DIM // BLK) + scale_off) + (offs // BLK)[None, :],
        mask=valid[:, None],
        other=0,
    ).to(tl.int32)
    # The arithmetic of _engram_gather_kernel, so the bf16 values are the same.
    scale = (exps << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exps == 0, E8M0_ZERO, scale)
    out = (vals * scale).to(tl.bfloat16)
    # The bf16 all-reduce over zero-padded shards turns -0 into +0 (-0 + +0 = +0).
    bits = out.to(tl.int16, bitcast=True)
    bits = tl.where(bits == -32768, tl.zeros_like(bits), bits)
    tl.store(
        out_ptr + r[:, None] * DIM + offs[None, :],
        bits.to(tl.bfloat16, bitcast=True),
        mask=valid[:, None],
    )


def engram_dequant_raw(
    raw: torch.Tensor,
    out: torch.Tensor,
    dim: int,
    block_size: int,
    block_rows: int = 16,
) -> torch.Tensor:
    """Dequantize one chunk of an ``engram_gather_raw`` buffer (uint8, payload then
    scales for ``out.shape[0]`` rows) into ``out`` ([rows, dim] bf16): the values of
    ``engram_gather`` followed by the sharded bf16 all-reduce, bit for bit."""
    assert dim > 0 and dim & (dim - 1) == 0 and dim % block_size == 0, (
        dim,
        block_size,
    )
    assert out.is_contiguous() and out.dtype == torch.bfloat16 and out.ndim == 2
    n = out.shape[0]
    assert out.shape[1] == dim, out.shape
    assert raw.is_contiguous() and raw.dtype == torch.uint8 and raw.device == out.device
    assert raw.numel() == engram_raw_chunk_bytes(n, dim, block_size), (raw.shape, n)
    if n:
        _engram_dequant_raw_kernel[(triton.cdiv(n, block_rows),)](
            raw,
            out,
            n,
            n * dim,
            DIM=dim,
            BLK=block_size,
            E8M0_ZERO=_E8M0_ZERO,
            BLOCK_R=block_rows,
            num_warps=4,
        )
    return out
