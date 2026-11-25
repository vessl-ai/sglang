"""Prefill-CP window KV through an FP8 all-gather.

Today every CP rank gathers the raw bf16 kv of all ranks (1,024 B/row), restores
the token order with ``index_select`` and runs ``fused_k_norm_rope_flashmla`` over
all T rows. RMSNorm, RoPE and the FP8 quantization are per row, so each rank can
instead run the same kernel on its own rows only, into a send buffer laid out as
V4 FlashMLA pages of 64 rows (584 B/row plus 64 B of padding per page), all-gather
that, and copy each row into the paged SWA cache. The cache bytes are those of the
current path: the same kernel computes every row, only the page it writes differs.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from .low_ratio_compress_triton import v4_page_bytes

# Rows per page of the send buffer; fused_k_norm_rope_flashmla needs a power of two.
SEND_PAGE_SIZE = 64
SEND_PAGE_BYTES = v4_page_bytes(SEND_PAGE_SIZE)


def send_buffer_pages(rank_rows: int) -> int:
    return triton.cdiv(rank_rows, SEND_PAGE_SIZE)


@triton.jit
def _scatter_v4_rows_kernel(
    Src,
    GatherIndex,
    DstLoc,
    Dst,
    num_tokens,
    rank_rows,
    pages_per_rank,
    CP: tl.constexpr,
    HAS_GATHER_INDEX: tl.constexpr,
    SRC_PAGE: tl.constexpr,
    SRC_PAGE_WORDS: tl.constexpr,
    DST_PAGE: tl.constexpr,
    DST_PAGE_WORDS: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Copy gathered row ``gather_index[t]`` (rank ``r``, local row ``j``) to SWA slot
    ``dst_loc[t]``: the 144 data words and the 7 scale bytes the store writes (the pad
    byte is never written). ``dst_loc < 0`` rows are skipped, as in the store."""
    t = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    live = t < num_tokens
    if HAS_GATHER_INDEX:
        src = tl.load(GatherIndex + t, mask=live, other=0).to(tl.int64)
    else:
        src = (t % CP) * rank_rows + t // CP
    r = src // rank_rows
    j = src - r * rank_rows
    spage = r * pages_per_rank + j // SRC_PAGE
    sslot = j % SRC_PAGE
    loc = tl.load(DstLoc + t, mask=live, other=-1).to(tl.int64)
    ok = live & (loc >= 0)
    dpage = loc // DST_PAGE
    dslot = loc - dpage * DST_PAGE

    w = tl.arange(0, 256)[None, :]
    m = ok[:, None] & (w < 144)
    src_words = Src.to(tl.pointer_type(tl.int32))
    dst_words = Dst.to(tl.pointer_type(tl.int32))
    data = tl.load(src_words + (spage * SRC_PAGE_WORDS + sslot * 144)[:, None] + w, mask=m)
    tl.store(dst_words + (dpage * DST_PAGE_WORDS + dslot * 144)[:, None] + w, data, mask=m)

    b = tl.arange(0, 8)[None, :]
    ms = ok[:, None] & (b < 7)
    s_off = spage * SRC_PAGE_WORDS * 4 + SRC_PAGE * 576 + sslot * 8
    d_off = dpage * DST_PAGE_WORDS * 4 + DST_PAGE * 576 + dslot * 8
    tl.store(Dst + d_off[:, None] + b, tl.load(Src + s_off[:, None] + b, mask=ms), mask=ms)


def scatter_gathered_window_kv(
    gathered: torch.Tensor,
    gather_index: Optional[torch.Tensor],
    dst_loc: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_page_size: int,
    *,
    num_tokens: int,
    rank_rows: int,
    cp_size: int,
) -> None:
    """``gathered``: ``[cp_size * pages_per_rank, SEND_PAGE_BYTES]`` uint8, rank r's
    send buffer at pages ``[r * pages_per_rank, (r + 1) * pages_per_rank)``. Token t is
    gathered row ``gather_index[t]`` (row ``r * rank_rows + j``), or, without an
    index, the equal-split interleave ``(t % cp_size) * rank_rows + t // cp_size``."""
    if num_tokens == 0:
        return
    assert gathered.dtype == torch.uint8 and gathered.is_contiguous()
    assert gathered.shape[1] == SEND_PAGE_BYTES
    pages_per_rank = gathered.shape[0] // cp_size
    assert pages_per_rank * cp_size == gathered.shape[0]
    assert pages_per_rank * SEND_PAGE_SIZE >= rank_rows
    dst_page_bytes = v4_page_bytes(swa_page_size)
    assert swa_cache.dtype == torch.uint8 and swa_cache.is_contiguous()
    assert swa_cache.shape[-1] == dst_page_bytes, (swa_cache.shape, dst_page_bytes)
    # 4 rows on one warp measured fastest on H200 (22.5 us for 64,968 rows; floor 17.8)
    rows = 4
    _scatter_v4_rows_kernel[(triton.cdiv(num_tokens, rows),)](
        gathered,
        gather_index if gather_index is not None else dst_loc,
        dst_loc,
        swa_cache,
        num_tokens,
        rank_rows,
        pages_per_rank,
        CP=cp_size,
        HAS_GATHER_INDEX=gather_index is not None,
        SRC_PAGE=SEND_PAGE_SIZE,
        SRC_PAGE_WORDS=SEND_PAGE_BYTES // 4,
        DST_PAGE=swa_page_size,
        DST_PAGE_WORDS=dst_page_bytes // 4,
        ROWS=rows,
        num_warps=1,
    )
