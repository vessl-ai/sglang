"""Hopper paged decode indexer, candidate consumers: score and select only the
candidate blocks (SGLANG_OPT_DSV4_DECODE_INDEXER_CANDIDATES_ONLY).

The tensor-core consumer path (``fp4_index_logits_paged`` with a per-block mask,
then ``topk_transform_paged_v2`` over the whole row) scores every visible position
and writes ``-inf`` outside the source's kept blocks. Here each verify group (the
``rows_per_request`` rows of one request) first lists the union of its rows' kept
blocks in ascending order; the group is scored on that compacted axis only, with
the same operands, tile shape and rounding points as ``_tc_score_pair``, so each
score equals the full path's score at the same position. The top-k runs on the
compacted rows, which keep the positions' order, and the finish maps compacted
columns back to positions before the slot lookup.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    INDEX_HEAD_DIM,
    INDEX_K_PAYLOAD_BYTES,
    _dequant_index_k_prmt,
    _e2m1_bits,
    _load_pair_q,
    _use_prmt_dequant,
)

_CANDIDATE_BLOCK = 8
_UNION_CHUNK = 1024


@triton.jit
def _candidate_union_kernel(
    mask_ptr,  # [rows, num_blocks + 1] bool per-block keep flags (last column: padding)
    mask_stride,
    lens_ptr,  # [rows] int32 visible positions
    ublk_ptr,  # [groups, UCAP] int32 out: the group's kept blocks, ascending
    ucount_ptr,  # [groups] int32 out
    compact_lens_ptr,  # [rows] int32 out: compacted positions each row sees
    num_blocks,
    UCAP: tl.constexpr,
    GROUP: tl.constexpr,
    GROUP_P2: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    g = tl.program_id(0)
    offs_r = tl.arange(0, GROUP_P2)
    live_r = offs_r < GROUP
    rows = g * GROUP + offs_r
    lens = tl.load(lens_ptr + rows, mask=live_r, other=0)
    # the block of each row's newest position; the source always keeps it
    newest = tl.where(lens > 0, (lens - 1) // BLOCK, -1)
    below = tl.zeros((GROUP_P2,), tl.int32)
    carry = 0
    # Only blocks below the group's longest row can be kept: the mask is as wide as
    # the graph capacity (the model's context), far past any live row.
    live_blocks = tl.minimum(num_blocks, (tl.max(lens, axis=0) + BLOCK - 1) // BLOCK)
    for start in range(0, live_blocks, CHUNK):
        b = start + tl.arange(0, CHUNK)
        in_range = b < num_blocks
        flags = tl.load(
            mask_ptr + rows[:, None].to(tl.int64) * mask_stride + b[None, :],
            mask=live_r[:, None] & in_range[None, :],
            other=0,
        ).to(tl.int32)
        f = tl.max(flags, axis=0)  # [CHUNK] union of the group's rows
        cs = tl.cumsum(f, axis=0)
        dst = carry + cs - 1
        tl.store(
            ublk_ptr + g.to(tl.int64) * UCAP + dst,
            b,
            mask=(f > 0) & (dst < UCAP),
        )
        below += tl.sum(
            ((b[None, :] < newest[:, None]) & (f[None, :] > 0)).to(tl.int32), axis=1
        )
        carry += tl.sum(f, axis=0)
    tl.store(ucount_ptr + g, tl.minimum(carry, UCAP))
    compact_lens = tl.where(lens > 0, BLOCK * below + (lens - BLOCK * newest), 0)
    tl.store(compact_lens_ptr + rows, compact_lens, mask=live_r)


@triton.jit
def _cand_score_pair(
    k_lo,
    k_hi,
    q_even,
    q_odd,
    w_ptr,
    row0,
    compact_lens_ptr,
    mask_ptr,
    mask_stride,
    out_ptr,
    out_stride,
    offs_c,
    pos,
    offs_m,
    offs_r,
    CAP,
    H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # The same operations as _tc_score_pair (bf16 dot -> relu -> * bf16 weight
    # -> bf16 -> head sum -> bf16), on compacted columns.
    acc = tl.dot(k_lo, tl.trans(q_even))  # [BLOCK_N, 2H] fp32
    acc = tl.dot(k_hi, tl.trans(q_odd), acc)
    s = tl.maximum(acc.to(tl.bfloat16).to(tl.float32), 0.0)
    w = tl.load(w_ptr + row0 * H + offs_m).to(tl.float32)
    s = (s * w[None, :]).to(tl.bfloat16).to(tl.float32)
    logit = tl.sum(tl.reshape(s, (BLOCK_N, 2, H)), axis=2)
    logit = logit.to(tl.bfloat16).to(tl.float32)  # [BLOCK_N, 2]
    rows = row0 + offs_r
    n_vis = tl.load(compact_lens_ptr + rows)
    valid = offs_c[:, None] < n_vis[None, :]
    valid = valid & tl.load(
        mask_ptr + rows[None, :].to(tl.int64) * mask_stride + pos[:, None] // BLOCK,
        mask=valid,
        other=False,
    )
    tl.store(
        out_ptr + rows[None, :].to(tl.int64) * out_stride + offs_c[:, None],
        tl.where(valid, logit, float("-inf")),
        mask=(offs_c < CAP)[:, None],
    )


@triton.jit
def _fp4_index_logits_cand_tc_kernel(
    qs_ptr,  # [rows, H, 2, HALF_D] bf16: each head's even elements, then odd
    w_ptr,  # [rows, H] bf16
    req_ptr,
    req_table_ptr,
    ublk_ptr,
    ucount_ptr,
    compact_lens_ptr,
    chunk_end_ptr,
    table_ptr,
    mask_ptr,
    out_ptr,
    num_groups,
    CAP,
    row_stride,
    req_stride,
    mask_stride,
    out_stride,
    PAGE_SIZE: tl.constexpr,
    H: tl.constexpr,
    HALF_D: tl.constexpr,
    RATIO: tl.constexpr,
    GROUP: tl.constexpr,
    UCAP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK_TILES: tl.constexpr,
    BLOCK: tl.constexpr,
    PRMT_DEQUANT: tl.constexpr,
):
    # _fp4_index_logits_paged_tc_kernel's (row group, chunk) striding over each
    # group's compacted axis: compacted column c is position
    # ublk[g, c // BLOCK] * BLOCK + c % BLOCK.
    PAIRS: tl.constexpr = GROUP // 2
    tl.static_assert(GROUP % 2 == 0 and GROUP <= 6)
    M: tl.constexpr = 2 * H
    offs_i = tl.arange(0, HALF_D)
    offs_m = tl.arange(0, M)
    offs_r = tl.arange(0, 2)
    total = tl.load(chunk_end_ptr + num_groups - 1)
    for unit in range(tl.program_id(0), total, tl.num_programs(0)):
        lo = 0
        hi = num_groups - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if tl.load(chunk_end_ptr + mid) > unit:
                hi = mid
            else:
                lo = mid + 1
        g = lo
        first = tl.load(chunk_end_ptr + g - 1, mask=g > 0, other=0)
        n_group = tl.load(ucount_ptr + g) * BLOCK
        b0 = g * GROUP
        req = tl.load(req_ptr + b0).to(tl.int64)
        tile0 = (unit - first) * CHUNK_TILES
        tile1 = tl.minimum(tile0 + CHUNK_TILES, tl.cdiv(n_group, BLOCK_N))
        qe0, qo0 = _load_pair_q(qs_ptr, b0, offs_m, offs_i, H, HALF_D, True)
        qe1, qo1 = _load_pair_q(qs_ptr, b0 + 2, offs_m, offs_i, H, HALF_D, PAIRS > 1)
        qe2, qo2 = _load_pair_q(qs_ptr, b0 + 4, offs_m, offs_i, H, HALF_D, PAIRS > 2)
        for tile in range(tile0, tile1):
            offs_c = tile * BLOCK_N + tl.arange(0, BLOCK_N)
            in_group = offs_c < n_group
            blk = tl.load(
                ublk_ptr + g.to(tl.int64) * UCAP + offs_c // BLOCK,
                mask=in_group,
                other=0,
            )
            pos = blk * BLOCK + offs_c % BLOCK
            slot = (
                tl.load(
                    req_table_ptr + req * req_stride + pos * RATIO,
                    mask=in_group,
                    other=0,
                )
                // RATIO
            )
            slot = tl.maximum(slot, 0)
            page = (slot // PAGE_SIZE).to(tl.int64)
            off = slot % PAGE_SIZE
            base = page * row_stride
            pay = tl.load(
                table_ptr
                + base[:, None]
                + off[:, None] * INDEX_K_PAYLOAD_BYTES
                + offs_i[None, :],
                mask=in_group[:, None],
                other=0,
            )
            word_ptr = (
                table_ptr + base + PAGE_SIZE * INDEX_K_PAYLOAD_BYTES + off * 4
            ).to(tl.pointer_type(tl.int32))
            word = tl.load(word_ptr, mask=in_group, other=0x7F7F7F7F)
            if PRMT_DEQUANT:
                k_lo, k_hi = _dequant_index_k_prmt(pay, word, offs_i)
            else:
                pay = pay.to(tl.int32)
                exps = (word[:, None] >> ((offs_i[None, :] // 16) * 8)) & 0xFF
                scale = (exps << 23).to(tl.float32, bitcast=True)
                k_lo = (_e2m1_bits(pay & 0xF).to(tl.float32, bitcast=True) * scale).to(
                    tl.bfloat16
                )
                k_hi = (_e2m1_bits(pay >> 4).to(tl.float32, bitcast=True) * scale).to(
                    tl.bfloat16
                )
            for p in tl.static_range(PAIRS):
                if p == 0:
                    q_even, q_odd = qe0, qo0
                elif p == 1:
                    q_even, q_odd = qe1, qo1
                else:
                    q_even, q_odd = qe2, qo2
                _cand_score_pair(
                    k_lo,
                    k_hi,
                    q_even,
                    q_odd,
                    w_ptr,
                    b0 + 2 * p,
                    compact_lens_ptr,
                    mask_ptr,
                    mask_stride,
                    out_ptr,
                    out_stride,
                    offs_c,
                    pos,
                    offs_m,
                    offs_r,
                    CAP,
                    H,
                    BLOCK_N,
                    BLOCK,
                )


@triton.jit
def _finish_cand_topk_kernel(
    Indices,
    Scores,
    CompactLens,
    UBlk,
    Requests,
    ReqTable,
    Pages,
    BATCH: tl.constexpr,
    K: tl.constexpr,
    WIDTH: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    IDX_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    RATIO: tl.constexpr,
    GROUP: tl.constexpr,
    UCAP: tl.constexpr,
    BLOCK: tl.constexpr,
    SORT_BLOCK: tl.constexpr,
):
    # _finish_paged_indexer_topk_kernel (masked scores dropped, ascending, slots)
    # on compacted columns, mapped back to positions before the slot lookup.
    row = tl.program_id(0)
    col = tl.arange(0, SORT_BLOCK)
    idx = tl.load(Indices + row * IDX_STRIDE + col, (row < BATCH) & (col < K), WIDTH)
    valid = (row < BATCH) & (col < K) & (idx >= 0) & (idx < WIDTH)
    score = tl.load(Scores + row * SCORE_STRIDE + idx, valid, -float("inf"))
    idx = tl.where(valid & (score > -float("inf")), idx, WIDTH)
    idx = tl.sort(tl.where(idx >= 0, idx, WIDTH), descending=False)
    length = tl.load(CompactLens + row, row < BATCH, 0)
    req = tl.load(Requests + row, row < BATCH, 0)
    valid = (row < BATCH) & (col < K) & (idx < length)
    g = row // GROUP
    blk = tl.load(UBlk + g.to(tl.int64) * UCAP + idx // BLOCK, valid, 0)
    pos = blk * BLOCK + idx % BLOCK
    slot = tl.load(
        ReqTable + req.to(tl.int64) * TABLE_STRIDE + pos.to(tl.int64) * RATIO,
        valid,
        0,
    )
    tl.store(
        Pages + row * PAGE_STRIDE + col,
        tl.where(valid, slot.to(tl.int64) // RATIO, -1),
        col < OUT_WIDTH,
    )


def candidate_union(
    mask: torch.Tensor, lens: torch.Tensor, rows_per_request: int, topk_blocks: int
):
    """Per verify group the union of its rows' kept blocks, ascending: (ublk
    ``[groups, rows_per_request * topk_blocks]`` int32, ucount ``[groups]`` int32,
    compacted lengths ``[rows]`` int32)."""
    rows = lens.shape[0]
    assert rows % rows_per_request == 0 and mask.stride(1) == 1
    groups = rows // rows_per_request
    ucap = rows_per_request * topk_blocks
    ublk = torch.empty((groups, ucap), dtype=torch.int32, device=lens.device)
    ucount = torch.empty(groups, dtype=torch.int32, device=lens.device)
    compact_lens = torch.empty(rows, dtype=torch.int32, device=lens.device)
    if groups:
        _candidate_union_kernel[(groups,)](
            mask,
            mask.stride(0),
            lens,
            ublk,
            ucount,
            compact_lens,
            mask.shape[1] - 1,
            UCAP=ucap,
            GROUP=rows_per_request,
            GROUP_P2=triton.next_power_of_2(rows_per_request),
            BLOCK=_CANDIDATE_BLOCK,
            CHUNK=_UNION_CHUNK,
            num_warps=4,
        )
    return ublk, ucount, compact_lens


def fp4_index_logits_candidates(
    q: torch.Tensor,
    weights: torch.Tensor,
    req: torch.Tensor,
    req_table: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
    ratio: int,
    mask: torch.Tensor,
    ublk: torch.Tensor,
    ucount: torch.Tensor,
    compact_lens: torch.Tensor,
    rows_per_request: int,
) -> torch.Tensor:
    """fp32 scores ``[rows, ublk.shape[1] * 8]`` on each group's compacted axis;
    columns below ``compact_lens`` hold the full path's score at the mapped position
    (``-inf`` outside the row's own kept blocks), later columns ``-inf`` or unset."""
    from sglang.kernels.jit.utils import aligned_new_empty
    from sglang.kernels.ops.attention.deep_select import (
        get_input_stride_alignment_bytes,
    )

    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    assert table.dtype == torch.uint8 and table.ndim == 2 and req_table.stride(1) == 1
    batch, heads, _ = q.shape
    assert rows_per_request in (2, 4, 6) and heads in (16, 32)
    cap = ublk.shape[1] * _CANDIDATE_BLOCK
    out = aligned_new_empty(
        shape=(batch, cap),
        dtype=torch.float32,
        device=q.device,
        alignment_bytes=get_input_stride_alignment_bytes(),
    )
    if not batch:
        return out
    block_n, chunk_tiles = 128, 8
    chunk_len = block_n * chunk_tiles
    chunk_end = torch.cumsum(
        (ucount * _CANDIDATE_BLOCK + chunk_len - 1) // chunk_len,
        dim=0,
        dtype=torch.int32,
    )
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    _fp4_index_logits_cand_tc_kernel[(num_sms * 2,)](
        q.reshape(batch, heads, INDEX_HEAD_DIM // 2, 2).transpose(-1, -2).contiguous(),
        weights.to(torch.bfloat16).contiguous(),
        req.contiguous(),
        req_table,
        ublk,
        ucount,
        compact_lens,
        chunk_end,
        table,
        mask,
        out,
        batch // rows_per_request,
        cap,
        table.stride(0),
        req_table.stride(0),
        mask.stride(0),
        out.stride(0),
        PAGE_SIZE=page_size,
        H=heads,
        HALF_D=INDEX_HEAD_DIM // 2,
        RATIO=ratio,
        GROUP=rows_per_request,
        UCAP=ublk.shape[1],
        BLOCK_N=block_n,
        CHUNK_TILES=chunk_tiles,
        BLOCK=_CANDIDATE_BLOCK,
        PRMT_DEQUANT=_use_prmt_dequant(),
        num_warps=4,
        num_stages=3,
    )
    return out


def finish_candidate_topk(
    indices: torch.Tensor,
    scores: torch.Tensor,
    compact_lens: torch.Tensor,
    ublk: torch.Tensor,
    req: torch.Tensor,
    req_table: torch.Tensor,
    page_indices: torch.Tensor,
    ratio: int,
    rows_per_request: int,
) -> None:
    if not page_indices.shape[0]:
        return
    _finish_cand_topk_kernel[(page_indices.shape[0],)](
        indices,
        scores,
        compact_lens,
        ublk,
        req,
        req_table,
        page_indices,
        compact_lens.numel(),
        indices.shape[1],
        scores.shape[1],
        page_indices.shape[1],
        indices.stride(0),
        scores.stride(0),
        req_table.stride(0),
        page_indices.stride(0),
        ratio,
        rows_per_request,
        ublk.shape[1],
        _CANDIDATE_BLOCK,
        triton.next_power_of_2(max(page_indices.shape[1], indices.shape[1])),
        num_warps=4,
    )
