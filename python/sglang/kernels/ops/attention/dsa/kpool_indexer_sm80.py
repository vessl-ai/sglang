"""SM80 (A100) kernels for the k-pool DSA indexer.

Replaces the DeepGEMM calls in ``dsa_indexer_kpool.py`` that have no SM80 build:
  * ``fp8_mqa_logits``        -> :func:`fp8_mqa_logits_sm80`        (ragged prefill)
  * ``fp8_paged_mqa_logits``  -> :func:`fp8_paged_mqa_logits_sm80`  (decode / target_verify / draft_extend_v2)
  * ``get_paged_mqa_logits_metadata``  -> not needed (static grid, no schedule)
  * ``get_num_sms``                    -> ``get_device_core_count``
and the Triton ``act_quant`` whose fp32 -> fp8 store does not lower below SM89:
  * ``act_quant``             -> :func:`act_quant_sm80`

logits[m, n] = k_scale[n] * sum_h w[m, h] * relu(q[m, h] . k[n])

SM80 Triton cannot convert to/from ``float8e4nv``, so fp8 is handled as raw bytes:
an exact integer decode to bf16 (every e4m3fn value is representable in bf16, so
the bf16 ``tl.dot`` products are exact and accumulate in fp32) and a bit-exact
RNE saturating encoder for the q quantizer.

All entry points use fixed launch configs (no autotune) and derive the grid only
from tensor shapes, so the decode path is CUDA-graph capturable.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sglang.kernels.ops.attention.dsa.fp8_e4m3_sw import f32_to_e4m3fn_u8
from sglang.srt.utils import get_device_core_count

INDEX_PAGE_SIZE = 64
INDEX_HEAD_DIM = 128
# One index-K page = 64 tokens x 128 fp8 bytes, then 64 fp32 scales (8448 bytes).
_PAGE_SCALE_OFFSET = INDEX_PAGE_SIZE * INDEX_HEAD_DIM

# Launch configs, picked by a config sweep on A100-SXM4-80GB.
PREFILL_CFG = dict(BLOCK_M=2, BLOCK_N=128, num_warps=4, num_stages=3)
PAGED_CFG = dict(num_warps=4, num_stages=2)


# --------------------------------------------------------------------------- codec


@triton.jit
def _e4m3fn_u8_to_bf16(u):
    """e4m3fn byte -> exact bf16 (0x7F/0xFF decode to +-480 instead of NaN).

    The byte shifted into an fp16 bit pattern is the e4m3 value times 2^-8 for both
    normals and subnormals (fp16 has the same 3-bit mantissa alignment and bias
    15 = 7 + 8); fp16 -> fp32 is exact and *256 restores the value.
    """
    u16 = u.to(tl.uint16)
    h = (((u16 & 0x80) << 8) | ((u16 & 0x7F) << 7)).to(tl.uint16)
    f = h.to(tl.float16, bitcast=True).to(tl.float32) * 256.0
    return f.to(tl.bfloat16)


# --------------------------------------------------------------------------- act_quant


@triton.jit
def _act_quant_sm80_kernel(
    X_ptr,
    Y_ptr,
    S_ptr,
    M,
    N,
    group_size: tl.constexpr,
    round_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Same math as triton_kernel._act_quant_kernel; only the fp8 store differs.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * group_size + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    mask = row_mask[:, None] & (cols < N)[None, :]

    x = tl.load(X_ptr + rows[:, None] * N + cols[None, :], mask=mask, other=0.0)
    x = x.to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    if round_scale:
        scale = tl.exp2(tl.ceil(tl.log2(amax * fp8_max_inv)))
    else:
        scale = amax * fp8_max_inv

    y = libdevice.div_rn(x, scale[:, None])
    y = tl.minimum(tl.maximum(y, -fp8_max), fp8_max)
    tl.store(Y_ptr + rows[:, None] * N + cols[None, :], f32_to_e4m3fn_u8(y), mask=mask)
    tl.store(S_ptr + rows * (N // group_size) + pid_n, scale, mask=row_mask)


def act_quant_sm80(x: torch.Tensor, block_size: int = 128, scale_fmt=None):
    """Drop-in for ``triton_kernel.act_quant`` on SM80."""
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0
    N = x.size(-1)
    x_flat = x.view(-1, N)
    M = x_flat.size(0)
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    round_scale = scale_fmt is not None
    _act_quant_sm80_kernel[(triton.cdiv(M, 32), triton.cdiv(N, block_size))](
        x_flat,
        y.view(torch.uint8).view(-1, N),
        s.view(-1, N // block_size),
        M,
        N,
        group_size=block_size,
        round_scale=round_scale,
        BLOCK_M=32,
        BLOCK_N=block_size,
        num_stages=0 if round_scale else 2,
    )
    return y, s


# --------------------------------------------------------------------------- prefill


@triton.jit
def _mqa_logits_tile(
    q,
    w,
    ks,
    ke,
    rows_ok,
    n0,
    k_ptr,
    k_scale_ptr,
    out_ptrs_row,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    n_ok = offs_n < N
    k = tl.load(
        k_ptr + offs_n.to(tl.int64)[:, None] * D + offs_d[None, :],
        mask=n_ok[:, None],
        other=0,
    )
    k_scale = tl.load(k_scale_ptr + offs_n, mask=n_ok, other=0.0)
    s = tl.dot(q, tl.trans(_e4m3fn_u8_to_bf16(k)))  # [BLOCK_M*H, BLOCK_N] fp32
    s = tl.maximum(s, 0.0) * w[:, None]
    out = tl.sum(tl.reshape(s, (BLOCK_M, H, BLOCK_N)), axis=1) * k_scale[None, :]
    valid = (offs_n[None, :] >= ks[:, None]) & (offs_n[None, :] < ke[:, None])
    out = tl.where(valid, out, float("-inf"))
    tl.store(out_ptrs_row + offs_n[None, :], out, mask=rows_ok[:, None] & n_ok[None, :])


@triton.jit
def _fill_neg_inf(
    out_ptrs_row, rows_ok, n0, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    offs_n = n0 + tl.arange(0, BLOCK_N)
    n_ok = offs_n < N
    vals = tl.full((BLOCK_M, BLOCK_N), float("-inf"), tl.float32)
    tl.store(
        out_ptrs_row + offs_n[None, :], vals, mask=rows_ok[:, None] & n_ok[None, :]
    )


@triton.jit
def _fp8_mqa_logits_sm80_kernel(
    q_ptr,  # [M, H, D] uint8 (e4m3fn), contiguous
    k_ptr,  # [N, D] uint8 (e4m3fn), contiguous
    k_scale_ptr,  # [N] fp32
    w_ptr,  # [M, H] fp32
    stride_w_m,
    ks_ptr,
    ke_ptr,
    out_ptr,  # [M, N] fp32, row stride stride_out_m
    stride_out_m,
    M,
    N,
    tiles_per_prog,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # BLOCK_M query rows x H heads form the M axis of one mma, so a K tile is
    # shared by BLOCK_M rows; the head reduction then happens per row.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows_ok = rows < M
    ks = tl.load(ks_ptr + rows, mask=rows_ok, other=0)
    ke = tl.load(ke_ptr + rows, mask=rows_ok, other=0)
    ke = tl.minimum(ke, N)
    lo = tl.min(tl.where(rows_ok & (ke > ks), ks, N), axis=0)
    hi = tl.max(tl.where(rows_ok & (ke > ks), ke, 0), axis=0)

    n_tiles = tl.cdiv(N, BLOCK_N)
    t_begin = pid_n * tiles_per_prog
    t_end = tl.minimum(t_begin + tiles_per_prog, n_tiles)
    t_lo = tl.minimum(tl.maximum(t_begin, lo // BLOCK_N), t_end)
    t_hi = tl.maximum(tl.minimum(t_end, tl.cdiv(hi, BLOCK_N)), t_lo)

    out_ptrs_row = out_ptr + rows.to(tl.int64)[:, None] * stride_out_m

    for t in range(t_begin, t_lo):
        _fill_neg_inf(out_ptrs_row, rows_ok, t * BLOCK_N, N, BLOCK_M, BLOCK_N)
    for t in range(t_hi, t_end):
        _fill_neg_inf(out_ptrs_row, rows_ok, t * BLOCK_N, N, BLOCK_M, BLOCK_N)

    if t_hi > t_lo:
        offs_mh = pid_m * BLOCK_M * H + tl.arange(0, BLOCK_M * H)
        mh_ok = offs_mh < M * H
        offs_d = tl.arange(0, D)
        q = tl.load(
            q_ptr + offs_mh.to(tl.int64)[:, None] * D + offs_d[None, :],
            mask=mh_ok[:, None],
            other=0,
        )
        q = _e4m3fn_u8_to_bf16(q)
        w = tl.load(
            w_ptr + (offs_mh // H).to(tl.int64) * stride_w_m + offs_mh % H,
            mask=mh_ok,
            other=0.0,
        )
        for t in range(t_lo, t_hi):
            _mqa_logits_tile(
                q,
                w,
                ks,
                ke,
                rows_ok,
                t * BLOCK_N,
                k_ptr,
                k_scale_ptr,
                out_ptrs_row,
                N,
                H,
                D,
                BLOCK_M,
                BLOCK_N,
            )


def fp8_mqa_logits_sm80(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Drop-in for ``deep_gemm.fp8_mqa_logits(q, (k, k_scale), w, ks, ke)``.

    q_fp8 [M, H, D] e4m3fn, k_fp8 [N, D] e4m3fn, k_scale [N] fp32, weights [M, H] fp32,
    starts/ends [M] int32. Returns [M, N] fp32 logits, -inf outside [ks, ke) (the
    full row is always written, so ``clean_logits`` is satisfied either way).
    """
    M, H, D = q_fp8.shape
    N = k_fp8.shape[0]
    assert D == INDEX_HEAD_DIM and k_fp8.shape[1] == D
    assert weights.shape == (M, H) and weights.stride(1) == 1
    # Row stride padded to 4 floats like DeepGEMM/torch paths for the topk readers.
    storage = torch.empty(
        (M, (N + 3) // 4 * 4), dtype=torch.float32, device=q_fp8.device
    )
    logits = storage[:, :N]
    if M == 0 or N == 0:
        return logits
    q_u8 = q_fp8.contiguous().view(torch.uint8)
    k_u8 = k_fp8.contiguous().view(torch.uint8)
    BM, BN = PREFILL_CFG["BLOCK_M"], PREFILL_CFG["BLOCK_N"]
    row_blocks = triton.cdiv(M, BM)
    n_tiles = triton.cdiv(N, BN)
    # Split the key axis only when the row blocks alone cannot fill the GPU.
    want = 4 * get_device_core_count(q_fp8.device.index)
    n_split = max(1, min(n_tiles, triton.cdiv(want, row_blocks)))
    tiles_per_prog = triton.cdiv(n_tiles, n_split)
    n_split = triton.cdiv(n_tiles, tiles_per_prog)
    _fp8_mqa_logits_sm80_kernel[(row_blocks, n_split)](
        q_u8,
        k_u8,
        k_scale.reshape(-1).contiguous(),
        weights,
        weights.stride(0),
        starts.to(torch.int32).contiguous(),
        ends.to(torch.int32).contiguous(),
        logits,
        logits.stride(0),
        M,
        N,
        tiles_per_prog,
        H=H,
        D=D,
        BLOCK_M=BM,
        BLOCK_N=BN,
        num_warps=PREFILL_CFG["num_warps"],
        num_stages=PREFILL_CFG["num_stages"],
    )
    return logits


# --------------------------------------------------------------------------- paged decode


@triton.jit
def _fp8_paged_mqa_logits_sm80_kernel(
    q_ptr,  # [R, H, D] uint8, row stride stride_q
    stride_q,
    kv_ptr,  # index-K buffer, uint8, page stride stride_page (bytes)
    stride_page,
    w_ptr,  # [R, H] fp32
    stride_w,
    ctx_ptr,  # [R] int32 pooled context length per row
    bt_ptr,  # [R, P] int32 pooled page table
    stride_bt,
    out_ptr,  # [R, P*64] fp32
    stride_out,
    P,
    PAGES_PER_PROG: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    SCALE_OFFSET: tl.constexpr,
):
    r = tl.program_id(0)
    pid = tl.program_id(1)
    ctx = tl.load(ctx_ptr + r)
    p_begin = pid * PAGES_PER_PROG
    p_end = tl.minimum(tl.minimum(p_begin + PAGES_PER_PROG, P), tl.cdiv(ctx, PAGE))
    if p_end <= p_begin:
        return

    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    offs_s = tl.arange(0, PAGE)
    q = tl.load(
        q_ptr + r.to(tl.int64) * stride_q + offs_h[:, None] * D + offs_d[None, :]
    )
    q = _e4m3fn_u8_to_bf16(q)
    w = tl.load(w_ptr + r.to(tl.int64) * stride_w + offs_h)
    out_row = out_ptr + r.to(tl.int64) * stride_out

    for p in range(p_begin, p_end):
        page = tl.load(bt_ptr + r.to(tl.int64) * stride_bt + p).to(tl.int64)
        base = kv_ptr + page * stride_page
        k = tl.load(base + offs_s[:, None] * D + offs_d[None, :])
        k_scale = tl.load(
            (base + SCALE_OFFSET).to(tl.pointer_type(tl.float32)) + offs_s
        )
        s = tl.dot(q, tl.trans(_e4m3fn_u8_to_bf16(k)))  # [H, PAGE]
        out = tl.sum(tl.maximum(s, 0.0) * w[:, None], axis=0) * k_scale
        pos = p * PAGE + offs_s
        tl.store(out_row + pos, tl.where(pos < ctx, out, float("-inf")))


def fp8_paged_mqa_logits_sm80(
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Drop-in for ``deep_gemm.fp8_paged_mqa_logits`` as the k-pool indexer calls it
    (next_n == 1, one context length per row).

    q_fp8 [R, 1, H, D] e4m3fn, kv_cache [pages, 64, 1, 132] uint8, weights [R, H] fp32,
    context_lens [R] or [R, 1] int32, block_tables [R, P] int32, max_seq_len = P*64.
    Writes [0, ceil(ctx/64)*64) of each row (-inf past ctx); columns past the last
    live page are left unwritten, as DeepGEMM does with clean_logits=False.
    """
    R, next_n, H, D = q_fp8.shape
    assert next_n == 1, "k-pool indexer always calls with next_n == 1"
    assert D == INDEX_HEAD_DIM and kv_cache.shape[1] == INDEX_PAGE_SIZE
    assert kv_cache.shape[-1] == D + 4
    assert block_tables.shape[0] == R and block_tables.stride(1) == 1
    P = block_tables.shape[1]
    assert max_seq_len == P * INDEX_PAGE_SIZE
    logits = torch.empty((R, max_seq_len), dtype=torch.float32, device=q_fp8.device)
    if R == 0 or P == 0:
        return logits
    kv_u8 = kv_cache.view(torch.uint8).reshape(kv_cache.shape[0], -1)
    assert kv_u8.stride(1) == 1
    q_u8 = q_fp8.view(torch.uint8)[:, 0]
    assert q_u8.stride(2) == 1 and q_u8.stride(1) == D
    assert weights.shape == (R, H) and weights.stride(1) == 1
    ctx = context_lens.reshape(R)
    assert ctx.dtype == torch.int32
    # Enough programs to cover the SMs; each one decodes q once for its pages.
    ppp = 1
    while ppp < 16 and R * triton.cdiv(P, ppp * 2) >= 2 * get_device_core_count(
        q_fp8.device.index
    ):
        ppp *= 2
    _fp8_paged_mqa_logits_sm80_kernel[(R, triton.cdiv(P, ppp))](
        q_u8,
        q_u8.stride(0),
        kv_u8,
        kv_u8.stride(0),
        weights,
        weights.stride(0),
        ctx.contiguous(),
        block_tables,
        block_tables.stride(0),
        logits,
        logits.stride(0),
        P,
        PAGES_PER_PROG=ppp,
        H=H,
        D=D,
        PAGE=INDEX_PAGE_SIZE,
        SCALE_OFFSET=_PAGE_SCALE_OFFSET,
        num_warps=PAGED_CFG["num_warps"],
        num_stages=PAGED_CFG["num_stages"],
    )
    return logits
