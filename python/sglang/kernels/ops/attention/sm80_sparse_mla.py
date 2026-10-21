"""SM80 (A100) tensor-core sparse MLA attention for the DSv4 paged fp8 KV cache.

Replaces `_tiled_sparse_decode_kernel` (flash_mla_sm120_triton.py) whose per-(token, head)
program re-gathers the same MQA KV for every head and never touches tensor cores.

Design (prior art: dsa/triton_sparse_mla_prefill.py, see prior-art.md):
  * one program per (token, 16-head block, kv-split); heads share one KV gather and ride
    the M=16 mma.sync tile (`tl.dot`, bf16 in / fp32 acc).
  * KV nope bytes are e4m3 * 2^k (ue8m0) => exactly representable in bf16. Dequant is pure
    integer/exponent math on uint8 (Triton refuses fp8e4nv on SM80).
  * kv-split (grid z) + `_combine_kernel` for decode (few tokens, long topk) so more than
    8 SMs are busy. Prefill uses KV_SPLITS=1 and writes the final result directly.

Interface = `_run_triton_sparse_decode` of the original module (drop-in).
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_NOPE_DIM = 448
_ROPE_DIM = 64
_D = 512
_TOKEN_BYTES = 576
_LOG2E = 1.4426950408889634
_BLOCK_H = 16


@triton.jit
def _sparse_mla_sm80_kernel(
    Q_ptr,  # [N, H, 512] bf16
    cache_u8_ptr,  # uint8 flat view of paged cache
    cache_bf16_ptr,  # bf16 flat view of same memory
    idx_ptr,  # [N, topk] int32
    len_ptr,  # [N] int32 (or dummy)
    O_ptr,  # KV_SPLITS==1: [N, H, 512] bf16 ; else [S, N, H, 512] fp32
    LSE_ptr,  # KV_SPLITS==1: [N, H] ; else [S, N, H]  fp32
    sm_scale,
    page_size,
    page_bytes,
    scale_off,  # page_size * 576
    N,
    H,
    topk,
    stride_qn,
    stride_qh,
    HAS_LEN: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    LOG2E: tl.constexpr = 1.4426950408889634
    n = tl.program_id(0)
    hb = tl.program_id(1)
    sp = tl.program_id(2)
    n64 = n.to(tl.int64)

    h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = h < H
    dn = tl.arange(0, 512)
    nmask = dn < 448
    dr = tl.arange(0, 64)

    qb = Q_ptr + n64 * stride_qn + h[:, None].to(tl.int64) * stride_qh
    q_nope = tl.load(qb + dn[None, :], mask=hmask[:, None] & nmask[None, :], other=0.0)
    q_rope = tl.load(qb + 448 + dr[None, :], mask=hmask[:, None], other=0.0)

    valid_len = topk
    if HAS_LEN:
        valid_len = tl.minimum(tl.load(len_ptr + n), topk)

    # kv-split: contiguous token range of whole tiles per split
    n_tiles = tl.cdiv(valid_len, BLOCK_T)
    per = tl.cdiv(n_tiles, KV_SPLITS)
    t_lo = sp * per * BLOCK_T
    t_hi = tl.minimum(valid_len, (sp + 1) * per * BLOCK_T)

    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc_n = tl.zeros([BLOCK_H, 512], tl.float32)
    acc_r = tl.zeros([BLOCK_H, 64], tl.float32)

    group = dn // 64
    tt = tl.arange(0, BLOCK_T)
    for t0 in range(t_lo, t_hi, BLOCK_T):
        t = t0 + tt
        raw = tl.load(idx_ptr + n64 * topk + t, mask=t < t_hi, other=-1)
        valid = raw >= 0
        safe = tl.where(valid, raw, 0)
        pg = (safe // page_size).to(tl.int64)
        off = (safe % page_size).to(tl.int64)
        base = pg * page_bytes + off * 576  # [BT] int64 byte offset of token data

        # --- nope: uint8 -> bf16 (exact), scale 2^(s-127) ---
        a = base[:, None] + dn[None, :].to(tl.int64)
        m2 = valid[:, None] & nmask[None, :]
        b = tl.load(cache_u8_ptr + a, mask=m2, other=0).to(tl.int32)
        bits = ((b & 0x7F) << 20) | ((b & 0x80) << 24)
        v = bits.to(tl.float32, bitcast=True) * 1.329227995784916e36  # * 2^120
        sa = (pg * page_bytes + scale_off + off * 8)[:, None] + group[None, :].to(tl.int64)
        s = tl.load(cache_u8_ptr + sa, mask=m2, other=127).to(tl.int32)
        sf = (s << 23).to(tl.float32, bitcast=True)
        k_nope = (v * sf).to(tl.bfloat16)

        # --- rope: bf16 direct ---
        ra = ((base + 448) // 2)[:, None] + dr[None, :].to(tl.int64)
        k_rope = tl.load(cache_bf16_ptr + ra, mask=valid[:, None], other=0.0)

        qk = tl.dot(q_nope, tl.trans(k_nope))
        qk = tl.dot(q_rope, tl.trans(k_rope), qk) * sm_scale
        qk = tl.where(valid[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.math.exp2((m_i - m_safe) * LOG2E)
        p = tl.math.exp2((qk - m_safe[:, None]) * LOG2E)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        pb = p.to(tl.bfloat16)
        acc_n = acc_n * alpha[:, None] + tl.dot(pb, k_nope)
        acc_r = acc_r * alpha[:, None] + tl.dot(pb, k_rope)
        m_i = m_new

    lse = tl.where(
        l_i > 0.0,
        tl.where(m_i == -float("inf"), 0.0, m_i) + tl.log(tl.where(l_i > 0.0, l_i, 1.0)),
        -float("inf"),
    )
    inv = 1.0 / tl.where(l_i > 0.0, l_i, 1.0)
    acc_n = acc_n * inv[:, None]
    acc_r = acc_r * inv[:, None]

    if KV_SPLITS == 1:
        ob = O_ptr + (n64 * H + h[:, None].to(tl.int64)) * 512
        tl.store(ob + dn[None, :], acc_n.to(tl.bfloat16), mask=hmask[:, None] & nmask[None, :])
        tl.store(ob + 448 + dr[None, :], acc_r.to(tl.bfloat16), mask=hmask[:, None])
        tl.store(LSE_ptr + n64 * H + h, lse, mask=hmask)
    else:
        row = (sp * N + n64) * H + h[:, None].to(tl.int64)
        ob = O_ptr + row * 512
        tl.store(ob + dn[None, :], acc_n, mask=hmask[:, None] & nmask[None, :])
        tl.store(ob + 448 + dr[None, :], acc_r, mask=hmask[:, None])
        tl.store(LSE_ptr + (sp * N + n64) * H + h, lse, mask=hmask)


@triton.jit
def _combine_kernel(
    Op_ptr, Lp_ptr, O_ptr, LSE_ptr, N, H, KV_SPLITS: tl.constexpr
):
    n = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    d = tl.arange(0, 512)
    sidx = tl.arange(0, KV_SPLITS)
    lses = tl.load(Lp_ptr + (sidx * N + n) * H + h)  # [S]
    mx = tl.max(lses, axis=0)
    mx_safe = tl.where(mx == -float("inf"), 0.0, mx)
    w = tl.exp(lses - mx_safe)  # -inf -> 0
    tot = tl.sum(w, axis=0)
    o = tl.load(Op_ptr + ((sidx[:, None] * N + n) * H + h) * 512 + d[None, :])  # [S,512]
    o = tl.sum(o * w[:, None], axis=0) / tl.where(tot > 0.0, tot, 1.0)
    tl.store(O_ptr + (n * H + h) * 512 + d, o.to(tl.bfloat16))
    lse = tl.where(tot > 0.0, mx_safe + tl.log(tl.where(tot > 0.0, tot, 1.0)), -float("inf"))
    tl.store(LSE_ptr + n * H + h, lse)


def _pick_splits(n_tokens: int, hblocks: int, topk: int, block_t: int, num_sm: int = 108) -> int:
    tiles = max(1, triton.cdiv(topk, block_t))
    want = max(1, (2 * num_sm) // max(1, n_tokens * hblocks))
    s = 1
    while s * 2 <= min(want, tiles, 16):
        s *= 2
    return s


def run_sparse_mla_sm80(
    q: torch.Tensor,  # [N, 1, H, 512] bf16
    k_cache: torch.Tensor,  # [num_pages, page_size, 1, bpt]
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    softmax_scale: float,
    block_t: int = 32,
    num_warps: int = 8,
    num_stages: int = 2,
    kv_splits: Optional[int] = None,
    block_h: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for `_run_triton_sparse_decode`: returns ([N,1,H,512] bf16, [N,1,H] fp32 lse)."""
    N, _, H, D = q.shape
    page_size = k_cache.shape[1]
    page_bytes = k_cache.stride(0)
    flat_idx = indices.reshape(N, -1).contiguous()
    topk = flat_idx.shape[1]
    raw = k_cache.as_strided((k_cache.shape[0] * page_bytes,), (1,)).view(torch.uint8)
    raw_bf16 = raw.view(torch.bfloat16)
    q3 = q.squeeze(1)
    if not q3.is_contiguous():
        q3 = q3.contiguous()

    # 헤드 타일: prefill(H=8)은 16, H=64 는 32행 타일×2 (64 단일 타일은 CTA 부족으로 0.6x, sweep 결과)
    BH = block_h or max(16, min(32, triton.next_power_of_2(H)))
    hblocks = triton.cdiv(H, BH)
    S = kv_splits if kv_splits is not None else _pick_splits(N, hblocks, topk, block_t)
    out = torch.empty(N, H, _D, dtype=torch.bfloat16, device=q.device)
    lse = torch.empty(N, H, dtype=torch.float32, device=q.device)
    if S == 1:
        o_buf, l_buf = out, lse
    else:
        o_buf = torch.empty(S, N, H, _D, dtype=torch.float32, device=q.device)
        l_buf = torch.empty(S, N, H, dtype=torch.float32, device=q.device)
    len_t = topk_length if topk_length is not None else flat_idx
    _sparse_mla_sm80_kernel[(N, hblocks, S)](
        q3, raw, raw_bf16, flat_idx, len_t, o_buf, l_buf,
        softmax_scale, page_size, int(page_bytes), int(page_size * _TOKEN_BYTES),
        N, H, topk, q3.stride(0), q3.stride(1),
        HAS_LEN=topk_length is not None, KV_SPLITS=S, BLOCK_H=BH, BLOCK_T=block_t,
        num_warps=num_warps, num_stages=num_stages,
    )
    if S > 1:
        _combine_kernel[(N, H)](o_buf, l_buf, out, lse, N, H, KV_SPLITS=S, num_warps=4)
    return out.unsqueeze(1), lse.unsqueeze(1)
