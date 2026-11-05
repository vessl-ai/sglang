# SPDX-License-Identifier: Apache-2.0
# 일부(union dedup 커널·ownership bitmask 구조)는 SGLang `dsa/triton_sparse_mla_prefill.py`(Apache-2.0)에서 가져와 D_TAIL=0(V4: d_qk=d_v=512)·attn_sink 융합으로 변형했다.
"""SM80 (A100) DSv4 sparse-MLA prefill, bf16 워크스페이스 경로 (env `SGLANG_SM80_SPARSE_PREFILL_WS`, 기본 OFF).

현 A100 prefill 은 `_tiled`/`sm80_sparse_mla` 커널이 query 토큰마다 fp8 KV 를 gather+dequant 하고(같은 compressed
엔트리를 수백 query 가 각자 재-dequant), SWA/extra 를 두 번 launch 한 뒤 torch 로 LSE merge + attn_sink 를 적용한다.
이 모듈은 upstream 의 `_forward_prefill_sparse` 구조를 SM80 에 여는 두 부품을 제공한다.

  1. `dequantize_k_cache_paged_sm80` — V4 paged fp8 KV(584B/token)의 필요한 슬롯만 bf16 `[n,1,512]` 워크스페이스로 1회 dequant.
     (upstream 커널은 `fp8e4nv` load 를 써서 SM80 Triton 이 컴파일 거부. 여기선 uint8 정수 연산, bit-exact.)
  2. `sparse_mla_prefill_sm80` — bf16 워크스페이스 + 재기준화된 combined index([compressed topk | SWA window | -1 pad], 길이
     `topk_length`) 위에서 토큰당 1 program(헤드 8개를 16행 MMA 타일에 패딩)으로 한 번에 softmax(+attn_sink 융합)까지 수행.
     `union_g` 2|4 는 인접 토큰이 gather 를 공유하는 opt-in (정렬+dedup 오버헤드가 있어 기본 OFF, μbench 로 판정).

인터페이스는 `sgl_kernel.flash_mla.flash_mla_sparse_fwd` 와 같은 의미(q [T,H,512], kv [S,512] bf16, indices [T,K] int32,
-1=무효, d_v=512=d_qk)에 attn_sink·topk_length 를 포함한다.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

DIM_NOPE = 448
DIM_ROPE = 64
D = 512
NOPE_ROPE_BYTES = 576
PADDED_SCALE_PER_TOKEN = 8
_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453


# ---------------------------------------------------------------------------
# 1. dequant: paged V4 fp8 KV -> bf16 workspace rows (uint8 정수 연산, SM80 컴파일 가능)
# ---------------------------------------------------------------------------
@triton.jit
def _dequant_v4_sm80_kernel(
    out_ptr,  # bf16 [n, out_stride]
    u8_ptr,  # uint8 flat view of paged cache
    bf16_ptr,  # bf16 flat view of the same memory
    ids_ptr,  # [n] int32/int64 slot ids
    n,
    out_stride,
    BYTES_PER_PAGE,
    PAGE_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    m = rows < n
    loc = tl.load(ids_ptr + rows, mask=m, other=0).to(tl.int64)
    page = loc // PAGE_SIZE
    inp = loc % PAGE_SIZE
    data = page * BYTES_PER_PAGE + inp * 576
    sbase = page * BYTES_PER_PAGE + PAGE_SIZE * 576 + inp * 8
    orow = rows.to(tl.int64) * out_stride
    c = tl.arange(0, 64)
    for g in tl.static_range(7):
        b = tl.load(u8_ptr + data[:, None] + g * 64 + c[None, :], mask=m[:, None], other=0).to(tl.int32)
        # e4m3 -> fp32 비트 재배치(지수 e 를 fp32 지수 자리에) 후 2^120 곱: 정규/비정규 모두 정확
        bits = ((b & 0x7F) << 20) | ((b & 0x80) << 24)
        v = bits.to(tl.float32, bitcast=True) * 1.329227995784916e36
        s = tl.load(u8_ptr + sbase + g, mask=m, other=127).to(tl.int32)
        sc = (s << 23).to(tl.float32, bitcast=True)  # 2^(s-127) 정확(ex2.approx 의 LSB 불확실성 회피; v1.1/v1.2 와 동일, s=0 은 0 으로 처리)
        tl.store(out_ptr + orow[:, None] + g * 64 + c[None, :], (v * sc[:, None]).to(tl.bfloat16), mask=m[:, None])
    rope = tl.load(bf16_ptr + ((data + 448) // 2)[:, None] + c[None, :], mask=m[:, None], other=0.0)
    tl.store(out_ptr + orow[:, None] + 448 + c[None, :], rope, mask=m[:, None])


def dequantize_k_cache_paged_sm80(
    quant_k_cache: torch.Tensor,
    token_ids: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
    layout=None,
) -> torch.Tensor:
    """`dsv4.dequant_k_cache.dequantize_k_cache_paged`(V4 레이아웃) 의 SM80 대체. out: [n,1,512] bf16 (슬라이스 가능).
    layout: V4(584B/token fp8) 는 dequant, V4_BF16(P2 bf16 풀)은 행 gather, 그 외(V4.1 fp8/fp4)는 거부."""
    lname = str(getattr(layout, "name", layout)).upper() if layout is not None else "V4"
    if lname.endswith("V4_BF16"):
        # [P2] V4_BF16 풀: 토큰 행 = 연속 bf16 512개(스케일·패딩 없음, page_bytes = page_size*1024) -> dequant 불필요, 행 gather 만.
        rows = quant_k_cache.view(torch.uint8).view(torch.bfloat16).reshape(-1, D)
        n = token_ids.shape[0]
        if out is None:
            out = torch.empty((n, 1, D), dtype=torch.bfloat16, device=quant_k_cache.device)
        else:
            assert out.shape == (n, 1, D) and out.dtype == torch.bfloat16
        if n:
            torch.index_select(rows, 0, token_ids.long(), out=out.view(n, D))
        return out
    assert lname.endswith("V4"), f"SM80 dequant: V4 / V4_BF16 layout only, got {layout}"
    assert quant_k_cache.is_contiguous()
    assert token_ids.dtype in (torch.int32, torch.int64)
    u8 = quant_k_cache.view(torch.uint8)
    n = token_ids.shape[0]
    bytes_per_page = u8.shape[-1]
    if out is None:
        out = torch.empty((n, 1, D), dtype=torch.bfloat16, device=quant_k_cache.device)
    else:
        assert out.shape == (n, 1, D) and out.dtype == torch.bfloat16
    if n == 0:
        return out
    block_t = 4
    _dequant_v4_sm80_kernel[(triton.cdiv(n, block_t),)](
        out,
        u8.reshape(-1),
        u8.view(torch.bfloat16).reshape(-1),
        token_ids,
        n,
        out.stride(0),
        bytes_per_page,
        PAGE_SIZE=page_size,
        BLOCK_T=block_t,
        num_warps=2,
    )
    return out


# ---------------------------------------------------------------------------
# 2. prefill attention: token 당 1 program (G=1)
# ---------------------------------------------------------------------------
@triton.jit
def _softmax_epilogue(acc, m_i, l_i, sink2, HAS_SINK: tl.constexpr):
    m_safe = tl.where(m_i == -float("inf"), 0.0, m_i)
    l_tot = l_i
    if HAS_SINK:
        l_tot = l_i + tl.exp2(sink2 - m_safe)
    l_safe = tl.where(l_tot > 0.0, l_tot, 1.0)
    lse = tl.where(l_tot > 0.0, (m_safe + tl.log2(l_safe)) * 0.6931471805599453, -float("inf"))
    return acc / l_safe[:, None], lse


@triton.jit
def _prefill_kernel(
    q_ptr,  # [T, H, 512] bf16 (strided)
    kv_ptr,  # [S, 512] bf16
    idx_ptr,  # [T, topk] int32
    len_ptr,  # [T] int32
    sink_ptr,  # [H] fp32 (or dummy)
    o_ptr,  # [T, H, 512] bf16
    lse_ptr,  # [T, H] fp32
    scale2,  # sm_scale * log2(e)
    topk,
    stride_qt,
    stride_qh,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BLOCK_H)
    hmask = h < H
    d = tl.arange(0, 512)
    q = tl.load(q_ptr + t * stride_qt + h[:, None] * stride_qh + d[None, :], mask=hmask[:, None], other=0.0)

    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, 512], tl.float32)

    n = tl.arange(0, BLOCK_N)
    k_len = tl.load(len_ptr + t)
    for k0 in range(0, k_len, BLOCK_N):
        idx = tl.load(idx_ptr + t * topk + k0 + n, mask=(k0 + n) < k_len, other=-1)
        valid = idx >= 0
        row = tl.where(valid, idx, 0).to(tl.int64)
        kv = tl.load(kv_ptr + row[:, None] * 512 + d[None, :], mask=valid[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(kv)) * scale2
        qk = tl.where(valid[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new

    sink2 = tl.zeros([BLOCK_H], tl.float32)
    if HAS_SINK:
        sink2 = tl.load(sink_ptr + h, mask=hmask, other=0.0) * 1.4426950408889634
    acc, lse = _softmax_epilogue(acc, m_i, l_i, sink2, HAS_SINK)
    tl.store(o_ptr + (t * H + h[:, None]) * 512 + d[None, :], acc.to(o_ptr.dtype.element_ty), mask=hmask[:, None])
    tl.store(lse_ptr + t * H + h, lse, mask=hmask)


# ---------------------------------------------------------------------------
# 3. union (G=2|4): 인접 토큰이 정렬+dedup 된 gather 를 공유 (opt-in)
# ---------------------------------------------------------------------------
@triton.jit
def _union_dedup_kernel(
    sorted_ptr, uidx_ptr, ubits_ptr, ulen_ptr, K: tl.constexpr, G: tl.constexpr, LOG_K: tl.constexpr
):
    # upstream 과 동일: 그룹당 1 program, 각 토큰 행은 오름차순 정렬(-1 패드가 앞). token i 는 이전 토큰이 고르지 않은 행만
    # 내보내고, ownership 비트는 그 행을 고른 모든 토큰을 OR.
    g = tl.program_id(0).to(tl.int64)
    grp = sorted_ptr + g * G * K
    n = tl.arange(0, K)
    cursor = tl.zeros([], tl.int32)
    for i in tl.static_range(G):
        v = tl.load(grp + i * K + n)
        valid = v >= 0
        bits = tl.where(valid, 1 << i, 0).to(tl.int32)
        first = valid
        for j in tl.static_range(G):
            if j != i:
                lst = grp + j * K
                lo = tl.zeros([K], tl.int32)
                hi = tl.full([K], K, tl.int32)
                for _ in tl.static_range(LOG_K):
                    mid = (lo + hi) // 2
                    mv = tl.load(lst + mid, mask=mid < K, other=2147483647)
                    lt = mv < v
                    lo = tl.where(lt, mid + 1, lo)
                    hi = tl.where(lt, hi, mid)
                found = (tl.load(lst + lo, mask=lo < K, other=-2) == v) & valid
                bits |= tl.where(found, 1 << j, 0).to(tl.int32)
                if j < i:
                    first = first & (found == 0)
        wpos = cursor + tl.cumsum(first.to(tl.int32), axis=0) - first.to(tl.int32)
        tl.store(uidx_ptr + g * (G * K) + wpos, v, mask=first)
        tl.store(ubits_ptr + g * (G * K) + wpos, bits, mask=first)
        cursor += tl.sum(first.to(tl.int32))
    tl.store(ulen_ptr + g, cursor)


@triton.jit
def _prefill_union_kernel(
    q_ptr,
    kv_ptr,
    uidx_ptr,
    ubits_ptr,
    ulen_ptr,
    sink_ptr,
    o_ptr,
    lse_ptr,
    scale2,
    U_STRIDE,
    stride_qt,
    stride_qh,
    H: tl.constexpr,
    G: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    g = tl.program_id(0).to(tl.int64)
    GH: tl.constexpr = G * H
    r = tl.arange(0, GH)
    tok = r // H
    hh = r % H
    d = tl.arange(0, 512)
    q = tl.load(q_ptr + (g * G + tok[:, None]) * stride_qt + hh[:, None] * stride_qh + d[None, :])

    m_i = tl.full([GH], -float("inf"), tl.float32)
    l_i = tl.zeros([GH], tl.float32)
    acc = tl.zeros([GH, 512], tl.float32)

    n = tl.arange(0, BLOCK_N)
    u_len = tl.load(ulen_ptr + g)
    ub = g * U_STRIDE
    for k0 in range(0, u_len, BLOCK_N):
        inb = (k0 + n) < u_len
        uidx = tl.load(uidx_ptr + ub + k0 + n, mask=inb, other=-1)
        bits = tl.load(ubits_ptr + ub + k0 + n, mask=inb, other=0)
        valid = uidx >= 0
        row = tl.where(valid, uidx, 0).to(tl.int64)
        kv = tl.load(kv_ptr + row[:, None] * 512 + d[None, :], mask=valid[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(kv)) * scale2
        sel = ((bits[None, :] >> tok[:, None]) & 1) != 0
        qk = tl.where(sel & valid[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new

    sink2 = tl.zeros([GH], tl.float32)
    if HAS_SINK:
        sink2 = tl.load(sink_ptr + hh) * 1.4426950408889634
    acc, lse = _softmax_epilogue(acc, m_i, l_i, sink2, HAS_SINK)
    tl.store(o_ptr + ((g * G + tok[:, None]) * H + hh[:, None]) * 512 + d[None, :], acc.to(o_ptr.dtype.element_ty))
    tl.store(lse_ptr + (g * G + tok) * H + hh, lse)


def _union_prep(idx: torch.Tensor, lens: torch.Tensor, G: int):
    """idx [T_main, K0] -> (uidx, ubits, ulen). K 는 2의 거듭제곱으로 -1 패딩, 각 행은 lens 이후를 -1 로 가린 뒤 정렬."""
    T_main, K0 = idx.shape
    K = triton.next_power_of_2(K0)
    pos = torch.arange(K0, device=idx.device, dtype=torch.int32)[None, :]
    masked = torch.where(pos < lens[:T_main, None], idx, torch.full_like(idx, -1))
    if K != K0:
        masked = torch.nn.functional.pad(masked, (0, K - K0), value=-1)
    srt = torch.sort(masked, dim=1)[0]
    NG = T_main // G
    uidx = torch.empty(NG, G * K, dtype=torch.int32, device=idx.device)
    ubits = torch.empty_like(uidx)
    ulen = torch.empty(NG, dtype=torch.int32, device=idx.device)
    _union_dedup_kernel[(NG,)](
        srt, uidx, ubits, ulen, K=K, G=G, LOG_K=K.bit_length(), num_warps=16 if G == 4 else 8
    )
    return uidx, ubits, ulen


# (BLOCK_N, num_warps, num_stages) 후보: A100 smem 164KB. bf16 KV 타일 BLOCK_N×512×2B = 64KB(BN=64)/stage.
# 앞에서부터 시도, OutOfResources 면 다음. 실측 순위는 bench_sm80_sparse_prefill_ws.py --sweep 로 확정.
# 오프라인 ptxas(sm_80): 64/8w/2s 219reg·112KB(1 CTA/SM), 32/8w/3s 148reg·64KB, 16/8w/3s 126reg·40KB(2 CTA/SM), 4warp 계열은 reg 241~255/스필 위험.
# GPU μbench(2026-10-20, A100): (32,4,2) 가 전 층 최속(8w 대비 1.3~1.5x), 이어서 (32,8,3).
_TILE_CANDIDATES = ((32, 4, 2), (32, 8, 3), (16, 4, 3), (64, 8, 2))
_FIT = {}


def _launch(fn, grid, key, cfg, **kw):
    cands = [cfg] if cfg is not None else ([_FIT[key]] if key in _FIT else list(_TILE_CANDIDATES))
    last = None
    for bn, w, s in cands:
        try:
            fn[grid](BLOCK_N=bn, num_warps=w, num_stages=s, **kw)
        except triton.runtime.errors.OutOfResources as e:
            last = e
            continue
        _FIT[key] = (bn, w, s)
        return
    raise last


def sparse_mla_prefill_sm80(
    q: torch.Tensor,  # [T, H, 512] bf16 (헤드 stride 허용)
    kv: torch.Tensor,  # [S, 512] bf16 contiguous (또는 [S,1,512])
    indices: torch.Tensor,  # [T, K] int32 (또는 [T,1,K]); -1 = 무효
    sm_scale: float,
    attn_sink: Optional[torch.Tensor] = None,  # [H] fp32 (자연로그 도메인 sink logit)
    topk_length: Optional[torch.Tensor] = None,  # [T] int32
    union_g: int = 1,
    tile: Optional[Tuple[int, int, int]] = None,  # (BLOCK_N, num_warps, num_stages)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """returns (out [T,H,512] bf16, lse [T,H] fp32 natural log; 전부 무효인 행은 out=0, lse=-inf)."""
    T, H, dq = q.shape
    # fp32/fp16 은 Triton 인터프리터(CPU) 논리 테스트용 (인터프리터의 bf16 tl.dot 은 오동작)
    assert dq == D and kv.shape[-1] == D and kv.dtype == q.dtype
    kv2 = kv.reshape(-1, D)
    assert kv2.is_contiguous()
    idx = indices.reshape(T, -1)
    if not idx.is_contiguous():
        idx = idx.contiguous()
    K = idx.shape[1]
    out = torch.empty(T, H, D, dtype=q.dtype, device=q.device)
    lse = torch.empty(T, H, dtype=torch.float32, device=q.device)
    if T == 0:
        return out, lse
    lens = topk_length if topk_length is not None else torch.full((T,), K, dtype=torch.int32, device=q.device)
    has_sink = attn_sink is not None
    sink = attn_sink.to(torch.float32).contiguous() if has_sink else lens  # dummy ptr
    scale2 = float(sm_scale) * _LOG2E
    BH = max(16, triton.next_power_of_2(H))

    T_main = 0
    if union_g > 1 and union_g * H in (16, 32):
        T_main = (T // union_g) * union_g
    if T_main:
        NG = T_main // union_g
        uidx, ubits, ulen = _union_prep(idx[:T_main], lens, union_g)
        _launch(
            _prefill_union_kernel, (NG,), ("u", H, union_g), tile,
            q_ptr=q, kv_ptr=kv2, uidx_ptr=uidx, ubits_ptr=ubits, ulen_ptr=ulen, sink_ptr=sink,
            o_ptr=out, lse_ptr=lse, scale2=scale2, U_STRIDE=uidx.stride(0),
            stride_qt=q.stride(0), stride_qh=q.stride(1), H=H, G=union_g, HAS_SINK=has_sink,
        )
    if T_main < T:
        r = T - T_main
        _launch(
            _prefill_kernel, (r,), ("p", H), tile,
            q_ptr=q[T_main:], kv_ptr=kv2, idx_ptr=idx[T_main:], len_ptr=lens[T_main:], sink_ptr=sink,
            o_ptr=out[T_main:], lse_ptr=lse[T_main:], scale2=scale2, topk=K,
            stride_qt=q.stride(0), stride_qh=q.stride(1), H=H, BLOCK_H=BH, HAS_SINK=has_sink,
        )
    return out, lse
