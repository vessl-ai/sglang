# Adapted from deepseek-ai/DeepInfer (Apache-2.0),
# deepinfer/model_executor/layers/sm90_ops/indexer_keys_major.py and indexer.py.
"""Keys-major SM90 prefill index logits over e4m3 re-encodings of the fp4 index-K pool
and the queries. The re-encodings and their power-of-two scales are exact, so the logits
equal the bf16 reference whenever the fp8 wgmma accumulates the 128-wide dot exactly;
its accumulator keeps fewer bits than fp32, so wide exponent spreads can round."""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    INDEX_HEAD_DIM,
    INDEX_K_PAYLOAD_BYTES,
    INDEX_K_SCALE_BYTES,
    _e2m1_decode,
)

# wgmma M over four warpgroups. Measured on H200; re-time when the kernel changes.
BLOCK_KEYS = 256
NUM_WARPS = 16
STAGES = 2
# wgmma N. A pass keeps HEADS_PER_PASS heads of a row block in shared memory:
# 32 heads * 32 rows * 128 B = 128 KiB next to STAGES * 32 KiB of keys.
BLOCK_ROWS = 32
HEADS_PER_PASS = 32


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@triton.jit
def _fp4_index_k_to_e4m3_kernel(
    table_ptr,
    slots_ptr,
    k8_ptr,  # [L, 128] e4m3
    sk_ptr,  # [L] fp32
    L,
    page_size,
    row_stride,
    HALF_D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    offs_l = tl.program_id(0) * BLOCK_L + tl.arange(0, BLOCK_L)
    valid = offs_l < L
    slot = tl.load(slots_ptr + offs_l, mask=valid, other=0)
    row_base = (slot // page_size) * row_stride
    off = slot % page_size
    offs_i = tl.arange(0, HALF_D)
    pay = tl.load(
        table_ptr
        + row_base[:, None]
        + off[:, None] * INDEX_K_PAYLOAD_BYTES
        + offs_i[None, :],
        mask=valid[:, None],
        other=0,
    )
    exps = tl.load(
        table_ptr
        + row_base[:, None]
        + page_size * INDEX_K_PAYLOAD_BYTES
        + off[:, None] * INDEX_K_SCALE_BYTES
        + offs_i[None, :] // 16,
        mask=valid[:, None],
        other=127,
    ).to(tl.int32)
    emax = tl.max(exps, axis=1)
    # e2m1 * 2**(e - emax + 6) is at most 384 and exact in e4m3 while the row's block
    # exponents span at most 14 binades.
    shift = tl.exp2((exps - emax[:, None] + 6).to(tl.float32))
    low = _e2m1_decode(pay & 0x0F) * shift
    high = _e2m1_decode((pay >> 4) & 0x0F) * shift
    k = tl.join(low, high).reshape(BLOCK_L, 2 * HALF_D)
    offs_d = tl.arange(0, 2 * HALF_D)
    tl.store(
        k8_ptr + offs_l[:, None].to(tl.int64) * (2 * HALF_D) + offs_d[None, :],
        k.to(tl.float8e4nv),
        mask=valid[:, None],
    )
    tl.store(sk_ptr + offs_l, tl.exp2((emax - 133).to(tl.float32)), mask=valid)


def _fp4_index_k_to_e4m3(
    table: torch.Tensor, slots: torch.Tensor, page_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    L = slots.shape[0]
    k8 = torch.empty(
        (L, INDEX_HEAD_DIM), dtype=torch.float8_e4m3fn, device=table.device
    )
    sk = torch.empty((L,), dtype=torch.float32, device=table.device)
    BLOCK_L = 64
    _fp4_index_k_to_e4m3_kernel[(triton.cdiv(L, BLOCK_L),)](
        table,
        slots,
        k8,
        sk,
        L,
        page_size,
        table.stride(0),
        HALF_D=INDEX_HEAD_DIM // 2,
        BLOCK_L=BLOCK_L,
        num_warps=4,
    )
    return k8, sk


@triton.jit
def _q_to_e4m3_kernel(
    q_ptr,  # [M, H, 128] bf16
    w_ptr,  # [M, H]
    q8_ptr,  # [M, H, 128] e4m3
    wq_ptr,  # [M, H] bf16
    H: tl.constexpr,
    D: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    offs_h = tl.arange(0, H)
    offs = m * H * D + offs_h[:, None] * D + tl.arange(0, D)[None, :]
    q = tl.load(q_ptr + offs).to(tl.float32)
    amax = tl.max(tl.abs(q), axis=1)
    # e = floor(log2(amax)) - 8 puts the head's largest |q| in [256, 512).
    e = ((amax.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 135
    e = tl.where(amax > 0, e, 0).to(tl.float32)
    tl.store(q8_ptr + offs, (q * tl.exp2(-e)[:, None]).to(tl.float8e4nv))
    w = tl.load(w_ptr + m * H + offs_h).to(tl.float32)
    tl.store(wq_ptr + m * H + offs_h, (w * tl.exp2(e)).to(tl.bfloat16))


def _q_to_e4m3(
    q: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    M, H, D = q.shape
    q8 = torch.empty((M, H, D), dtype=torch.float8_e4m3fn, device=q.device)
    wq = torch.empty((M, H), dtype=torch.bfloat16, device=q.device)
    _q_to_e4m3_kernel[(M,)](
        q.contiguous(), weights.contiguous(), q8, wq, H=H, D=D, num_warps=4
    )
    return q8, wq


@triton.jit
def _visible_tiles_kernel(
    lens_ptr,
    tiles_ptr,  # [n_row_blocks + 1] int32 prefix of visible key tiles per row block
    M,
    n_row_blocks,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CHUNK: tl.constexpr,
):
    offs_c = tl.arange(0, CHUNK)
    offs_r = tl.arange(0, BLOCK_R)
    tl.store(tiles_ptr, 0)
    carry = tl.zeros([], dtype=tl.int32)
    for c0 in range(0, n_row_blocks, CHUNK):
        rbs = c0 + offs_c
        rows = rbs[:, None] * BLOCK_R + offs_r[None, :]
        lens = tl.load(lens_ptr + rows, mask=rows < M, other=0)
        tiles = ((tl.max(lens, axis=1) + BLOCK_K - 1) // BLOCK_K).to(tl.int32)
        tiles = tl.where(rbs < n_row_blocks, tiles, 0)
        tl.store(
            tiles_ptr + 1 + rbs,
            tl.cumsum(tiles, axis=0) + carry,
            mask=rbs < n_row_blocks,
        )
        carry += tl.sum(tiles, axis=0)


@triton.jit
def _fill_hidden_tiles_kernel(
    tiles_ptr,
    out_ptr,
    M,
    L,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    kt = tl.program_id(0)
    rb = tl.program_id(1)
    if kt >= tl.load(tiles_ptr + rb + 1) - tl.load(tiles_ptr + rb):
        rows = rb * BLOCK_R + tl.arange(0, BLOCK_R)
        keys = kt * BLOCK_K + tl.arange(0, BLOCK_K)
        tl.store(
            out_ptr + rows[:, None].to(tl.int64) * L + keys[None, :],
            float("-inf"),
            mask=(rows[:, None] < M) & (keys[None, :] < L),
        )


@gluon.jit
def _last_row_block_at(prefix_ptr, n_row_blocks, v):
    lo = 0
    hi = n_row_blocks
    while hi - lo > 1:
        mid = (lo + hi) // 2
        right = gl.load(prefix_ptr + mid) <= v
        lo = gl.where(right, mid, lo)
        hi = gl.where(right, hi, mid)
    return lo


@gluon.jit
def _issue_head(k0, k1, k2, k3, q_head, acc):
    acc = warpgroup_mma(
        k0,
        q_head.slice(0, 32, dim=1).permute((1, 0)),
        acc,
        use_acc=False,
        is_async=True,
    )
    acc = warpgroup_mma(
        k1, q_head.slice(32, 32, dim=1).permute((1, 0)), acc, is_async=True
    )
    acc = warpgroup_mma(
        k2, q_head.slice(64, 32, dim=1).permute((1, 0)), acc, is_async=True
    )
    return warpgroup_mma(
        k3, q_head.slice(96, 32, dim=1).permute((1, 0)), acc, is_async=True
    )


@gluon.jit
def _head_epilogue(acc, s, w):
    # acc += rn_bf16(relu(rn_bf16(dot)) * w) on bf16x2 pairs; bf16 * bf16 is exact
    # in fp32, so mul.rn.bf16x2 rounds the same product the reference rounds.
    return gl.inline_asm_elementwise(
        """
        {
        .reg .b32 t;
        cvt.rn.relu.bf16x2.f32 t, $3, $2;
        mul.rn.bf16x2 t, t, $4;
        shl.b32 $0, t, 16;
        and.b32 $1, t, 0xffff0000;
        add.f32 $0, $5, $0;
        add.f32 $1, $6, $1;
        }
        """,
        "=r,=r,r,r,r,r,r",
        [s, w[None, :], acc],
        dtype=gl.float32,
        is_pure=True,
        pack=2,
    )


@gluon.jit
def _load_row_block(
    q_desc,
    q_bar,
    q_smem,
    wq_ptr,
    w_smem,
    lens_ptr,
    M,
    rb,
    q_phase,
    offs_r0,
    offs_w,
    H: gl.constexpr,
    H_TOTAL: gl.constexpr,
    HEAD0: gl.constexpr,
    D: gl.constexpr,
    BLOCK_R: gl.constexpr,
):
    rows = rb * BLOCK_R + offs_r0
    lens = gl.load(lens_ptr + rows, mask=rows < M, other=0)
    # w_smem[h * BLOCK_R + r] = weight of (row r, head h)
    offs_wr = rb * BLOCK_R + offs_w % BLOCK_R
    w_blk = gl.load(
        wq_ptr + offs_wr.to(gl.int64) * H_TOTAL + HEAD0 + offs_w // BLOCK_R,
        mask=offs_wr < M,
        other=0.0,
    )
    w_smem.store(w_blk)
    mbarrier.expect(q_bar, H * BLOCK_R * D)
    for h in gl.static_range(H):
        tma.async_copy_global_to_shared(
            q_desc, [rb * BLOCK_R, (HEAD0 + h) * D], q_bar, q_smem.index(h)
        )
    mbarrier.wait(q_bar, q_phase)
    return lens


@gluon.jit
def _score_tile(
    k_tile,
    q_smem,
    w_smem,
    hz,
    acc,
    a,
    b,
    H: gl.constexpr,
    BLOCK_R: gl.constexpr,
    row_layout: gl.constexpr,
):
    # The key tile is read once into registers and reused by every head's wgmmas.
    k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=a.type.layout, k_width=4
    )
    k0 = k_tile.slice(0, 32, dim=1).load(k_layout)
    k1 = k_tile.slice(32, 32, dim=1).load(k_layout)
    k2 = k_tile.slice(64, 32, dim=1).load(k_layout)
    k3 = k_tile.slice(96, 32, dim=1).load(k_layout)
    # Unrolled and double-buffered over heads so head h's epilogue overlaps head
    # h + 1's wgmmas; a runtime head loop makes ptxas serialize every wgmma.
    a = _issue_head(k0, k1, k2, k3, q_smem.index(hz), a)
    for h in gl.static_range(0, H - 2, 2):
        b = _issue_head(k0, k1, k2, k3, q_smem.index(hz + h + 1), b)
        w = w_smem.slice(h * BLOCK_R, BLOCK_R).load(row_layout)
        a = warpgroup_mma_wait(4, deps=[a])
        acc = _head_epilogue(acc, a, w)
        a = _issue_head(k0, k1, k2, k3, q_smem.index(hz + h + 2), a)
        w = w_smem.slice((h + 1) * BLOCK_R, BLOCK_R).load(row_layout)
        b = warpgroup_mma_wait(4, deps=[b])
        acc = _head_epilogue(acc, b, w)
    b = _issue_head(k0, k1, k2, k3, q_smem.index(hz + H - 1), b)
    w = w_smem.slice((H - 2) * BLOCK_R, BLOCK_R).load(row_layout)
    a = warpgroup_mma_wait(4, deps=[a])
    acc = _head_epilogue(acc, a, w)
    w = w_smem.slice((H - 1) * BLOCK_R, BLOCK_R).load(row_layout)
    b = warpgroup_mma_wait(0, deps=[b])
    acc = _head_epilogue(acc, b, w)
    return acc, a, b


@gluon.jit
def _prefetch_keys(
    k_desc,
    k_bar,
    k_smem,
    tiles_ptr,
    n_row_blocks,
    v,
    stage,
    p_lo,
    p_hi,
    BLOCK_K: gl.constexpr,
    D: gl.constexpr,
):
    if v >= p_hi:
        rb = _last_row_block_at(tiles_ptr, n_row_blocks, v)
        p_lo = gl.load(tiles_ptr + rb)
        p_hi = gl.load(tiles_ptr + rb + 1)
    mbarrier.expect(k_bar.index(stage), BLOCK_K * D)
    tma.async_copy_global_to_shared(
        k_desc, [(v - p_lo) * BLOCK_K, 0], k_bar.index(stage), k_smem.index(stage)
    )
    return p_lo, p_hi


@gluon.jit
def _keys_major_kernel(
    q_desc,  # [M, H_TOTAL * D] e4m3, block [BLOCK_R, D]
    k_desc,  # [L, D] e4m3, block [BLOCK_K, D]
    wq_ptr,  # [M, H_TOTAL] bf16
    sk_ptr,  # [L] fp32
    lens_ptr,  # [M]
    tiles_ptr,  # [n_row_blocks + 1] int32 from _visible_tiles_kernel
    out_ptr,  # [M, L] fp32; carries the raw head sum between passes
    M,
    L,
    n_row_blocks,
    H: gl.constexpr,
    H_TOTAL: gl.constexpr,
    HEAD0: gl.constexpr,
    FIRST: gl.constexpr,
    LAST: gl.constexpr,
    D: gl.constexpr,
    BLOCK_R: gl.constexpr,
    BLOCK_K: gl.constexpr,
    STAGES: gl.constexpr,
    num_warps: gl.constexpr,
):
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, BLOCK_R, 32]
    )
    key_layout: gl.constexpr = gl.SliceLayout(1, mma_layout)
    row_layout: gl.constexpr = gl.SliceLayout(0, mma_layout)
    w_layout: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0])

    q_smem = gl.allocate_shared_memory(gl.float8e4nv, [H, BLOCK_R, D], q_desc.layout)
    k_smem = gl.allocate_shared_memory(
        gl.float8e4nv, [STAGES, BLOCK_K, D], k_desc.layout
    )
    # Flattened [H, BLOCK_R]: per-head slices need constexpr offsets.
    w_smem = gl.allocate_shared_memory(
        gl.bfloat16, [H * BLOCK_R], gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    )
    q_bar = mbarrier.allocate_mbarrier()
    k_bar = mbarrier.allocate_mbarrier(batch=STAGES)
    mbarrier.init(q_bar, count=1)
    for s in gl.static_range(STAGES):
        mbarrier.init(k_bar.index(s), count=1)

    pid = gl.program_id(0)
    grid = gl.num_programs(0)
    n_vis_tiles = gl.load(tiles_ptr + n_row_blocks)
    v0 = (pid.to(gl.int64) * n_vis_tiles // grid).to(gl.int32)
    v1 = ((pid.to(gl.int64) + 1) * n_vis_tiles // grid).to(gl.int32)
    p_lo = 0
    p_hi = 0
    for s in gl.static_range(STAGES):
        if v0 + s < v1:
            p_lo, p_hi = _prefetch_keys(
                k_desc,
                k_bar,
                k_smem,
                tiles_ptr,
                n_row_blocks,
                v0 + s,
                s,
                p_lo,
                p_hi,
                BLOCK_K,
                D,
            )

    a = gl.zeros([BLOCK_K, BLOCK_R], gl.float32, layout=mma_layout)
    b = a
    offs_r0 = gl.arange(0, BLOCK_R, layout=row_layout)
    offs_k0 = gl.arange(0, BLOCK_K, layout=key_layout)
    offs_w = gl.arange(0, H * BLOCK_R, layout=w_layout)
    rb = 0
    lo = 0
    hi = 0
    q_phase = 0
    row_lens = gl.zeros([BLOCK_R], gl.int32, layout=row_layout)
    for v in range(v0, v1):
        if v >= hi:
            # Every wgmma reading q_smem / w_smem retired at the last tile's wait(0) + barrier.
            rb = _last_row_block_at(tiles_ptr, n_row_blocks, v)
            lo = gl.load(tiles_ptr + rb)
            hi = gl.load(tiles_ptr + rb + 1)
            row_lens = _load_row_block(
                q_desc,
                q_bar,
                q_smem,
                wq_ptr,
                w_smem,
                lens_ptr,
                M,
                rb,
                q_phase,
                offs_r0,
                offs_w,
                H,
                H_TOTAL,
                HEAD0,
                D,
                BLOCK_R,
            )
            q_phase ^= 1
            gl.barrier()

        j = v - v0
        stage = j % STAGES
        mbarrier.wait(k_bar.index(stage), (j // STAGES) & 1)
        # Opaque per-tile zero: keeps the 4 * H q descriptors from being hoisted out
        # of the tile loop, where they spill.
        hz = gl.minimum(v, 0)
        keys = (v - lo) * BLOCK_K + offs_k0
        rows = rb * BLOCK_R + offs_r0
        out_ptrs = out_ptr + rows.to(gl.int64)[None, :] * L + keys[:, None]
        out_mask = (rows[None, :] < M) & (keys[:, None] < L)
        if FIRST:
            acc = gl.zeros([BLOCK_K, BLOCK_R], gl.float32, layout=mma_layout)
        else:
            acc = gl.load(out_ptrs, mask=out_mask, other=0.0)
        acc, a, b = _score_tile(
            k_smem.index(stage), q_smem, w_smem, hz, acc, a, b, H, BLOCK_R, row_layout
        )
        gl.barrier()
        if v + STAGES < v1:
            p_lo, p_hi = _prefetch_keys(
                k_desc,
                k_bar,
                k_smem,
                tiles_ptr,
                n_row_blocks,
                v + STAGES,
                stage,
                p_lo,
                p_hi,
                BLOCK_K,
                D,
            )
        if LAST:
            sk = gl.load(sk_ptr + keys, mask=keys < L, other=0.0)
            acc = (acc * sk[:, None]).to(gl.bfloat16).to(gl.float32)
            acc = gl.where(keys[:, None] < row_lens[None, :], acc, float("-inf"))
        gl.store(out_ptrs, acc, mask=out_mask)

    mbarrier.invalidate(q_bar)
    for s in gl.static_range(STAGES):
        mbarrier.invalidate(k_bar.index(s))


def fp4_index_logits_prefill_keys_major(
    q: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Same contract as fp4_index_logits_prefill; SM90 only, even head count."""
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    assert table.dtype == torch.uint8 and table.dim() == 2
    M, H, D = q.shape
    assert H % 2 == 0
    L = slots.shape[0]
    BLOCK_R = BLOCK_ROWS
    k8, sk = _fp4_index_k_to_e4m3(table, slots.to(torch.int64).contiguous(), page_size)
    q8, wq = _q_to_e4m3(q, weights)
    lens = lens.to(torch.int32).contiguous()
    n_row_blocks = triton.cdiv(M, BLOCK_R)
    tiles = torch.empty(n_row_blocks + 1, dtype=torch.int32, device=q.device)
    _visible_tiles_kernel[(1,)](
        lens, tiles, M, n_row_blocks, BLOCK_R=BLOCK_R, BLOCK_K=BLOCK_KEYS, CHUNK=256
    )
    out = torch.empty((M, L), dtype=torch.float32, device=q.device)
    n_key_tiles = triton.cdiv(L, BLOCK_KEYS)
    _fill_hidden_tiles_kernel[(n_key_tiles, n_row_blocks)](
        tiles, out, M, L, BLOCK_R=BLOCK_R, BLOCK_K=BLOCK_KEYS, num_warps=4
    )
    q_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_R, D], gl.float8e4nv)
    k_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_KEYS, D], gl.float8e4nv)
    q_desc = TensorDescriptor.from_tensor(q8.view(M, H * D), [BLOCK_R, D], q_layout)
    k_desc = TensorDescriptor.from_tensor(k8, [BLOCK_KEYS, D], k_layout)
    grid = min(n_row_blocks * n_key_tiles, _num_sms(q.device.index))
    for head0 in range(0, H, HEADS_PER_PASS):
        heads = min(HEADS_PER_PASS, H - head0)
        _keys_major_kernel[(grid,)](
            q_desc,
            k_desc,
            wq,
            sk,
            lens,
            tiles,
            out,
            M,
            L,
            n_row_blocks,
            H=heads,
            H_TOTAL=H,
            HEAD0=head0,
            FIRST=head0 == 0,
            LAST=head0 + heads == H,
            D=D,
            BLOCK_R=BLOCK_R,
            BLOCK_K=BLOCK_KEYS,
            STAGES=STAGES,
            num_warps=NUM_WARPS,
        )
    return out
