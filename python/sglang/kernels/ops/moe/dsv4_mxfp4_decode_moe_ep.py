"""INF-730: the SM90 W4A8 decode MoE (``dsv4_mxfp4_decode_moe_v2``, #115) on expert-parallel layers.

Under expert parallelism (``--ep-size`` > 1, MoE TP 1) a rank holds whole experts (DeepSeek-V4.1-Flash:
intermediate 2304) instead of a TP slice of every expert (576 at TP4), and the v2 kernel covers 576-channel
experts only.

An expert of intermediate S x 576 is the sum of S "virtual experts", each one 576-channel slice of it: SwiGLU
acts per channel, so the gate/up rows and the down columns of slice s form an expert of their own, and the S down
outputs add up to the whole expert's output. Slice s is exactly what TP rank s of a TP-S layer holds of that
expert, so every virtual slot's output row (bf16 ``y``, the kernel's per-slot result) has the bits TP-S rank s
computes for that slot. Only the order of the final sums differs: here a rank sums its own (k, s) rows of a token
in fp32 (k outer, s inner) and the EP reduce-scatter adds the ranks; at TP-S each rank sums its k rows and the
reduce-scatter adds the S slices.

- ``pack_experts_ep`` (load time): the rank's E local experts in sglang's [w1; w3] layout -> ``PackedExperts``
  of E * S virtual experts (virtual id = e * S + s), the same bytes reordered.
- ``decode_moe_ep``: hidden [T, 5120] bf16, topk_ids [T, K] LOCAL expert ids (-1: the expert lives on another
  rank, or a masked dp-attention pad row), topk_weights [T, K] -> out[t] = sum over this rank's slots of
  w * expert(x[t]), i.e. this rank's part of the MoE output (the caller's reduce-scatter adds the ranks).

The prologue expands each local slot into its S virtual slots without a sort: an atomic count per local expert
gives every slot its rank among the slots of its expert, and the slot writes its S entries at
S * offset(e) + s * count(e) + rank. The order of the slots inside an expert is therefore not stable from call
to call; the v2 kernel's output does not depend on it (see the header of ``dsv4_mxfp4_decode_moe_v2.cuh``).

Buffers are sized for the worst case (every slot of every token on this rank): ``y`` is T * K * S x 5120 bf16,
1.0 GB at T = 4096, K = 6, S = 4 (DP8 x 128 requests x 4 verify rows). Calls above ``max_virtual_slots()`` run
in token chunks, which is allowed in eager mode only (prefill); below it ``decode_moe_ep`` is CUDA-graph safe (no
host sync).

Kernels (SGLANG_OPT_DSV4_EP_DECODE_MOE_KERNEL):
- ``v2w`` (default): ``dsv4_mxfp4_decode_moe_v2w`` (INF-730 track a-moe), the v2 kernel compiled for whole 2304-wide
  experts (64-row passes, grid-wide h quantization). S = 1: one slot per (token, k), y = T * K rows. h is quantized
  once per 2304-wide row (TP4 quantizes each 576 slice), so the numerics differ from TP4 by ~0.4% (a-moe).
- ``virtual``: the v2 kernel on S = 4 virtual 576-channel experts per local expert (above), bit-identical rows to TP4.
Both use this module's prologue (routing + row quantization) and combine, so the pre-quantized input of a
dispatch (``row_quant``) works for both.

This module is the seam for the EP expert kernel: the engine calls only ``kernel_mode``, ``supports_ep``,
``pack_experts_ep``, ``prepare_ep`` and ``decode_moe_ep``.
"""

from __future__ import annotations

import functools
from typing import NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2 as v2
from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2w as v2w
from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

HIDDEN = ws.HIDDEN
INTER = ws.INTER  # channels of one virtual expert
MAX_VIRTUAL_EXPERTS = 512  # the v2 kernel's kMaxExperts
MAX_E8M0 = ws.MAX_E8M0
PackedExperts = ws.PackedExperts

_ROUTE_BLOCK = 1024
_COMBINE_BLOCK = 1024


V2W_WIDTHS = (2304,)  # whole-expert widths the v2w kernel is built and tested for


def kernel_mode() -> str:
    from sglang.srt.environ import envs

    mode = envs.SGLANG_OPT_DSV4_EP_DECODE_MOE_KERNEL.get()
    if mode not in ("v2w", "virtual"):
        raise ValueError(
            f"SGLANG_OPT_DSV4_EP_DECODE_MOE_KERNEL must be 'v2w' or 'virtual', got {mode!r}"
        )
    return mode


def split_of(inter_per_rank: int) -> int:
    """Virtual experts per local expert (0 when the intermediate size is not a multiple of 576)."""
    if inter_per_rank <= 0 or inter_per_rank % INTER:
        return 0
    return inter_per_rank // INTER


def supports_ep(
    hidden: int, inter_per_rank: int, num_local_experts: int, mode: str = "virtual"
) -> bool:
    if mode == "v2w":
        return (
            hidden == HIDDEN
            and inter_per_rank in V2W_WIDTHS
            and 0 < num_local_experts <= MAX_VIRTUAL_EXPERTS
        )
    split = split_of(inter_per_rank)
    return (
        hidden == HIDDEN
        and split > 0
        and 0 < num_local_experts * split <= MAX_VIRTUAL_EXPERTS
    )


def max_virtual_slots() -> int:
    """Largest T * K * S handled in one call (the buffers scale with it); larger calls are chunked."""
    from sglang.srt.environ import envs

    return envs.SGLANG_OPT_DSV4_EP_DECODE_MOE_MAX_VSLOTS.get()


def virtual_expert_weights(w13, s13, w2, s2):
    """[E, 2I, H/2] / [E, 2I, H/32] / [E, H, I/2] / [E, H, I/32] (MXFP4 bytes and e8m0 scales of whole
    experts, gate rows first) -> the same four for E * S experts of 576 channels; virtual expert e * S + s =
    gate rows [s*576, (s+1)*576) and up rows [I + s*576, ...) of expert e, and its down columns [s*576, ...).
    Pure index permutation (also on CPU)."""
    E, n2, kb = w13.shape
    inter = n2 // 2
    split = split_of(inter)
    if split == 0:
        raise ValueError(f"intermediate {inter} is not a multiple of {INTER}")
    hidden = w2.shape[1]
    assert w2.shape == (E, hidden, inter // 2), w2.shape
    assert s13.shape[:2] == (E, n2) and s2.shape[:2] == (E, hidden), (s13.shape, s2.shape)

    def gate_up(t):  # [E, 2I, X] -> [E*S, 2*576, X]
        x = t.shape[2]
        return (
            t.reshape(E, 2, split, INTER, x)
            .transpose(1, 2)
            .reshape(E * split, 2 * INTER, x)
        )

    def down(t):  # [E, H, I/g] -> [E*S, H, 576/g]
        x = t.shape[2] // split
        return (
            t.reshape(E, hidden, split, x)
            .transpose(1, 2)
            .reshape(E * split, hidden, x)
        )

    return gate_up(w13), gate_up(s13), down(w2), down(s2)


def pack_experts_ep(w13, s13, w2, s2, expert_chunk: int = 8, mode: str = "virtual"):
    """One layer's local whole experts -> (packed, S): ``virtual`` = ws.PackedExperts over E * S virtual 576
    experts; ``v2w`` = v2w.PackedExperts over the E whole experts, S = 1. Packs ``expert_chunk`` experts at a time,
    so the transient copy is a few experts, not the layer."""
    E = w13.shape[0]
    # Byte views: the reorder copies bytes, and e8m0 has few kernels.
    w13, s13, w2, s2 = (t.view(torch.uint8) for t in (w13, s13, w2, s2))
    out13 = out2 = None  # filled chunk by chunk (no final cat: its copy left ~7 GB of fragments at EP8 v2w)
    for e0 in range(0, E, expert_chunk):
        e1 = min(E, e0 + expert_chunk)
        if mode == "v2w":
            p = v2w.pack_w13_w2(w13[e0:e1], s13[e0:e1], w2[e0:e1], s2[e0:e1])
        else:
            v13, vs13, vw2, vs2 = virtual_expert_weights(
                w13[e0:e1], s13[e0:e1], w2[e0:e1], s2[e0:e1]
            )
            p = ws.pack_experts(v13, vs13, vw2, vs2)
            del v13, vs13, vw2, vs2
        per = p.w13.shape[0] // (e1 - e0)  # packed experts per local expert (virtual: S, v2w: 1)
        if out13 is None:
            out13 = p.w13.new_empty((E * per,) + tuple(p.w13.shape[1:]))
            out2 = p.w2.new_empty((E * per,) + tuple(p.w2.shape[1:]))
        n, lo = p.w13.shape[0], e0 * per
        out13[lo:lo + n].copy_(p.w13)
        out2[lo:lo + n].copy_(p.w2)
        del p
        torch.cuda.empty_cache()
    if mode == "v2w":
        return v2w.PackedExperts(out13, out2, w13.shape[1] // 2), 1
    return PackedExperts(out13, out2), split_of(w13.shape[1] // 2)


# ---------------------------------------------------------------------------------------------------------
# Prologue: local slots -> virtual slots sorted by virtual expert, plus the e4m3 row quantization of the tokens.
# Three launches: one memset (counts + the experts kernel's workspace), count + row quantization, expand.


@triton.jit(do_not_specialize=["num_slots", "num_slot_blocks"])
def _ep_count_quant_kernel(
    ids_ptr,
    cursor_ptr,
    rank_ptr,
    num_slots,
    num_slot_blocks,
    x_ptr,
    x8_ptr,
    xs_ptr,
    HID: tl.constexpr,
    QBLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
    QUANT: tl.constexpr,
):
    """Programs below num_slot_blocks: every local slot takes its rank among its expert's slots (atomic count).
    The others: one token row each, quantized exactly as mxfp4_w4a8_prefill._row_quant does it (e4m3, one
    power-of-2 scale per row)."""
    pid = tl.program_id(0)
    if pid < num_slot_blocks:
        slots = pid * BLOCK + tl.arange(0, BLOCK)
        in_range = slots < num_slots
        ids = tl.load(ids_ptr + slots, mask=in_range, other=-1)
        valid = in_range & (ids >= 0)
        rank = tl.atomic_add(cursor_ptr + tl.where(valid, ids, 0), 1, mask=valid)
        tl.store(rank_ptr + slots, rank, mask=valid)
    elif QUANT:
        r = (pid - num_slot_blocks).to(tl.int64)
        o = tl.arange(0, QBLOCK)
        x = tl.load(x_ptr + r * HID + o, mask=o < HID, other=0.0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x)), 1e-4)
        sc = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
        tl.store(x8_ptr + r * HID + o, (x / sc).to(tl.float8e4nv), mask=o < HID)
        tl.store(xs_ptr + r, sc)


@triton.jit(do_not_specialize=["num_slots"])
def _ep_expand_kernel(
    ids_ptr,
    weights_ptr,
    rank_ptr,
    counts_ptr,
    counts_v_ptr,
    sorted_slots_ptr,
    sorted_tokens_ptr,
    sorted_weights_ptr,
    num_slots,
    NUM_LOCAL: tl.constexpr,
    E_BLOCK: tl.constexpr,
    TOP_K: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    j = tl.arange(0, E_BLOCK)
    cnt = tl.load(counts_ptr + j, mask=j < NUM_LOCAL, other=0)
    first = tl.cumsum(cnt, 0) - cnt  # first sorted position of each local expert's slots
    if tl.program_id(0) == 0:
        for s in tl.static_range(SPLIT):
            tl.store(counts_v_ptr + j * SPLIT + s, cnt, mask=j < NUM_LOCAL)
    slots = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = slots < num_slots
    ids = tl.load(ids_ptr + slots, mask=in_range, other=-1)
    valid = in_range & (ids >= 0)
    e = tl.where(valid, ids, 0)
    rank = tl.load(rank_ptr + slots, mask=valid, other=0)
    count = tl.gather(cnt, e, 0)
    offset = tl.gather(first, e, 0)
    w = tl.load(weights_ptr + slots, mask=valid, other=0.0)
    token = slots // TOP_K
    for s in tl.static_range(SPLIT):
        pos = SPLIT * offset + s * count + rank
        tl.store(sorted_slots_ptr + pos, slots * SPLIT + s, mask=valid)
        tl.store(sorted_tokens_ptr + pos, token, mask=valid)
        tl.store(sorted_weights_ptr + pos, w, mask=valid)


class EpRouting(NamedTuple):
    counts: torch.Tensor  # [E*S] int32 slots of each virtual expert
    sorted_slots: torch.Tensor  # [T*K*S] int32 virtual slot (t * K + k) * S + s = the row of y it writes
    sorted_tokens: torch.Tensor  # [T*K*S] int32 token row of x
    sorted_weights: torch.Tensor  # [T*K*S] fp32 routing weight
    work: torch.Tensor  # the experts kernel's zeroed workspace


def route_ep(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_local: int,
    split: int,
    hidden_states: Optional[torch.Tensor] = None,
):
    """The v2 kernel's sorted routing over the virtual experts (entries past sum(counts) unwritten) and its
    zeroed workspace; with ``hidden_states`` also its row quantization -> (EpRouting, (x8, xscale) or None)."""
    num_tokens, top_k = topk_ids.shape
    num_slots = num_tokens * top_k
    dev = topk_ids.device
    ids = topk_ids.to(torch.int32).contiguous()
    weights = topk_weights.float().contiguous()
    num_virtual = num_local * split
    vslots = num_slots * split
    max_passes = num_virtual + vslots // 32 + 1
    # one memset: the experts kernel's workspace (rounded to 16 bytes, so that the cursors after it keep the
    # pointer alignment Triton specialized on), then the per-expert cursors
    # v2 needs 32 + 2 x passes ints, v2w 32 + 5 x passes; one size for both
    work_len = (v2.WORKSPACE_BASE + 5 * max_passes + 3) // 4 * 4
    zeroed = torch.zeros(work_len + num_local, dtype=torch.int32, device=dev)
    work, cursor = zeroed[:work_len], zeroed[work_len:]
    rank = torch.empty(num_slots, dtype=torch.int32, device=dev)
    slot_blocks = triton.cdiv(num_slots, _ROUTE_BLOCK)
    quant = None
    if hidden_states is not None:
        quant = (
            torch.empty(num_tokens, HIDDEN, dtype=torch.float8_e4m3fn, device=dev),
            torch.empty(num_tokens, dtype=torch.float32, device=dev),
        )
        x = hidden_states.contiguous()
    else:  # placeholders of the right types; QUANT=False compiles no access to them
        x = torch.empty(0, dtype=torch.bfloat16, device=dev)
        empty = (torch.empty(0, dtype=torch.float8_e4m3fn, device=dev), torch.empty(0, dtype=torch.float32, device=dev))
    _ep_count_quant_kernel[(slot_blocks + (num_tokens if quant is not None else 0),)](
        ids,
        cursor,
        rank,
        num_slots,
        slot_blocks,
        x,
        *(quant if quant is not None else empty),
        HID=HIDDEN,
        QBLOCK=triton.next_power_of_2(HIDDEN),
        BLOCK=_ROUTE_BLOCK,
        QUANT=quant is not None,
        num_warps=8,
    )
    counts = torch.empty(num_virtual, dtype=torch.int32, device=dev)
    sorted_slots = torch.empty(vslots, dtype=torch.int32, device=dev)
    sorted_tokens = torch.empty(vslots, dtype=torch.int32, device=dev)
    sorted_weights = torch.empty(vslots, dtype=torch.float32, device=dev)
    e_block = triton.next_power_of_2(num_local)
    block = max(64, min(1024, 32768 // e_block))
    _ep_expand_kernel[(triton.cdiv(num_slots, block),)](
        ids, weights, rank, cursor, counts, sorted_slots, sorted_tokens, sorted_weights, num_slots,
        NUM_LOCAL=num_local, E_BLOCK=e_block, TOP_K=top_k, SPLIT=split, BLOCK=block,
    )
    return EpRouting(counts, sorted_slots, sorted_tokens, sorted_weights, work), quant


# ---------------------------------------------------------------------------------------------------------
# Combine: out[t] = sum over (k, s) of y[(t * K + k) * S + s] for the token's local slots, fp32, k outer.


@triton.jit(do_not_specialize=["num_tokens"])
def _ep_combine_kernel(
    y_ptr,
    ids_ptr,
    out_ptr,
    num_tokens,
    HIDDEN_: tl.constexpr,
    TOP_K: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cmask = cols < HIDDEN_
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        use = tl.load(ids_ptr + t * TOP_K + k) >= 0
        for s in tl.static_range(SPLIT):
            row = (t * TOP_K + k) * SPLIT + s
            v = tl.load(y_ptr + row * HIDDEN_ + cols, mask=cmask & use, other=0.0)
            # A skipped slot adds nothing (not +0.0, which would turn a -0.0 sum into +0.0).
            acc = tl.where(use, acc + v.to(tl.float32), acc)
    tl.store(out_ptr + t * HIDDEN_ + cols, acc.to(tl.bfloat16), mask=cmask)


def combine_ep(y: torch.Tensor, topk_ids: torch.Tensor, split: int, out: torch.Tensor) -> torch.Tensor:
    num_tokens, top_k = topk_ids.shape
    assert y.shape == (num_tokens * top_k * split, HIDDEN), y.shape
    assert out.shape == (num_tokens, HIDDEN) and out.is_contiguous(), out.shape
    if num_tokens == 0:
        return out
    grid = (num_tokens, triton.cdiv(HIDDEN, _COMBINE_BLOCK))
    _ep_combine_kernel[grid](
        y,
        topk_ids.to(torch.int32).contiguous(),
        out,
        num_tokens,
        HIDDEN_=HIDDEN,
        TOP_K=top_k,
        SPLIT=split,
        BLOCK=_COMBINE_BLOCK,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------------------------------------


def _row_quant(x: torch.Tensor):
    return ws._row_quant(x)


def _decode_moe_ep_call(
    hidden_states,
    topk_ids,
    topk_weights,
    packed: PackedExperts,
    swiglu_limit: float,
    split: int,
    output: torch.Tensor,
    row_quant,
    return_y: bool,
):
    num_tokens, top_k = topk_ids.shape
    num_virtual = packed.w13.shape[0]
    assert num_virtual % split == 0, (num_virtual, split)
    dev = output.device
    routing, quant = route_ep(
        topk_ids,
        topk_weights,
        num_virtual // split,
        split,
        hidden_states=hidden_states if row_quant is None else None,
    )
    x8f, xs = quant if row_quant is None else row_quant
    vslots = num_tokens * top_k * split
    if isinstance(packed, v2w.PackedExperts):
        assert split == 1, split
        inter, mod = packed.inter, v2w._module(packed.inter)
    else:
        inter, mod = INTER, v2._module()
    h = torch.empty(vslots, inter, dtype=torch.bfloat16, device=dev)
    h8 = torch.empty(vslots, inter, dtype=torch.uint8, device=dev)
    hs = torch.empty(vslots, dtype=torch.float32, device=dev)
    y = torch.empty(vslots, HIDDEN, dtype=torch.bfloat16, device=dev)
    mod.experts(
        x8f.view(torch.uint8),
        xs,
        packed.w13,
        packed.w2,
        routing.counts,
        routing.sorted_tokens,
        routing.sorted_slots,
        routing.sorted_weights,
        h,
        h8,
        hs,
        y,
        routing.work,
        float(swiglu_limit),
        *((y,) if mod is not v2._module() else ()),  # v2w: its gather mode's token-row output (unused here)
    )
    combine_ep(y, topk_ids, split, output)
    return y if return_y else None


def decode_moe_ep(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    packed: PackedExperts,
    swiglu_limit: float,
    split: int,
    output: Optional[torch.Tensor] = None,
    row_quant: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """This rank's routed-expert output: out[t] = sum over the slots k with topk_ids[t, k] >= 0 (a LOCAL
    expert id) of w[t, k] * expert(x[t]). ``row_quant`` = (e4m3 rows, power-of-2 fp32 scale per row) already
    computed as ``_row_quant`` does (then ``hidden_states`` only gives the row count); None quantizes
    ``hidden_states`` here."""
    num_tokens, top_k = topk_ids.shape
    if output is None:
        output = torch.empty(
            num_tokens, HIDDEN, dtype=torch.bfloat16, device=topk_ids.device
        )
    if num_tokens == 0:
        return output
    assert row_quant is not None or hidden_states.dtype == torch.bfloat16
    per_call = max(1, max_virtual_slots() // (top_k * split))
    if num_tokens <= per_call:
        _decode_moe_ep_call(
            hidden_states,
            topk_ids,
            topk_weights,
            packed,
            swiglu_limit,
            split,
            output,
            row_quant,
            False,
        )
        return output
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"EP decode MoE: {num_tokens} tokens x top {top_k} x {split} virtual experts exceed "
            f"SGLANG_OPT_DSV4_EP_DECODE_MOE_MAX_VSLOTS={max_virtual_slots()} inside a CUDA graph; "
            "raise it (the buffers grow with it) or lower the decode graph batch size."
        )
    for t0 in range(0, num_tokens, per_call):
        t1 = min(num_tokens, t0 + per_call)
        rq = None if row_quant is None else (row_quant[0][t0:t1], row_quant[1][t0:t1])
        _decode_moe_ep_call(
            None if hidden_states is None else hidden_states[t0:t1],
            topk_ids[t0:t1],
            topk_weights[t0:t1],
            packed,
            swiglu_limit,
            split,
            output[t0:t1],
            rq,
            False,
        )
    return output


@functools.cache
def prepare_ep(
    device_index: int, num_virtual: int, split: int, top_k: int, inter: Optional[int] = None
) -> None:
    """Build and load everything ``decode_moe_ep`` launches for this shape, at weight-processing time (the expert
    kernel's JIT module and kernels, the Triton prologue and combine for this top-k and split), with calls on a
    routing whose every slot is masked (no expert work, no weight read). The first MoE call of CUDA graph capture
    then compiles and loads nothing. ``inter``: the v2w kernel's whole-expert width (None: the virtual v2 path)."""
    if inter is None:
        v2.prepare(device_index)
    else:
        v2w.prepare(device_index, inter)
    with torch.cuda.device(device_index):
        one = torch.zeros(1, dtype=torch.uint8, device="cuda")
        if inter is None:
            packed = PackedExperts(
                one.expand(num_virtual, 9, 348160), one.expand(num_virtual, 40, 39168)
            )
        else:
            packed = v2w.PackedExperts(
                one.expand(num_virtual, inter // 64, 348160),
                one.expand(num_virtual, 40, inter * 68),
                inter,
            )
        for num_tokens in (2, 16):  # Triton specializes pointers' alignment, not these ints
            x = torch.zeros(num_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
            ids = torch.full((num_tokens, top_k), -1, dtype=torch.int32, device="cuda")
            w = torch.ones(ids.shape, device="cuda")
            decode_moe_ep(x, ids, w, packed, 10.0, split)
            # the pre-quantized input path (a dispatch that sends e4m3 rows)
            decode_moe_ep(None, ids, w, packed, 10.0, split, row_quant=_row_quant(x))
        torch.cuda.synchronize()
