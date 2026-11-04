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

This module is the seam for a dedicated EP kernel: the engine calls only ``supports_ep``, ``pack_experts_ep``,
``prepare_ep`` and ``decode_moe_ep``.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2 as v2
from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

HIDDEN = ws.HIDDEN
INTER = ws.INTER  # channels of one virtual expert
MAX_VIRTUAL_EXPERTS = 512  # the v2 kernel's kMaxExperts
MAX_E8M0 = ws.MAX_E8M0
PackedExperts = ws.PackedExperts

_ROUTE_BLOCK = 1024
_COMBINE_BLOCK = 1024


def split_of(inter_per_rank: int) -> int:
    """Virtual experts per local expert (0 when the intermediate size is not a multiple of 576)."""
    if inter_per_rank <= 0 or inter_per_rank % INTER:
        return 0
    return inter_per_rank // INTER


def supports_ep(hidden: int, inter_per_rank: int, num_local_experts: int) -> bool:
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


def pack_experts_ep(w13, s13, w2, s2, expert_chunk: int = 8) -> Tuple[PackedExperts, int]:
    """One layer's local whole experts -> (PackedExperts over E * S virtual experts, S). Packs ``expert_chunk``
    whole experts at a time, so the transient copy is a few experts, not the layer."""
    E = w13.shape[0]
    split = split_of(w13.shape[1] // 2)
    # Byte views: the reorder copies bytes, and e8m0 has few kernels.
    w13, s13, w2, s2 = (t.view(torch.uint8) for t in (w13, s13, w2, s2))
    parts13, parts2 = [], []
    for e0 in range(0, E, expert_chunk):
        e1 = min(E, e0 + expert_chunk)
        v13, vs13, v2w, vs2 = virtual_expert_weights(
            w13[e0:e1], s13[e0:e1], w2[e0:e1], s2[e0:e1]
        )
        p = ws.pack_experts(v13, vs13, v2w, vs2)
        parts13.append(p.w13)
        parts2.append(p.w2)
        del v13, vs13, v2w, vs2, p
    packed = PackedExperts(torch.cat(parts13), torch.cat(parts2))
    return packed, split


# ---------------------------------------------------------------------------------------------------------
# Routing: local slots -> virtual slots sorted by virtual expert.


@triton.jit(do_not_specialize=["num_slots"])
def _ep_count_kernel(ids_ptr, cursor_ptr, rank_ptr, num_slots, BLOCK: tl.constexpr):
    slots = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = slots < num_slots
    ids = tl.load(ids_ptr + slots, mask=in_range, other=-1)
    valid = in_range & (ids >= 0)
    rank = tl.atomic_add(cursor_ptr + tl.where(valid, ids, 0), 1, mask=valid)
    tl.store(rank_ptr + slots, rank, mask=valid)


@triton.jit(do_not_specialize=["num_slots"])
def _ep_expand_kernel(
    ids_ptr,
    weights_ptr,
    rank_ptr,
    counts_ptr,
    offsets_ptr,
    sorted_slots_ptr,
    sorted_tokens_ptr,
    sorted_weights_ptr,
    num_slots,
    TOP_K: tl.constexpr,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slots = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    in_range = slots < num_slots
    ids = tl.load(ids_ptr + slots, mask=in_range, other=-1)
    valid = in_range & (ids >= 0)
    e = tl.where(valid, ids, 0)
    rank = tl.load(rank_ptr + slots, mask=valid, other=0)
    count = tl.load(counts_ptr + e, mask=valid, other=0)
    offset = tl.load(offsets_ptr + e, mask=valid, other=0)
    w = tl.load(weights_ptr + slots, mask=valid, other=0.0)
    token = slots // TOP_K
    for s in tl.static_range(SPLIT):
        pos = SPLIT * offset + s * count + rank
        tl.store(sorted_slots_ptr + pos, slots * SPLIT + s, mask=valid)
        tl.store(sorted_tokens_ptr + pos, token, mask=valid)
        tl.store(sorted_weights_ptr + pos, w, mask=valid)


def route_ep(topk_ids: torch.Tensor, topk_weights: torch.Tensor, num_local: int, split: int):
    """(counts [E*S], sorted_slots, sorted_tokens, sorted_weights [T*K*S]): the v2 kernel's sorted routing
    over the virtual experts. Virtual slot (t, k, s) = (t * K + k) * S + s, the row of y it writes. Entries past
    sum(counts) are left unwritten."""
    num_tokens, top_k = topk_ids.shape
    num_slots = num_tokens * top_k
    dev = topk_ids.device
    ids = topk_ids.to(torch.int32).contiguous()
    weights = topk_weights.float().contiguous()
    cursor = torch.zeros(num_local, dtype=torch.int32, device=dev)
    rank = torch.empty(num_slots, dtype=torch.int32, device=dev)
    grid = (triton.cdiv(num_slots, _ROUTE_BLOCK),)
    _ep_count_kernel[grid](ids, cursor, rank, num_slots, BLOCK=_ROUTE_BLOCK)
    offsets = torch.cumsum(cursor, 0, dtype=torch.int32) - cursor
    vslots = num_slots * split
    sorted_slots = torch.empty(vslots, dtype=torch.int32, device=dev)
    sorted_tokens = torch.empty(vslots, dtype=torch.int32, device=dev)
    sorted_weights = torch.empty(vslots, dtype=torch.float32, device=dev)
    _ep_expand_kernel[grid](
        ids,
        weights,
        rank,
        cursor,
        offsets,
        sorted_slots,
        sorted_tokens,
        sorted_weights,
        num_slots,
        TOP_K=top_k,
        SPLIT=split,
        BLOCK=_ROUTE_BLOCK,
    )
    counts = cursor.view(num_local, 1).expand(num_local, split).reshape(-1)
    return counts, sorted_slots, sorted_tokens, sorted_weights


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
    counts, sorted_slots, sorted_tokens, sorted_weights = route_ep(
        topk_ids, topk_weights, num_virtual // split, split
    )
    if row_quant is None:
        x8f, xs = _row_quant(hidden_states.contiguous())
    else:
        x8f, xs = row_quant
    vslots = num_tokens * top_k * split
    max_passes = num_virtual + vslots // 32 + 1
    work = torch.zeros(v2.WORKSPACE_BASE + 2 * max_passes, dtype=torch.int32, device=dev)
    h = torch.empty(vslots, INTER, dtype=torch.bfloat16, device=dev)
    h8 = torch.empty(vslots, INTER, dtype=torch.uint8, device=dev)
    hs = torch.empty(vslots, dtype=torch.float32, device=dev)
    y = torch.empty(vslots, HIDDEN, dtype=torch.bfloat16, device=dev)
    v2._module().experts(
        x8f.view(torch.uint8),
        xs,
        packed.w13,
        packed.w2,
        counts,
        sorted_tokens,
        sorted_slots,
        sorted_weights,
        h,
        h8,
        hs,
        y,
        work,
        float(swiglu_limit),
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
def prepare_ep(device_index: int, num_virtual: int, split: int, top_k: int) -> None:
    """Build and load everything ``decode_moe_ep`` launches for this shape, at weight-processing time (the v2
    module and its kernels, the Triton routing and combine kernels for this top-k and split), with one call on
    a routing whose every slot is masked (no expert work, no weight read). The first MoE call of CUDA graph
    capture then compiles and loads nothing."""
    v2.prepare(device_index)
    with torch.cuda.device(device_index):
        one = torch.zeros(1, dtype=torch.uint8, device="cuda")
        packed = PackedExperts(
            one.expand(num_virtual, 9, 348160), one.expand(num_virtual, 40, 39168)
        )
        for num_tokens in (2, 16):  # Triton specializes pointers' alignment, not these ints
            x = torch.zeros(num_tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
            ids = torch.full((num_tokens, top_k), -1, dtype=torch.int32, device="cuda")
            decode_moe_ep(x, ids, torch.ones(ids.shape, device="cuda"), packed, 10.0, split)
        torch.cuda.synchronize()
