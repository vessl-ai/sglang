"""Stable sort of routed slots by expert for small (decode-sized) batches, in one launch.

Replaces ``argsort`` + histogram + gathers in front of the triton_kernels grouped
matmuls. Program ``e`` scans every slot: its output offset is the number of slots
routed to experts below ``e``, and its slots keep their original order. The scan is
O(num_experts x num_slots), so this is meant for decode batches (a few thousand
slots); it reads only device memory and is CUDA-graph safe.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _expert_sort_small_kernel(
    topk_ids_ptr,
    topk_weights_ptr,
    num_slots,
    top_k,
    counts_ptr,
    sorted_slots_ptr,
    sorted_tokens_ptr,
    sorted_weights_ptr,
    BLOCK: tl.constexpr,
    DROP_MASKED: tl.constexpr,
):
    expert = tl.program_id(0)
    # DROP_MASKED: masked slots (id -1) get the id num_experts, so no program counts or writes them.
    masked_id = tl.num_programs(0) if DROP_MASKED else 0
    # Pass 1: this expert's output offset (slots routed to lower experts) and count.
    offset = 0
    count = 0
    for start in range(0, num_slots, BLOCK):
        slots = start + tl.arange(0, BLOCK)
        in_range = slots < num_slots
        # Masked slots (id -1) go to expert 0 with weight 0 (or to no expert under DROP_MASKED).
        raw = tl.load(topk_ids_ptr + slots, mask=in_range, other=0)
        ids = tl.where(raw < 0, masked_id, raw)
        offset += tl.sum((in_range & (ids < expert)).to(tl.int32), 0)
        count += tl.sum((in_range & (ids == expert)).to(tl.int32), 0)
    tl.store(counts_ptr + expert, count)
    # Pass 2: write this expert's slots in their original order.
    pos_base = offset
    for start in range(0, num_slots, BLOCK):
        slots = start + tl.arange(0, BLOCK)
        in_range = slots < num_slots
        raw_ids = tl.load(topk_ids_ptr + slots, mask=in_range, other=0)
        hit = (in_range & (tl.where(raw_ids < 0, masked_id, raw_ids) == expert)).to(
            tl.int32
        )
        pos = pos_base + tl.cumsum(hit, 0) - hit
        pos_base += tl.sum(hit, 0)
        keep = hit != 0
        weights = tl.load(topk_weights_ptr + slots, mask=keep, other=0.0)
        weights = tl.where(raw_ids < 0, 0.0, weights)
        tl.store(sorted_slots_ptr + pos, slots, mask=keep)
        tl.store(sorted_tokens_ptr + pos, slots // top_k, mask=keep)
        tl.store(sorted_weights_ptr + pos, weights, mask=keep)


def expert_sort_small(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    drop_masked: bool = False,
):
    """Return (counts[E] int32, sorted_slots[S] int32, sorted_tokens[S] int32,
    sorted_weights[S] float32) for ``topk_ids``/``topk_weights`` of shape [T, top_k].

    ``sorted_slots`` equals ``argsort(ids, stable=True)`` with ids -1 mapped to 0. With
    ``drop_masked``, slots with id -1 are counted for no expert and sort after all the others
    (their entries are left unwritten).
    Ids must lie in [-1, num_experts): a larger id is counted for no expert and
    leaves output entries unwritten, without an error.
    """
    num_tokens, top_k = topk_ids.shape
    num_slots = num_tokens * top_k
    device = topk_ids.device
    counts = torch.empty(num_experts, dtype=torch.int32, device=device)
    sorted_slots = torch.empty(num_slots, dtype=torch.int32, device=device)
    sorted_tokens = torch.empty(num_slots, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(num_slots, dtype=torch.float32, device=device)
    _expert_sort_small_kernel[(num_experts,)](
        topk_ids.reshape(-1),
        topk_weights.reshape(-1),
        num_slots,
        top_k,
        counts,
        sorted_slots,
        sorted_tokens,
        sorted_weights,
        BLOCK=1024,
        DROP_MASKED=drop_masked,
        num_warps=4,
    )
    return counts, sorted_slots, sorted_tokens, sorted_weights
