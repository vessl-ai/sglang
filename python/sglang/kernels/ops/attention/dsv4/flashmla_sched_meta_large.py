"""FlashMLA's split-KV schedule for a large sparse batch (INF-681 k-fuse-p).

Same output as FlashMLA's own get_mla_metadata kernel (and as `flashmla_sched_meta`),
computed by a whole block instead of one thread: the per-row cost is prefix-summed in
parallel and the partition walk jumps over whole rows with a binary search. Meant for
the SM90 CP prefill, where every query row is one "request" of the decode kernel and
FlashMLA's own walk costs ~7 ms per call at 16k rows per rank.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

from .utils import make_name

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# sizeof(DecodingSchedMeta) / 4, fixed by FlashMLA's params.h.
META_INTS = 8


@cache_once
def _module() -> Module:
    return load_jit(
        make_name("flashmla_sched_meta_large"),
        cuda_files=["deepseek_v4/flashmla_sched_meta_large.cuh"],
        cuda_wrappers=[
            ("flashmla_sched_meta_large", "flashmla_sched_meta_large"),
            ("flashmla_sched_meta_chain", "flashmla_sched_meta_chain"),
        ],
    )


def smem_bytes(batch_size: int, num_sm_parts: int) -> int:
    """Shared memory the kernel needs: the row prefix, the metadata, the split rows."""
    return 4 * (batch_size + 1 + num_sm_parts * META_INTS + num_sm_parts)


@functools.lru_cache(maxsize=None)
def max_smem_bytes(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin


def fits(batch_size: int, num_sm_parts: int, device_index: int) -> bool:
    # Leave 1 KB for the kernel's static shared memory (warp sums, two flags).
    return smem_bytes(batch_size, num_sm_parts) <= max_smem_bytes(device_index) - 1024


def flashmla_sched_meta_large(
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    block_size_n: int,
    fixed_overhead_num_blocks: int,
    topk: int,
    extra_topk: int = 0,
) -> None:
    """Fill ``tile_scheduler_metadata`` ``[num_sm_parts, 8]`` and ``num_splits``
    ``[batch_size + 1]`` (int32) in place, for a sparse schedule (``topk >= 0``).

    ``topk_length`` / ``extra_topk_length``: ``[batch_size]`` int32 per-row lengths,
    or None for ``topk`` / ``extra_topk`` on every row. ``block_size_n`` and
    ``fixed_overhead_num_blocks`` are the decode kernel's (64 and 5 for the 64-head
    sparse FP8 kernel on SM90 and SM100).
    """
    _module().flashmla_sched_meta_large(
        tile_scheduler_metadata,
        num_splits,
        topk_length,
        extra_topk_length,
        block_size_n,
        fixed_overhead_num_blocks,
        topk,
        extra_topk,
    )


def chain_capacity(
    batch_size: int,
    *,
    block_size_n: int,
    fixed_overhead_num_blocks: int,
    topk: int,
    extra_topk: int = 0,
) -> int:
    """Upper bound on the schedule's total cost (every row's num_blocks + fixed
    overhead) when no row is longer than the index widths ``topk`` / ``extra_topk``;
    the same length rule as the kernels' seqlen_k_of."""
    seqlen = max(topk, 1)
    if extra_topk:
        seqlen = -(-seqlen // block_size_n) * block_size_n + extra_topk
    num_blocks = (seqlen - 1) // block_size_n + 1
    return batch_size * (num_blocks + fixed_overhead_num_blocks)


# The chain kernel's thread count and jump length (flashmla_sched_meta_large.cuh).
_CHAIN_THREADS = 1024
_CHAIN_JUMP = 4


def chain_smem_bytes(batch_size: int, num_sm_parts: int, capacity: int) -> int:
    """Shared memory of the chain kernel: the row prefix, the metadata, the split
    rows, the per-part start states, the jump starts and three (capacity + 1)
    step tables."""
    num_hops = -(-num_sm_parts // _CHAIN_JUMP) + 1
    return 4 * (
        batch_size
        + 1
        + num_sm_parts * META_INTS
        + num_sm_parts
        + num_sm_parts
        + 1
        + num_hops
        + 3 * (capacity + 1)
    )


def chain_fits(
    batch_size: int, num_sm_parts: int, capacity: int, device_index: int
) -> bool:
    # Rows and blocks are packed in 19 + 12 bits; 1 KB stays for static shared memory.
    return (
        batch_size < (1 << 19)
        and num_sm_parts < _CHAIN_THREADS
        and capacity < (1 << 30)
        and capacity // max(batch_size, 1) < (1 << 12)
        and chain_smem_bytes(batch_size, num_sm_parts, capacity)
        <= max_smem_bytes(device_index) - 1024
    )


def flashmla_sched_meta_chain(
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    *,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    block_size_n: int,
    fixed_overhead_num_blocks: int,
    topk: int,
    extra_topk: int = 0,
    capacity: int,
) -> None:
    """INF-874 decode attention A2: the same output as ``flashmla_sched_meta_large``,
    a few microseconds for a decode batch (hundreds of rows) instead of tens: every
    walk state's one-part step is computed in parallel, and the serial walk only
    follows that table. ``capacity`` (``chain_capacity``) sizes the table; if the
    real total is larger (lengths past the index widths) the kernel falls back to
    the serial walk, so the output is the same either way."""
    _module().flashmla_sched_meta_chain(
        tile_scheduler_metadata,
        num_splits,
        topk_length,
        extra_topk_length,
        block_size_n,
        fixed_overhead_num_blocks,
        topk,
        extra_topk,
        capacity,
    )
