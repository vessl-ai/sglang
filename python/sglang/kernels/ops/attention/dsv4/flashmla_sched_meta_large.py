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
        cuda_wrappers=[("flashmla_sched_meta_large", "flashmla_sched_meta_large")],
    )


def smem_bytes(batch_size: int, num_sm_parts: int) -> int:
    """Shared memory the kernel needs: the row prefix, the metadata, the split rows."""
    return 4 * (batch_size + 1 + num_sm_parts * META_INTS + num_sm_parts)


@functools.lru_cache(maxsize=None)
def max_smem_bytes(device_index: int) -> int:
    return torch.cuda.get_device_properties(
        device_index
    ).shared_memory_per_block_optin


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
