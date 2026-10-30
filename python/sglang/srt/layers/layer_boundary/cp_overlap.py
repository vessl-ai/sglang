"""A sparse FFN that runs its own prefill CP gather and reduce-scatter in row
chunks, so the collectives of one chunk overlap the MoE of another."""

from typing import Callable, List, Tuple

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (
    _tbo_event,
    attn_cp_all_gather_into_tensor,
    attn_cp_reduce_scatter_tensor,
    is_allocation_symmetric,
)
from sglang.srt.layers.layer_boundary.layout import (
    _cp_gathers_over_attn_cp,
    _prefill_cp_shards_tokens,
    _same_ranks,
)
from sglang.srt.layers.moe.utils import (
    is_moe_input_scattered_across_dp_ranks,
    post_experts_reduction_group,
)
from sglang.srt.runtime_context import get_exec, get_parallel, get_stream

_COMM_STREAM = "ffn_cp_overlap_comm"


def ffn_owns_cp_move(chunks: int, *, sparse: bool) -> bool:
    """Whether a sparse FFN pipelines its prefill CP collectives over
    ``chunks`` row chunks. Off without a prefill CP that shards tokens."""
    if chunks <= 1 or not sparse or not _prefill_cp_shards_tokens():
        return False
    parallel = get_parallel()
    if (
        not _cp_gathers_over_attn_cp()
        or parallel.attn_tp_size != 1
        or parallel.attn_dp_size != 1
        or is_moe_input_scattered_across_dp_ranks()
        or not _same_ranks(post_experts_reduction_group(), parallel.attn_cp_group)
    ):
        raise NotImplementedError(
            "the FFN CP overlap needs DSA or MLA prefill CP without attention TP "
            "or DP, and a MoE without all-to-all whose output sums over the "
            "attention-CP group"
        )
    # Per-token recording overwrites its first rows on every MoE call.
    if get_exec().moe.expert_distribution_recorder_mode == "per_token":
        raise NotImplementedError(
            "the FFN CP overlap with the per-token expert distribution recorder"
        )
    return True


def chunk_bounds(rows: int, chunks: int) -> List[Tuple[int, int]]:
    """Equal row chunks, the last one possibly shorter. Every CP rank holds
    the same padded row count, so all ranks cut the same chunks."""
    size = max(-(-rows // chunks), 1)
    return [(start, min(start + size, rows)) for start in range(0, rows, size)]


def run_ffn_cp_overlap(
    compute: Callable[[torch.Tensor], torch.Tensor],
    hidden_states: torch.Tensor,
    *,
    chunks: int,
) -> torch.Tensor:
    """``compute`` over this rank's CP shard gathered over attention CP,
    summed and scattered back onto the shard, one row chunk at a time."""
    if torch.cuda.is_current_stream_capturing():
        raise NotImplementedError("the FFN CP overlap under CUDA graph capture")
    hidden_states = hidden_states.contiguous()
    bounds = chunk_bounds(hidden_states.shape[0], chunks)
    gathered = _gathered_chunks(hidden_states, bounds)
    output = torch.empty_like(hidden_states)
    main = torch.cuda.current_stream()
    comm = get_stream(_COMM_STREAM)
    comm.wait_stream(main)
    # Every tensor the comm stream touches stays referenced here until main
    # waits on the last reduce-scatter.
    computed = []
    done = None
    ready = _all_gather(gathered, hidden_states, bounds, 0, comm=comm)
    for k, (start, end) in enumerate(bounds):
        # Queue the next gather before compute: the MoE syncs the host.
        following = _all_gather(gathered, hidden_states, bounds, k + 1, comm=comm)
        main.wait_event(ready)
        computed.append(compute(gathered[k]))
        done = _reduce_scatter(output[start:end], computed[k], k, main=main, comm=comm)
        ready = following
    if done is not None:
        main.wait_event(done)
    return output


def _gathered_chunks(hidden_states, bounds):
    """One buffer for the gathered rows, viewed as each chunk's rank-major
    rows."""
    parallel = get_parallel()
    size = parallel.attn_cp_size
    with use_symmetric_memory(
        parallel.attn_cp_group, disabled=not is_allocation_symmetric()
    ):
        gathered = hidden_states.new_empty(
            (hidden_states.shape[0] * size, *hidden_states.shape[1:])
        )
    return [gathered[start * size : end * size] for start, end in bounds]


def _all_gather(gathered, hidden_states, bounds, k, *, comm):
    if k >= len(bounds):
        return None
    start, end = bounds[k]
    event = _tbo_event(("ffn_cp_overlap_gather", k))
    with torch.cuda.stream(comm):
        attn_cp_all_gather_into_tensor(gathered[k], hidden_states[start:end])
        event.record(comm)
    return event


def _reduce_scatter(output, computed, k, *, main, comm):
    computed_event = _tbo_event(("ffn_cp_overlap_compute", k))
    computed_event.record(main)
    event = _tbo_event(("ffn_cp_overlap_scatter", k))
    with torch.cuda.stream(comm):
        comm.wait_event(computed_event)
        attn_cp_reduce_scatter_tensor(output, computed)
        event.record(comm)
    return event
