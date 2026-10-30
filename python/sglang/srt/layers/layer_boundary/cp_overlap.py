"""A sparse FFN that runs its own prefill CP gather and reduce-scatter in row
chunks, so the collectives of one chunk overlap the MoE of another."""

from typing import Callable, Union

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.cp.interleave import shard_chunk_bounds
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
from sglang.srt.layers.layer_boundary.ops import attn_cp_interleave_reduce_scatter
from sglang.srt.layers.layer_boundary.output import UnreducedArrival
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


def run_ffn_cp_overlap(
    compute: Callable[[torch.Tensor], torch.Tensor],
    hidden_states: Union[torch.Tensor, UnreducedArrival],
    *,
    chunks: int,
) -> torch.Tensor:
    """``compute`` over this rank's CP shard gathered over attention CP,
    summed and scattered back onto the shard, one row chunk at a time. An
    input still owing its attention-CP sum is completed and read per chunk."""
    if torch.cuda.is_current_stream_capturing():
        raise NotImplementedError("the FFN CP overlap under CUDA graph capture")
    main = torch.cuda.current_stream()
    comm = get_stream(_COMM_STREAM)
    comm.wait_stream(main)
    if isinstance(hidden_states, UnreducedArrival):
        source = _ArrivingInput(hidden_states, chunks=chunks)
    else:
        source = _ShardInput(hidden_states, chunks=chunks)
    bounds = source.bounds
    gathered = _gathered_chunks(source.template, rows=source.rows, bounds=bounds)
    output = source.template.new_empty((source.rows, source.template.shape[-1]))
    # Every tensor the comm stream touches stays referenced here until main
    # waits on the last reduce-scatter.
    inputs, computed = [], []
    done = None
    source.reduce(0, comm=comm)
    source.reduce(1, comm=comm)
    ready = _all_gather(source, gathered, inputs, 0, main=main, comm=comm)
    for k, (start, end) in enumerate(bounds):
        # Queue the next chunk's read and gather before compute: the MoE syncs
        # the host.
        following = _all_gather(source, gathered, inputs, k + 1, main=main, comm=comm)
        source.reduce(k + 2, comm=comm)
        main.wait_event(ready)
        computed.append(compute(gathered[k]))
        done = _reduce_scatter(output[start:end], computed[k], k, main=main, comm=comm)
        ready = following
    if done is not None:
        main.wait_event(done)
    return output


class _ShardInput:
    """An input already on this rank's CP shard."""

    def __init__(self, hidden_states, *, chunks):
        self.template = hidden_states.contiguous()
        self.rows = self.template.shape[0]
        self.bounds = shard_chunk_bounds(self.rows, chunks)

    def reduce(self, k, *, comm):
        pass

    def read(self, k, *, main):
        start, end = self.bounds[k]
        return self.template[start:end], None


class _ArrivingInput:
    """An input owing its attention-CP sum: each row chunk is reduce-scattered
    on the comm stream, then read on main."""

    def __init__(self, arrival: UnreducedArrival, *, chunks):
        self.arrival = arrival
        self.template = arrival.partial
        self.size = get_parallel().attn_cp_size
        self.rows = arrival.partial.shape[0] // self.size
        self.bounds = shard_chunk_bounds(self.rows, chunks)
        self.reduced = {}

    def reduce(self, k, *, comm):
        if k >= len(self.bounds):
            return
        start, end = self.bounds[k]
        event = _tbo_event(("ffn_cp_overlap_reduce", k))
        with torch.cuda.stream(comm):
            shard = attn_cp_interleave_reduce_scatter(
                self.arrival.partial[start * self.size : end * self.size]
            )
            event.record(comm)
        self.reduced[k] = (shard, event)

    def read(self, k, *, main):
        start, end = self.bounds[k]
        shard, reduced = self.reduced[k]
        main.wait_event(reduced)
        local = self.arrival.read_rows(shard, start=start, end=end)
        event = _tbo_event(("ffn_cp_overlap_read", k))
        event.record(main)
        return local, event


def _gathered_chunks(template, *, rows, bounds):
    """One buffer for the gathered rows, viewed as each chunk's rank-major
    rows."""
    parallel = get_parallel()
    size = parallel.attn_cp_size
    with use_symmetric_memory(
        parallel.attn_cp_group, disabled=not is_allocation_symmetric()
    ):
        gathered = template.new_empty((rows * size, template.shape[-1]))
    return [gathered[start * size : end * size] for start, end in bounds]


def _all_gather(source, gathered, inputs, k, *, main, comm):
    if k >= len(source.bounds):
        return None
    local, read = source.read(k, main=main)
    inputs.append(local)
    event = _tbo_event(("ffn_cp_overlap_gather", k))
    with torch.cuda.stream(comm):
        if read is not None:
            comm.wait_event(read)
        attn_cp_all_gather_into_tensor(gathered[k], local)
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
