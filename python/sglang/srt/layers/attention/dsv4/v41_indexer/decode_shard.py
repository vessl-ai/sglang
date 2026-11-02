"""Request-sharded decode indexer (``SGLANG_OPT_DSV4_DECODE_INDEXER_SHARD``).

The V4.1 indexer heads are replicated across attention TP, so without this every
rank scores every decode / verify row against the same index K and selects the
same top-k. Here each rank takes a contiguous run of whole requests (all verify
rows of a request stay on one rank, so they still share each K tile), runs the
unchanged backend on those rows, and the ranks all-gather the selected pool slots
into the full ``out_page_indices``. A candidate source's block ids stay
rank-local: the consumers of the same forward score the same rows on the same
rank. Every rank calls the gather for every layer, whatever its share, so the
collective is uniform under CUDA graphs.
"""

from __future__ import annotations

import functools
from typing import Callable, Optional, TypeVar

import msgspec
import torch

from .types import DecodeInputs

T = TypeVar("T")


class RowShard(msgspec.Struct, frozen=True):
    rows: slice  # this rank's rows of the batch
    rows_per_rank: int  # rows each rank contributes to the gather (equal on all)


def plan_row_shard(
    num_rows: int, rows_per_request: int, rank: int, size: int
) -> RowShard:
    """Whole requests per rank: ``ceil(requests / size)`` each, the last ranks
    possibly fewer or none."""
    assert num_rows % rows_per_request == 0, (num_rows, rows_per_request)
    requests = num_rows // rows_per_request
    per_rank = -(-requests // size)
    start = min(rank * per_rank, requests) * rows_per_request
    stop = min((rank + 1) * per_rank, requests) * rows_per_request
    return RowShard(rows=slice(start, stop), rows_per_rank=per_rank * rows_per_request)


class RowSlicedMetadata:
    """The fields of ``PagedIndexerMetadata`` the Hopper decode paths read
    (``max_compressed_seq_len``, ``compressed_page_size``, ``page_table``), for a
    run of rows. The DeepGEMM plans of the full metadata describe all rows, so this
    view is only valid where DeepGEMM does not serve the decode indexer."""

    def __init__(self, metadata, rows: slice):
        self.page_size = metadata.page_size
        self.compressed_page_size = metadata.compressed_page_size
        self.page_table = metadata.page_table[rows]
        self.compressed_seq_lens = metadata.compressed_seq_lens[rows]
        # Graph-stable capacity: the full table's width, not the slice's rows.
        self.max_compressed_seq_len = metadata.max_compressed_seq_len


class DecodeRowSharder:
    def __init__(self, group):
        self.group = group  # GroupCoordinator of the attention TP ranks
        self.rank = group.rank_in_group
        self.size = group.world_size

    def run(self, inputs: DecodeInputs, fn: Callable[[DecodeInputs], T]) -> T:
        """``fn`` on this rank's rows of ``inputs``, then every rank's selection
        gathered into ``inputs.out_page_indices``."""
        num_rows = inputs.x.shape[0]
        shard = plan_row_shard(num_rows, inputs.rows_per_request, self.rank, self.size)
        out = inputs.out_page_indices
        # -1 padded like the full buffer; rows past this rank's share stay -1.
        local_out = out.new_full((shard.rows_per_rank, out.shape[1]), -1)
        rows = shard.rows
        local = DecodeInputs(
            indexer=inputs.indexer,
            layer_id=inputs.layer_id,
            compress_ratio=inputs.compress_ratio,
            freqs_cis=inputs.freqs_cis,
            x=inputs.x[rows],
            q_lora=inputs.q_lora[rows],
            positions=inputs.positions[rows],
            req_rows=inputs.req_rows[rows],
            paged_metadata=RowSlicedMetadata(inputs.paged_metadata, rows),
            is_verify=inputs.is_verify,
            rows_per_request=inputs.rows_per_request,
            out_page_indices=local_out,
        )
        result = fn(local)
        self._gather(local_out, out, num_rows)
        return result

    def _gather(self, local_out: torch.Tensor, out: torch.Tensor, num_rows: int):
        gathered_rows = local_out.shape[0] * self.size
        if out.is_contiguous() and gathered_rows == num_rows:
            # The requests split evenly: gather straight into the selection buffer.
            self.group.all_gather_into_tensor(out[:num_rows], local_out)
            return
        gathered = local_out.new_empty((gathered_rows, local_out.shape[1]))
        self.group.all_gather_into_tensor(gathered, local_out)
        out[:num_rows].copy_(gathered[:num_rows])


@functools.cache
def _decode_row_sharder_for_process() -> Optional[DecodeRowSharder]:
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import get_parallel

    if not envs.SGLANG_OPT_DSV4_DECODE_INDEXER_SHARD.get():
        return None
    par = get_parallel()
    if par.attn_dp_size != 1 or par.attn_cp_size != 1 or par.attn_tp_size <= 1:
        return None
    return DecodeRowSharder(par.attn_tp_group)


def get_decode_row_sharder(use_deep_gemm_decode: bool) -> Optional[DecodeRowSharder]:
    """The sharder when the env is on, the device is Hopper and the topology allows
    it, else None. The DeepGEMM decode path (SM100) keeps its whole-batch plans and
    is never sharded."""
    from sglang.srt.runtime_context import get_platform

    # Validated on the Hopper paged scorer only.
    if use_deep_gemm_decode or not get_platform().is_sm90:
        return None
    return _decode_row_sharder_for_process()
