"""INF-874 PCOLL: prefill collectives that the model does not need.

Three paths, each behind its own environment switch (all off by default) and each
giving the bits of the path it replaces:

* P1 ``SGLANG_OPT_DSV4_CP_REPLICATED_EMBEDDING``: ``ReplicatedVocabEmbedding`` keeps
  the whole token-embedding table on every rank, so the step-start vocab-parallel
  all-reduce goes away.
* P2 ``SGLANG_OPT_DSV4_ENGRAM_PREFETCH_SHARED_LOCAL``: see ``layers/engram.py``; each
  rank copies its own engram rows from the shared host table (no integer
  reduce-scatter).
* P3-bit ``SGLANG_OPT_DSV4_CP_KV_SOURCE_PROJECTION_ALLGATHER``:
  ``kv_source_projection_allgather`` all-gathers a kv_source compressor's projections
  instead of the hidden state.

``enabled(name)`` is the per-forward gate of a path already switched on by its env
(names "embed", "engram", "kvsrc"). It is always True here; it exists so that a
measurement build can run the replaced path in the same process (the replaced path
still works on the new weight layouts and gives the same outputs).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.environ import envs
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding


def enabled(name: str) -> bool:
    """Per-forward gate of an env-enabled PCOLL path; True outside measurement builds."""
    del name
    return True


# ---- P1: replicated token embedding ------------------------------------------------


class ReplicatedVocabEmbedding(VocabParallelEmbedding):
    """The token embedding with the whole table on every rank (``enable_tp=False``).

    The vocab-parallel lookup it replaces returns the owner rank's row plus zeros
    from the other ranks (bf16 all-reduce), which is the row itself except that -0
    comes out +0. ``finish_load`` rewrites the table's -0 entries as +0 once, so the
    plain lookup here gives the same bits.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        tp_size: int,
        tp_rank: int,
        prefix: str = "",
    ):
        super().__init__(num_embeddings, embedding_dim, enable_tp=False, prefix=prefix)
        # The id range this rank owned under the vocab-parallel layout; only the
        # replaced path (enabled("embed") False) uses it, and only for its cost:
        # any partition gives the same sum.
        self._legacy_lo = num_embeddings * tp_rank // tp_size
        self._legacy_hi = num_embeddings * (tp_rank + 1) // tp_size
        self._legacy_tp_size = tp_size

    @torch.no_grad()
    def finish_load(self) -> None:
        w = self.weight.data
        # (w == 0) is True for -0 and +0; writing +0 turns -0 into +0.
        w.masked_fill_(w == 0, 0.0)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if enabled("embed") or self._legacy_tp_size == 1:
            return super().forward(input_)
        # The replaced path: this rank's id range, zeros elsewhere, all-reduce.
        owned = (input_ >= self._legacy_lo) & (input_ < self._legacy_hi)
        out = F.embedding(torch.where(owned, input_, 0).long(), self.weight)
        out.masked_fill_(~owned.unsqueeze(-1), 0)
        return tensor_model_parallel_all_reduce(out)


def replicated_embedding_wanted(config, quant_config, tp_size: int) -> bool:
    """P1 applies to an unquantized, untied embedding under TP > 1."""
    return (
        envs.SGLANG_OPT_DSV4_CP_REPLICATED_EMBEDDING.get()
        and tp_size > 1
        and quant_config is None
        and not getattr(config, "tie_word_embeddings", False)
    )


# ---- P3-bit: kv_source projections instead of the hidden state ---------------------


def local_rows_global_index(forward_batch, cp_rank: int, cp_size: int) -> Optional[torch.Tensor]:
    """Global token positions of this rank's logical rows under the interleave split
    (token t lives on rank t % cp_size as local row t // cp_size), or None for a
    layout this module does not handle (a packed tail gather index)."""
    md = forward_batch.attn_cp_metadata
    if getattr(md, "gather_index", None) is not None:
        return None
    logical = md.per_rank_logical_token or md.per_rank_actual_token
    n = int(logical[cp_rank])
    total = int(md.total_seq_lens)
    idx = torch.arange(n, dtype=torch.int64, device=forward_batch.positions.device)
    idx = idx * cp_size + cp_rank
    assert n == 0 or int(cp_rank + (n - 1) * cp_size) < total, (n, cp_rank, total)
    return idx


def kv_source_projection_allgather(
    compressor,
    x_local: torch.Tensor,
    forward_batch,
    *,
    cp_rank: int,
    cp_size: int,
    gather,
) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """``compressor.project`` of the all-gathered hidden rows, without gathering them.

    ``gather(t)`` is the CP all-gather into global token order (the caller passes
    ``cp_materialize_global_token_order``). Returns (kv, score) over the ``total``
    global rows, or None when the layout is not handled (the caller then gathers the
    hidden state as before).

    Bits: by default each rank runs the projection on a buffer of the all-gathered
    shape [total, hidden] holding its own rows at their global positions (other rows
    uninitialized). A GEMM row depends only on its input row and the launch, and the
    launch (shape, strides, dtype, alignment, tile placement of the row) is the one the
    all-gathered path makes, so each own row gets the same bits; the all-gather then
    hands every rank every row. SGLANG_OPT_DSV4_CP_KV_SOURCE_PROJECTION_LOCAL_M projects
    only the own rows (M / cp_size) for steps of at least ..._LOCAL_M_MIN_TOTAL rows:
    cheaper, but a different launch, so the bits are the same only where a sweep on the
    deployed GPU and cuBLAS showed it.
    """
    md = forward_batch.attn_cp_metadata
    total = int(md.total_seq_lens)
    rows = local_rows_global_index(forward_batch, cp_rank, cp_size)
    if rows is None:
        return None
    n = rows.shape[0]
    x_mine = x_local[:n]
    if (
        envs.SGLANG_OPT_DSV4_CP_KV_SOURCE_PROJECTION_LOCAL_M.get()
        and total >= envs.SGLANG_OPT_DSV4_CP_KV_SOURCE_PROJECTION_LOCAL_M_MIN_TOTAL.get()
    ):
        kv, score = compressor.project(x_mine.contiguous())
    else:
        # [phys, cp, hidden]: global row t = j * cp + rank sits at [j, rank]; the GEMM
        # reads the first `total` rows of the contiguous buffer, the all-gathered shape.
        phys = -(-total // cp_size)
        buf = x_local.new_empty((phys * cp_size, x_local.shape[1]))
        buf.view(phys, cp_size, -1)[:n, cp_rank].copy_(x_mine)
        kv, score = compressor.project(buf[:total])
        kv = kv[cp_rank::cp_size][:n]
        score = None if score is None else score[cp_rank::cp_size][:n]
    if score is None:
        return gather(kv.contiguous())[:total], None
    width = kv.shape[1]
    assert kv.dtype == score.dtype, (kv.dtype, score.dtype)
    both = gather(torch.cat([kv, score], dim=1))[:total]
    return both[:, :width].contiguous(), both[:, width:].contiguous()
