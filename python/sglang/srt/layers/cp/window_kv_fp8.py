"""Prefill-CP (interleave) window KV store through an FP8 all-gather.

``MqaAttention._cp_store_window_kv`` gathers the raw bf16 kv of every rank, puts it
back in token order and norm/RoPE/quantizes all T rows on every rank. Here each rank
runs ``fused_k_norm_rope_flashmla`` on its own rows into a send buffer of V4 pages,
the ranks all-gather those pages (584 B/row instead of 1,024 B), and one copy kernel
moves every token's row into the paged SWA cache. The cache bytes do not change.
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.attention.dsv4.cp_window_kv_fp8 import (
    SEND_PAGE_BYTES,
    SEND_PAGE_SIZE,
    scatter_gathered_window_kv,
    send_buffer_pages,
)
from sglang.kernels.ops.attention.dsv4.elementwise import fused_k_norm_rope_flashmla
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    is_allocation_symmetric,
)
from sglang.srt.runtime_context import get_parallel


def quantize_local_window_kv(
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    local_positions: torch.Tensor,
    num_local: int,
    rank_rows: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Norm + RoPE + FP8 of this rank's ``num_local`` rows into ``out``, the send
    buffer ``[send_buffer_pages(rank_rows), SEND_PAGE_BYTES]`` uint8; local row j is
    page ``j // 64``, slot ``j % 64``. Rows past ``num_local`` are left unwritten:
    no token's gather index points at them."""
    assert out.shape == (send_buffer_pages(rank_rows), SEND_PAGE_BYTES), out.shape
    if num_local:
        fused_k_norm_rope_flashmla(
            kv=kv[:num_local],
            kv_weight=kv_weight,
            eps=eps,
            freqs_cis=freqs_cis,
            positions=local_positions[:num_local],
            out_loc=torch.arange(num_local, dtype=torch.int32, device=kv.device),
            kvcache=out,
            page_size=SEND_PAGE_SIZE,
            layout=KVLayout.V4,
        )
    return out


def cp_store_window_kv_fp8(
    *,
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    local_positions: torch.Tensor,
    swa_loc: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_page_size: int,
    forward_batch,
) -> None:
    """The window-KV store of every token of the CP chunk, from this rank's rows.

    :param kv: this rank's ``[rows, 512]`` bf16 kv before the norm (rows past the
        rank's logical count are padding).
    :param local_positions: this rank's token positions, aligned with ``kv``.
    :param swa_loc: ``[total]`` SWA slot of every token in global order.
    """
    metadata = forward_batch.attn_cp_metadata
    parallel = get_parallel()
    cp_size, cp_rank = parallel.attn_cp_size, parallel.attn_cp_rank
    total = int(metadata.total_seq_lens)
    logical = metadata.per_rank_logical_token or metadata.per_rank_actual_token
    num_local = int(logical[cp_rank])
    rank_rows = max(metadata.per_rank_actual_token)
    if rank_rows == 0:
        return
    pages = send_buffer_pages(rank_rows)
    with use_symmetric_memory(
        parallel.attn_cp_group, disabled=not is_allocation_symmetric()
    ):
        send = torch.empty((pages, SEND_PAGE_BYTES), dtype=torch.uint8, device=kv.device)
        gathered = torch.empty(
            (cp_size * pages, SEND_PAGE_BYTES), dtype=torch.uint8, device=kv.device
        )
    quantize_local_window_kv(
        kv, kv_weight, eps, freqs_cis, local_positions, num_local, rank_rows, send
    )
    attn_cp_all_gather_into_tensor(gathered, send)
    scatter_gathered_window_kv(
        gathered,
        metadata.gather_index,
        swa_loc,
        swa_cache,
        swa_page_size,
        num_tokens=total,
        rank_rows=rank_rows,
        cp_size=cp_size,
    )
