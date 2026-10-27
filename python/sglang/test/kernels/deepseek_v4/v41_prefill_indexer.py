"""The DeepSeek-V4.1 torch prefill indexer over a real fp4 index-K pool, with the
query and head-weight projections replaced by identity."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import List, Tuple

import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer import store_fp4_index_k_cache
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer
from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import prefill_requests
from sglang.srt.layers.attention.dsv4.v41_indexer.types import PrefillInputs
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4IndexerPool

PAGE_SIZE = 64


def _multiples_of_eighth(*shape: int, device: str) -> torch.Tensor:
    return torch.randint(-8, 9, shape, device=device).to(torch.bfloat16) / 8


def make_prefill_indexer_case(
    *,
    heads: int,
    ratio: int,
    requests: List[Tuple[int, int]],
    topk: int,
    device: str = "cuda",
) -> Tuple[PrefillInputs, SimpleNamespace, torch.Tensor]:
    """``requests`` holds (cached tokens, new query rows) per request. Returns the
    inputs, the token-to-KV pool and req_to_token. Queries, head weights and keys
    are multiples of 1/8, so every dot product and head sum is exact in fp32."""
    lens_per_request = [(prefix + rows) // ratio for prefix, rows in requests]
    total = sum(lens_per_request)
    num_pages = max(1, -(-total // PAGE_SIZE))
    pages = torch.randperm(num_pages, device=device)
    locations = (
        pages[:, None] * PAGE_SIZE + torch.arange(PAGE_SIZE, device=device)
    ).flatten()[:total]
    table = torch.zeros(num_pages, PAGE_SIZE * 68, device=device, dtype=torch.uint8)
    store_fp4_index_k_cache(
        _multiples_of_eighth(total, 128, device=device),
        table,
        locations,
        page_size=PAGE_SIZE,
        rne=True,
    )
    index_pool = DeepSeekV4IndexerPool.__new__(DeepSeekV4IndexerPool)
    index_pool.use_fp4_indexer = True
    index_pool.index_k_with_scale_buffer = [table]
    index_pool.start_layer = 0
    index_pool.page_size = PAGE_SIZE
    pool = SimpleNamespace(
        get_low_ratio_index_k_dequant=index_pool.get_index_k_dequant,
        get_index_k_with_scale_buffer=index_pool.get_index_k_with_scale_buffer,
    )

    capacity = max(prefix + rows for prefix, rows in requests)
    req_to_token = torch.zeros(
        (len(requests), capacity), device=device, dtype=torch.int32
    )
    start = 0
    for r, lc in enumerate(lens_per_request):
        req_to_token[r, : lc * ratio : ratio] = (
            locations[start : start + lc] * ratio
        ).to(torch.int32)
        start += lc

    positions = torch.cat(
        [
            torch.arange(prefix, prefix + rows, device=device)
            for prefix, rows in requests
        ]
    )
    req_rows = torch.cat(
        [
            torch.full((rows,), r, device=device, dtype=torch.int64)
            for r, (_, rows) in enumerate(requests)
        ]
    )
    num_rows = positions.numel()
    indexer = SimpleNamespace(
        queries=lambda q_lora, freqs: q_lora,
        head_weights=lambda x: x,
        # The torch reference; it reads nothing from the module.
        scores=partial(DeepseekV41Indexer.scores, None),
        index_topk=topk,
    )
    rows_per_request = [rows for _, rows in requests]
    inputs = PrefillInputs(
        indexer=indexer,
        layer_id=0,
        compress_ratio=ratio,
        freqs_cis=torch.zeros(capacity, device=device),
        x=_multiples_of_eighth(num_rows, heads, device=device),
        q_lora=_multiples_of_eighth(num_rows, heads, 128, device=device),
        positions=positions,
        req_rows=req_rows,
        req_pool_indices=torch.arange(len(requests), device=device),
        kv_page_table=torch.empty((num_rows, 0), device=device, dtype=torch.int32),
        seq_lens_cpu=[prefix + rows for prefix, rows in requests],
        rows_per_request=rows_per_request,
        rows_per_request_device=torch.tensor(rows_per_request, device=device),
        out_raw_indices=torch.empty((num_rows, topk), device=device, dtype=torch.int32),
        out_page_indices=torch.empty(
            (num_rows, topk), device=device, dtype=torch.int32
        ),
    )
    return inputs, pool, req_to_token


def prefill_scores(
    *,
    inputs: PrefillInputs,
    token_to_kv_pool: SimpleNamespace,
    req_to_token: torch.Tensor,
    fused: bool,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """(query rows, [rows, lc] scores) of every chunk the prefill indexer scores."""
    with envs.SGLANG_OPT_DSV41_FUSED_PREFILL_INDEXER.override(fused):
        return [
            (chunk.tok, chunk.scores)
            for _, chunks in prefill_requests(
                inputs=inputs,
                token_to_kv_pool=token_to_kv_pool,
                req_to_token=req_to_token,
            )
            for chunk in chunks
        ]
