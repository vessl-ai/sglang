"""SGLANG_OPT_DSV4_SPARSE_PREFILL_ASYNC_LENGTHS: SparsePrefillChunkCache.ensure_compressed
builds the same gather with the switch on and off, and with it on does not wait for
the stream (the pageable torch.tensor(list, device=cuda) copy did)."""

import time

import pytest
import torch

from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    SparsePrefillChunkCache,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_ENV = "SGLANG_OPT_DSV4_SPARSE_PREFILL_ASYNC_LENGTHS"


def _cache(seq_lens_list, query_lens_list):
    dev = "cuda"
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=dev)
    query_lens = torch.tensor(query_lens_list, dtype=torch.int32, device=dev)
    num_reqs = len(seq_lens_list)
    qsl = torch.zeros(num_reqs + 1, dtype=torch.int32, device=dev)
    qsl[1:] = torch.cumsum(query_lens, 0)
    num_qo = int(sum(query_lens_list))
    swa_offsets = torch.zeros(num_reqs + 1, dtype=torch.int32, device=dev)
    swa_offsets[1:] = torch.cumsum(torch.full((num_reqs,), 128, device=dev), 0)
    empty = torch.empty(0, dtype=torch.int32, device=dev)
    return SparsePrefillChunkCache(
        num_reqs=num_reqs,
        num_qo_tokens=num_qo,
        max_seq_len=max(seq_lens_list),
        swa_window_size=128,
        swa_page_size=256,
        seq_lens=seq_lens,
        query_start_loc=qsl,
        query_pos=torch.zeros(num_qo, dtype=torch.int32, device=dev),
        swa_token_ids=empty,
        swa_first_pos=torch.zeros_like(seq_lens),
        swa_gather_lens=torch.zeros_like(seq_lens),
        swa_offsets=swa_offsets,
        seq_lens_cpu=list(seq_lens_list),
    )


def _page_table(cache, ratio, c_page_size):
    # Rows within a request are duplicates; distinct random page ids per request.
    g = torch.Generator(device="cpu").manual_seed(ratio)
    blocks = (cache.max_seq_len // ratio) // c_page_size + 2
    per_req = torch.randint(0, 1 << 20, (cache.num_reqs, blocks), generator=g)
    rows = torch.repeat_interleave(
        per_req, torch.diff(cache.query_start_loc.cpu()).long(), dim=0
    )
    return rows.to(torch.int32).cuda()


@pytest.mark.parametrize("ratio,c_page_size", [(1, 256), (2, 128), (4, 64)])
def test_same_gather(monkeypatch, ratio, c_page_size):
    seq = [75000, 68352, 6912, 300, 1]
    q = [6904, 6904, 6912, 300, 1]
    out = {}
    for on in (False, True):
        monkeypatch.setenv(_ENV, "1" if on else "0")
        cache = _cache(seq, q)
        pt = _page_table(cache, ratio, c_page_size)
        g = cache.ensure_compressed(ratio, pt, c_page_size)
        torch.cuda.synchronize()
        out[on] = (g.flat_token_ids, g.compressed_base, g.swa_base)
    for a, b in zip(out[False], out[True]):
        assert torch.equal(a, b)


def test_no_stream_wait(monkeypatch):
    seq = [75000, 68352, 6912]
    q = [6904, 6904, 6912]
    waits = {}
    for on in (False, True):
        monkeypatch.setenv(_ENV, "1" if on else "0")
        cache = _cache(seq, q)
        pt = _page_table(cache, 2, 128)
        cache.ensure_compressed(2, pt, 128)  # warm the allocators
        torch.cuda.synchronize()
        cache.compressed.clear()
        torch.cuda._sleep(2_000_000_000)  # ~1 s of GPU time queued ahead
        t0 = time.perf_counter()
        g = cache.ensure_compressed(2, pt, 128)
        waits[on] = time.perf_counter() - t0
        torch.cuda.synchronize()
        assert g.flat_token_ids.numel() == sum(s // 2 for s in seq)
    assert waits[False] > 0.3, waits
    assert waits[True] < 0.05, waits
