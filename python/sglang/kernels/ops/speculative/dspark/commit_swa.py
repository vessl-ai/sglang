"""Translate committed cache locations to the draft SWA pool."""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
_INF834_LOGGED = []


@triton.jit
def _committed_swa_locations(
    LOC,
    MAP,
    LENS,
    OUT,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    MAP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(LENS + i // WIDTH, i < N, other=0)
    committed = (i < N) & (i % WIDTH < length)
    loc = tl.load(LOC + i, committed, other=0).to(tl.int64)
    # Preserve torch indexing for an unused negative location.
    loc = tl.where(loc < 0, loc + MAP_SIZE, loc)
    swa = tl.load(MAP + loc, committed, other=-1).to(tl.int32)
    tl.store(OUT + i, swa, i < N)


@triton.jit
def _committed_swa_locations_rt(
    LOC,
    MAP,
    LENS,
    OUT,
    N,
    WIDTH: tl.constexpr,
    MAP_SIZE,
    BLOCK: tl.constexpr,
):
    # Same body as _committed_swa_locations with N and MAP_SIZE as runtime values:
    # one compiled kernel serves every batch size and KV pool size.
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(LENS + i // WIDTH, i < N, other=0)
    committed = (i < N) & (i % WIDTH < length)
    loc = tl.load(LOC + i, committed, other=0).to(tl.int64)
    # Preserve torch indexing for an unused negative location.
    loc = tl.where(loc < 0, loc + MAP_SIZE, loc)
    swa = tl.load(MAP + loc, committed, other=-1).to(tl.int32)
    tl.store(OUT + i, swa, i < N)


def committed_swa_locations(cache_loc, full_to_swa_mapping, commit_lens, width):
    assert cache_loc.numel() == commit_lens.numel() * width
    out = torch.empty_like(cache_loc, dtype=torch.int32)
    if cache_loc.numel():
        from sglang.srt.environ import envs

        kernel = (
            _committed_swa_locations_rt
            if envs.SGLANG_OPT_INF834_COMMIT_SWA_RUNTIME_N.get()
            else _committed_swa_locations
        )
        if kernel is _committed_swa_locations_rt and not _INF834_LOGGED:
            _INF834_LOGGED.append(True)
            logger.info("INF834 commit_swa: runtime N and MAP_SIZE (one kernel for all batch sizes)")
        kernel[(triton.cdiv(cache_loc.numel(), 256),)](
            cache_loc,
            full_to_swa_mapping,
            commit_lens,
            out,
            cache_loc.numel(),
            width,
            full_to_swa_mapping.numel(),
            256,
        )
    return out
