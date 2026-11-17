"""INF-834 helpers for the MoE expert repack at load (decoder segment W6).

SGLANG_OPT_INF834_MOE_PACK_DEFER_EMPTY_CACHE (REJECTED on 2026-11-17, do not ship: on the EP8 decoder it
shrank "avail mem" at load end from 84.10 to 74.55 GB and max_total_num_tokens by 15.5%; freeing per layer
matters for fragmentation): skip the torch.cuda.empty_cache() calls inside the
per-layer expert repack (two per layer plus one per 8-expert chunk). The freed raw-weight blocks stay in the
caching allocator and are reused by the next layer's packed tensors; get_available_gpu_memory() at
"Load weight end" calls empty_cache() itself, so the memory reported there and the KV pool sizing still see
every fully free segment returned, but not the segments that reuse left partly occupied.
SGLANG_OPT_INF834_MOE_PACK_SKIP_CHUNK_EMPTY_CACHE: the narrower variant, see dsv4_mxfp4_decode_moe_ep.pack_experts_ep.
Gate for both: max_total_num_tokens and "avail mem" at load end equal to native.

SGLANG_OPT_INF834_W6_PROFILE (diagnostic, not a lever): per-layer timing of the repack steps with a device
synchronize around each, logged on rank 0.
"""

import time

import torch

from sglang.srt.environ import envs


def defer_empty_cache() -> bool:
    return envs.SGLANG_OPT_INF834_MOE_PACK_DEFER_EMPTY_CACHE.get()


def repack_empty_cache() -> None:
    if not defer_empty_cache():
        torch.cuda.empty_cache()


class StepTimer:
    """Accumulates synchronized step times when SGLANG_OPT_INF834_W6_PROFILE=1; a no-op otherwise."""

    def __init__(self):
        self.on = envs.SGLANG_OPT_INF834_W6_PROFILE.get()
        self.steps = {}
        if self.on:
            torch.cuda.synchronize()
            self._t = time.perf_counter()

    def mark(self, name: str) -> None:
        if not self.on:
            return
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.steps[name] = self.steps.get(name, 0.0) + (now - self._t)
        self._t = now

    def summary(self) -> str:
        total = sum(self.steps.values())
        parts = ", ".join(f"{k}={v * 1000:.1f}" for k, v in self.steps.items())
        return f"total={total * 1000:.1f} ms ({parts})"
