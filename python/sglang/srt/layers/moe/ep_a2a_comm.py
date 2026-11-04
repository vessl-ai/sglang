"""INF-730: the NVLink peer-pointer EP dispatch/combine (``sglang.kernels.ops.moe.ep_a2a``) behind the
``EpDecodeComm`` protocol of the EP decode MoE seam (``sglang.srt.layers.moe.ep_decode_comm``, branch
inf730/ep8-decode).

Select it with ``SGLANG_OPT_DSV4_EP_MOE_COMM=sglang.srt.layers.moe.ep_a2a_comm:make``.

Compared with the all-gather reference (``ag``), every row goes only to the ranks that own one of its experts
(about 4.4 of 8 ranks at top-6 of 384 experts, rows deduplicated per rank), and the combine sums only those
ranks' partial rows (pulled over NVLink, fp32 in rank order, so the result does not depend on timing). The
receive arrays keep the all-gather shape [W * n, ...] (W = EP ranks, n = rows per rank of this call, the same on
every rank under MAX_LEN padding): source s's rows sit at the front of block s in token order, the rest of the
block has expert ids -1 and is skipped by the experts.

One ``EpA2A`` (one symmetric buffer) per (local experts, top-k) shape, built at the first eager call of that
shape; every rank reaches that call at the same layer, so the collective construction lines up. The capacity per
source block is ``SGLANG_OPT_DSV4_EP_A2A_MAX_ROWS`` rows (default 1024 = 256 requests x 4 verify rows).
"""

from __future__ import annotations

import os
from typing import Any, NamedTuple

import torch

from sglang.kernels.ops.moe.ep_a2a import EpA2A

try:  # the protocol's output type, when the seam is present
    from sglang.srt.layers.moe.ep_decode_comm import EpDispatchOutput
except ImportError:  # pragma: no cover - standalone use

    class EpDispatchOutput(NamedTuple):
        x8: torch.Tensor
        xscale: torch.Tensor
        topk_ids: torch.Tensor
        topk_weights: torch.Tensor
        handle: Any


class EpA2ADecodeComm:
    name = "ep_a2a"

    def __init__(self, group, hidden: int = 5120, max_rows: int | None = None):
        # group: sglang GroupCoordinator (its NCCL device group carries the symmetric-memory rendezvous) or a
        # torch ProcessGroup.
        self.pg = getattr(group, "device_group", group)
        self.hidden = hidden
        self.max_rows = int(max_rows or os.environ.get("SGLANG_OPT_DSV4_EP_A2A_MAX_ROWS", "1024"))
        self._a2a: dict = {}

    def _get(self, num_local_experts: int, top_k: int) -> EpA2A:
        key = (num_local_experts, top_k)
        a2a = self._a2a.get(key)
        if a2a is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "ep_a2a: the first call of a (local experts, top-k) shape must run eagerly (symmetric-memory "
                    "rendezvous), not inside CUDA graph capture"
                )
            world = self.pg.size()
            a2a = EpA2A(
                self.pg,
                num_experts=num_local_experts * world,
                top_k=top_k,
                max_tokens=self.max_rows,
                x_bytes=self.hidden,
                s_bytes=4,
                hidden=self.hidden,
                dispatch_blocks=int(os.environ.get("SGLANG_OPT_DSV4_EP_A2A_DISPATCH_BLOCKS", "16")),
                combine_blocks=int(os.environ.get("SGLANG_OPT_DSV4_EP_A2A_COMBINE_BLOCKS", "32")),
            )
            self._a2a[key] = a2a
        return a2a

    def dispatch(self, x8, xscale, topk_ids, topk_weights, *, num_local_experts, ep_rank) -> EpDispatchOutput:
        a2a = self._get(num_local_experts, topk_ids.shape[1])
        if ep_rank != a2a.rank:
            raise ValueError(f"ep_a2a: ep_rank {ep_rank} != group rank {a2a.rank}")
        n = x8.shape[0]
        recv = a2a.dispatch(
            x8, xscale.reshape(n, 1), topk_ids, topk_weights, mask_stale=True, region_rows=n
        )
        return EpDispatchOutput(
            recv.x, recv.scale.reshape(-1), recv.topk_ids, recv.topk_weights, a2a
        )

    def combine_buffer(self, dispatched: EpDispatchOutput) -> torch.Tensor:
        return dispatched.handle.partial_out

    def combine(self, partial, dispatched, out) -> torch.Tensor:
        a2a: EpA2A = dispatched.handle
        if partial.data_ptr() != a2a.partial_out.data_ptr():
            raise ValueError("ep_a2a: the experts must write into combine_buffer() (symmetric memory)")
        a2a.combine(out=out)
        return out


def make(group, hidden: int = 5120) -> EpA2ADecodeComm:
    return EpA2ADecodeComm(group, hidden=hidden)
