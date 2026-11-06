"""INF-730: intra-node expert-parallel dispatch / combine over NVLink peer pointers. See ``jit/csrc/moe/ep_a2a.cuh``.

One kernel per direction, no host sync, CUDA-graph safe, deterministic (receive order = token order of each source;
combine sums the destinations in rank order in fp32).

Usage on every rank of an EP group (all ranks must make the same calls in the same order)::

    a2a = EpA2A(group, num_experts=384, top_k=6, max_tokens=640)        # once, collective
    recv = a2a.dispatch(x8, xscale, topk_ids, topk_weights)              # x8 [T, 5120] e4m3, xscale [T, 1] fp32
    # recv.x / recv.scale / recv.topk_ids (local ids, -1 = not here) / recv.topk_weights: [W * max_tokens, ...]
    # rows of source s live at [s * S, s * S + recv.counts[s]) with S = region_rows (default max_tokens); every other
    # row has ids -1. With region_rows = T (all ranks pass the same T, e.g. MAX_LEN padding) the receive arrays are
    # [W * T, ...], the shape an all-gather would give.
    expert_kernel(recv..., out=a2a.partial_out)    # one BF16 row per received row: sum_k w_k * expert_k(x) (local k)
    y = a2a.combine()                              # [T, 5120] BF16 on the source rank: sum over destinations

The payload format is opaque bytes (row of x_bytes and row of s_bytes), so FP8 + per-row scale (the SM90 W4A8 decode
kernel's own quantization), FP8 + per-128 scales, or BF16 without scale all work.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

MAX_RANKS = 8
MAX_BLOCKS = 160
MAX_TOKENS = 2048
_ALIGN = 256


@cache_once
def _module() -> Module:
    return load_jit(
        "ep_a2a",
        cuda_files=["moe/ep_a2a.cuh"],
        cuda_wrappers=[
            ("dispatch", "ep_a2a::dispatch"),
            ("combine", "ep_a2a::combine"),
        ],
    )


def _align(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


@dataclasses.dataclass
class EpRecv:
    x: (
        torch.Tensor
    )  # [W * max_tokens, x_cols] (input dtype), rows in the receive layout
    scale: torch.Tensor  # [W * max_tokens, s_cols] (input dtype)
    topk_ids: torch.Tensor  # [W * max_tokens, K] int32, this rank's local expert ids, -1 = elsewhere / stale row
    topk_weights: torch.Tensor  # [W * max_tokens, K] fp32
    src_token: torch.Tensor  # [W * max_tokens] int32, token index on the source rank
    counts: torch.Tensor  # [W] int32, rows received from each source this call


class EpA2A:
    def __init__(
        self,
        group: dist.ProcessGroup,
        *,
        num_experts: int,
        top_k: int,
        max_tokens: int,
        x_bytes: int = 5120,
        s_bytes: int = 4,
        hidden: int = 5120,
        expert_rank: Optional[torch.Tensor] = None,
        expert_local: Optional[torch.Tensor] = None,
        dispatch_blocks: int = 32,
        combine_blocks: int = 64,
        device: Optional[torch.device] = None,
    ) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        self.group = group
        self.world = group.size()
        self.rank = group.rank()
        W = self.world
        if not 1 <= W <= MAX_RANKS:
            raise ValueError(f"EpA2A: 1..{MAX_RANKS} ranks, got {W}")
        if not 0 < max_tokens <= MAX_TOKENS:
            raise ValueError(f"EpA2A: max_tokens 1..{MAX_TOKENS}, got {max_tokens}")
        if x_bytes % 16 or s_bytes % 4:
            raise ValueError(
                "EpA2A: x row must be a multiple of 16 bytes, s row of 4 bytes"
            )
        if not (
            1 <= dispatch_blocks <= MAX_BLOCKS and 1 <= combine_blocks <= MAX_BLOCKS
        ):
            raise ValueError(f"EpA2A: 1..{MAX_BLOCKS} blocks")
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.num_experts, self.top_k, self.tmax = num_experts, top_k, max_tokens
        self.x_bytes, self.s_bytes, self.hidden = x_bytes, s_bytes, hidden
        self.dispatch_blocks, self.combine_blocks = dispatch_blocks, combine_blocks
        rows = W * max_tokens
        off = 0
        lay = {}
        for name, nbytes in (
            ("x", rows * x_bytes),
            ("s", rows * s_bytes),
            ("ids", rows * top_k * 4),
            ("w", rows * top_k * 4),
            ("src", rows * 4),
            ("cnt", MAX_BLOCKS * MAX_RANKS * 4),
            ("dflag", MAX_BLOCKS * MAX_RANKS * 4),
            ("cflag", MAX_BLOCKS * MAX_RANKS * 4),
            ("y", rows * hidden * 2),
        ):
            lay[name] = off
            off = _align(off + nbytes)
        self.layout = lay
        self.nbytes = off
        self.buf = symm_mem.empty(off, dtype=torch.uint8, device=self.device)
        self.buf.zero_()
        torch.cuda.synchronize(self.device)
        handle = symm_mem.rendezvous(self.buf, group.group_name)
        self.peers = torch.tensor(
            [int(handle.buffer_ptrs[i]) for i in range(W)],
            dtype=torch.int64,
            device=self.device,
        )
        per = num_experts // W
        if expert_rank is None:
            if num_experts % W:
                raise ValueError(
                    "EpA2A: give expert_rank/expert_local when experts do not split evenly"
                )
            e = torch.arange(num_experts, dtype=torch.int32, device=self.device)
            expert_rank, expert_local = e // per, e % per
        self.expert_rank = expert_rank.to(self.device, torch.int32).contiguous()
        self.expert_local = expert_local.to(self.device, torch.int32).contiguous()
        self.slot_map = torch.full(
            (max_tokens, W), -1, dtype=torch.int32, device=self.device
        )
        self.dctr = torch.zeros(MAX_BLOCKS, dtype=torch.int32, device=self.device)
        self.cctr = torch.zeros(MAX_BLOCKS, dtype=torch.int32, device=self.device)
        self.recv_counts = torch.zeros(W, dtype=torch.int32, device=self.device)
        self._num_tokens = 0
        self._stride = max_tokens
        _module()
        torch.cuda.synchronize(self.device)
        dist.barrier(group=group)

    def _region(self, name: str, nbytes: int) -> torch.Tensor:
        o = self.layout[name]
        return self.buf[o : o + nbytes]

    def _offsets(self):
        L = self.layout
        return (
            L["x"],
            L["s"],
            L["ids"],
            L["w"],
            L["src"],
            L["cnt"],
            L["dflag"],
            L["cflag"],
            L["y"],
        )

    @property
    def partial_out(self) -> torch.Tensor:
        """[W * S, hidden] BF16 (S = region rows of the last dispatch), in the receive layout: the expert kernel
        writes one row per received row (the weighted sum over this rank's experts of that token)."""
        rows = self.world * self._stride
        return (
            self._region("y", rows * self.hidden * 2)
            .view(torch.bfloat16)
            .view(rows, self.hidden)
        )

    def dispatch(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        mask_stale: bool = True,
        region_rows: Optional[int] = None,
    ) -> EpRecv:
        """region_rows: rows per source region (same on every rank, T <= region_rows <= max_tokens); default
        max_tokens. x [T, x_cols], scale [T, s_cols] (any dtypes, rows of x_bytes / s_bytes), topk_ids [T, K]
        int32/int64 global expert ids (-1 = none), topk_weights [T, K] fp32."""
        T = x.shape[0]
        stride = self.tmax if region_rows is None else int(region_rows)
        if not T <= stride <= self.tmax:
            raise ValueError(
                f"EpA2A.dispatch: need tokens {T} <= region_rows {stride} <= max_tokens {self.tmax}"
            )
        if (
            x.shape[1] * x.element_size() != self.x_bytes
            or scale.shape[1] * scale.element_size() != self.s_bytes
        ):
            raise ValueError(
                "EpA2A.dispatch: x / scale row bytes differ from the constructor's"
            )
        if topk_ids.shape[1] != self.top_k or topk_weights.dtype != torch.float32:
            raise ValueError("EpA2A.dispatch: topk_ids [T, top_k], topk_weights fp32")
        _module().dispatch(
            self.peers,
            x.contiguous(),
            scale.contiguous(),
            topk_ids.contiguous(),
            topk_weights.contiguous(),
            self.expert_rank,
            self.expert_local,
            self.slot_map,
            self.dctr,
            self.recv_counts,
            self.rank,
            stride,
            *self._offsets(),
            self.dispatch_blocks,
            int(mask_stale),
        )
        self._num_tokens = T
        self._stride = stride
        rows = self.world * stride
        K = self.top_k
        return EpRecv(
            x=self._region("x", rows * self.x_bytes)
            .view(x.dtype)
            .view(rows, x.shape[1]),
            scale=self._region("s", rows * self.s_bytes)
            .view(scale.dtype)
            .view(rows, scale.shape[1]),
            topk_ids=self._region("ids", rows * K * 4).view(torch.int32).view(rows, K),
            topk_weights=self._region("w", rows * K * 4)
            .view(torch.float32)
            .view(rows, K),
            src_token=self._region("src", rows * 4).view(torch.int32),
            counts=self.recv_counts,
        )

    def combine(self, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sum, for every token of the last dispatch, the partial_out rows of the ranks it went to (fp32, rank
        order) -> out [T, hidden] BF16. Rows of out past T are not written."""
        T = self._num_tokens
        if out is not None and (
            out.shape[0] < T
            or out.shape[1] != self.hidden
            or out.dtype != torch.bfloat16
        ):
            raise ValueError("EpA2A.combine: out must be bf16 [>= T, hidden]")
        if out is not None and out.shape[0] != T:
            out = out[:T]
        if out is None:
            out = torch.empty(T, self.hidden, dtype=torch.bfloat16, device=self.device)
        _module().combine(
            self.peers,
            self.slot_map,
            self.cctr,
            out,
            self.rank,
            self._stride,
            *self._offsets(),
            self.combine_blocks,
        )
        return out
