"""INF-730: the communication seam of the expert-parallel DeepSeek-V4 decode MoE under attention DP.

With ``--moe-a2a-backend none`` and ``--ep-size == --tp-size == --attn-dp-size`` (attention TP 1), the MoE of
``DeepseekV4DecoderLayer._run_moe_ffn_dp_sync`` all-gathers every DP rank's BF16 rows, routes all of them on
every rank, runs this rank's experts on every row and reduce-scatters the partial sums. With
``SGLANG_OPT_DSV4_EP_MOE_COMM`` set, a decode forward (MAX_LEN padding) of a layer whose experts run
``dsv4_mxfp4_decode_moe_ep`` (SGLANG_OPT_DSV4_SM90_DECODE_MOE_W4A8_EP) instead does:

  1. route the rank's own rows (router GEMM + top-k on its n rows instead of all ranks' rows);
  2. quantize them the way the experts read them: e4m3 with one power-of-2 fp32 scale per row (``_row_quant``);
  3. ``dispatch`` -> the rows this rank's experts need, with their LOCAL expert ids;
  4. this rank's experts on those rows (``decode_moe_ep`` with the received quantization);
  5. ``combine`` -> every local row's sum over the ranks (bf16 [n, 5120]);

the shared expert stays on the local rows as before (SGLANG_DP_SHARED_EXPERT_LOCAL is required).

The dispatch/combine pair is pluggable (``EpDecodeComm``):

- ``ag``: all-gather of (rows, scales, ids, weights) over the TP group and reduce-scatter of the partial sums.
  Every rank receives every row as in the default path, but in e4m3 (5,124 + 8K bytes per row instead of
  10,240) and with the router on the local rows only. The reference implementation of the protocol.
- ``<module>:<factory>``: any implementation of the protocol (INF-730 track b-a2a's dispatch/combine kernels),
  constructed once per process as ``factory(group=<TP GroupCoordinator>, hidden=5120)``.

An implementation must be CUDA-graph capturable (no host sync, fixed shapes per batch size) and make the same
symmetric-memory allocations on every rank in the same order.
"""

from __future__ import annotations

import functools
import importlib
import logging
from typing import Any, NamedTuple, Optional, Protocol

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class EpDispatchOutput(NamedTuple):
    """What ``dispatch`` hands to this rank's experts. R rows (fixed per batch size)."""

    x8: torch.Tensor  # [R, 5120] uint8: e4m3 bits of the rows
    xscale: torch.Tensor  # [R] fp32: the power-of-2 scale of each row
    topk_ids: torch.Tensor  # [R, K] int32: LOCAL expert ids on this rank, -1 for other ranks' experts / pad rows
    topk_weights: torch.Tensor  # [R, K] fp32 (routed scaling factor already applied)
    handle: Any  # whatever ``combine`` needs


class EpDecodeComm(Protocol):
    name: str

    def dispatch(
        self,
        x8: torch.Tensor,
        xscale: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        num_local_experts: int,
        ep_rank: int,
    ) -> EpDispatchOutput:
        """x8 [n, 5120] uint8, xscale [n] fp32, topk_ids [n, K] int32 PHYSICAL expert ids (-1: no expert, e.g.
        a pad row), topk_weights [n, K] fp32. Expert p lives on rank p // num_local_experts as local expert
        p % num_local_experts. n is the same on every rank (MAX_LEN padding)."""
        ...

    def combine_buffer(self, dispatched: EpDispatchOutput) -> torch.Tensor:
        """[R, 5120] bf16 that the experts write this rank's partial sums into (e.g. symmetric memory)."""
        ...

    def combine(
        self, partial: torch.Tensor, dispatched: EpDispatchOutput, out: torch.Tensor
    ) -> torch.Tensor:
        """out [n, 5120] bf16: for every local row, the sum over the ranks of that row's partial sums."""
        ...


def local_expert_ids(
    topk_ids: torch.Tensor, num_local_experts: int, ep_rank: int
) -> torch.Tensor:
    """Physical expert ids -> this rank's local ids (-1 for experts of other ranks and for -1)."""
    lo = ep_rank * num_local_experts
    mine = (topk_ids >= lo) & (topk_ids < lo + num_local_experts)
    return torch.where(mine, topk_ids - lo, -1).to(torch.int32)


class AllGatherEpComm:
    """Reference ``EpDecodeComm``: one grouped all-gather of the four arrays and one reduce-scatter, both over
    the TP group (= the EP group here)."""

    name = "ag"

    def __init__(self, group, hidden: int = 5120):
        self.group = group
        self.hidden = hidden

    def dispatch(
        self, x8, xscale, topk_ids, topk_weights, *, num_local_experts, ep_rank
    ) -> EpDispatchOutput:
        gx8, gxs, gids, gw = self.group.all_gatherv(
            [x8.contiguous(), xscale.contiguous(), topk_ids, topk_weights]
        )
        local = local_expert_ids(gids, num_local_experts, ep_rank)
        return EpDispatchOutput(gx8, gxs, local, gw, None)

    def combine_buffer(self, dispatched: EpDispatchOutput) -> torch.Tensor:
        from sglang.srt.distributed.device_communicators.pynccl_allocator import (
            use_symmetric_memory,
        )
        from sglang.srt.layers.dp_attention import is_allocation_symmetric

        with use_symmetric_memory(self.group, disabled=not is_allocation_symmetric()):
            return torch.empty(
                dispatched.topk_ids.shape[0],
                self.hidden,
                dtype=torch.bfloat16,
                device=dispatched.topk_ids.device,
            )

    def combine(self, partial, dispatched, out) -> torch.Tensor:
        self.group.reduce_scatter_tensor(out, partial)
        return out


@functools.cache
def get_ep_decode_comm() -> Optional[EpDecodeComm]:
    spec = envs.SGLANG_OPT_DSV4_EP_MOE_COMM.get()
    if not spec:
        return None
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group
    if spec == "ag":
        comm = AllGatherEpComm(group)
    else:
        module, _, attr = spec.partition(":")
        if not attr:
            raise ValueError(
                "SGLANG_OPT_DSV4_EP_MOE_COMM must be 'ag' or '<module>:<factory>', "
                f"got {spec!r}"
            )
        comm = getattr(importlib.import_module(module), attr)(group=group, hidden=5120)
    logger.info("INF-730 EP decode MoE communication: %s", getattr(comm, "name", spec))
    return comm


_WARNED_SHARED = False


def applies(mlp, forward_batch, *, shared_local: bool) -> bool:
    """Whether this MoE call takes the EP decode path: the switch is set, the layer's experts run
    dsv4_mxfp4_decode_moe_ep, the forward is a MAX_LEN-padded DP forward (decode), attention TP is 1 and the
    TP group is the EP group. The answer must be the same on every rank (it only reads configuration and the
    padding mode, which all ranks share)."""
    global _WARNED_SHARED
    if get_ep_decode_comm() is None:
        return False
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend
    from sglang.srt.runtime_context import get_parallel

    method = getattr(getattr(mlp, "experts", None), "quant_method", None)
    if not getattr(method, "_ep_split", 0):
        return False
    p = get_parallel()
    if not (
        get_moe_a2a_backend().is_none()
        and p.attn_tp_size == 1
        and p.attn_cp_size == 1
        and p.moe_ep_size == p.tp_size == p.attn_dp_size
        and forward_batch.dp_padding_mode is not None
        and forward_batch.dp_padding_mode.is_max_len()
    ):
        return False
    if not shared_local and getattr(mlp, "shared_experts", None) is not None:
        if not _WARNED_SHARED:
            _WARNED_SHARED = True
            logger.warning(
                "SGLANG_OPT_DSV4_EP_MOE_COMM needs the shared expert on the local rows "
                "(SGLANG_SHARED_EXPERT_TP1=1 and SGLANG_DP_SHARED_EXPERT_LOCAL=1); "
                "keeping the all-gather path."
            )
        return False
    return True


def _mask_local_pad_rows(topk_ids: torch.Tensor) -> None:
    """MAX_LEN padding: rows at or past this rank's token count carry stale values; give them no expert."""
    from sglang.srt.layers.dp_attention import get_dp_global_num_tokens_gpu
    from sglang.srt.runtime_context import get_parallel

    counts = get_dp_global_num_tokens_gpu()
    if counts is None:
        return
    real = counts[get_parallel().attn_dp_rank]
    rows = torch.arange(topk_ids.shape[0], device=topk_ids.device)
    topk_ids.masked_fill_((rows >= real).unsqueeze(1), -1)


def forward_ep_decode_moe(
    mlp, hidden_states: torch.Tensor, input_ids: Optional[torch.Tensor]
) -> torch.Tensor:
    """The routed-expert output of this rank's own rows ([n, 5120] bf16, MAX_LEN-padded local buffer); the
    caller adds the shared expert."""
    from sglang.kernels.ops.moe.dsv4_mxfp4_decode_moe_ep import decode_moe_ep
    from sglang.srt.layers.dp_attention import get_local_dp_buffer
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import _row_quant
    from sglang.srt.runtime_context import get_parallel

    comm = get_ep_decode_comm()
    method = mlp.experts.quant_method
    p = get_parallel()
    num_local = method._sm90_packed.w13.shape[0] // method._ep_split
    topk_output = mlp.route_rows(hidden_states, input_ids)
    topk_ids = topk_output.topk_ids.to(torch.int32).contiguous()
    _mask_local_pad_rows(topk_ids)
    x8, xs = _row_quant(hidden_states.contiguous())
    dispatched = comm.dispatch(
        x8.view(torch.uint8),
        xs,
        topk_ids,
        topk_output.topk_weights.float().contiguous(),
        num_local_experts=num_local,
        ep_rank=p.moe_ep_rank,
    )
    partial = comm.combine_buffer(dispatched)
    decode_moe_ep(
        None,
        dispatched.topk_ids,
        dispatched.topk_weights,
        method._sm90_packed,
        method.runner.config.swiglu_limit,
        method._ep_split,
        output=partial,
        row_quant=(dispatched.x8, dispatched.xscale),
    )
    out = get_local_dp_buffer(p.tp_group)
    assert out.shape[0] == hidden_states.shape[0], (out.shape, hidden_states.shape)
    return comm.combine(partial, dispatched, out)
