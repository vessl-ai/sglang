"""INF-730 SGLANG_OPT_DSV4_DP_FP8_MOE_GATHER: the DeepSeek-V4 TP decode MoE input all-gathered in FP8.

The i10 decoder (DP attention, attention TP 1, TP MoE, ``--moe-a2a-backend none``, MAX_LEN padding) all-gathers
every rank's BF16 rows into the global DP buffer (``tp_group.all_gather_into_tensor``, NCCL symmetric kernel), runs
the router and top-k on all W*n rows, and the SM90 W4A8 v2 decode experts quantize all W*n rows in their prologue
(e4m3, one power-of-2 fp32 scale per row) before the expert GEMMs. With this switch a decode forward instead does:

  1. ``begin``: copy this rank's rows into its slot of the global buffer (where the all-gather would put them) and,
     in the same kernel, quantize them with the v2 prologue's own code into a packed row
     [e4m3 5120 B | scale | K ids | K weights] (``fp8ag_quant``);
  2. the MoE's router and top-k run unchanged on the global buffer: same shape, same kernels, this rank's rows at
     the same position, so this rank's rows get the BF16 path's top-k (every per-row function -- the router GEMM,
     top-k -- gives a row the same bits whatever the other rows hold; the other slots hold stale rows here);
  3. ``RoutedRowsHook``: this rank's top-k into its packed rows, ONE all-gather of the packed rows over the TP group
     (same NCCL call as today, symmetric buffers, 5,184 B instead of 10,240 B per row at top 6), unpack, and the
     experts take the gathered e4m3 rows as their input (``decode_moe_prequant``: prep's sort, same experts kernel).

Hence top-k, expert input and MoE output are the BF16 path's bit for bit; only the bytes on the wire change. One
NCCL collective per layer as before, every allocation's shape depends only on the padded row count (the same on
every rank), so CUDA graph capture is unaffected. ``SGLANG_OPT_DSV4_DP_FP8_MOE_GATHER_VERIFY`` additionally runs the
BF16 path on eager forwards and logs layers whose MoE output differs (debug only, doubles the MoE).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_LOGGED = False
_VERIFY_STATS = {"layers": 0, "mismatch": 0}


def _v2_method(mlp):
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    method = getattr(getattr(mlp, "experts", None), "quant_method", None)
    if not isinstance(method, Mxfp4MarlinMoEMethod):
        return None
    if getattr(method, "_use_w4a8", False):
        return None
    if not (getattr(method, "_sm90_decode_w4a8", False) and getattr(method, "_sm90_decode_w4a8_v2", False)):
        return None
    return method


def _top_k(mlp) -> Optional[int]:
    """Columns of the MoE's top-k output (routed experts plus fused shared experts, 0 for DeepSeek-V4)."""
    k = getattr(mlp, "top_k", None)
    return None if k is None else k + int(getattr(mlp, "num_fused_shared_experts", 0) or 0)


def applies(mlp, hidden_states, local_hidden_states, forward_batch, *, shared_local: bool) -> bool:
    """Whether this decode MoE call takes the FP8 all-gather. Reads only configuration, the padding mode and the
    row counts, which every rank shares, so every rank decides the same."""
    global _LOGGED
    if not envs.SGLANG_OPT_DSV4_DP_FP8_MOE_GATHER.get():
        return False
    from sglang.kernels.ops.moe.dsv4_mxfp4_decode_moe_v2 import PREP_MAX_SLOTS
    from sglang.srt.layers.moe.mega_moe import should_use_mega_moe
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend
    from sglang.srt.runtime_context import get_exec, get_parallel

    p = get_parallel()
    W = p.tp_size
    n = local_hidden_states.shape[0]
    top_k = _top_k(mlp)
    ok = (
        shared_local
        and _v2_method(mlp) is not None
        and get_moe_a2a_backend().is_none()
        and not getattr(mlp, "_enable_a2a_moe", True)
        and p.attn_tp_size == 1
        and p.attn_cp_size == 1
        and p.attn_dp_size == W
        and forward_batch.dp_padding_mode is not None
        and forward_batch.dp_padding_mode.is_max_len()
        and not get_exec().moe.enable_eplb
        and n > 0
        and hidden_states.shape[0] == W * n
        and hidden_states.dtype == torch.bfloat16
        and local_hidden_states.dtype == torch.bfloat16
        and hidden_states.shape[1] == 5120
        and top_k is not None
        and W * n * top_k <= PREP_MAX_SLOTS
        and not should_use_mega_moe(mlp, hidden_states)
    )
    if ok and not _LOGGED:
        _LOGGED = True
        logger.info(
            "INF-730 DP FP8 MoE all-gather on (verify %s, first layer %s)",
            envs.SGLANG_OPT_DSV4_DP_FP8_MOE_GATHER_VERIFY.get(),
            getattr(mlp.experts.quant_method, "prefix", "?"),
        )
    return bool(ok)


class RoutedRowsHook:
    """Called by DeepseekV2MoE.forward_normal right after top-k on the global buffer: gathers this rank's packed
    rows with their top-k and returns (top-k over the gathered rows, W4A8RowQuant of the gathered rows)."""

    def __init__(self, packed: torch.Tensor, slot: int, n: int, top_k: int):
        self.packed = packed
        self.slot = slot
        self.n = n
        self.top_k = top_k

    def __call__(self, topk_output):
        from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2 as v2
        from sglang.srt.distributed.device_communicators.pynccl_allocator import (
            use_symmetric_memory,
        )
        from sglang.srt.layers.dp_attention import is_allocation_symmetric
        from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import W4A8RowQuant
        from sglang.srt.runtime_context import get_parallel

        ids, wts = topk_output.topk_ids, topk_output.topk_weights
        if wts.dtype != torch.float32 or ids.shape[1] != self.top_k:
            raise RuntimeError(
                f"DP FP8 MoE gather: top-k weights must be fp32 [*, {self.top_k}], got {wts.dtype} {tuple(ids.shape)}"
            )
        lo = self.slot * self.n
        v2.fp8ag_meta(ids[lo : lo + self.n], wts[lo : lo + self.n], self.packed)
        group = get_parallel().tp_group
        with use_symmetric_memory(group, disabled=not is_allocation_symmetric()):
            gathered = torch.empty(
                group.world_size * self.n, self.packed.shape[1], dtype=torch.uint8, device=self.packed.device
            )
        group.all_gather_into_tensor(gathered, self.packed)
        x8, xs, gids, gw = v2.fp8ag_unpack(gathered, self.top_k)
        if gids.dtype != ids.dtype:
            gids = gids.to(ids.dtype)
        return (
            topk_output._replace(topk_ids=gids, topk_weights=gw),
            W4A8RowQuant(x8.view(torch.float8_e4m3fn), xs),
        )


def begin(mlp, hidden_states: torch.Tensor, local_hidden_states: torch.Tensor) -> RoutedRowsHook:
    """Step 1: this rank's rows -> its slot of the global buffer (bf16) and its packed e4m3 rows. Returns the hook
    that forward_normal calls after top-k."""
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2 as v2
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group
    slot = group.rank_in_group
    n = local_hidden_states.shape[0]
    top_k = _top_k(mlp)
    with use_symmetric_memory(group, disabled=not is_allocation_symmetric()):
        packed = torch.empty(n, v2.fp8ag_row_bytes(top_k), dtype=torch.uint8, device=local_hidden_states.device)
    v2.fp8ag_quant(local_hidden_states.contiguous(), packed, hidden_states[slot * n : (slot + 1) * n])
    return RoutedRowsHook(packed, slot, n, top_k)


def verify_enabled() -> bool:
    return envs.SGLANG_OPT_DSV4_DP_FP8_MOE_GATHER_VERIFY.get() and not torch.cuda.is_current_stream_capturing()


def verify_compare(layer_id, out_fp8: torch.Tensor, out_ref: torch.Tensor) -> None:
    """Log a layer whose FP8-gather MoE output differs from the BF16-gather one (host sync: debug only)."""
    _VERIFY_STATS["layers"] += 1
    same = torch.equal(out_fp8, out_ref)
    if not same:
        _VERIFY_STATS["mismatch"] += 1
        diff = (out_fp8.float() - out_ref.float()).abs()
        rows = int((diff.amax(-1) > 0).sum())
        logger.warning(
            "INF-730 DP FP8 MoE gather VERIFY layer %s: MoE output differs in %d of %d rows (max abs %.3g); "
            "%d/%d layers differ so far",
            layer_id,
            rows,
            out_fp8.shape[0],
            float(diff.max()),
            _VERIFY_STATS["mismatch"],
            _VERIFY_STATS["layers"],
        )
    elif _VERIFY_STATS["layers"] % 500 == 1:
        logger.info(
            "INF-730 DP FP8 MoE gather VERIFY: %d/%d layers differ so far",
            _VERIFY_STATS["mismatch"],
            _VERIFY_STATS["layers"],
        )


def maybe_begin(mlp, hidden_states, local_hidden_states, forward_batch, *, shared_local: bool) -> Optional[RoutedRowsHook]:
    if not applies(mlp, hidden_states, local_hidden_states, forward_batch, shared_local=shared_local):
        return None
    return begin(mlp, hidden_states, local_hidden_states)
