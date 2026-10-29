"""SM90 W4A8 decode MoE v2 for DeepSeek-V4 TP4 MXFP4 experts (hidden 5120, 576 intermediate channels per
rank). See ``jit/csrc/moe/dsv4_mxfp4_decode_moe_v2.cuh``.

Same packed weights and the same numerics as ``dsv4_mxfp4_decode_moe_ws`` (#104), bit for bit; one
persistent kernel runs gate/up, the h row quantization and down for all experts with dynamically handed-out
items instead of three kernels with static splits, and one prologue launch replaces the expert sort, the token
row quantization and the workspace fill. ``decode_moe`` is CUDA-graph safe (no host sync).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

if TYPE_CHECKING:
    from tvm_ffi.module import Module

HIDDEN = ws.HIDDEN
INTER = ws.INTER
WORKSPACE_BASE = 32  # queue head + padding; then done[E], ready[E]
PREP_MAX_SLOTS = 16384  # the prologue kernel's limit (routed slots per call)

supports = ws.supports
pack_experts = ws.pack_experts
PackedExperts = ws.PackedExperts
MAX_E8M0 = ws.MAX_E8M0


@cache_once
def _module() -> Module:
    return load_jit(
        "dsv4_mxfp4_decode_moe_v2",
        cuda_files=["moe/dsv4_mxfp4_decode_moe_v2.cuh"],
        cuda_wrappers=[
            ("prep", "dsv4_moe_v2::prep"),
            ("experts", "dsv4_moe_v2::experts"),
        ],
    )


def decode_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    packed: ws.PackedExperts,
    swiglu_limit: float,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Routed experts, W4A8: out[t] = sum_k w[t,k] * expert_{id[t,k]}(x[t]). Masked ids (-1) add 0
    and cost no expert work."""
    num_tokens, top_k = topk_ids.shape
    if output is None:
        output = torch.empty_like(hidden_states)
    if num_tokens == 0:
        return output
    assert hidden_states.dtype == torch.bfloat16, hidden_states.dtype
    num_experts = packed.w13.shape[0]
    num_slots = num_tokens * top_k
    dev = hidden_states.device
    ids = topk_ids.to(torch.int32).contiguous()
    counts = torch.empty(num_experts, dtype=torch.int32, device=dev)
    sorted_slots = torch.empty(num_slots, dtype=torch.int32, device=dev)
    sorted_tokens = torch.empty(num_slots, dtype=torch.int32, device=dev)
    sorted_weights = torch.empty(num_slots, dtype=torch.float32, device=dev)
    x8 = torch.empty(num_tokens, HIDDEN, dtype=torch.uint8, device=dev)
    xs = torch.empty(num_tokens, dtype=torch.float32, device=dev)
    work = torch.empty(WORKSPACE_BASE + 2 * num_experts, dtype=torch.int32, device=dev)
    mod = _module()
    if num_slots <= PREP_MAX_SLOTS:
        # One launch: stable expert sort (as #104's), token row quantization (as _row_quant), zeroed workspace.
        mod.prep(
            ids,
            topk_weights.float().contiguous(),
            hidden_states.contiguous(),
            counts,
            sorted_slots,
            sorted_tokens,
            sorted_weights,
            x8,
            xs,
            work,
        )
    else:  # larger than decode batches: #104's prologue (same outputs)
        counts, sorted_slots, sorted_tokens, sorted_weights = ws._sort_slots(
            topk_ids, topk_weights, num_experts
        )
        x8f, xs = ws._row_quant(hidden_states.contiguous())
        x8 = x8f.view(torch.uint8)
        work.zero_()
    h = hidden_states.new_empty(num_slots, INTER)
    h8 = torch.empty(num_slots, INTER, dtype=torch.uint8, device=dev)
    hs = torch.empty(num_slots, dtype=torch.float32, device=dev)
    y = hidden_states.new_empty(num_slots, HIDDEN)
    mod.experts(
        x8,
        xs,
        packed.w13,
        packed.w2,
        counts,
        sorted_tokens,
        sorted_slots,
        sorted_weights,
        h,
        h8,
        hs,
        y,
        work,
        float(swiglu_limit),
    )
    ws._module().combine(y, ids, output)
    return output
