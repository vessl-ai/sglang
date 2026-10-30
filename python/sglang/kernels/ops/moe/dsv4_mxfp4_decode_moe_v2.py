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
WORKSPACE_BASE = 32  # queue head + padding; then per 32-row pass: done[P], ready[P]
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
            ("prepare_kernels", "dsv4_moe_v2::prepare_kernels"),
        ],
    )


@cache_once
def prepare(device_index: int) -> None:
    """Build and load everything ``decode_moe`` launches, once per process and device, at weight-processing
    time (as ``dsv4_mxfp4_decode_moe_ws.prepare``, which this calls for the top-k combine it shares and for
    the fallback prologue): this module's JIT build (under a lock shared by the node's ranks on a cold cache),
    its CUDA kernels (loaded lazily at the first launch otherwise), and one launch of the whole call on a
    routing with every slot masked (no expert work, no weight read). The first MoE call of CUDA graph capture
    then compiles and loads nothing."""
    ws.prepare(device_index)
    with torch.cuda.device(device_index):
        _module().prepare_kernels()
        for num_experts in (384, 128):  # target and DSpark draft experts
            # Zero-stride stand-ins for the packed weights: with every slot masked nothing reads them.
            one = torch.zeros(1, dtype=torch.uint8, device="cuda")
            packed = ws.PackedExperts(
                one.expand(num_experts, 9, 348160), one.expand(num_experts, 40, 39168)
            )
            x = torch.zeros(2, HIDDEN, dtype=torch.bfloat16, device="cuda")
            ids = torch.full((2, 6), -1, dtype=torch.int32, device="cuda")
            decode_moe(x, ids, torch.ones(ids.shape, device="cuda"), packed, 10.0)
        torch.cuda.synchronize()


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
    # per 32-row pass of an expert: done items and h8 ready (passes <= experts + slots / 32)
    max_passes = num_experts + num_slots // 32 + 1
    work = torch.empty(WORKSPACE_BASE + 2 * max_passes, dtype=torch.int32, device=dev)
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
