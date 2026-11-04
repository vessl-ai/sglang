"""SM90 W4A8 decode MoE for DeepSeek-V4 MXFP4 experts with a per-rank intermediate width other than TP4's 576
(INF-730). See ``jit/csrc/moe/dsv4_mxfp4_decode_moe_v2w.cuh``: the v2 kernel (#115) with the width a compile-time
constant, one JIT module per width.

- ``inter=2304``: expert parallel. A rank holds whole experts; pass the rank's (token, local expert) pairs as rows
  with ``top_k=1`` (ids = local expert index, weights = routing weight). Each row's output is that expert's
  weighted output; with ``top_k=1`` the kernel writes it straight into ``output`` (no top-k combine).
- ``inter=320``: a TP8 slice (288 channels) zero-padded to 320 (``pack_tp_slice``); the padded channels are exact
  zeros, so the result equals a 288-wide slice's.
- ``inter=576``: the v2 kernel's TP4 shape (for checking this file against v2).

Same numerics as v2 per (row, expert) except that h is row-quantized over the whole width (2304: one e4m3 scale per
expert row instead of one per 576-wide slice).
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

HIDDEN = 5120
WORKSPACE_BASE = 32
PREP_MAX_SLOTS = 32768


class PackedExperts(NamedTuple):
    w13: torch.Tensor  # uint8 [E, inter / 64, 348160]
    w2: torch.Tensor  # uint8 [E, 40, inter * 68]
    inter: int


@cache_once
def _module(inter: int, defines: tuple = ()):
    """defines: extra compile-time switches (ablation builds for measurement only, e.g. "DSV4_ABL_CONSUMER_NOP")."""
    return load_jit(
        "dsv4_mxfp4_decode_moe_v2w",
        str(inter),
        *(d.replace("=", "_") for d in defines),
        cuda_files=["moe/dsv4_mxfp4_decode_moe_v2w.cuh"],
        cuda_wrappers=[
            ("prep", "dsv4_moe_v2w::prep"),
            ("experts", "dsv4_moe_v2w::experts"),
            ("prepare_kernels", "dsv4_moe_v2w::prepare_kernels"),
        ],
        extra_cuda_cflags=[f"-DDSV4_MOE_INTER={inter}"] + [f"-D{d}" for d in defines],
    )


def _specs(inter: int):
    w13 = (inter // 64, 2, lambda j, w, t: t * inter + 64 * j + 16 * w)  # tile 0 gate, tile 1 up
    w2 = (40, 2, lambda j, w, t: 128 * j + 64 * t + 16 * w)
    return w13, w2


def pack_w13_w2(w13, s13, w2, s2) -> PackedExperts:
    """[E, 2I, H/2] (gate rows then up rows, low nibble = even k) + e8m0 scales, [E, H, I/2] + scales."""
    inter = w13.shape[1] // 2
    assert w13.shape[2] * 2 == HIDDEN and w2.shape[1] == HIDDEN and w2.shape[2] * 2 == inter, (w13.shape, w2.shape)
    spec13, spec2 = _specs(inter)
    return PackedExperts(ws._pack(w13, s13, spec13), ws._pack(w2, s2, spec2), inter)


def pack_experts(w1, s1, w3, s3, w2, s2) -> PackedExperts:
    """Whole experts (checkpoint layout): w1/w3 [E, I, H/2], w2 [E, H, I/2] and their e8m0 scales."""
    u8 = torch.uint8  # no CUDA cat for float8_e8m0fnu: concatenate the scale bytes
    s13 = torch.cat([s1.view(u8), s3.view(u8)], 1).view(s1.dtype)
    return pack_w13_w2(torch.cat([w1, w3], 1), s13, w2, s2)


def pack_tp_slice(w1, s1, w3, s3, w2, s2, rank: int, tp: int, pad_to: int) -> PackedExperts:
    """TP slice `rank` of `tp` of whole experts, zero-padded from I/tp to `pad_to` channels (zero e2m1 weights)."""
    I = w1.shape[1]
    sl = I // tp
    c0 = rank * sl
    E = w1.shape[0]

    def pad_rows(t):
        out = torch.zeros((E, pad_to) + tuple(t.shape[2:]), dtype=t.dtype, device=t.device)
        out[:, :sl] = t[:, c0:c0 + sl]
        return out

    def pad_k(t, per_byte):  # pad the last dim (K) from sl / per_byte to pad_to / per_byte
        out = torch.zeros(t.shape[:2] + (pad_to // per_byte,), dtype=t.dtype, device=t.device)
        out[:, :, : sl // per_byte] = t[:, :, c0 // per_byte:(c0 + sl) // per_byte]
        return out

    u8, e8 = torch.uint8, s1.dtype
    w13 = torch.cat([pad_rows(w1), pad_rows(w3)], 1)
    s13 = torch.cat([pad_rows(s1.view(u8)), pad_rows(s3.view(u8))], 1).view(e8)
    return pack_w13_w2(w13, s13, pad_k(w2, 2), pad_k(s2.view(u8), 32).view(e8))


@cache_once
def prepare(device_index: int, inter: int) -> None:
    ws.prepare(device_index)
    with torch.cuda.device(device_index):
        _module(inter).prepare_kernels()
        torch.cuda.synchronize()


def decode_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    packed: PackedExperts,
    swiglu_limit: float,
    output: torch.Tensor | None = None,
    defines: tuple = (),
) -> torch.Tensor:
    """out[t] = sum_k w[t,k] * expert_{id[t,k]}(x[t]); masked ids (-1) add 0 and cost no expert work. With
    top_k == 1 the kernel writes rows into ``output`` directly: rows whose id is masked are left unwritten."""
    num_tokens, top_k = topk_ids.shape
    if output is None:
        output = torch.empty_like(hidden_states)
    if num_tokens == 0:
        return output
    assert hidden_states.dtype == torch.bfloat16, hidden_states.dtype
    inter = packed.inter
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
    max_passes = num_experts + num_slots // 32 + 1
    work = torch.empty(WORKSPACE_BASE + 5 * max_passes, dtype=torch.int32, device=dev)
    mod = _module(inter, defines)
    assert num_slots <= PREP_MAX_SLOTS, num_slots
    mod.prep(ids, topk_weights.float().contiguous(), hidden_states.contiguous(), counts, sorted_slots,
             sorted_tokens, sorted_weights, x8, xs, work)
    h = hidden_states.new_empty(num_slots, inter)
    h8 = torch.empty(num_slots, inter, dtype=torch.uint8, device=dev)
    hs = torch.empty(num_slots, dtype=torch.float32, device=dev)
    y = output if top_k == 1 else hidden_states.new_empty(num_slots, HIDDEN)
    mod.experts(x8, xs, packed.w13, packed.w2, counts, sorted_tokens, sorted_slots, sorted_weights, h, h8, hs, y,
                work, float(swiglu_limit))
    if top_k != 1:
        ws._module().combine(y, ids, output)
    return output
