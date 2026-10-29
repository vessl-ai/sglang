"""SM90 decode MoE for DeepSeek-V4 MXFP4 experts at TP4 (hidden 5120, intermediate 576 per rank).

W4A16 like the current decode paths: exact e2m1 x e8m0 weights in bf16, bf16 activations, fp32
accumulation. One CTA streams one (expert, channel block) of weights once for all rows routed to
the expert (up to 32 rows per pass), instead of once per 16-row tile. See
``jit/csrc/moe/dsv4_mxfp4_decode_moe.cuh`` for the kernel and the repacked weight layout.

Entry points:
  ``pack_w13`` / ``pack_w2``: repack checkpoint-layout MXFP4 experts at load time.
  ``decode_moe``: routed experts for a decode batch (CUDA-graph safe, no host sync).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

HIDDEN = 5120
INTER = 576
# Bit positions (sign, e1, e0, m) of the first value of each bf16x2 A register in the 32-bit
# word of one k16 step; the second value sits 16 bits higher. Must match dequant8() in the .cuh.
_PLACE = ((15, 8, 7, 6), (12, 5, 4, 3), (9, 2, 1, 0), (14, 13, 11, 10))
# Gate/up: 3 blocks x 6 warps x 4 tiles (gate g0, gate g1, up g0, up g1) of 16 channels.
W13_LAYOUT = (3, 6, 4)
# Down: 20 blocks x 4 warps x 4 tiles of 16 output channels.
W2_LAYOUT = (20, 4, 4)
SCALE_BIAS = 126  # stored scale byte = e8m0 exponent + 126 (bf16 multiplier 2^(s-1))
MAX_E8M0 = 128


class PackedExperts(NamedTuple):
    w13_q: torch.Tensor  # uint8, [E, 3, 6, 80, 4, 32, 16]
    w13_s: torch.Tensor  # uint8, [E, 3, 6, 80, 32, 16]
    w2_q: torch.Tensor  # uint8, [E, 20, 4, 9, 4, 32, 16]
    w2_s: torch.Tensor  # uint8, [E, 20, 4, 9, 32, 16]


def supports(hidden: int, inter_per_rank: int) -> bool:
    return hidden == HIDDEN and inter_per_rank == INTER


@cache_once
def _module() -> Module:
    return load_jit(
        "dsv4_mxfp4_decode_moe",
        cuda_files=["moe/dsv4_mxfp4_decode_moe.cuh"],
        cuda_wrappers=[
            ("gate_up", "dsv4_decode_moe::gate_up"),
            ("down", "dsv4_decode_moe::down"),
        ],
    )


def _w13_channel_base(b, w, t):
    return (t // 2) * INTER + 192 * b + 32 * w + 16 * (t % 2)


def _w2_channel_base(b, w, t):
    return 256 * b + 64 * w + 16 * t


def _place(nib: torch.Tensor, bits) -> torch.Tensor:
    s, e1, e0, m = bits
    return (
        (((nib >> 3) & 1) << s)
        | (((nib >> 2) & 1) << e1)
        | (((nib >> 1) & 1) << e0)
        | ((nib & 1) << m)
    )


def _repack(weight, scale, layout, chan_base, expert_chunk=16):
    """weight [E, N, K/2] (low nibble = even k), scale [E, N, K/32] e8m0 bytes ->
    (q [E, NB, W, K/64, T, 32, 16] uint8, s [E, NB, W, K/64, 32, 16] uint8).

    Word j of a lane's 16 bytes = k16 step j of the k64 step, bits per _PLACE; scale byte q of a
    lane = tile q // 4, k32 block (q // 2) % 2, channel g + 8 * (q % 2), stored as e8m0 + 126."""
    nb, nw, nt = layout
    E, N, Kb = weight.shape
    K = 2 * Kb
    steps = K // 64
    dev = weight.device
    weight = weight.view(torch.uint8)
    scale = scale.view(torch.uint8)
    smax = int(scale.max())
    if smax > MAX_E8M0:
        raise ValueError(f"e8m0 scale exponent {smax} > {MAX_E8M0} is not supported")

    ar = lambda n: torch.arange(n, device=dev, dtype=torch.int64)  # noqa: E731
    b = ar(nb).view(nb, 1, 1, 1, 1, 1, 1)
    w = ar(nw).view(1, nw, 1, 1, 1, 1, 1)
    st = ar(steps).view(1, 1, steps, 1, 1, 1, 1)
    t = ar(nt).view(1, 1, 1, nt, 1, 1, 1)
    lane = ar(32).view(1, 1, 1, 1, 32, 1, 1)
    j = ar(4).view(1, 1, 1, 1, 1, 4, 1)
    i = ar(4).view(1, 1, 1, 1, 1, 1, 4)
    g, c = lane // 4, lane % 4
    chan = chan_base(b, w, t) + g + 8 * (i % 2)
    kbyte = (64 * st + 16 * j + 2 * c + 8 * (i // 2)) // 2
    flat = (chan * Kb + kbyte).reshape(-1)

    # scale bytes: [nb, nw, steps, 32, 16], byte q = t*4 + jb*2 + h
    q = ar(16).view(1, 1, 1, 1, 16)
    ts, jb, h = q // 4, (q // 2) % 2, q % 2
    sb = b.view(nb, 1, 1, 1, 1)
    sw = w.view(1, nw, 1, 1, 1)
    sst = st.view(1, 1, steps, 1, 1)
    sg = (lane // 4).view(1, 1, 1, 32, 1)
    schan = chan_base(sb, sw, ts) + sg + 8 * h
    sflat = (schan * (K // 32) + 2 * sst + jb).reshape(-1)

    out_q = torch.empty((E, nb, nw, steps, nt, 32, 16), dtype=torch.uint8, device=dev)
    out_s = torch.empty((E, nb, nw, steps, 32, 16), dtype=torch.uint8, device=dev)
    for e0 in range(0, E, expert_chunk):
        e1 = min(E, e0 + expert_chunk)
        ne = e1 - e0
        by = weight[e0:e1].reshape(ne, -1)[:, flat].to(torch.int64)
        by = by.view(ne, nb, nw, steps, nt, 32, 4, 4)
        lo, hi = by & 0xF, by >> 4
        word = torch.zeros(by.shape[:-1], dtype=torch.int64, device=dev)
        for ii, bits in enumerate(_PLACE):
            word |= _place(lo[..., ii], bits) | (_place(hi[..., ii], bits) << 16)
        word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
        out_q[e0:e1] = word.contiguous().view(torch.uint8).view(ne, nb, nw, steps, nt, 32, 16)
        sv = scale[e0:e1].reshape(ne, -1)[:, sflat].to(torch.int32) + SCALE_BIAS
        out_s[e0:e1] = sv.to(torch.uint8).view(ne, nb, nw, steps, 32, 16)
    return out_q, out_s


def pack_w13(w13: torch.Tensor, s13: torch.Tensor):
    """w13 [E, 2*576, 5120/2] ([w1; w3] rows), s13 [E, 1152, 160] e8m0."""
    assert w13.shape[1] == 2 * INTER and w13.shape[2] * 2 == HIDDEN, w13.shape
    return _repack(w13, s13, W13_LAYOUT, _w13_channel_base)


def pack_w2(w2: torch.Tensor, s2: torch.Tensor):
    """w2 [E, 5120, 576/2], s2 [E, 5120, 18] e8m0."""
    assert w2.shape[1] == HIDDEN and w2.shape[2] * 2 == INTER, w2.shape
    return _repack(w2, s2, W2_LAYOUT, _w2_channel_base)


def pack_experts(w13, s13, w2, s2) -> PackedExperts:
    """Repack one layer's local experts (sglang [w1; w3] layout) for decode_moe; the packed
    tensors hold exactly the MXFP4 bytes (no padding)."""
    q13, sc13 = pack_w13(w13, s13)
    q2, sc2 = pack_w2(w2, s2)
    return PackedExperts(q13, sc13, q2, sc2)


def _sort_slots(topk_ids: torch.Tensor, topk_weights: torch.Tensor, num_experts: int):
    """Same outputs as expert_sort_small (stable, masked ids -> expert 0 with weight 0), for
    large batches; no host sync."""
    top_k = topk_ids.shape[1]
    valid = topk_ids >= 0
    ids = torch.where(valid, topk_ids, 0).flatten().long()
    weights = torch.where(valid, topk_weights, 0).flatten().float()
    order = torch.argsort(ids, stable=True)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=ids.device)
    counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
    return counts, order.to(torch.int32), (order // top_k).to(torch.int32), weights[order]


def decode_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    packed: PackedExperts,
    swiglu_limit: float,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Routed experts: out[t] = sum_k w[t,k] * expert_{id[t,k]}(x[t]). Masked ids (-1) add 0."""
    from sglang.kernels.ops.moe.expert_sort_small import expert_sort_small
    from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum

    num_tokens, top_k = topk_ids.shape
    if output is None:
        output = torch.empty_like(hidden_states)
    if num_tokens == 0:
        return output
    num_experts = packed.w13_q.shape[0]
    num_slots = num_tokens * top_k
    if num_slots <= 16 * num_experts:
        # One launch; its O(experts x slots) scan is meant for decode-sized batches.
        counts, sorted_slots, sorted_tokens, sorted_weights = expert_sort_small(
            topk_ids, topk_weights, num_experts
        )
    else:
        counts, sorted_slots, sorted_tokens, sorted_weights = _sort_slots(
            topk_ids, topk_weights, num_experts
        )
    mod = _module()
    h = hidden_states.new_empty(num_slots, INTER)
    y = hidden_states.new_empty(num_slots, HIDDEN)
    mod.gate_up(
        hidden_states.contiguous(),
        packed.w13_q,
        packed.w13_s,
        counts,
        sorted_tokens,
        h,
        float(swiglu_limit),
    )
    mod.down(h, packed.w2_q, packed.w2_s, counts, sorted_slots, sorted_weights, y)
    moe_topk_sum(y.view(num_tokens, top_k, HIDDEN), output)
    return output
