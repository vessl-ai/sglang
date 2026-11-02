"""SM90 persistent, warp-specialized W4A8 decode MoE for DeepSeek-V4 TP4 MXFP4 experts (hidden 5120,
576 intermediate channels per rank). See ``jit/csrc/moe/dsv4_mxfp4_decode_moe_ws.cuh``.

Numerics are the W4A8 prefill path's: activations of both GEMMs in e4m3 with one power-of-2 scale per
row, exact MXFP4 weights (each 32-k block's product promoted to fp32 with its e8m0 scale).

Entry points: ``pack_experts`` (load time, raw MXFP4 bytes -> stage-ordered layout of the same size) and
``decode_moe`` (CUDA-graph safe, no host sync). Masked slots (topk id -1, e.g. the dp-attention pad rows
that SGLANG_OPT_MASK_DP_PAD_MOE masks) are sorted to no expert and skipped by the top-k combine, so they
cost no expert work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

HIDDEN = 5120
INTER = 576
SCALE_BIAS = 6  # stored scale byte = e8m0 + 6: fp32 2^(byte-127) = scale x 2^6 (undoes the e4m3 placement)
MAX_E8M0 = (
    254 - SCALE_BIAS
)  # the stored byte must stay below 255 (fp32 exponent of inf)
CONSUMER_WARPS = 4
# (items per expert, 16-channel tiles per warp, channel base of (item j, warp, tile)).
W13 = (9, 2, lambda j, w, t: t * INTER + 64 * j + 16 * w)  # tile 0 gate, tile 1 up
W2 = (40, 2, lambda j, w, t: 128 * j + 64 * t + 16 * w)


class PackedExperts(NamedTuple):
    w13: torch.Tensor  # uint8 [E, 9, 348160]
    w2: torch.Tensor  # uint8 [E, 40, 39168]


def supports(hidden: int, inter_per_rank: int) -> bool:
    return hidden == HIDDEN and inter_per_rank == INTER


@cache_once
def _module() -> Module:
    return load_jit(
        "dsv4_mxfp4_decode_moe_ws",
        cuda_files=["moe/dsv4_mxfp4_decode_moe_ws.cuh"],
        cuda_wrappers=[
            ("gate_up", "dsv4_moe_ws::gate_up"),
            ("down", "dsv4_moe_ws::down"),
            ("combine", "dsv4_moe_ws::combine"),
            ("sort_slots", "dsv4_moe_ws::sort_slots"),
            ("prepare_kernels", "dsv4_moe_ws::prepare_kernels"),
        ],
    )


def _place(nib: torch.Tensor, bits) -> torch.Tensor:
    s, e1, e0, m = bits
    return (
        (((nib >> 3) & 1) << s)
        | (((nib >> 2) & 1) << e1)
        | (((nib >> 1) & 1) << e0)
        | ((nib & 1) << m)
    )


def _pack(
    weight: torch.Tensor, scale: torch.Tensor, spec, expert_chunk: int = 8
) -> torch.Tensor:
    """weight [E, N, K/2] (low nibble = even k), scale [E, N, K/32] e8m0 -> [E, NB, item bytes] uint8.

    Item = 128-k stages; a stage = [steps][warp][tile][lane] 16-byte weight words, then
    [steps][warp][8 groups][tile][blk][row g / g+8] scale bytes. Word j of a lane: MX block j // 2, rows
    g (+8 for odd j); byte i: A = k 4c+i, B = k 4c+16+i of the block."""
    nb, nt, chan_base = spec
    E, N, Kb = weight.shape
    K = 2 * Kb
    nstep = K // 64
    dev = weight.device
    weight = weight.view(torch.uint8)
    scale = scale.view(torch.uint8)
    if int(scale.max()) > MAX_E8M0:
        raise ValueError(f"e8m0 scale exponent above {MAX_E8M0} is not supported")
    ar = lambda n: torch.arange(n, device=dev, dtype=torch.int64)  # noqa: E731
    # weights: [nb, nstep, warp, tile, lane, word, byte, ab]
    j = ar(nb).view(nb, 1, 1, 1, 1, 1, 1, 1)
    st = ar(nstep).view(1, nstep, 1, 1, 1, 1, 1, 1)
    w = ar(CONSUMER_WARPS).view(1, 1, CONSUMER_WARPS, 1, 1, 1, 1, 1)
    t = ar(nt).view(1, 1, 1, nt, 1, 1, 1, 1)
    lane = ar(32).view(1, 1, 1, 1, 32, 1, 1, 1)
    word = ar(4).view(1, 1, 1, 1, 1, 4, 1, 1)
    byte = ar(4).view(1, 1, 1, 1, 1, 1, 4, 1)
    ab = ar(2).view(1, 1, 1, 1, 1, 1, 1, 2)
    g, c = lane // 4, lane % 4
    chan = chan_base(j, w, t) + g + 8 * (word % 2)
    k = 64 * st + 32 * (word // 2) + 4 * c + byte + 16 * ab
    shape = (nb, nstep, CONSUMER_WARPS, nt, 32, 4, 4, 2)
    flat = (chan * Kb + k // 2).expand(shape).reshape(-1)
    shift = (4 * (k % 2)).expand(shape).reshape(-1)
    # scales: [nb, nstep, warp, group, q = tile*4 + blk*2 + h]
    q = ar(4 * nt).view(1, 1, 1, 1, 4 * nt)
    ts, blk, h = q // 4, (q // 2) % 2, q % 2
    schan = (
        chan_base(j.view(nb, 1, 1, 1, 1), w.view(1, 1, CONSUMER_WARPS, 1, 1), ts)
        + ar(8).view(1, 1, 1, 8, 1)
        + 8 * h
    )
    sflat = (
        (schan * (K // 32) + 2 * st.view(1, nstep, 1, 1, 1) + blk)
        .expand(nb, nstep, CONSUMER_WARPS, 8, 4 * nt)
        .reshape(-1)
    )

    wbytes = torch.empty(
        (E, nb, nstep, CONSUMER_WARPS, nt, 32, 16), dtype=torch.uint8, device=dev
    )
    sbytes = torch.empty(
        (E, nb, nstep, CONSUMER_WARPS, 8, 4 * nt), dtype=torch.uint8, device=dev
    )
    shifts = 8 * ar(4)
    for e0 in range(0, E, expert_chunk):
        e1 = min(E, e0 + expert_chunk)
        ne = e1 - e0
        by = weight[e0:e1].reshape(ne, -1)[:, flat].to(torch.int64)
        nib = ((by >> shift) & 0xF).view(ne, *shape)
        b8 = _place(nib[..., 0], (7, 4, 3, 2)) | _place(nib[..., 1], (5, 6, 1, 0))
        words = (b8 << shifts).sum(-1)
        words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
        wbytes[e0:e1] = (
            words.contiguous()
            .view(torch.uint8)
            .view(ne, nb, nstep, CONSUMER_WARPS, nt, 32, 16)
        )
        sv = scale[e0:e1].reshape(ne, -1)[:, sflat].to(torch.int32) + SCALE_BIAS
        sbytes[e0:e1] = sv.to(torch.uint8).view(
            ne, nb, nstep, CONSUMER_WARPS, 8, 4 * nt
        )
    parts = []
    for s0 in range(0, nstep, 2):
        steps = slice(s0, min(nstep, s0 + 2))
        parts.append(wbytes[:, :, steps].reshape(E, nb, -1))
        parts.append(sbytes[:, :, steps].reshape(E, nb, -1))
    return torch.cat(parts, dim=2).contiguous()


def pack_experts(w13, s13, w2, s2) -> PackedExperts:
    """One layer's local experts in sglang's [w1; w3] layout -> PackedExperts (same byte count)."""
    assert w13.shape[1] == 2 * INTER and w13.shape[2] * 2 == HIDDEN, w13.shape
    assert w2.shape[1] == HIDDEN and w2.shape[2] * 2 == INTER, w2.shape
    return PackedExperts(_pack(w13, s13, W13), _pack(w2, s2, W2))


def _row_quant(x: torch.Tensor):
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import _row_quant

    return _row_quant(x)


SORT_MAX_SLOTS = 16384  # one-CTA sort kernel limit


@cache_once
def prepare(device_index: int) -> None:
    """Build and load everything ``decode_moe`` launches, once per process and device: the JIT module
    (compiled under a lock shared by the node's ranks on a cold cache), its CUDA kernels (loaded lazily at
    the first launch otherwise), and the Triton expert sort and row quantization specializations. Called at
    weight-processing time, so that the first MoE call of CUDA graph capture compiles and loads nothing while
    other ranks have collectives in flight."""
    from sglang.kernels.ops.moe.expert_sort_small import expert_sort_small

    with torch.cuda.device(device_index):
        _module().prepare_kernels()
        for num_tokens in (
            4,
            16,
        ):  # num_slots not / divisible by 16: Triton specializes on it
            ids = torch.zeros(num_tokens, 6, dtype=torch.int32, device="cuda")
            expert_sort_small(
                ids, torch.ones(ids.shape, device="cuda"), 384, drop_masked=True
            )
        for width in (HIDDEN, INTER):
            _row_quant(torch.zeros(1, width, dtype=torch.bfloat16, device="cuda"))
        torch.cuda.synchronize()


def _sort_slots(topk_ids, topk_weights, num_experts):
    """Slots sorted by expert, stable: (counts [E], sorted_slots, sorted_tokens, sorted_weights).
    Masked slots (id -1, e.g. dp-attention pad rows under SGLANG_OPT_MASK_DP_PAD_MOE) belong to
    no expert: they are not counted, sort after every real slot and are never computed."""
    num_tokens, top_k = topk_ids.shape
    num_slots = num_tokens * top_k
    device = topk_ids.device
    if num_slots <= 16 * num_experts:
        from sglang.kernels.ops.moe.expert_sort_small import expert_sort_small

        return expert_sort_small(topk_ids, topk_weights, num_experts, drop_masked=True)
    if num_slots <= SORT_MAX_SLOTS and num_experts <= 1024:
        counts = torch.empty(num_experts, dtype=torch.int32, device=device)
        sorted_slots = torch.empty(num_slots, dtype=torch.int32, device=device)
        sorted_tokens = torch.empty(num_slots, dtype=torch.int32, device=device)
        sorted_weights = torch.empty(num_slots, dtype=torch.float32, device=device)
        _module().sort_slots(
            topk_ids.to(torch.int32).contiguous(),
            topk_weights.float().contiguous(),
            counts,
            sorted_slots,
            sorted_tokens,
            sorted_weights,
        )
        return counts, sorted_slots, sorted_tokens, sorted_weights
    ids = torch.where(topk_ids >= 0, topk_ids, num_experts).flatten().long()
    order = torch.argsort(ids, stable=True)
    counts = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
    return (
        counts[:num_experts],
        order.to(torch.int32),
        (order // top_k).to(torch.int32),
        topk_weights.flatten().float()[order],
    )


def decode_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    packed: PackedExperts,
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
    num_experts = packed.w13.shape[0]
    num_slots = num_tokens * top_k
    counts, sorted_slots, sorted_tokens, sorted_weights = _sort_slots(
        topk_ids, topk_weights, num_experts
    )
    mod = _module()
    x8, xs = _row_quant(hidden_states.contiguous())
    h = hidden_states.new_empty(num_slots, INTER)
    mod.gate_up(
        x8.view(torch.uint8),
        xs,
        packed.w13,
        counts,
        sorted_tokens,
        h,
        float(swiglu_limit),
    )
    h8, hs = _row_quant(h)
    y = hidden_states.new_empty(num_slots, HIDDEN)
    mod.down(
        h8.view(torch.uint8), hs, packed.w2, counts, sorted_slots, sorted_weights, y
    )
    mod.combine(y, topk_ids.to(torch.int32).contiguous(), output)
    return output
