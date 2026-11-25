"""K1 (INF-874): SM90 W4A8 MoE expert kernels for DeepSeek-V4 MXFP4 experts in expert parallel (hidden 5120,
intermediate 2304). See ``jit/csrc/moe/dsv4_mxfp4_decode_moe_k1.cuh``.

Engine seam (``dsv4_mxfp4_decode_moe_ep``, ``SGLANG_OPT_DSV4_EP_DECODE_MOE_KERNEL=k1``, default ``v2w``):
- ``pack_layer`` (load time): a layer's local whole experts -> ``PackedK1`` (block scales folded per channel, the same
  number of bytes as the checkpoint's MXFP4), or None when a channel's block exponents span more than 11 (the layer then
  keeps v2w).
- ``decode_ep``: received e4m3 token rows + power-of-2 row scales, LOCAL top-k ids [T, K] (-1: another rank or a pad
  row), weights [T, K] -> output [T, 5120] bf16, every row written (zero for a token without a local slot, as
  ``combine_ep``). CUDA-graph safe (no host sync).
- ``prepare_ep``: JIT build + one all-masked call at weight-processing time.

``decode_moe`` is the general entry (top-k layout or a pair list with a device-side count).
"""

from __future__ import annotations

import math
import os
from typing import NamedTuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

HIDDEN = 5120
INTER = 2304
JB = INTER // 64
STAGES = HIDDEN // 128
WSTAGE = 8192 + 512
MAX_FOLD = 11  # largest exact exponent offset of a block inside its channel
NMAX = 128  # activation rows per tile


@cache_once
def _module(defines: tuple = ()):
    return load_jit(
        "dsv4_mxfp4_decode_moe_k1",
        *(d.replace("=", "_") for d in defines),
        cuda_files=["moe/dsv4_mxfp4_decode_moe_k1.cuh"],
        cuda_wrappers=[
            ("x_prep", "dsv4_moe_k1::x_prep"),
            ("x_gather", "dsv4_moe_k1::x_gather"),
            ("x_quant", "dsv4_moe_k1::x_quant"),
            ("x_run", "dsv4_moe_k1::x_run"),
            ("x_reduce", "dsv4_moe_k1::x_reduce"),
        ],
        extra_cuda_cflags=[f"-D{d}" for d in defines],
    )


class PackedW13(NamedTuple):
    w13: torch.Tensor  # uint8 [E, 36, 40, 8704]
    cs13: torch.Tensor  # f32 [E, 2, 2304]: 2^(max block exponent - 133) per gate / up channel
    lossy_blocks: int  # blocks whose offset exceeded MAX_FOLD and had nonzero values (clamped: not exact)


def _codes(w: torch.Tensor) -> torch.Tensor:
    """[N, K/2] two e2m1 per byte (low nibble = even k) -> [N, K] codes 0..15 (bit 3 = sign)."""
    b = w.view(torch.uint8)
    return torch.stack([b & 0xF, b >> 4], -1).flatten(-2)


def _encode(v: torch.Tensor) -> torch.Tensor:
    """[..., 8] codes -> [...] int32 words: magnitude i < 4 in bits 4i..4i+2, its sign at 8i+7; magnitude 4 + i in
    bits 16+4i.., its sign at 8i+3."""
    mag = (v & 7).long()
    sgn = (v >> 3).long()
    w = torch.zeros(v.shape[:-1], dtype=torch.long, device=v.device)
    for i in range(4):
        w |= mag[..., i] << (4 * i)
        w |= sgn[..., i] << (8 * i + 7)
        w |= mag[..., 4 + i] << (16 + 4 * i)
        w |= sgn[..., 4 + i] << (8 * i + 3)
    w = torch.where(w >= 2**31, w - 2**32, w)
    return w.to(torch.int32)


def pack_w13(w1, s1, w3, s3) -> PackedW13:
    """Whole experts: w1/w3 [E, 2304, 2560] (two e2m1 per byte), s1/s3 [E, 2304, 160] e8m0."""
    E = w1.shape[0]
    assert w1.shape[1:] == (INTER, HIDDEN // 2) and s1.shape[1:] == (INTER, HIDDEN // 32), (w1.shape, s1.shape)
    dev = w1.device
    out = torch.empty(E, JB, STAGES, WSTAGE, dtype=torch.uint8, device=dev)
    cs = torch.empty(E, 2, INTER, dtype=torch.float32, device=dev)
    lossy = 0
    for e in range(E):
        codes = torch.stack([_codes(w1[e]), _codes(w3[e])])  # [2(h), 2304, 5120]
        # (h, j, cw, q, grp, st, i, bb, half, c, v) -> (j, st, cw, q, i, grp, c, bb, h, half, v)
        v = codes.view(2, JB, 2, 4, 8, STAGES, 2, 2, 2, 4, 4).permute(1, 5, 2, 3, 6, 4, 9, 7, 0, 8, 10)
        words = _encode(v.reshape(JB, STAGES, 2048, 8))
        out[e, :, :, :8192] = words.view(torch.uint8).view(JB, STAGES, 8192)
        s = torch.stack([s1[e].view(torch.uint8), s3[e].view(torch.uint8)]).int()  # [2, 2304, 160]
        smax = s.amax(-1)
        k = smax[..., None] - s
        mag = (codes & 7).view(2, INTER, HIDDEN // 32, 32).amax(-1)
        over = k > MAX_FOLD
        lossy += int((over & (mag > 0)).sum())
        k = k.clamp_max(MAX_FOLD)
        # (h, j, cw, q, grp, st, b) -> (j, st, cw, q, grp, h, b)
        kb = k.view(2, JB, 2, 4, 8, STAGES, 4).permute(1, 5, 2, 3, 4, 0, 6)
        out[e, :, :, 8192:] = kb.to(torch.uint8).reshape(JB, STAGES, 512)
        cs[e] = torch.exp2(smax.float() - 133.0)
        del codes, v, words
    return PackedW13(out, cs, lossy)


DN_STAGES = INTER // 128  # 18
DN_JB = HIDDEN // 128  # 40


class PackedW2(NamedTuple):
    w2: torch.Tensor  # uint8 [E, 40, 18, 8704]
    cs2: torch.Tensor  # f32 [E, 5120]
    lossy_blocks: int


def pack_w2(w2, s2) -> PackedW2:
    """w2 [E, 5120, 1152] (two e2m1 per byte, K = 2304), s2 [E, 5120, 72] e8m0. Output channel of A row grp (+ 8 h) of
    warp q of consumer warpgroup cw in block jd: 128 jd + 64 cw + 16 q + 8 h + grp."""
    E = w2.shape[0]
    assert w2.shape[1:] == (HIDDEN, INTER // 2) and s2.shape[1:] == (HIDDEN, INTER // 32), (w2.shape, s2.shape)
    dev = w2.device
    out = torch.empty(E, DN_JB, DN_STAGES, WSTAGE, dtype=torch.uint8, device=dev)
    cs = torch.empty(E, HIDDEN, dtype=torch.float32, device=dev)
    lossy = 0
    for e in range(E):
        codes = _codes(w2[e])  # [5120, 2304]
        # (jd, cw, q, h, grp, st, i, bb, half, c, v) -> (jd, st, cw, q, i, grp, c, bb, h, half, v)
        v = codes.view(DN_JB, 2, 4, 2, 8, DN_STAGES, 2, 2, 2, 4, 4).permute(0, 5, 1, 2, 6, 4, 9, 7, 3, 8, 10)
        words = _encode(v.reshape(DN_JB, DN_STAGES, 2048, 8))
        out[e, :, :, :8192] = words.view(torch.uint8).view(DN_JB, DN_STAGES, 8192)
        sc = s2[e].view(torch.uint8).int()  # [5120, 72]
        smax = sc.amax(-1)
        k = smax[:, None] - sc
        mag = (codes & 7).view(HIDDEN, INTER // 32, 32).amax(-1)
        lossy += int(((k > MAX_FOLD) & (mag > 0)).sum())
        k = k.clamp_max(MAX_FOLD).to(torch.uint8)
        # (jd, cw, q, h, grp, st, b) -> (jd, st, cw, q, grp, h, b)
        kb = k.view(DN_JB, 2, 4, 2, 8, DN_STAGES, 4).permute(0, 5, 1, 2, 4, 3, 6)
        out[e, :, :, 8192:] = kb.reshape(DN_JB, DN_STAGES, 512)
        cs[e] = torch.exp2(smax.float() - 133.0)
        del codes, v, words
    return PackedW2(out, cs, lossy)


class XScratch(NamedTuple):
    rowpos: torch.Tensor
    gam: torch.Tensor
    dst: torch.Tensor
    tcnt: torch.Tensor
    tfirst: torch.Tensor
    tiles_gu: torch.Tensor
    tiles_dn: torch.Tensor
    counters: torch.Tensor
    xa: torch.Tensor
    xrs: torch.Tensor
    h: torch.Tensor
    ha: torch.Tensor
    hs: torch.Tensor
    vlist: torch.Tensor


COUNTER_INTS = 8 + 2 * 512


def x_scratch(num_slots: int, num_tokens: int, num_experts: int, device) -> XScratch:
    """Per-call scratch (no memsets: every element read is written first by prep / gather / quant). Rows <= slots +
    16 E, chunks <= slots / 128 + E."""
    rpad = num_slots + 16 * num_experts
    chunks = num_slots // NMAX + num_experts + 1
    i32 = torch.int32
    return XScratch(
        torch.empty(num_slots, dtype=i32, device=device),
        torch.empty(rpad, dtype=torch.float32, device=device),
        torch.empty(rpad, dtype=i32, device=device),
        torch.empty(num_tokens, dtype=i32, device=device),
        torch.empty(2 * num_tokens, dtype=i32, device=device),
        torch.empty(chunks * JB, 4, dtype=i32, device=device),
        torch.empty(chunks * DN_JB, 4, dtype=i32, device=device),
        torch.empty(COUNTER_INTS, dtype=i32, device=device),
        torch.empty(STAGES, rpad, 128, dtype=torch.uint8, device=device),
        torch.empty(rpad, dtype=torch.float32, device=device),
        torch.empty(rpad, INTER, dtype=torch.bfloat16, device=device),
        torch.empty(DN_STAGES, rpad, 128, dtype=torch.uint8, device=device),
        torch.empty(rpad, dtype=torch.float32, device=device),
        torch.empty(num_slots, dtype=i32, device=device),
    )


def decode_moe(
    x8: torch.Tensor,
    xscale: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    p13: PackedW13,
    p2: PackedW2,
    swiglu_limit: float,
    out: torch.Tensor,
    y: torch.Tensor,
    sc: XScratch,
    tok: torch.Tensor | None = None,
    num_valid: torch.Tensor | None = None,
    top_k: int = 1,
    defines: tuple = (),
):
    """out[t] (bf16 [T, 5120]) = sum over the valid slots s of token t of w[s] * expert_{ids[s]}(x[t]); tokens
    without a valid slot: zero row in the top-k layout (as combine_ep), not written in the pair-list layout. Slots: top-k layout (ids / weights [T * top_k], token = s // top_k) or a pair
    list (tok [S], slots of a token adjacent, num_valid on the device). y: [S, 5120] bf16 scratch (slots of tokens
    with several slots)."""
    mod = _module(defines)
    E = p13.w13.shape[0]
    T = out.shape[0]
    use_tok = tok is not None
    tk = tok if use_tok else ids
    nv = num_valid if num_valid is not None else ids
    use_nv = int(num_valid is not None)
    sc.counters.zero_()
    mod.x_prep(ids, weights, tk, int(use_tok), top_k, nv, use_nv, E, T, sc.rowpos, sc.gam, sc.dst, sc.tcnt,
                sc.tfirst, sc.vlist, sc.tiles_gu, sc.tiles_dn, sc.counters)
    mod.x_gather(x8, xscale, sc.rowpos, tk, int(use_tok), top_k, sc.vlist, sc.counters, sc.xa, sc.xrs)
    mod.x_run(0, sc.xa, sc.xrs, p13.w13, p13.cs13, sc.tiles_gu, sc.counters, sc.h, sc.gam, sc.dst, y, out,
              float(swiglu_limit))
    mod.x_quant(sc.h, sc.rowpos, nv, use_nv, sc.ha, sc.hs, sc.vlist, 1, sc.counters)
    mod.x_run(1, sc.ha, sc.hs, p2.w2, p2.cs2, sc.tiles_dn, sc.counters, sc.h, sc.gam, sc.dst, y, out,
              float(swiglu_limit))
    mod.x_reduce(y, ids, sc.tcnt, sc.tfirst, sc.counters, out)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Engine seam (dsv4_mxfp4_decode_moe_ep, SGLANG_OPT_DSV4_EP_DECODE_MOE_KERNEL=k1)
class PackedK1(NamedTuple):
    w13: torch.Tensor  # [E, 36, 40, 8704] (shape[0] = local experts, as the other packings)
    cs13: torch.Tensor
    w2: torch.Tensor
    cs2: torch.Tensor


def pack_layer(w13, s13, w2, s2, expert_chunk: int = 8):
    """Engine layout ([E, 2I, H/2] gate rows then up rows, [E, 2I, H/32], [E, H, I/2], [E, H, I/32]) -> PackedK1,
    or None when a block's exponent lies more than 11 below its channel's largest (the fold is then not exact)."""
    E, n2, _ = w13.shape
    inter = n2 // 2
    assert inter == INTER, inter
    w13, s13, w2, s2 = (t.view(torch.uint8) for t in (w13, s13, w2, s2))
    out = None
    lossy = 0
    for e0 in range(0, E, expert_chunk):
        e1 = min(E, e0 + expert_chunk)
        p13 = pack_w13(w13[e0:e1, :inter], s13[e0:e1, :inter], w13[e0:e1, inter:], s13[e0:e1, inter:])
        p2 = pack_w2(w2[e0:e1], s2[e0:e1])
        lossy += p13.lossy_blocks + p2.lossy_blocks
        if out is None:
            out = PackedK1(
                p13.w13.new_empty((E,) + tuple(p13.w13.shape[1:])),
                p13.cs13.new_empty((E,) + tuple(p13.cs13.shape[1:])),
                p2.w2.new_empty((E,) + tuple(p2.w2.shape[1:])),
                p2.cs2.new_empty((E,) + tuple(p2.cs2.shape[1:])),
            )
        out.w13[e0:e1].copy_(p13.w13)
        out.cs13[e0:e1].copy_(p13.cs13)
        out.w2[e0:e1].copy_(p2.w2)
        out.cs2[e0:e1].copy_(p2.cs2)
        del p13, p2
        torch.cuda.empty_cache()
    return None if lossy else out


def decode_ep(x8, xscale, topk_ids, topk_weights, packed: PackedK1, swiglu_limit: float, output):
    """The engine's EP call: received e4m3 token rows [T, 5120] + power-of-2 row scales, LOCAL topk_ids [T, K]
    (-1: not on this rank / pad row), weights [T, K] -> output [T, 5120] bf16 (every row written). CUDA-graph safe
    (no host sync; scratch from the caching allocator). Returns y (the per-slot rows of multi-slot tokens)."""
    T, K = topk_ids.shape
    E = packed.w13.shape[0]
    dev = output.device
    S = T * K
    sc = x_scratch(S, T, E, dev)
    y = torch.empty(S, HIDDEN, dtype=torch.bfloat16, device=dev)
    p13 = PackedW13(packed.w13, packed.cs13, 0)
    p2 = PackedW2(packed.w2, packed.cs2, 0)
    decode_moe(x8.view(torch.uint8), xscale, topk_ids.to(torch.int32).contiguous().view(-1),
               topk_weights.float().contiguous().view(-1), p13, p2, swiglu_limit, output, y, sc, top_k=K)
    return y


def prepare_ep(device_index: int, num_local: int, top_k: int) -> None:
    """Build and load the K1 module at weight-processing time and run one all-masked call (no expert work)."""
    with torch.cuda.device(device_index):
        _module()
        z = torch.zeros(1, dtype=torch.uint8, device="cuda")
        f = torch.zeros(1, dtype=torch.float32, device="cuda")
        packed = PackedK1(z.expand(num_local, JB, STAGES, WSTAGE), f.expand(num_local, 2, INTER),
                          z.expand(num_local, DN_JB, DN_STAGES, WSTAGE), f.expand(num_local, HIDDEN))
        for T in (2, 16):
            x8 = torch.zeros(T, HIDDEN, dtype=torch.uint8, device="cuda")
            xs = torch.ones(T, device="cuda")
            ids = torch.full((T, top_k), -1, dtype=torch.int32, device="cuda")
            w = torch.ones(ids.shape, device="cuda")
            out = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device="cuda")
            decode_ep(x8, xs, ids, w, packed, 10.0, out)
        torch.cuda.synchronize()
