"""W4A8 MXFP4 MoE experts for SM90 prefill (INF-674).

Hopper has no FP4 tensor cores, so the W4A16 paths (Marlin, triton_kernels) upcast the
weights to BF16 inside the GEMM and run at BF16 tensor-core rate. This path instead:

1. converts the layer's MXFP4 experts to FP8 e4m3 just in time, into one scratch buffer
   shared by all layers (a bandwidth-bound pass, ~1.3 ms per layer on H200 at the TP4
   DeepSeek-V4.1-Flash expert shapes);
2. quantizes the activations to FP8 with one power-of-2 scale per row;
3. runs both expert GEMMs as triton_kernels FP8 x FP8 ragged matmuls (FP8 tensor cores),
   with the row scale and a per-output-channel weight scale applied in the epilogue.

Numerics:
- Weights are exact. An e2m1 value times 2^d is exact in e4m3 for d in [-5, 6] with the
  integer conversion below, so with d = s - max_k(s) + 6 the per-32 e8m0 scale folds into
  one per-channel scale 2^(max_k(s) - 133) as long as the block exponents of an output
  channel span at most 11 binades (the DeepSeek-V4.1-Flash checkpoint spans at most 4).
  Layers that do not satisfy this raise at load time.
- Activations follow the model's own reference inference code, which quantizes every
  expert GEMM input to FP8 with a power-of-2 (UE8M0) scale per 1x32 block: a power-of-2
  scale per row rounds every value that stays in the e4m3 normal range identically.
  Against a BF16-activation computation the MoE output differs by ~4.5-5% (relative L2),
  which is the model's native FP8 activation quantization, not an error of this kernel.

Prefetch (SGLANG_DSV4_W4A8_PREFETCH, on by default): the conversion does not depend on the
activations, so right after a layer's experts finish, the next layer's FP8 weights are
converted on a side stream and overlap that layer's attention instead of sitting on the
critical path. The next layer is the one observed to follow last time (construction order
before that); a wrong guess only costs an in-line conversion. Two scratch buffers (w13, w2;
3.4 GB per GPU at the TP4 shapes) are shared by all layers.

Use for prefill only (no CUDA graph support: the routing sort and ragged metadata are
computed eagerly). Enabled by SGLANG_OPT_DSV4_W4A8_PREFILL_MOE=1 on SM90.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_LUT: dict[torch.device, torch.Tensor] = {}

# A FP4 nibble scaled by 2^d with d in [_MIN_SHIFT, _MAX_SHIFT] is a normal e4m3 number.
_MAX_SHIFT = 6
_MIN_SHIFT = -5
# Conversion tile: scale blocks (16 packed bytes each) per program.
_CONVERT_BLOCK = 256
# An overlapped prefetch (SGLANG_DSV4_W4A8_PREFETCH_CHUNKS > 0) is launched only after a
# layer with at least this many rows, whose MoE all-reduce is long enough to cover the
# conversion. A smaller layer keeps the single launch before its top-k sum.
_OVERLAP_MIN_ROWS = 8192


class _Scratch:
    """FP8 weight buffers shared by every layer on one device, plus the stream state that
    says which layer they currently hold. Never freed: the size is fixed by the model.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.w13 = torch.empty(0, dtype=torch.float8_e4m3fn, device=device)
        self.w2 = torch.empty(0, dtype=torch.float8_e4m3fn, device=device)
        # The experts whose FP8 weights are (being) written into the buffers.
        self.holder = None
        self.stream = None
        self.ready = None  # recorded after the last conversion into the buffers
        self.free = None  # recorded after the last GEMM that read the buffers
        self.last_called = None

    def reserve(self, n13: int, n2: int):
        if self.w13.numel() < n13:
            self.w13 = torch.empty(n13, dtype=torch.float8_e4m3fn, device=self.device)
        if self.w2.numel() < n2:
            self.w2 = torch.empty(n2, dtype=torch.float8_e4m3fn, device=self.device)


_SCRATCH: dict[torch.device, _Scratch] = {}
# Construction order of the experts; the first guess for "next layer".
_INSTANCES: list = []


def _scratch(device: torch.device) -> _Scratch:
    sc = _SCRATCH.get(device)
    if sc is None:
        sc = _SCRATCH[device] = _Scratch(device)
    return sc


def _lut(device: torch.device) -> torch.Tensor:
    """int32 per packed byte. Low 16 bits: the two e4m3 bytes of its nibbles at d = 0
    (low nibble = even k in the low byte). High 16 bits: 0x08 in each byte whose nibble is
    nonzero, so `lut16 + d * mask16` adds d to both exponent fields and keeps zeros zero.
    """
    lut = _LUT.get(device)
    if lut is None:
        b = torch.arange(256, dtype=torch.int32)

        def nibble(c):
            mag = c & 7
            exp = torch.where(mag >= 2, (mag >> 1) + 6, torch.full_like(mag, 6))
            bits = (exp << 3) | (((mag & 1) << 2) * (mag >= 2))
            bits = torch.where(mag == 0, torch.zeros_like(bits), bits)
            return bits | ((c & 8) << 4), (mag != 0).int() * 8

        lo, lo_mask = nibble(b & 0xF)
        hi, hi_mask = nibble(b >> 4)
        lut = ((lo | (hi << 8)) | ((lo_mask | (hi_mask << 8)) << 16)).to(device)
        _LUT[device] = lut
    return lut


@triton.jit
def _mxfp4_to_fp8_kernel(
    W,
    SCALE,
    SMAX,
    LUT,
    OUT,
    first_block,
    n_blocks,
    BLOCKS_PER_ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # W viewed as [n_blocks, 16] bytes: one 32-element scale block per row of 16 bytes.
    # This launch converts scale blocks [first_block, n_blocks).
    pid = tl.program_id(0).to(tl.int64)
    blk = first_block + pid * BLOCK + tl.arange(0, BLOCK)
    mask = blk < n_blocks
    j = tl.arange(0, 16)
    b = tl.load(W + blk[:, None] * 16 + j[None, :], mask=mask[:, None], other=0).to(
        tl.int32
    )
    s = tl.load(SCALE + blk, mask=mask, other=0).to(tl.int32)
    smax = tl.load(SMAX + blk // BLOCKS_PER_ROW, mask=mask, other=0).to(tl.int32)
    d = (s - smax + 6)[:, None]
    lut = tl.load(LUT + b)
    v = (lut & 0xFFFF) + d * (lut >> 16)
    tl.store(OUT + blk[:, None] * 16 + j[None, :], v.to(tl.int16), mask=mask[:, None])


# e4m3 bytes of the e2m1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 at exponent shift 0, as
# the two source registers of PTX prmt (byte i of the pair = magnitude code i).
_PRMT_LUT_LO = tl.constexpr(0x3C383000)
_PRMT_LUT_HI = tl.constexpr(0x4C484440)
_WORDS_BLOCK = 1024


@triton.jit
def _prmt(a, b, c):
    return tl.inline_asm_elementwise(
        "prmt.b32 $0, $1, $2, $3;",
        "=r,r,r,r",
        [a, b, c],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _nibbles_to_fp8(x16, dd, lut_lo, lut_hi):
    """4 e2m1 nibbles (bits 0..15, nibble i = element i) -> 4 e4m3 bytes (byte i).
    dd = 8 * d + 128 with d the exponent shift of the 32-element block."""
    mag = _prmt(lut_lo, lut_hi, x16 & 0x7777)
    # 0x01 in each byte whose magnitude is nonzero (bytes are <= 0x4C: no carries).
    nz = ((mag + 0x7F7F7F7F) >> 7) & 0x01010101
    # Add 8d to each nonzero exponent field: +128 keeps every byte positive and below 256,
    # so bytes do not carry into each other; the xor removes the 128 again.
    v = (mag + nz * dd) ^ (nz << 7)
    # Sign: nibble bit 3 -> byte bit 7.
    t = (x16 & 0xFF) | ((x16 & 0xFF00) << 8)
    spread = (t & 0x000F000F) | ((t & 0x00F000F0) << 4)
    return v | ((spread << 4) & -2139062144)  # 0x80808080 as int32


@triton.jit
def _mxfp4_to_fp8_words_kernel(
    W,
    SCALE,
    SMAX,
    OUT,
    first_word,
    n_words,
    BLOCKS_PER_ROW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Same result as _mxfp4_to_fp8_kernel, bit for bit, but on 32-bit words (8 nibbles,
    # 4 words per scale block) with the nibble table in registers (prmt) instead of a
    # per-byte table gather: HBM-bound (~4.1 TB/s on H200) instead of SM-bound (~3.9),
    # so it keeps its speed beside other kernels on fewer SMs.
    pid = tl.program_id(0).to(tl.int64)
    wi = first_word + pid * BLOCK + tl.arange(0, BLOCK)
    m = wi < n_words
    w = tl.load(W + wi, mask=m, other=0)
    blk = wi // 4
    s = tl.load(SCALE + blk, mask=m, other=0).to(tl.int32)
    smax = tl.load(SMAX + blk // BLOCKS_PER_ROW, mask=m, other=0)
    dd = (s - smax + 6) * 8 + 128
    lut_lo = tl.full([BLOCK], _PRMT_LUT_LO, tl.int32)
    lut_hi = tl.full([BLOCK], _PRMT_LUT_HI, tl.int32)
    lo = _nibbles_to_fp8(w & 0xFFFF, dd, lut_lo, lut_hi)
    hi = _nibbles_to_fp8((w >> 16) & 0xFFFF, dd, lut_lo, lut_hi)
    tl.store(OUT + wi, (hi.to(tl.int64) << 32) | (lo.to(tl.int64) & 0xFFFFFFFF), mask=m)


def _mxfp4_to_fp8(
    w: torch.Tensor, scale: torch.Tensor, smax: torch.Tensor, out, num_launches: int = 0
):
    """[E, N, K/2] packed e2m1 + [E, N, K/32] e8m0 -> [E, N, K] e4m3 (scaled by 2^(133 - smax)).

    num_launches = 0: the INF-674 byte kernel in one launch. num_launches >= 1: the word
    kernel (same bits) split into that many back-to-back launches on the current stream.
    One launch takes every SM until all its programs have been dispatched, so a kernel
    another stream launches meanwhile (e.g. an all-reduce) waits for the whole conversion.
    Between two launches the hardware can dispatch such a kernel, so it waits for at most
    one slice and then runs beside the remaining slices.
    """
    if num_launches <= 0:
        n_blocks = w.numel() // 16
        _mxfp4_to_fp8_kernel[(triton.cdiv(n_blocks, _CONVERT_BLOCK),)](
            w.view(torch.uint8),
            scale.view(torch.uint8),
            smax,
            _lut(w.device),
            out.view(torch.int16),
            0,
            n_blocks,
            scale.shape[-1],
            _CONVERT_BLOCK,
            num_warps=4,
        )
        return out
    n_words = w.numel() // 4
    n_programs = triton.cdiv(n_words, _WORDS_BLOCK)
    per_launch = triton.cdiv(n_programs, num_launches)
    for first in range(0, n_programs, per_launch):
        _mxfp4_to_fp8_words_kernel[(min(per_launch, n_programs - first),)](
            w.view(torch.int32),
            scale.view(torch.uint8),
            smax,
            out.view(torch.int64),
            first * _WORDS_BLOCK,
            n_words,
            scale.shape[-1],
            _WORDS_BLOCK,
            num_warps=8,
        )
    return out


@triton.jit
def _row_quant_kernel(X, Y, S, N: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    o = tl.arange(0, BLOCK)
    x = tl.load(X + r * N + o, mask=o < N, other=0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x)), 1e-4)
    s = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
    tl.store(Y + r * N + o, (x / s).to(tl.float8e4nv), mask=o < N)
    tl.store(S + r, s)


def _row_quant(x: torch.Tensor):
    M, N = x.shape
    y = torch.empty(M, N, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(M, device=x.device, dtype=torch.float32)
    _row_quant_kernel[(M,)](x, y, s, N, triton.next_power_of_2(N), num_warps=8)
    return y, s


@triton.jit
def _swiglu_quant_kernel(
    GU, Y, S, I: tl.constexpr, LIMIT: tl.constexpr, BLOCK: tl.constexpr
):
    r = tl.program_id(0).to(tl.int64)
    o = tl.arange(0, BLOCK)
    m = o < I
    g = tl.load(GU + r * (2 * I) + o, mask=m, other=0.0).to(tl.float32)
    u = tl.load(GU + r * (2 * I) + I + o, mask=m, other=0.0).to(tl.float32)
    g = tl.minimum(g, LIMIT)
    u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
    a = (g / (1.0 + tl.exp(-g)) * u).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(a)), 1e-4)
    s = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
    tl.store(Y + r * I + o, (a / s).to(tl.float8e4nv), mask=m)
    tl.store(S + r, s)


def _swiglu_quant(gate_up: torch.Tensor, limit: float):
    R, N2 = gate_up.shape
    inter = N2 // 2
    y = torch.empty(R, inter, device=gate_up.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(R, device=gate_up.device, dtype=torch.float32)
    _swiglu_quant_kernel[(R,)](
        gate_up, y, s, inter, limit, triton.next_power_of_2(inter), num_warps=4
    )
    return y, s


@triton.jit
def _gather_topk_sum_kernel(
    Y, POS, OUT, H: tl.constexpr, TOPK: tl.constexpr, BLOCK_H: tl.constexpr
):
    # OUT[t] = sum_k Y[POS[t * TOPK + k]] with fp32 accumulation in k order, the same
    # arithmetic as moe_topk_sum over token-major rows.
    t = tl.program_id(0).to(tl.int64)
    o = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    m = o < H
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        r = tl.load(POS + t * TOPK + k).to(tl.int64)
        acc += tl.load(Y + r * H + o, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + t * H + o, acc.to(tl.bfloat16), mask=m)


def _gather_topk_sum(down_sorted, combine_indx, out, top_k: int):
    """out[t] = sum over the top-k slots of token t of down_sorted[sorted position of slot],
    where combine_indx[i] is the slot (token * top_k + k) at sorted position i."""
    n_slots = combine_indx.numel()
    pos = torch.empty_like(combine_indx)
    pos[combine_indx.long()] = torch.arange(
        n_slots, dtype=combine_indx.dtype, device=combine_indx.device
    )
    num_tokens, hidden = out.shape
    block_h = 1024
    _gather_topk_sum_kernel[(num_tokens, triton.cdiv(hidden, block_h))](
        down_sorted, pos, out, hidden, top_k, block_h, num_warps=4
    )
    return out


def _fold(scale: torch.Tensor, name: str):
    """Per-output-channel max exponent and the matching fp32 channel scale."""
    s = scale.view(torch.uint8).to(torch.int16)
    smax = s.amax(-1)
    spread = int((smax - s.amin(-1)).max())
    if spread > _MAX_SHIFT - _MIN_SHIFT:
        raise ValueError(
            f"SGLANG_OPT_DSV4_W4A8_PREFILL_MOE: {name} block exponents span {spread} "
            f"binades in one output channel (max {_MAX_SHIFT - _MIN_SHIFT}); FP8 weights "
            "would not be exact. Unset the flag for this model."
        )
    channel_scale = torch.exp2(smax.float() - 127.0 - _MAX_SHIFT).contiguous()
    return smax.to(torch.int32).contiguous(), channel_scale


class Mxfp4W4A8PrefillExperts:
    """Keeps the checkpoint's MXFP4 experts ([E, N, K/2] int8 + [E, N, K/32] e8m0) and runs
    them as FP8 x FP8 GEMMs."""

    def __init__(
        self,
        w13,
        w13_scale,
        w2,
        w2_scale,
        swiglu_limit: float,
        prefetch: bool,
        prefetch_chunks: int = 0,
        sorted_sum: bool = False,
    ):
        self.w13, self.w13_scale = w13, w13_scale
        self.w2, self.w2_scale = w2, w2_scale
        self.w13_smax, self.w13_channel_scale = _fold(w13_scale, "w13")
        self.w2_smax, self.w2_channel_scale = _fold(w2_scale, "w2")
        self.num_experts = w13.shape[0]
        self.swiglu_limit = swiglu_limit
        self.prefetch = prefetch
        # > 0: after a large layer, prefetch in this many launches per weight, issued
        # after the layer's top-k sum so the all-reduce does not queue behind it.
        self.prefetch_chunks = prefetch_chunks
        # W2 writes its rows in expert order (no epilogue scatter) and the top-k sum
        # gathers them by position: same sums, bit for bit, ~12% less W2 time at 32k rows.
        self.sorted_sum = sorted_sum
        self.successor = None  # the experts observed to run after these ones
        self.scratch = _scratch(w13.device)
        # Reserve the buffers now so KV-cache sizing sees them.
        self.scratch.reserve(w13.numel() * 2, w2.numel() * 2)
        _lut(w13.device)
        _INSTANCES.append(self)

    def _convert_into_scratch(self, num_launches: int = 0):
        sc = self.scratch
        for w, scale, smax, buf in (
            (self.w13, self.w13_scale, self.w13_smax, sc.w13),
            (self.w2, self.w2_scale, self.w2_smax, sc.w2),
        ):
            E, N, KH = w.shape
            _mxfp4_to_fp8(
                w, scale, smax, buf[: E * N * KH * 2].view(E, N, KH * 2), num_launches
            )

    def _single_launch(self) -> int:
        """num_launches for a conversion that is not sliced: the word kernel when the
        overlapped prefetch is on (same bits, faster), else the INF-674 byte kernel."""
        return 1 if self.prefetch_chunks > 0 else 0

    def _fp8_view(self, w, buf):
        E, N, KH = w.shape
        # triton_kernels takes an FP8 rhs as [E, K, N] column-major.
        return buf[: E * N * KH * 2].view(E, N, KH * 2).transpose(-1, -2)

    def _acquire(self):
        """Make the scratch hold this layer's FP8 weights, in stream order."""
        sc = self.scratch
        current = torch.cuda.current_stream(sc.device)
        if sc.ready is not None:
            # A prefetch may still be writing the buffers.
            current.wait_event(sc.ready)
        if sc.holder is not self:
            self._convert_into_scratch(self._single_launch())
            sc.holder = self
        if sc.last_called is not None and sc.last_called is not self:
            sc.last_called.successor = self
        sc.last_called = self

    def _prefetch_next(self, num_launches: int = 0):
        sc = self.scratch
        nxt = self.successor
        if nxt is None:
            i = _INSTANCES.index(self)
            nxt = _INSTANCES[(i + 1) % len(_INSTANCES)]
        if nxt is self or nxt.scratch is not sc:
            return
        if sc.stream is None:
            sc.stream = torch.cuda.Stream(sc.device)
            sc.ready = torch.cuda.Event()
            sc.free = torch.cuda.Event()
        sc.free.record(torch.cuda.current_stream(sc.device))
        with torch.cuda.stream(sc.stream):
            sc.stream.wait_event(sc.free)
            nxt._convert_into_scratch(num_launches)
            sc.ready.record(sc.stream)
        sc.holder = nxt

    def __call__(self, hidden_states, topk_ids, topk_weights) -> torch.Tensor:
        import triton_kernels.matmul_details.opt_flags as opt_flags
        from triton_kernels.matmul import PrecisionConfig, matmul
        from triton_kernels.tensor import make_ragged_tensor_metadata

        from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum
        from sglang.srt.layers.quantization.mxfp4_marlin_moe import (
            route_slots_by_expert,
        )
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )

        # The Python state of the shared scratch (which layer it holds, the prefetch
        # events) is not replayed by a CUDA graph: a captured call would freeze it and
        # could read another layer's FP8 weights, or fail capture on the events.
        if torch.cuda.is_current_stream_capturing() or get_is_capture_mode():
            raise RuntimeError(
                "SGLANG_OPT_DSV4_W4A8_PREFILL_MOE does not support CUDA graphs; enable "
                "it only on a PD prefill server whose MoE runs eagerly."
            )
        num_tokens, top_k = topk_ids.shape
        output = torch.empty_like(hidden_states)
        if num_tokens == 0:
            return output
        routing = route_slots_by_expert(topk_ids, topk_weights, self.num_experts)
        ragged_metadata = make_ragged_tensor_metadata(
            routing.expert_counts, num_tokens * top_k
        )

        x_fp8, x_scale = _row_quant(hidden_states.contiguous())
        self._acquire()
        # Use triton_kernels' own tile heuristics with no constraint at all: constraints
        # set by others (the W4A16 path's num_stages pin, the MXFP4 swizzle's split_k=1)
        # are tuned for BF16 x MXFP4 kernels. scoped_opt_flags_constraints merges into
        # them, so replace the ContextVar directly (triton_kernels 3.8 internals).
        token = opt_flags._opt_flags_constraints.set({})
        try:
            gate_up = matmul(
                x_fp8,
                self._fp8_view(self.w13, self.scratch.w13),
                None,
                a_ragged_metadata=ragged_metadata,
                gather_indx=routing.gather_indx,
                precision_config=PrecisionConfig(
                    a_mx_tensor_scale=x_scale,
                    b_mx_tensor_scale=self.w13_channel_scale,
                    out_dtype=torch.bfloat16,
                ),
            )
            a_fp8, a_scale = _swiglu_quant(gate_up, self.swiglu_limit)
            del gate_up
            down = matmul(
                a_fp8,
                self._fp8_view(self.w2, self.scratch.w2),
                None,
                a_ragged_metadata=ragged_metadata,
                scatter_indx=None if self.sorted_sum else routing.combine_indx,
                precision_config=PrecisionConfig(
                    a_mx_tensor_scale=a_scale,
                    b_mx_tensor_scale=self.w2_channel_scale,
                    out_dtype=torch.bfloat16,
                ),
                gammas=routing.sorted_weights,
            )
        finally:
            opt_flags._opt_flags_constraints.reset(token)
        overlapped = self.prefetch_chunks > 0 and num_tokens >= _OVERLAP_MIN_ROWS
        if self.prefetch and not overlapped:
            self._prefetch_next(self._single_launch())
        if self.sorted_sum:
            _gather_topk_sum(down, routing.combine_indx, output, top_k)
        else:
            moe_topk_sum(down.view(num_tokens, top_k, -1), output)
        if self.prefetch and overlapped:
            # Launched before the top-k sum in one piece, the conversion takes every SM
            # and the main stream's next kernels (shared add, all-reduce) queue behind it
            # (equal stream priority): p1 measured 0.8 of 57 ms overlapped. Issued after
            # the top-k sum in slices, the all-reduce gets its SMs after one slice and the
            # rest of the conversion runs beside it.
            self._prefetch_next(self.prefetch_chunks)
        return output
