"""INF-681 k-qcp: quant/copy fusions for the DeepSeek-V4.1 Hopper prefill (CP4 and TP4).

Every function here computes exactly what the unfused path computes (same per-element
arithmetic, same rounding points), only with fewer passes over HBM:

* `grouped_ue8m0_quant_fp8_mn` + `wo_b_from_grouped` (SGLANG_OPT_DSV4_WO_B_FUSED_ACT_QUANT):
  wo_a's grouped output o [T, G, R] is the einsum's [G, T, R] buffer seen token-major, so
  wo_b's ``o.flatten(1)`` copied it before its 1x128 activation quant read it again. The
  quant reads the strided view directly and wo_b's DeepGEMM takes the (q, s) it writes.
* `gather_topk_sum_add` (SGLANG_OPT_DSV4_MOE_SHARED_ADD_IN_TOPK_SUM): the W4A8 prefill
  experts' top-k sum adds the shared expert's output in its epilogue instead of a separate
  ``shared.add_(routed, alpha)`` pass over [T, hidden] (three bf16 reads/writes).
* `dual_quant_fp8` + `offer_dual_quant` (SGLANG_OPT_DSV4_MOE_INPUT_DUAL_QUANT): the MoE
  input's 1x128 quant (GRAN128 shared expert) and per-row quant (W4A8 routed experts) in one
  pass over [T, hidden].
"""

from typing import NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl

_GROUP = 128


@triton.jit
def _grouped_ue8m0_quant_mn_kernel(
    Y,
    Q,
    S,
    M,
    stride_ym,
    stride_yg,
    stride_sk,
    NUM_GROUPS,
    R: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUPS_PER_PROG: tl.constexpr,
):
    # Column c of the flattened [M, G * R] row is y[m, c // R, c % R]; one program covers
    # BLOCK_K columns inside one group g (R % BLOCK_K == 0). Arithmetic as
    # fp8_block32_gran128._ue8m0_quant_mn_kernel (bit for bit).
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    BLOCK_K: tl.constexpr = GROUP * GROUPS_PER_PROG
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    g = (pid_k * BLOCK_K) // R
    r = cols - g * R
    row_mask = rows < M
    mask = row_mask[:, None]
    x = tl.load(
        Y + rows[:, None].to(tl.int64) * stride_ym + g.to(tl.int64) * stride_yg + r[None, :],
        mask,
        other=0.0,
    ).to(tl.float32)
    xg = tl.reshape(x, (BLOCK_M, GROUPS_PER_PROG, GROUP))
    amax = tl.maximum(tl.max(tl.abs(xg), axis=2), 1e-10)
    bits = (amax * (1.0 / 448.0)).to(tl.uint32, bitcast=True)
    exp = (
        ((bits >> 23) & 0xFF).to(tl.int32) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    )
    scale_inv = ((exp + 127) << 23).to(tl.float32, bitcast=True)
    scale = ((127 - exp) << 23).to(tl.float32, bitcast=True)
    q = xg * scale[:, :, None]
    q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    q = tl.reshape(q, (BLOCK_M, BLOCK_K)).to(Q.dtype.element_ty)
    K = NUM_GROUPS * GROUP
    tl.store(Q + rows[:, None].to(tl.int64) * K + cols[None, :], q, mask)
    groups = pid_k * GROUPS_PER_PROG + tl.arange(0, GROUPS_PER_PROG)
    tl.store(
        S + groups[None, :].to(tl.int64) * stride_sk + rows[:, None],
        scale_inv,
        row_mask[:, None] & (groups[None, :] < NUM_GROUPS),
    )


def grouped_ue8m0_quant_fp8_mn(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``ue8m0_group_quant_fp8_mn(y.flatten(1).contiguous(), 128)`` for y [M, G, R] with a
    unit last stride, without materializing the flattened copy. Returns (q [M, G * R] e4m3,
    s [M, G * R / 128] fp32, MN-major with the M stride aligned to 4)."""
    assert y.dim() == 3 and y.stride(2) == 1
    m, groups, r = y.shape
    assert r % 512 == 0, "one program covers 512 columns inside one group"
    k = groups * r
    num_groups = k // _GROUP
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=y.device)
    m_al = (m + 3) // 4 * 4
    s_buf = torch.empty((num_groups, m_al), dtype=torch.float32, device=y.device)
    s = s_buf[:, :m].t()
    if m == 0:
        return q, s
    block_m = 16
    grid = (triton.cdiv(m, block_m), num_groups // 4)
    _grouped_ue8m0_quant_mn_kernel[grid](
        y,
        q,
        s_buf,
        m,
        y.stride(0),
        y.stride(1),
        m_al,
        num_groups,
        R=r,
        GROUP=_GROUP,
        BLOCK_M=block_m,
        GROUPS_PER_PROG=4,
        num_warps=4,
    )
    return q, s


def wo_b_from_grouped(o: torch.Tensor, wo_b) -> Optional[torch.Tensor]:
    """wo_b(o.flatten(1)) for a GRAN128 (fp8_block32_gran128) wo_b without a TP reduction,
    reading wo_a's grouped output in place; None when this path does not apply (the
    caller runs the regular wo_b)."""
    from sglang.kernels.ops.gemm.fp8_kernel import deep_gemm_fp8_fp8_bf16_nt
    from sglang.srt.layers.quantization.fp8_block32_gran128 import prefer_gran32_kernel

    if not (
        isinstance(o, torch.Tensor)
        and o.dim() == 3
        and o.dtype == torch.bfloat16
        and o.stride(2) == 1
        and o.shape[2] % 512 == 0
        and getattr(wo_b, "dense_gran128", False)
        and getattr(wo_b, "bias", None) is None
        and getattr(wo_b, "tp_size", 1) == 1
    ):
        return None
    m, k = o.shape[0], o.shape[1] * o.shape[2]
    weight = wo_b.weight
    n = weight.shape[0]
    if m == 0 or weight.shape[1] != k or prefer_gran32_kernel(m, n, k):
        return None
    q, s = grouped_ue8m0_quant_fp8_mn(o)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=o.device)
    deep_gemm_fp8_fp8_bf16_nt(q, s, weight, wo_b.weight_scale_inv, out, _GROUP, _GROUP)
    return out


@triton.jit
def _gather_topk_sum_add_kernel(
    Y,
    POS,
    OUT,
    ALPHA,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # mxfp4_w4a8_prefill._gather_topk_sum_kernel (fp32 sum in k order, rounded to bf16),
    # then torch's shared.add_(routed, alpha=ALPHA): bf16(shared + ALPHA * routed) in fp32.
    t = tl.program_id(0).to(tl.int64)
    o = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    m = o < H
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        r = tl.load(POS + t * TOPK + k).to(tl.int64)
        acc += tl.load(Y + r * H + o, mask=m & (r >= 0), other=0.0).to(tl.float32)
    routed = acc.to(tl.bfloat16).to(tl.float32)
    shared = tl.load(OUT + t * H + o, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + t * H + o, (shared + ALPHA * routed).to(tl.bfloat16), mask=m)


def topk_sum_by_pos_add(down_sorted, pos, shared, top_k: int, alpha: float):
    """shared[t] = bf16(shared[t] + alpha * bf16(sum_k down_sorted[pos[t * top_k + k]])),
    in place; pos < 0 slots contribute nothing."""
    num_tokens, hidden = shared.shape
    assert shared.is_contiguous() and shared.dtype == torch.bfloat16
    block_h = 1024
    _gather_topk_sum_add_kernel[(num_tokens, triton.cdiv(hidden, block_h))](
        down_sorted, pos, shared, float(alpha), hidden, top_k, block_h, num_warps=4
    )
    return shared


def gather_topk_sum_add(down_sorted, combine_indx, shared, top_k: int, alpha: float):
    """`_gather_topk_sum` (combine_indx[i] = the slot at sorted position i) with the
    shared-expert add folded in; writes into ``shared``."""
    n_slots = combine_indx.numel()
    pos = torch.empty_like(combine_indx)
    pos[combine_indx.long()] = torch.arange(
        n_slots, dtype=combine_indx.dtype, device=combine_indx.device
    )
    return topk_sum_by_pos_add(down_sorted, pos, shared, top_k, alpha)


# --- the shared-add handshake between DeepseekV2MoE and the W4A8 prefill experts -------


def arm_shared_add(experts, shared: torch.Tensor, routed_scaling_factor: float) -> bool:
    """Offer ``shared`` to the W4A8 prefill experts' top-k sum; True when offered. The
    experts take it only on the paths that implement it (see `take_shared_add`)."""
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    method = getattr(experts, "quant_method", None)
    runner = getattr(method, "_w4a8", None)
    if not (
        isinstance(method, Mxfp4MarlinMoEMethod)
        and getattr(method, "_use_w4a8", False)
        and runner is not None
        and not getattr(experts, "reduce_results", True)
        and shared.dtype == torch.bfloat16
        and shared.is_contiguous()
        and shared.dim() == 2
    ):
        return False
    # maybe_fuse_routed_scale_and_shared_add: Marlin's method is the fused kind, so the
    # add is shared.add_(routed, alpha) with alpha 1 when the top-k already scaled.
    alpha = (
        1.0
        if experts.should_fuse_routed_scaling_factor_in_topk
        else float(routed_scaling_factor)
    )
    runner.qcp_shared_add = (shared, alpha)
    runner.qcp_shared_add_taken = False
    return True


def disarm_shared_add(experts) -> bool:
    """Withdraw the offer; True when the experts folded the add (``shared`` now holds
    shared + alpha * routed)."""
    runner = experts.quant_method._w4a8
    taken = runner.qcp_shared_add_taken
    runner.qcp_shared_add = None
    runner.qcp_shared_add_taken = False
    return taken


def take_shared_add(runner, num_tokens: int, hidden: int):
    """For the experts: the armed (shared, alpha) if it fits this call, else None."""
    offer = getattr(runner, "qcp_shared_add", None)
    if offer is None:
        return None
    shared, alpha = offer
    if shared.shape != (num_tokens, hidden):
        return None
    runner.qcp_shared_add_taken = True
    return shared, alpha


# --- SGLANG_OPT_DSV4_MOE_INPUT_DUAL_QUANT: one pass for both quants of the MoE input -----
#
# The [T, hidden] MoE input is quantized twice: per 1x128 group (UE8M0, MN-major scales) for
# the GRAN128 shared expert's gate_up GEMM, and per row (power-of-two scale) for the W4A8
# routed experts. One kernel reads the row once and writes both, with the arithmetic of
# fp8_block32_gran128._ue8m0_quant_mn_kernel and mxfp4_w4a8_prefill._row_quant_kernel.


class Gran128Activation(NamedTuple):
    """A pre-quantized activation for a GRAN128 linear (fp8_block32_gran128.apply):
    q [M, K] e4m3 and s [M, K / 128] fp32 powers of two, MN-major (M stride aligned to 4)."""

    q: torch.Tensor
    s: torch.Tensor


@triton.jit
def _dual_quant_kernel(
    X,
    Q1,
    S1,
    Q2,
    S2,
    stride_s1,
    N: tl.constexpr,
    GROUP: tl.constexpr,
    CHUNK: tl.constexpr,
):
    # Pass 1 finds the row's amax chunk by chunk; pass 2 re-reads the row (from cache)
    # and writes both quantizations. No masked lanes for N = 5 x 1024.
    r = tl.program_id(0).to(tl.int64)
    o = tl.arange(0, CHUNK)
    amax = tl.zeros((), tl.float32)
    for c in tl.static_range(N // CHUNK):
        x = tl.load(X + r * N + c * CHUNK + o).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x)))
    # per row (mxfp4_w4a8_prefill._row_quant_kernel)
    amax = tl.maximum(amax, 1e-4)
    s = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
    tl.store(S2 + r, s)
    g = tl.arange(0, CHUNK // GROUP)
    for c in tl.static_range(N // CHUNK):
        x = tl.load(X + r * N + c * CHUNK + o).to(tl.float32)
        tl.store(Q2 + r * N + c * CHUNK + o, (x / s).to(tl.float8e4nv))
        # per 1x128 group (fp8_block32_gran128._ue8m0_quant_mn_kernel)
        xg = tl.reshape(x, (CHUNK // GROUP, GROUP))
        gamax = tl.maximum(tl.max(tl.abs(xg), axis=1), 1e-10)
        bits = (gamax * (1.0 / 448.0)).to(tl.uint32, bitcast=True)
        exp = (
            ((bits >> 23) & 0xFF).to(tl.int32)
            - 127
            + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        )
        scale_inv = ((exp + 127) << 23).to(tl.float32, bitcast=True)
        scale = ((127 - exp) << 23).to(tl.float32, bitcast=True)
        q = xg * scale[:, None]
        q = tl.minimum(tl.maximum(q, -448.0), 448.0)
        q = tl.reshape(q, (CHUNK,)).to(Q1.dtype.element_ty)
        tl.store(Q1 + r * N + c * CHUNK + o, q)
        gi = c * (CHUNK // GROUP) + g
        tl.store(S1 + gi.to(tl.int64) * stride_s1 + r, scale_inv)


def dual_quant_fp8(x: torch.Tensor):
    """(Gran128Activation(q1, s1), (q2, s2)): q1/s1 equal ue8m0_group_quant_fp8_mn(x, 128),
    q2/s2 equal mxfp4_w4a8_prefill._row_quant(x), bit for bit."""
    assert x.dim() == 2 and x.is_contiguous() and x.dtype == torch.bfloat16
    m, n = x.shape
    assert n % _GROUP == 0
    num_groups = n // _GROUP
    q1 = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    m_al = (m + 3) // 4 * 4
    s1_buf = torch.empty((num_groups, m_al), dtype=torch.float32, device=x.device)
    q2 = torch.empty((m, n), dtype=torch.float8_e4m3fn, device=x.device)
    s2 = torch.empty(m, dtype=torch.float32, device=x.device)
    if m > 0:
        chunk = 1024 if n % 1024 == 0 else _GROUP
        _dual_quant_kernel[(m,)](
            x, q1, s1_buf, q2, s2, m_al, N=n, GROUP=_GROUP, CHUNK=chunk, num_warps=4
        )
    return Gran128Activation(q1, s1_buf[:, :m].t()), (q2, s2)


def gran128_linear_prequant(layer, x: Gran128Activation, bias) -> torch.Tensor:
    """fp8_block32_gran128.block128_fp8_linear for an already quantized activation (the
    producer checked the GRAN32 windows)."""
    from sglang.kernels.ops.gemm.fp8_kernel import deep_gemm_fp8_fp8_bf16_nt

    m, n = x.q.shape[0], layer.weight.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.q.device)
    if m > 0:
        deep_gemm_fp8_fp8_bf16_nt(
            x.q, x.s, layer.weight, layer.weight_scale_inv, out, _GROUP, _GROUP
        )
    if bias is not None:
        out += bias
    return out


def offer_dual_quant(moe, hidden_states: torch.Tensor) -> Optional[Gran128Activation]:
    """For DeepseekV2MoE.forward_normal: when the shared expert's gate_up is a GRAN128
    linear and the routed experts are the W4A8 prefill experts on their row-quant path,
    quantize the MoE input once, hand the row quant to the experts and return the shared
    expert's pre-quantized gate_up input; None leaves both paths as they are."""
    from sglang.srt.layers.quantization.fp8_block32_gran128 import prefer_gran32_kernel
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    shared = getattr(moe, "shared_experts", None)
    gate_up = getattr(shared, "gate_up_proj", None)
    method = getattr(getattr(moe, "experts", None), "quant_method", None)
    runner = getattr(method, "_w4a8", None)
    x = hidden_states
    if not (
        gate_up is not None
        and getattr(gate_up, "dense_gran128", False)
        and isinstance(method, Mxfp4MarlinMoEMethod)
        and getattr(method, "_use_w4a8", False)
        and runner is not None
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.shape[0] > max(0, runner.small_step_rows)  # the row-quant (copy) path
        and x.shape[1] == gate_up.weight.shape[1]
        and x.shape[1] % _GROUP == 0
        and not prefer_gran32_kernel(x.shape[0], *gate_up.weight.shape)
    ):
        return None
    g128, row = dual_quant_fp8(x)
    runner.qcp_row_quant = (x.data_ptr(), tuple(x.shape), row)
    return g128


def take_row_quant(runner, hidden_states: torch.Tensor):
    """For the W4A8 experts: the offered (q, s) row quant of exactly this input, once."""
    offer = getattr(runner, "qcp_row_quant", None)
    if offer is None:
        return None
    runner.qcp_row_quant = None
    ptr, shape, row = offer
    if ptr != hidden_states.data_ptr() or shape != tuple(hidden_states.shape):
        return None
    return row


def withdraw_row_quant(moe) -> None:
    runner = getattr(getattr(moe.experts, "quant_method", None), "_w4a8", None)
    if runner is not None:
        runner.qcp_row_quant = None
