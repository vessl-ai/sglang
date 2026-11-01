"""INF-681 k-pfp8: gather less than the BF16 MoE input in the DeepSeek-V4 prefill-CP MoE.

With prefill context parallelism (attention CP x MoE TP, ``--enable-prefill-cp``) every
rank all-gathers the BF16 MoE input of all T rows of the step (rank-major), then routes
all T rows (router GEMM + top-k) and quantizes all T rows to FP8 for the W4A8 prefill
experts. Router, top-k and the experts' row quantization are per-row functions, so the
same rows come out when each rank computes them on its own T/cp shard first and the
results are gathered instead:

- SGLANG_OPT_DSV4_CP_MOE_LOCAL_ROUTER: route the shard, gather the top-k ids and weights
  (48 B per row at top 6) next to the BF16 rows.
- SGLANG_OPT_DSV4_CP_MOE_FP8_ALLGATHER: also quantize the shard with the experts' own
  _row_quant and gather the FP8 rows and their scales (5,124 B per row instead of
  10,240 B at hidden 5120). The experts take the gathered quantization as is
  (Mxfp4W4A8PrefillExperts row_quant); no BF16 copy of the gathered rows exists.

Same bits as the BF16 path, by construction, when each per-row function gives the same
bits for a row whatever the number of rows in the call: _row_quant and the top-k kernels
(one row per program) do; the router GEMM (cuBLAS bf16 x bf16 -> fp32) must be checked on
the target GPU (test_dsv4_cp_moe_fp8ag_inf681.py); the gate takes the same branch on both
sides only above its tiny-GEMM row limit (checked in plan()).
SGLANG_OPT_DSV4_CP_MOE_FP8AG_VERIFY also runs the BF16 path on every such layer and logs
the layers whose top-k, FP8 input or MoE output differ (debug only, doubles the MoE).

The same pattern (quantize before the all-gather, gather the top-k with it) is the
flashinfer_cutlass FP4 all-gather of the standard token dispatcher.
"""

from __future__ import annotations

import logging
from typing import Callable, NamedTuple, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import W4A8RowQuant, _row_quant

logger = logging.getLogger(__name__)


class PreRouted(NamedTuple):
    """What DeepseekV2MoE.forward_prerouted takes instead of running the router: the
    top-k of the gathered rows and, in FP8 mode, their W4A8 row quantization."""

    topk_output: object  # StandardTopKOutput over the gathered rows
    row_quant: Optional[W4A8RowQuant]


_LOGGED = False


def _w4a8_runner(mlp):
    """The W4A8 prefill experts of this MoE when it runs them on its FP8-copy path."""
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    method = getattr(getattr(mlp, "experts", None), "quant_method", None)
    if not (
        isinstance(method, Mxfp4MarlinMoEMethod) and getattr(method, "_use_w4a8", False)
    ):
        return None
    return getattr(method, "_w4a8", None)


def plan(mlp, x_local: torch.Tensor, *, shared_local: bool) -> Optional[str]:
    """ "fp8", "router" or None (the BF16 all-gather with the router on all rows) for
    this layer's prefill-CP MoE. ``x_local``: this rank's padded CP shard (the same row
    count on every rank, so every rank decides the same and the collectives match).
    ``shared_local``: the caller computes the shared expert on the shard
    (SGLANG_OPT_DSV4_CP_SHARED_EXPERT_LOCAL), so the MoE needs no BF16 rows for it."""
    global _LOGGED
    fp8 = envs.SGLANG_OPT_DSV4_CP_MOE_FP8_ALLGATHER.get()
    if not (fp8 or envs.SGLANG_OPT_DSV4_CP_MOE_LOCAL_ROUTER.get()):
        return None
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_is_capture_mode,
    )
    from sglang.srt.runtime_context import get_exec, get_parallel

    runner = _w4a8_runner(mlp)
    gate = getattr(mlp, "gate", None)
    rows = x_local.shape[0]
    total = rows * get_parallel().attn_cp_size
    ok = (
        shared_local
        # The W4A8 prefill layers only (the late layers keep their graph-captured path).
        and runner is not None
        and total > max(0, runner.small_step_rows)  # its FP8-copy (row-quant) path
        and torch.cuda.is_available()
        and not torch.cuda.is_current_stream_capturing()
        and not get_is_capture_mode()
        and not getattr(mlp, "_enable_a2a_moe", True)
        and gate is not None
        and getattr(gate, "e_score_correction_bias_vl", None) is None  # no vision top-k
        # The shard and the gathered rows both take the gate's cuBLAS branch.
        and rows > getattr(gate, "tiny_router_gemm_max_tokens", 1 << 30)
        and not get_exec().moe.enable_eplb
        and x_local.dim() == 2
        and x_local.dtype == torch.bfloat16
    )
    if not ok:
        return None
    if not _LOGGED:
        _LOGGED = True
        logger.info(
            "k-pfp8 prefill-CP MoE: local router on, FP8 all-gather %s, verify %s "
            "(first layer: %s)",
            "on" if fp8 else "off",
            envs.SGLANG_OPT_DSV4_CP_MOE_FP8AG_VERIFY.get(),
            getattr(getattr(mlp.experts, "quant_method", None), "prefix", "?"),
        )
    return "fp8" if fp8 else "router"


def _route(mlp, x: torch.Tensor, num_token_non_padded, input_ids):
    """DeepseekV2MoE.forward_normal's router on these rows (no vision top-k, no EPLB:
    see plan())."""
    logits = mlp.gate(x)
    kwargs = {"input_ids": input_ids} if getattr(mlp, "is_hash", False) else {}
    return mlp.topk(
        x,
        logits,
        num_token_non_padded=num_token_non_padded,
        expert_location_dispatch_info=None,
        **kwargs,
    )


@triton.jit
def _pack_kernel(
    S, IDS, W, META, R, K: tl.constexpr, HAS_S: tl.constexpr, BLOCK_R: tl.constexpr
):
    # META[r] = [bits(S[r]) if HAS_S] + IDS[r, :K] + bits(W[r, :K]), int32
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    rm = rows < R
    r64 = rows.to(tl.int64)
    base = r64 * (HAS_S + 2 * K)
    if HAS_S:
        s = tl.load(S + r64, mask=rm, other=0.0)
        tl.store(META + base, s.to(tl.int32, bitcast=True), mask=rm)
    for k in tl.static_range(K):
        ids = tl.load(IDS + r64 * K + k, mask=rm, other=0)
        tl.store(META + base + HAS_S + k, ids, mask=rm)
        w = tl.load(W + r64 * K + k, mask=rm, other=0.0)
        tl.store(META + base + HAS_S + K + k, w.to(tl.int32, bitcast=True), mask=rm)


@triton.jit
def _unpack_kernel(
    META, S, IDS, W, R, K: tl.constexpr, HAS_S: tl.constexpr, BLOCK_R: tl.constexpr
):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    rm = rows < R
    r64 = rows.to(tl.int64)
    base = r64 * (HAS_S + 2 * K)
    if HAS_S:
        s = tl.load(META + base, mask=rm, other=0)
        tl.store(S + r64, s.to(tl.float32, bitcast=True), mask=rm)
    for k in tl.static_range(K):
        ids = tl.load(META + base + HAS_S + k, mask=rm, other=0)
        tl.store(IDS + r64 * K + k, ids, mask=rm)
        w = tl.load(META + base + HAS_S + K + k, mask=rm, other=0)
        tl.store(W + r64 * K + k, w.to(tl.float32, bitcast=True), mask=rm)


_BLOCK_R = 256


def pack_meta(
    s: Optional[torch.Tensor], ids: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """[R, has_s + 2K] int32 rows: the row scale's bits (if given), the K top-k ids and
    the K top-k weights' bits."""
    assert ids.dtype == torch.int32 and w.dtype == torch.float32, (ids.dtype, w.dtype)
    assert ids.shape == w.shape and ids.dim() == 2
    rows, k = ids.shape
    has_s = int(s is not None)
    if s is not None:
        assert s.dtype == torch.float32 and s.shape == (rows,)
    meta = torch.empty((rows, has_s + 2 * k), dtype=torch.int32, device=ids.device)
    if rows > 0:
        ids, w = ids.contiguous(), w.contiguous()
        _pack_kernel[(triton.cdiv(rows, _BLOCK_R),)](
            s if s is not None else w, ids, w, meta, rows, k, has_s, _BLOCK_R
        )
    return meta


def unpack_meta(
    meta: torch.Tensor, k: int, has_s: bool
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    rows = meta.shape[0]
    assert meta.shape[1] == int(has_s) + 2 * k and meta.is_contiguous()
    s = torch.empty(rows, dtype=torch.float32, device=meta.device) if has_s else None
    ids = torch.empty((rows, k), dtype=torch.int32, device=meta.device)
    w = torch.empty((rows, k), dtype=torch.float32, device=meta.device)
    if rows > 0:
        _unpack_kernel[(triton.cdiv(rows, _BLOCK_R),)](
            meta, s if s is not None else w, ids, w, rows, k, int(has_s), _BLOCK_R
        )
    return s, ids, w


def gather(
    mlp,
    x_local: torch.Tensor,
    *,
    fp8: bool,
    num_token_non_padded,
    input_ids_global: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, PreRouted]:
    """Route (and in FP8 mode quantize) this rank's CP shard, then all-gather the
    results in rank-major order, the order attn_cp_interleave_gather gives the BF16
    rows. Returns (rows for the MoE call, PreRouted). In FP8 mode the rows are a
    stride-0 [T, hidden] bf16 shape carrier that nothing reads.

    ``num_token_non_padded``: the MoE row bound for the shard (every row routed, as for
    the gathered rows); ``input_ids_global``: the step's rank-major token ids (all T
    rows; the hash-routed layers route this rank's slice of them)."""
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )
    from sglang.srt.layers.dp_attention import is_allocation_symmetric
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    group, cp, rank = (
        parallel.attn_cp_group,
        parallel.attn_cp_size,
        parallel.attn_cp_rank,
    )
    rows, hidden = x_local.shape
    total = rows * cp
    ids_local = None
    if getattr(mlp, "is_hash", False):
        assert input_ids_global is not None and input_ids_global.shape[0] == total, (
            None if input_ids_global is None else input_ids_global.shape,
            total,
        )
        ids_local = input_ids_global[rank * rows : (rank + 1) * rows]
    x_local = x_local.contiguous()
    q = s = None
    if fp8:
        q, s = _row_quant(x_local)  # the experts' own quantization, on the shard only
    topk = _route(mlp, x_local, num_token_non_padded, ids_local)
    k = topk.topk_ids.shape[1]
    meta = pack_meta(s, topk.topk_ids, topk.topk_weights)
    with use_symmetric_memory(group, disabled=not is_allocation_symmetric()):
        meta_all = meta.new_empty((total, meta.shape[1]))
        if fp8:
            rows_all = torch.empty(
                (total, hidden), dtype=torch.uint8, device=x_local.device
            )
        else:
            rows_all = x_local.new_empty((total, hidden))
    group.all_gather_into_tensor(rows_all, q.view(torch.uint8) if fp8 else x_local)
    group.all_gather_into_tensor(meta_all, meta)
    s_all, ids_all, w_all = unpack_meta(meta_all, k, fp8)
    topk_output = StandardTopKOutput(w_all, ids_all, None)
    if not fp8:
        return rows_all, PreRouted(topk_output, None)
    carrier = x_local.new_empty((1, hidden)).expand(total, hidden)
    row_quant = W4A8RowQuant(rows_all.view(torch.float8_e4m3fn), s_all)
    return carrier, PreRouted(topk_output, row_quant)


# --- SGLANG_OPT_DSV4_CP_MOE_FP8AG_VERIFY ------------------------------------------------

_VERIFY = {"layers": 0, "ids": 0, "weights": 0, "q": 0, "s": 0, "out": 0}


def verify(
    mlp,
    x_local: torch.Tensor,
    prerouted: PreRouted,
    out: torch.Tensor,
    *,
    num_token_non_padded,
    input_ids_global: Optional[torch.Tensor],
    run_moe: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Run the BF16 path on the same shard (gather, router and the MoE on all rows,
    ``run_moe``) and count the differences from the k-pfp8 path's ``prerouted`` and
    ``out`` (bitwise). Called inside the caller's MoE contexts (all rows routed)."""
    from sglang.srt.layers.cp.interleave import attn_cp_interleave_gather
    from sglang.srt.runtime_context import get_parallel

    ref_rows = attn_cp_interleave_gather(x_local)
    ids = input_ids_global if getattr(mlp, "is_hash", False) else None
    ref_topk = _route(mlp, ref_rows, num_token_non_padded, ids)
    got = prerouted.topk_output
    diff = {
        "ids": not torch.equal(ref_topk.topk_ids, got.topk_ids),
        "weights": not torch.equal(ref_topk.topk_weights, got.topk_weights),
        "q": False,
        "s": False,
    }
    if prerouted.row_quant is not None:
        q_ref, s_ref = _row_quant(ref_rows)
        diff["q"] = not torch.equal(
            q_ref.view(torch.uint8), prerouted.row_quant.q.view(torch.uint8)
        )
        diff["s"] = not torch.equal(s_ref, prerouted.row_quant.s)
    ref_out = run_moe(ref_rows)
    diff["out"] = not torch.equal(ref_out, out)
    _VERIFY["layers"] += 1
    for key, bad in diff.items():
        _VERIFY[key] += int(bad)
    prefix = getattr(getattr(mlp.experts, "quant_method", None), "prefix", "?")
    if any(diff.values()):
        logger.warning(
            "k-pfp8 verify MISMATCH rank %d %s rows %d: %s; totals %s",
            get_parallel().attn_cp_rank,
            prefix,
            ref_rows.shape[0],
            [key for key, bad in diff.items() if bad],
            _VERIFY,
        )
    elif _VERIFY["layers"] % 64 == 0 or _VERIFY["layers"] <= 2:
        logger.info(
            "k-pfp8 verify rank %d: %s (last %s, rows %d)",
            get_parallel().attn_cp_rank,
            _VERIFY,
            prefix,
            ref_rows.shape[0],
        )
