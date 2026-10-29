from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, NamedTuple

import torch
from torch.nn import Module

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils import log_info_on_rank0, round_up, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)


def build_marlin_moe_quant_info(layer: Module) -> MarlinMoeQuantInfo:
    """Build the Marlin quant_info for an MXFP4 MoE layer.

    Single source for the runner inputs shared by the marlin path of
    ``Mxfp4MoEMethod.apply`` and :class:`Mxfp4MarlinMoEMethod`, including
    the dispatcher's EP mapping (global -> local expert ids) when EP is on.
    """
    expert_map = getattr(layer.dispatcher, "local_expert_mapping", None)
    global_num_experts = layer.dispatcher.num_experts if expert_map is not None else -1
    return MarlinMoeQuantInfo(
        w13_qweight=layer.w13_weight,
        w2_qweight=layer.w2_weight,
        w13_scales=layer.w13_weight_scale,
        w2_scales=layer.w2_weight_scale,
        w13_g_idx_sort_indices=None,
        w2_g_idx_sort_indices=None,
        weight_bits=4,
        is_k_full=True,
        w13_bias=getattr(layer, "w13_weight_bias", None),
        w2_bias=getattr(layer, "w2_weight_bias", None),
        expert_map=expert_map,
        global_num_experts=global_num_experts,
    )


@functools.cache
def _warn_w4a8_ignored() -> None:
    logger.warning("SGLANG_OPT_DSV4_W4A8_PREFILL_MOE is set but needs SM90; ignored.")


@functools.cache
def _warn_late_w4a16_ignored() -> None:
    logger.warning(
        "SGLANG_OPT_DSV4_W4A8_LATE_W4A16_FROM_LAYER is set but "
        "--enable-decoder-swa-bounded-replay is off: the late layers see every row, "
        "so they keep the W4A8 path."
    )


def _is_late_w4a16_layer(prefix: str) -> bool:
    """Whether this W4A8 MoE layer runs W4A16 instead (bounded replay's late layers)."""
    from sglang.srt.layers.utils.common import get_layer_id
    from sglang.srt.runtime_context import get_exec

    start = envs.SGLANG_OPT_DSV4_W4A8_LATE_W4A16_FROM_LAYER.get()
    if start < 0:
        return False
    # DSpark draft stages have no "layers.<id>." in their prefix and stay W4A8.
    layer_id = get_layer_id(prefix)
    if layer_id is None or layer_id < start:
        return False
    if not get_exec().features.enable_decoder_swa_bounded_replay:
        _warn_late_w4a16_ignored()
        return False
    return True


class ExpertRouting(NamedTuple):
    """Top-k slots sorted by expert, for triton_kernels' ragged expert matmuls."""

    expert_counts: torch.Tensor  # [num_experts] int32 slots routed to each expert
    combine_indx: torch.Tensor  # [slots] int32 slot (token * top_k + k) in sorted order
    gather_indx: torch.Tensor  # [slots] int32 token of each sorted slot
    sorted_weights: torch.Tensor  # [slots] top-k weight of each sorted slot


def route_slots_by_expert(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor, num_experts: int
) -> ExpertRouting:
    """Sort the top-k slots by expert without a host sync.

    Masked slots (id -1) contribute nothing: they are routed to expert 0 with weight 0.
    The sort is stable, so slots of one expert keep token order. Expert ids must be in
    [-1, num_experts). The per-expert counts use a fixed-size scatter-add histogram:
    torch.bincount reads its max on the host, which stalls the CPU's kernel launches at
    every MoE layer and cannot be captured in a CUDA graph.
    """
    top_k = topk_ids.shape[1]
    valid = topk_ids >= 0
    expert_ids = torch.where(valid, topk_ids, 0).flatten().long()
    weights = torch.where(valid, topk_weights, 0).flatten()
    # triton_kernels' own topk needs a power-of-2 top_k, so sort here.
    combine_indx = torch.argsort(expert_ids, stable=True).to(torch.int32)
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    expert_counts.scatter_add_(
        0, expert_ids, torch.ones_like(expert_ids, dtype=torch.int32)
    )
    return ExpertRouting(
        expert_counts=expert_counts,
        combine_indx=combine_indx,
        gather_indx=combine_indx // top_k,
        sorted_weights=weights[combine_indx.long()],
    )


class Mxfp4MarlinMoEMethod:
    """MXFP4 (E8M0 scales) MoE quantization method using the Marlin backend."""

    fuse_routed_scaling_factor_in_topk = True

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix
        # W4A8 (prefill only) takes precedence over both triton_kernels switches.
        self._use_w4a8 = (
            envs.SGLANG_OPT_DSV4_W4A8_PREFILL_MOE.get() and get_platform().is_sm90
        )
        if envs.SGLANG_OPT_DSV4_W4A8_PREFILL_MOE.get() and not self._use_w4a8:
            _warn_w4a8_ignored()
        self._triton_kernels_decode = (
            envs.SGLANG_OPT_DSV4_TRITON_KERNELS_MOE_DECODE.get()
            and get_platform().is_sm90
            and not self._use_w4a8
        )
        self._use_triton_kernels = self._triton_kernels_decode or (
            envs.SGLANG_OPT_DSV4_TRITON_KERNELS_MOE.get()
            and get_platform().is_sm90
            and not self._use_w4a8
        )
        self._rows_aware_tiles = envs.SGLANG_OPT_DSV4_DECODE_MOE_ROWS_AWARE_TILES.get()
        # INF-681 SM90 decode MoE kernel; cleared in process_weights_after_loading when the
        # layer's shape is not the one the kernel covers.
        self._sm90_decode_kernel = (
            envs.SGLANG_OPT_DSV4_SM90_DECODE_MOE_KERNEL.get()
            and get_platform().is_sm90
            and not self._use_w4a8
        )
        self._sm90_decode_w4a8 = (
            envs.SGLANG_OPT_DSV4_SM90_DECODE_MOE_W4A8.get()
            and get_platform().is_sm90
            and not self._use_w4a8
        )

    def create_moe_runner(self, layer, moe_runner_config):
        from sglang.srt.layers.moe.moe_runner import MoeRunner

        self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoeWeightScaleSupported,
        )

        layer._dsv4_mxfp4_backend = None  # set in process_weights_after_loading
        fp4_block_k = 32
        # Hopper repacking pads the loaded gate/up halves and down columns to
        # the required tile sizes. Keep their logical widths until then: early
        # padding to 128 turns a TP8 width of 288 into 384 instead of 320.
        if not get_platform().is_sm90:
            intermediate_size_per_partition = round_up(
                intermediate_size_per_partition, 128
            )
        hidden_size = round_up(hidden_size, 256)
        self.hidden_pad = hidden_size - layer.hidden_size

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Store loader scales in E8M0; uint8 127 encodes 1.0.
        def _e8m0_ones(*shape: int) -> torch.Tensor:
            return torch.full(shape, 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)

        w13_weight_scale = torch.nn.Parameter(
            _e8m0_ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // fp4_block_k,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            _e8m0_ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // fp4_block_k,
            ),
            requires_grad=False,
        )
        w13_weight_scale.format_ue8m0 = False
        w2_weight_scale.format_ue8m0 = False
        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, scale_attrs)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, scale_attrs)

    def process_weights_after_loading(self, layer: Module) -> None:
        from sglang.srt.layers.quantization.marlin_utils import (
            check_moe_marlin_supports_layer,
        )
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            deinterleave_moe_mxfp4_w13_for_marlin,
            prepare_moe_mxfp4_layer_for_marlin,
        )

        # Let the FP8 base method handle ROCm normalization, etc.
        self._fp8.process_weights_after_loading(layer)

        if getattr(layer, "_mega_moe_weights_built", False):
            return

        if not get_platform().is_sm90 and not get_platform().is_sm120:
            raise RuntimeError("MXFP4 Marlin requires SM90 or SM120.")

        if not check_moe_marlin_supports_layer(layer, 32, allow_tile_padding=True):
            raise RuntimeError(
                "Current MXFP4 MoE layer does not satisfy Marlin constraints."
            )

        # NOTE: the Marlin MoE runner consumes w13 in the checkpoint's
        # native ``[w1; w3]`` order -- see ``silu_and_mul`` in
        # fused_marlin_moe.py which expects ``gate = intermediate[:, :N]``
        # (first half) and ``up = intermediate[:, N:]`` (second half).
        # Unlike the flashinfer trtllm_fp4 kernel (which wants [w3, w1]),
        # we must *not* call ``reorder_w1w3_to_w3w1`` here.

        if self._use_w4a8 and _is_late_w4a16_layer(self.prefix):
            self._use_late_w4a16()
        if self._use_w4a8:
            self._prepare_w4a8_weights(layer)
            return
        if self._sm90_decode_w4a8 and self._prepare_sm90_decode_weights(
            layer, w4a8=True
        ):
            return
        self._sm90_decode_w4a8 = False
        if self._sm90_decode_kernel and self._prepare_sm90_decode_weights(layer):
            return
        self._sm90_decode_kernel = False
        if self._use_triton_kernels:
            self._prepare_triton_kernels_weights(layer)
            return

        log_info_on_rank0(
            logger,
            f"Preparing MXFP4 experts for Marlin backend (layer: {self.prefix})...",
        )
        if self.runner.config.gemm1_alpha is not None:
            deinterleave_moe_mxfp4_w13_for_marlin(layer)
        prepare_moe_mxfp4_layer_for_marlin(layer)
        layer._dsv4_mxfp4_backend = "marlin"

    # INF-681 k-pmoe (SGLANG_OPT_DSV4_W4A8_FUSED_SHARED_ADD): the MoE block may hand the
    # shared expert's output to the next apply(), which then returns shared + routed
    # (added in place by the W4A8 top-k sum) instead of the routed output alone.
    _pending_shared_add = None
    _logged_shared_add = False

    def accepts_shared_add(self, experts) -> bool:
        """True when the next apply() can return shared + routed for this layer: the W4A8
        prefill experts with the fused add on, and the routed scaling factor already in
        the top-k weights (so the add has no scale)."""
        w4a8 = getattr(self, "_w4a8", None)
        return (
            self._use_w4a8
            and w4a8 is not None
            and w4a8.fused_shared_add
            and bool(
                getattr(experts, "should_fuse_routed_scaling_factor_in_topk", False)
            )
        )

    def set_pending_shared_add(self, shared_output) -> None:
        if not Mxfp4MarlinMoEMethod._logged_shared_add:
            Mxfp4MarlinMoEMethod._logged_shared_add = True
            log_info_on_rank0(
                logger,
                "SGLANG_OPT_DSV4_W4A8_FUSED_SHARED_ADD: the shared expert output is added "
                f"inside the routed top-k sum (first layer: {self.prefix})",
            )
        self._pending_shared_add = shared_output

    def take_unconsumed_shared_add(self):
        """Clear and return a shared output that no apply() consumed (None if consumed)."""
        pending, self._pending_shared_add = self._pending_shared_add, None
        return pending

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        topk_output = dispatch_output.topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError(f"Unsupported topk output format: {topk_output.format}")
        hidden_states = dispatch_output.hidden_states
        if self._use_w4a8:
            add_to, self._pending_shared_add = self._pending_shared_add, None
            return StandardCombineInput(
                hidden_states=self._w4a8(
                    hidden_states,
                    topk_output.topk_ids,
                    topk_output.topk_weights,
                    add_to=add_to,
                )
            )
        if self._sm90_decode_w4a8:
            from sglang.kernels.ops.moe.dsv4_mxfp4_decode_moe_ws import (
                decode_moe as decode_moe_w4a8,
            )

            return StandardCombineInput(
                hidden_states=decode_moe_w4a8(
                    hidden_states,
                    topk_output.topk_ids,
                    topk_output.topk_weights,
                    self._sm90_packed,
                    self.runner.config.swiglu_limit,
                )
            )
        if self._sm90_decode_kernel:
            from sglang.kernels.ops.moe.dsv4_mxfp4_decode_moe import decode_moe

            return StandardCombineInput(
                hidden_states=decode_moe(
                    hidden_states,
                    topk_output.topk_ids,
                    topk_output.topk_weights,
                    self._sm90_packed,
                    self.runner.config.swiglu_limit,
                )
            )
        if self._use_triton_kernels:
            return StandardCombineInput(
                hidden_states=self._apply_triton_kernels(hidden_states, topk_output)
            )
        target_hidden_size = layer.w13_weight.shape[1] * 16
        if hidden_states.shape[-1] == target_hidden_size:
            hidden_states_padded = hidden_states
        else:
            hidden_states_padded = torch.nn.functional.pad(
                hidden_states,
                (0, target_hidden_size - hidden_states.shape[-1]),
                mode="constant",
                value=0.0,
            )

        quant_info = build_marlin_moe_quant_info(layer)
        runner_output = self.runner.run(
            dispatch_output._replace(hidden_states=hidden_states_padded),
            quant_info=quant_info,
        )

        return StandardCombineInput(hidden_states=runner_output.hidden_states)

    def _use_late_w4a16(self) -> None:
        """Switch this layer from W4A8 to a W4A16 backend (before its weights are prepared)."""
        backend = envs.SGLANG_OPT_DSV4_W4A8_LATE_W4A16_BACKEND.get()
        if backend not in ("triton", "marlin", "sm90", "sm90_w4a8"):
            raise ValueError(
                "SGLANG_OPT_DSV4_W4A8_LATE_W4A16_BACKEND must be 'triton', 'marlin' or "
                f"'sm90' (W4A16), or 'sm90_w4a8', got {backend!r}"
            )
        log_info_on_rank0(
            logger,
            f"Bounded replay late layer: W4A16 {backend} MXFP4 experts instead of W4A8 "
            f"(layer: {self.prefix})",
        )
        self._use_w4a8 = False
        # The decode tiles cover the late layers' few rows per expert, and the triton
        # path falls back to its default tiles above that (up to 64 x 128 rows).
        self._triton_kernels_decode = backend == "triton"
        self._use_triton_kernels = backend == "triton"
        # "sm90": the SGLANG_OPT_DSV4_SM90_DECODE_MOE_KERNEL kernels; a layer shape they do
        # not cover falls back to Marlin in process_weights_after_loading.
        self._sm90_decode_kernel = backend == "sm90"
        # INF-681 k-pmoe "sm90_w4a8": the SGLANG_OPT_DSV4_SM90_DECODE_MOE_W4A8 kernels (W4A8,
        # FP8 activations like the early layers, raw-MXFP4 byte count; same fallback).
        self._sm90_decode_w4a8 = backend == "sm90_w4a8"

    def _prepare_w4a8_weights(self, layer: Module) -> None:
        from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import (
            Mxfp4W4A8PrefillExperts,
        )

        config = self.runner.config
        if not (
            config.activation == "silu"
            and config.is_gated
            and config.gemm1_alpha is None
            and config.swiglu_limit is not None
            and self.hidden_pad == 0
        ):
            raise ValueError(
                "SGLANG_OPT_DSV4_W4A8_PREFILL_MOE supports only the DeepSeek-V4 "
                "clamped SwiGLU experts with an unpadded hidden size."
            )
        log_info_on_rank0(
            logger,
            f"Using W4A8 (FP8 tensor core) MXFP4 experts (layer: {self.prefix}; k-pmoe "
            f"fused_swiglu={envs.SGLANG_OPT_DSV4_W4A8_FUSED_SWIGLU.get()} "
            f"fused_shared_add={envs.SGLANG_OPT_DSV4_W4A8_FUSED_SHARED_ADD.get()} "
            f"counting_route={envs.SGLANG_OPT_DSV4_W4A8_COUNTING_ROUTE.get()} "
            f"w2_two_cta_min_rows={envs.SGLANG_OPT_DSV4_W4A8_W2_TWO_CTA_MIN_ROWS.get()})",
        )
        self._w4a8 = Mxfp4W4A8PrefillExperts(
            layer.w13_weight.data,
            layer.w13_weight_scale_inv.data,
            layer.w2_weight.data,
            layer.w2_weight_scale_inv.data,
            config.swiglu_limit,
            prefetch=envs.SGLANG_DSV4_W4A8_PREFETCH.get(),
            prefetch_chunks=envs.SGLANG_DSV4_W4A8_PREFETCH_CHUNKS.get(),
            sorted_sum=envs.SGLANG_OPT_DSV4_W4A8_W2_SORTED_SUM.get(),
            small_step_rows=envs.SGLANG_OPT_DSV4_W4A8_SMALL_STEP_ROWS.get(),
            fused_swiglu=envs.SGLANG_OPT_DSV4_W4A8_FUSED_SWIGLU.get(),
            fused_shared_add=envs.SGLANG_OPT_DSV4_W4A8_FUSED_SHARED_ADD.get(),
            counting_route=envs.SGLANG_OPT_DSV4_W4A8_COUNTING_ROUTE.get(),
            w2_two_cta_min_rows=envs.SGLANG_OPT_DSV4_W4A8_W2_TWO_CTA_MIN_ROWS.get(),
        )
        layer._dsv4_mxfp4_backend = "w4a8_fp8"

    def _prepare_sm90_decode_weights(self, layer: Module, w4a8: bool = False) -> bool:
        """Repack the experts for the dsv4_mxfp4_decode_moe kernels (W4A16) or the
        dsv4_mxfp4_decode_moe_ws kernels (W4A8). Returns False (weights untouched) when the layer
        is not the DeepSeek-V4 TP4 shape the kernels cover."""
        if w4a8:
            from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as dk
        else:
            from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe as dk

        config = self.runner.config
        hidden = layer.w2_weight.shape[1]
        inter = layer.w2_weight.shape[2] * 2
        expert_map = getattr(layer.dispatcher, "local_expert_mapping", None)
        scale_max = max(
            int(layer.w13_weight_scale_inv.data.view(torch.uint8).max()),
            int(layer.w2_weight_scale_inv.data.view(torch.uint8).max()),
        )
        if not (
            dk.supports(hidden, inter)
            and config.activation == "silu"
            and config.is_gated
            and config.gemm1_alpha is None
            and config.swiglu_limit is not None
            and self.hidden_pad == 0
            and expert_map is None
            and scale_max <= dk.MAX_E8M0
        ):
            log_info_on_rank0(
                logger,
                "SGLANG_OPT_DSV4_SM90_DECODE_MOE_KERNEL: layer shape not covered "
                f"(hidden {hidden}, intermediate {inter}, EP {expert_map is not None}); "
                f"keeping the configured path (layer: {self.prefix})",
            )
            return False
        log_info_on_rank0(
            logger,
            f"Using the SM90 decode MoE kernel ({'W4A8 warp-specialized wgmma' if w4a8 else 'W4A16'}) "
            f"for MXFP4 experts (layer: {self.prefix})",
        )
        torch.cuda.empty_cache()
        self._sm90_packed = dk.pack_experts(
            layer.w13_weight.data,
            layer.w13_weight_scale_inv.data,
            layer.w2_weight.data,
            layer.w2_weight_scale_inv.data,
        )
        del layer.w13_weight, layer.w2_weight
        del layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
        torch.cuda.empty_cache()
        layer._dsv4_mxfp4_backend = "sm90_decode_w4a8" if w4a8 else "sm90_decode"
        return True

    def _prepare_triton_kernels_weights(self, layer: Module) -> None:
        import triton_kernels.matmul_details.opt_flags as opt_flags
        from triton_kernels.matmul import FlexCtx, PrecisionConfig

        from sglang.srt.layers.quantization.mxfp4 import (
            _mx_block_kwargs,
            _swizzle_mxfp4,
        )

        config = self.runner.config
        if not (
            config.activation == "silu"
            and config.is_gated
            and config.gemm1_alpha is None
            and config.swiglu_limit is not None
            and self.hidden_pad == 0
        ):
            raise ValueError(
                "SGLANG_OPT_DSV4_TRITON_KERNELS_MOE supports only the DeepSeek-V4 "
                "clamped SwiGLU experts with an unpadded hidden size."
            )
        log_info_on_rank0(
            logger,
            f"Preparing MXFP4 experts for triton_kernels (layer: {self.prefix})...",
        )
        mx_block = _mx_block_kwargs()
        if self._triton_kernels_decode:
            # The swizzled scale layout fixes the warp count of every later
            # matmul; 4 warps lets decode take 128-wide tiles (see _DECODE_TILES).
            swizzle_num_warps = 4
        else:
            swizzle_num_warps = 8
            # Measured on H200 at the DSV4-Flash TP4 expert shapes: 3 stages beat
            # the default by ~16% at 16k-32k rows with bit-identical output.
            opt_flags.update_opt_flags_constraints({"num_stages": 3})
        self._tk_num_experts = layer.w13_weight.shape[0]
        self._tk_weights = []
        for weight, scale in (
            (layer.w13_weight, layer.w13_weight_scale_inv),
            (layer.w2_weight, layer.w2_weight_scale_inv),
        ):
            # Return the previous swizzle's cached scratch before the next swizzle, and
            # this swizzle's scratch right after it. Otherwise a swizzle's output can be
            # placed inside a cached scratch segment: the rest of that segment can then
            # never be returned to the driver, and KV-cache sizing (which empties the
            # cache and reads driver free memory) loses ~0.5 GB per TP4 layer on H200.
            torch.cuda.empty_cache()
            tk_weight, flex, tk_scale = _swizzle_mxfp4(
                weight.data.view(torch.uint8),
                scale.data.view(torch.uint8),
                swizzle_num_warps,
            )
            torch.cuda.empty_cache()
            precision = PrecisionConfig(
                b_mx_scale=tk_scale, flex_ctx=FlexCtx(rhs_data=flex), **mx_block
            )
            self._tk_weights.append((tk_weight, precision))
        self._tk_intermediate_size = layer.w2_weight.shape[-1] * 2
        del layer.w13_weight, layer.w2_weight
        del layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
        # Return the original MXFP4 weights' segments too, so later allocations do not
        # mix into them.
        torch.cuda.empty_cache()
        layer._dsv4_mxfp4_backend = "triton_kernels"

    # H200, DSV4-Flash TP4 experts (hidden 5120, intermediate 576), 16 routed rows
    # per tile: best of a block/stage sweep at 48-128 decode requests x 6 verify
    # rows (target, 384 experts top 6) and x 5 draft rows (128 experts top 3).
    # Measured at TP4 only; other TP shard widths use the same tiles unmeasured.
    _DECODE_TILES = (
        dict(block_m=16, block_n=128, block_k=128, num_warps=4, num_stages=4),
        dict(block_m=16, block_n=256, block_k=64, num_warps=4, num_stages=4),
    )
    # Above this many routed rows per expert on average, leave the tile choice to
    # triton_kernels (only the warp count is fixed by the swizzled scales).
    _DECODE_MAX_ROWS_PER_EXPERT = 16

    def _uses_decode_tiles(self, num_slots: int) -> bool:
        """Whether this call takes the decode routing and tiles: the decode switch is
        on and the routed rows per expert stay small on average."""
        return (
            self._triton_kernels_decode
            and num_slots <= self._DECODE_MAX_ROWS_PER_EXPERT * self._tk_num_experts
        )

    # INF-681 SGLANG_OPT_DSV4_DECODE_MOE_ROWS_AWARE_TILES: (max routed rows per expert on
    # average, tiles). Measured on H200 at the TP4 shapes with DSpark verify routing (6 rows
    # per request, correlated) at bs 32-192 and with independent rows at 768-8192 rows. The
    # first limit is 10 under CUDA graph capture (decode: a request's verify rows share
    # experts, so hot experts pass 16 rows early) and 16 in eager mode (independent prefill
    # rows: 16-row tiles stay ~6% faster at 12 rows per expert).
    _DECODE_TILES_32 = (
        dict(block_m=32, block_n=128, block_k=128, num_warps=4, num_stages=4),
        dict(block_m=32, block_n=256, block_k=64, num_warps=4, num_stages=4),
    )
    _DECODE_TILES_64 = (
        dict(block_m=64, block_n=128, block_k=128, num_warps=4, num_stages=3),
        dict(block_m=64, block_n=256, block_k=64, num_warps=4, num_stages=3),
    )
    _ROWS_AWARE_TILES = (
        (10, _DECODE_TILES),
        (36, _DECODE_TILES_32),
        (160, _DECODE_TILES_64),
    )

    def _decode_tiles(self, decode_tiles: bool, num_slots: int = 0):
        if not self._triton_kernels_decode:
            return {}, {}
        if self._rows_aware_tiles:
            rows_per_expert = num_slots / self._tk_num_experts
            first = 10 if torch.cuda.is_current_stream_capturing() else 16
            for i, (limit, tiles) in enumerate(self._ROWS_AWARE_TILES):
                if rows_per_expert <= (first if i == 0 else limit):
                    return tiles
        if not decode_tiles:
            return {"num_warps": 4}, {"num_warps": 4}
        return self._DECODE_TILES

    def _apply_triton_kernels(self, hidden_states, topk_output) -> torch.Tensor:
        from triton_kernels.matmul import matmul, scoped_opt_flags_constraints
        from triton_kernels.tensor import make_ragged_tensor_metadata

        from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            swiglu_limit_func,
        )

        if torch.cuda.is_current_stream_capturing() and not self._triton_kernels_decode:
            raise RuntimeError(
                "SGLANG_OPT_DSV4_TRITON_KERNELS_MOE is prefill-only and does not support "
                "CUDA graphs; enable it only where the MoE runs eagerly (a PD prefill "
                "server without a prefill CUDA graph), or use "
                "SGLANG_OPT_DSV4_TRITON_KERNELS_MOE_DECODE for CUDA-graph decode."
            )
        num_tokens, top_k = topk_output.topk_ids.shape
        if envs.SGLANG_OPT_DSV4_DP_SYMM_MOE_OUTPUT.get():
            # INF-681: the DP-attention reduce-scatter reads this buffer; from the
            # NCCL symmetric pool (with --enable-symm-mem) NCCL takes its symmetric
            # kernel. A no-op context without symmetric memory or MAX_LEN padding.
            from sglang.srt.distributed.device_communicators.pynccl_allocator import (
                use_symmetric_memory,
            )
            from sglang.srt.layers.dp_attention import is_allocation_symmetric
            from sglang.srt.runtime_context import get_parallel

            with use_symmetric_memory(
                get_parallel().tp_group, disabled=not is_allocation_symmetric()
            ):
                output = torch.empty_like(hidden_states)
        else:
            output = torch.empty_like(hidden_states)
        if num_tokens == 0:
            return output
        num_slots = num_tokens * top_k
        decode_tiles = self._uses_decode_tiles(num_slots)
        if decode_tiles:
            from sglang.kernels.ops.moe.expert_sort_small import expert_sort_small

            # One launch instead of argsort + histogram + gathers (~45 us at decode).
            routing = ExpertRouting(
                *expert_sort_small(
                    topk_output.topk_ids,
                    topk_output.topk_weights,
                    self._tk_num_experts,
                )
            )
        else:
            routing = route_slots_by_expert(
                topk_output.topk_ids, topk_output.topk_weights, self._tk_num_experts
            )
        ragged_metadata = make_ragged_tensor_metadata(routing.expert_counts, num_slots)
        (w13, w13_precision), (w2, w2_precision) = self._tk_weights
        w13_tiles, w2_tiles = self._decode_tiles(decode_tiles, num_slots)
        with scoped_opt_flags_constraints(w13_tiles):
            gate_up = matmul(
                hidden_states,
                w13,
                None,
                a_ragged_metadata=ragged_metadata,
                gather_indx=routing.gather_indx,
                precision_config=w13_precision,
            )
        activated = gate_up.new_empty(gate_up.shape[0], self._tk_intermediate_size)
        swiglu_limit_func(
            activated, gate_up, swiglu_limit=self.runner.config.swiglu_limit
        )
        with scoped_opt_flags_constraints(w2_tiles):
            down = matmul(
                activated,
                w2,
                None,
                a_ragged_metadata=ragged_metadata,
                scatter_indx=routing.combine_indx,
                precision_config=w2_precision,
                gammas=routing.sorted_weights,
            )
        moe_topk_sum(down.view(num_tokens, top_k, -1), output)
        return output
