from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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


class Mxfp4MarlinMoEMethod:
    """MXFP4 (E8M0 scales) MoE quantization method using the Marlin backend."""

    fuse_routed_scaling_factor_in_topk = True

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix
        self._use_triton_kernels = (
            envs.SGLANG_OPT_DSV4_TRITON_KERNELS_MOE.get() and get_platform().is_sm90
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
        # Measured on H200 at the DSV4-Flash TP4 expert shapes: 3 stages beat
        # the default by ~16% at 16k-32k rows with bit-identical output.
        opt_flags.update_opt_flags_constraints({"num_stages": 3})
        self._tk_num_experts = layer.w13_weight.shape[0]
        self._tk_weights = []
        for weight, scale in (
            (layer.w13_weight, layer.w13_weight_scale_inv),
            (layer.w2_weight, layer.w2_weight_scale_inv),
        ):
            tk_weight, flex, tk_scale = _swizzle_mxfp4(
                weight.data.view(torch.uint8), scale.data.view(torch.uint8), 8
            )
            precision = PrecisionConfig(
                b_mx_scale=tk_scale, flex_ctx=FlexCtx(rhs_data=flex), **mx_block
            )
            self._tk_weights.append((tk_weight, precision))
        self._tk_intermediate_size = layer.w2_weight.shape[-1] * 2
        del layer.w13_weight, layer.w2_weight
        del layer.w13_weight_scale_inv, layer.w2_weight_scale_inv
        # The swizzle's scratch stays reserved otherwise, and KV-cache sizing
        # counts reserved memory as used.
        torch.cuda.empty_cache()
        layer._dsv4_mxfp4_backend = "triton_kernels"

    def _apply_triton_kernels(self, hidden_states, topk_output) -> torch.Tensor:
        from triton_kernels.matmul import matmul
        from triton_kernels.tensor import make_ragged_tensor_metadata

        from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            swiglu_limit_func,
        )

        num_tokens, top_k = topk_output.topk_ids.shape
        output = torch.empty_like(hidden_states)
        if num_tokens == 0:
            return output
        # Masked slots (id -1) contribute nothing; route them to expert 0 with weight 0.
        valid = topk_output.topk_ids >= 0
        expert_ids = torch.where(valid, topk_output.topk_ids, 0).flatten().long()
        weights = torch.where(valid, topk_output.topk_weights, 0).flatten()
        # Slots sorted by expert; triton_kernels' own topk needs a power-of-2 top_k.
        combine_indx = torch.argsort(expert_ids, stable=True).to(torch.int32)
        # bincount sizes its output from the max id, a device-to-host sync per
        # layer that leaves the GPU idle while the host relaunches; the counts
        # are the same from a fixed-size scatter_add.
        counts = torch.zeros(
            self._tk_num_experts, dtype=torch.int32, device=expert_ids.device
        )
        counts.scatter_add_(
            0, expert_ids, torch.ones_like(expert_ids, dtype=torch.int32)
        )
        ragged_metadata = make_ragged_tensor_metadata(counts, expert_ids.numel())
        (w13, w13_precision), (w2, w2_precision) = self._tk_weights
        gate_up = matmul(
            hidden_states,
            w13,
            None,
            a_ragged_metadata=ragged_metadata,
            gather_indx=combine_indx // top_k,
            precision_config=w13_precision,
        )
        activated = gate_up.new_empty(gate_up.shape[0], self._tk_intermediate_size)
        swiglu_limit_func(
            activated, gate_up, swiglu_limit=self.runner.config.swiglu_limit
        )
        down = matmul(
            activated,
            w2,
            None,
            a_ragged_metadata=ragged_metadata,
            scatter_indx=combine_indx,
            precision_config=w2_precision,
            gammas=weights[combine_indx.long()],
        )
        moe_topk_sum(down.view(num_tokens, top_k, -1), output)
        return output
