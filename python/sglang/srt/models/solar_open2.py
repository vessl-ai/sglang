# SPDX-License-Identifier: Apache-2.0
"""Inference-only SolarOpen2 (Upstage) model for SGLang.

Topology (48 layers, all MoE):
  * ``config.gqa_layers`` (0-based, 12 of 48) -> softmax GQA + NoPE + output gate
  * every other layer (36)                   -> KDA linear attention, reused
    verbatim from ``srt/models/kimi_linear.py``'s ``KimiDeltaAttention``

Reference implementation: the Upstage vLLM fork
(``vllm/model_executor/models/solar_open2.py``, commit 00907fc9).
Skeleton: ``srt/models/kimi_linear.py`` (sglang v0.5.17), whose full-attention
half is MLA and is replaced here by RadixAttention GQA.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn

from sglang.srt.configs.solar_open2 import SolarOpen2Config
from sglang.srt.distributed import get_pp_group, tensor_model_parallel_all_reduce
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.kimi_linear import (
    KimiDeltaAttention,
    _materialize_residual_stream,
)
from sglang.srt.models.llama import LlamaMLP as SolarOpen2MLP
from sglang.srt.models.transformers import maybe_prefix
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import make_layers
from sglang.srt.utils.common import BumpAllocator, add_prefix

logger = logging.getLogger(__name__)


class SolarOpen2Attention(nn.Module):
    """Softmax GQA with NoPE and a per-head sigmoid output gate.

    NoPE is implemented by *not building* a rotary embedding at all - SGLang's
    RadixAttention never applies rope itself, the model does. ``positions`` is
    accepted for signature parity and deliberately unused.
    """

    def __init__(
        self,
        config: SolarOpen2Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if getattr(config, "use_rope", False):
            raise NotImplementedError(
                "SolarOpen2: use_rope=True is not wired in the SGLang port "
                "(the released checkpoint is NoPE)."
            )
        if getattr(config, "use_qk_norm", False):
            raise NotImplementedError(
                "SolarOpen2: use_qk_norm=True is not wired in the SGLang port."
            )

        attn_tp_size = get_parallel().attn_tp_size
        attn_tp_rank = get_parallel().attn_tp_rank

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size

        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= attn_tp_size:
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)

        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("o_proj", prefix),
        )

        # Output gate. bf16 on disk (listed in the checkpoint's quant ignore),
        # so it is built unquantized. Dropping it boots and produces plausible
        # text - the loss is silent, hence the explicit load-count gate below.
        self.use_gqa_gate = bool(getattr(config, "use_gqa_gate", True))
        if self.use_gqa_gate:
            self.g_proj = ColumnParallelLinear(
                config.hidden_size,
                self.total_num_heads * self.head_dim,
                bias=bool(getattr(config, "use_gqa_gate_bias", False)),
                quant_config=None,
                prefix=add_prefix("g_proj", prefix),
            )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,  # unused: NoPE
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v, forward_batch)
        if self.use_gqa_gate:
            gate, _ = self.g_proj(hidden_states)
            attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)[0]


class SolarOpen2MoE(nn.Module):
    """320 routed experts (int4/fp8 CUTLASS W4AFP8) + 1 bf16 shared expert.

    Mirrors ``KimiMoE`` (srt/models/kimi_linear.py:70-182) but keeps the module
    attribute named ``mlp`` only - Kimi aliases ``block_sparse_moe``/``mlp`` to
    the same module, and ``named_parameters()`` would then report the
    ``block_sparse_moe.*`` names, which do not exist in the Solar checkpoint.
    """

    def __init__(
        self,
        config: SolarOpen2Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_parallel().tp_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_shared_experts = config.num_shared_experts

        # Router stays unquantized (checkpoint ignores ``mlp.gate$``).
        # SOLAR_GATE_FP32=1 reproduces the Upstage vLLM fork, which runs the
        # router in float32 (nn.Linear(dtype=float32) + float32
        # e_score_correction_bias + router_logits_dtype=float32). Same shape as
        # glm4_moe's "GLM requires FP32 gate projection". Default 0 keeps the
        # KimiMoE convention (bf16) this port started from, so the two are an
        # OFAT pair.
        self.gate_fp32 = os.environ.get("SOLAR_GATE_FP32", "0") == "1"
        gate_dtype = torch.float32 if self.gate_fp32 else None
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            params_dtype=gate_dtype,
            prefix=add_prefix("gate", prefix),
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(
                config.num_experts,
                dtype=torch.float32 if self.gate_fp32 else torch.get_default_dtype(),
            )
        )
        if layer_id == 0:
            logger.info(
                "[SOLAR-GATE] MoE router dtype: gate_fp32=%s (weight=%s bias=%s)",
                self.gate_fp32,
                self.gate.weight.dtype,
                self.gate.e_score_correction_bias.dtype,
            )

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_token,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            activation=config.hidden_act,
            prefix=add_prefix("experts", prefix),
        )

        # Same call shape as KimiMoE: sigmoid routing is implied by
        # correction_bias (noaux_tc path), matching the fork's
        # scoring_func="sigmoid" + e_score_correction_bias.
        self.topk = TopK(
            top_k=config.num_experts_per_token,
            renormalize=config.moe_renormalize,
            use_grouped_topk=True,
            num_expert_group=config.num_expert_group,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
            output_format=TopKOutputFormat.STANDARD if quant_config is None else None,
        )

        if self.num_shared_experts:
            self.shared_experts = SolarOpen2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size
                * self.num_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
            )
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        shared_output = None
        if self.shared_experts is not None and hidden_states.shape[0] > 0:
            shared_output = self.shared_experts(hidden_states)

        gate_in = hidden_states.to(torch.float32) if self.gate_fp32 else hidden_states
        router_logits, _ = self.gate(gate_in)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = self.experts(hidden_states, topk_output)

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_size)


class SolarOpen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: SolarOpen2Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_kda = config.is_kda_layer(layer_id)

        if self.is_kda:
            # Reused verbatim. quant_config is non-None here, which keeps
            # KimiDeltaAttention on its *unfused* q/k/v/b/f/g path; the
            # projections themselves resolve to unquantized because the config
            # class widened the compressed-tensors ignore list for KDA layers.
            self.self_attn = KimiDeltaAttention(
                layer_idx=layer_id,
                hidden_size=config.hidden_size,
                config=config,
                quant_config=quant_config,
                rms_norm_eps=config.rms_norm_eps,
                prefix=add_prefix("self_attn", prefix),
            )
        else:
            self.self_attn = SolarOpen2Attention(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
            )

        # first_k_dense_replace == 0 in the release config: every layer is MoE.
        if layer_id >= config.first_k_dense_replace:
            self.mlp = SolarOpen2MoE(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = SolarOpen2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class SolarOpen2Model(nn.Module):
    def __init__(
        self,
        config: SolarOpen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()
        self.dspark_layers_to_capture: Optional[list[int]] = None

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: SolarOpen2DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )

        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        world_size = get_parallel().tp_size
        assert (
            config.num_attention_heads % world_size == 0
        ), "num_attention_heads must be divisible by world_size"

        # Risk #1 gate, authoritative instance: this is the config the layers
        # were actually built from (HF also constructs throwaway defaults).
        built_full = [
            i
            for i in range(self.start_layer, self.end_layer)
            if not self.layers[i].is_kda
        ]
        logger.info(
            "[SOLAR-GATE] layers built: full_attention=%s (n=%d) kda_count=%d",
            built_full,
            len(built_full),
            (self.end_layer - self.start_layer) - len(built_full),
        )
        expected_full = [
            i
            for i in config.full_attention_layer_ids
            if self.start_layer <= i < self.end_layer
        ]
        if built_full != expected_full:
            raise ValueError(
                "SolarOpen2 layer split mismatch: built full-attention layers "
                f"{built_full} != config.full_attention_layer_ids {expected_full}"
            )

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if self.pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        total_num_layers = self.end_layer - self.start_layer
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2,
            dtype=torch.float32,
            device=hidden_states.device,
        )

        aux_hidden_states = []
        for i in range(self.start_layer, self.end_layer):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                hidden_states, residual = self.layers[i](
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                    residual=residual,
                    zero_allocator=zero_allocator,
                )
            if (
                self.dspark_layers_to_capture is not None
                and i in self.dspark_layers_to_capture
            ):
                aux_hidden_states.append(
                    _materialize_residual_stream(hidden_states, residual)
                )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        if self.dspark_layers_to_capture is not None:
            return hidden_states, aux_hidden_states
        return hidden_states


class SolarOpen2ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "qkv_conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
    }

    def __init__(
        self,
        config: SolarOpen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = SolarOpen2Model(
            config, quant_config, prefix=maybe_prefix(prefix, "model")
        )
        self.start_layer = self.model.start_layer
        self.end_layer = self.model.end_layer
        self.pp_group = get_pp_group()
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config=config)
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_dspark_layers_to_capture(self, layer_ids: list[int]) -> None:
        if self.pp_group.world_size > 1:
            raise NotImplementedError("DSPARK aux hidden capture requires PP=1.")
        if not self.pp_group.is_last_rank:
            return
        if layer_ids is None:
            raise ValueError(
                "DSPARK requires explicit layer_ids for aux hidden capture."
            )
        full_attn = set(self.config.full_attention_layer_ids)
        offenders = [i for i in layer_ids if i not in full_attn]
        # Risk #2 in the porting delta: an aux tap on a KDA layer boots fine and
        # only shows up as a dead accept rate.
        logger.info(
            "[SOLAR-GATE] dspark aux tap layer_ids=%s all_full_attention=%s",
            list(layer_ids),
            not offenders,
        )
        if offenders:
            raise ValueError(
                "SolarOpen2 DSPARK aux hidden capture must tap full-attention "
                f"(GQA) layers only; got non-GQA layer ids {offenders}. "
                f"full_attention_layer_ids={sorted(full_attn)}"
            )
        self.capture_aux_hidden_states = True
        self.model.dspark_layers_to_capture = list(layer_ids)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            inputs_embeds,
            pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank:
            aux_hidden_states = None
            if self.capture_aux_hidden_states:
                hidden_states, aux_hidden_states = hidden_states
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                aux_hidden_states,
            )
        return hidden_states

    def _is_non_local_pp_weight(self, name: str) -> bool:
        if self.pp_group.world_size == 1:
            return False
        layer_id = get_layer_id(name)
        if layer_id is not None:
            return not (self.model.start_layer <= layer_id < self.model.end_layer)
        if name.startswith("model.embed_tokens."):
            return not self.pp_group.is_first_rank
        if name.startswith(("model.norm.", "lm_head.")):
            return not self.pp_group.is_last_rank
        return False

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            (".qkv_conv1d", ".q_conv1d", 0),
            (".qkv_conv1d", ".k_conv1d", 1),
            (".qkv_conv1d", ".v_conv1d", 2),
        ]
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for args in weights:
            name, loaded_weight = args[:2]
            kwargs = args[2] if len(args) > 2 else {}
            if self._is_non_local_pp_weight(name):
                continue
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Experts are handled below; skip before the name is rewritten.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped.endswith(".bias") and mapped not in params_dict:
                    continue
                if mapped not in params_dict:
                    continue
                name = mapped
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for param_name, weight_name, expert_id, shard_id in (
                    expert_params_mapping
                ):
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        name,
                        expert_id=expert_id,
                        shard_id=shard_id,
                    )
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                    if name not in params_dict:
                        logger.warning(
                            "[SOLAR-GATE] unexpected checkpoint tensor with no "
                            "matching parameter: %s",
                            name,
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, **kwargs)
            loaded_params.add(name)

        self._log_load_gates(params_dict, loaded_params)

    def _log_load_gates(self, params_dict, loaded_params: set[str]) -> None:
        """Boot-time gates for the three silent-failure modes of this port."""
        # Risk #3: a missing output gate boots and reads fine.
        g_proj_loaded = sorted(
            n for n in loaded_params if n.endswith(".self_attn.g_proj.weight")
        )
        expected = len(self.config.full_attention_layer_ids)
        logger.info(
            "[SOLAR-GATE] g_proj weights loaded=%d expected=%d ok=%s",
            len(g_proj_loaded),
            expected,
            len(g_proj_loaded) == expected,
        )
        if len(g_proj_loaded) != expected:
            raise ValueError(
                f"SolarOpen2: loaded {len(g_proj_loaded)} self_attn.g_proj weights, "
                f"expected {expected} (one per GQA layer)."
            )

        # Risk #4: MoE sharded on TP instead of EP silently floors the number of
        # int4 scale groups (moe_intermediate 1280 / 128 == 10; TP4 would give
        # 320 // 128 == 2).
        try:
            experts = self.model.layers[self.model.start_layer].mlp.experts
            scale = getattr(experts, "w2_weight_scale", None)
            if scale is not None:
                expected_groups = self.config.moe_intermediate_size // 128
                logger.info(
                    "[SOLAR-GATE] w2_weight_scale.shape=%s last_dim=%d expected=%d ok=%s",
                    tuple(scale.shape),
                    scale.shape[-1],
                    expected_groups,
                    scale.shape[-1] == expected_groups,
                )
                if scale.shape[-1] != expected_groups:
                    raise ValueError(
                        "SolarOpen2 MoE int4 scale groups mismatch: "
                        f"w2_weight_scale.shape={tuple(scale.shape)} "
                        f"(last dim {scale.shape[-1]}, expected {expected_groups}). "
                        "Shard the MoE with expert parallelism (--ep-size == --tp), "
                        "not tensor parallelism."
                    )
            else:
                logger.info(
                    "[SOLAR-GATE] w2_weight_scale absent on %s (unquantized MoE?)",
                    type(experts).__name__,
                )
        except (AttributeError, IndexError) as e:  # pragma: no cover
            logger.warning("[SOLAR-GATE] could not inspect MoE scales: %s", e)


EntryClass = SolarOpen2ForCausalLM
