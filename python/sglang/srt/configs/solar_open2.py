# SPDX-License-Identifier: Apache-2.0
"""SolarOpen2 (Upstage) configuration for SGLang.

Hybrid architecture: ``gqa_layers`` (0-based) are softmax GQA (full attention);
every other layer is KDA (Kimi Gated DeltaNet linear attention).

This subclasses ``KimiLinearConfig`` on purpose: the KV-cache configurator, the
mamba pool builder and the DSPARK/ReplaySSM paths all gate on
``kimi_linear_config(...) is not None``, which is an ``isinstance`` check
against ``KimiLinearConfig`` (sglang v0.5.17:
``srt/configs/hybrid_arch.py:101`` consumed at ``mem_cache/kv_cache_configurator.py``
:730/:771/:809/:1851/:1856 and ``mem_cache/kv_cache_builder.py:163``).
Registering only through ``linear_attn_model_registry`` would leave those paths
silently off.

Two conventions differ from KimiLinear and are bridged here:
  * Kimi's ``is_kda_layer`` uses ``(layer_idx + 1) in kda_layers`` (1-based);
    Solar ships ``gqa_layers`` 0-based. We synthesise a 1-based
    ``linear_attn_config['kda_layers']`` *and* override ``is_kda_layer`` so both
    conventions agree.
  * The checkpoint's compressed-tensors ``ignore`` list does not mention the KDA
    layers' ``q/k/v/o_proj`` (they are plain bf16 on disk - no ``weight_scale``
    tensor exists for them). ``find_matched_target`` raises on an unmatched
    layer, so we append an explicit ignore regex for them.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sglang.srt.configs.kimi_linear import KimiLinearConfig

logger = logging.getLogger(__name__)

# KDA-layer projections that live unquantized on disk but would otherwise fall
# through compressed-tensors' target matching and raise.
_KDA_UNQUANTIZED_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")


class SolarOpen2Config(KimiLinearConfig):
    model_type = "solar_open2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 196608,
        hidden_size: int = 4096,
        intermediate_size: int = 10240,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 64,
        head_dim: int = 128,
        num_key_value_heads: int = 8,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1_048_576,
        rms_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        moe_intermediate_size: int = 1280,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 1,
        n_routed_experts: int = 320,
        routed_scaling_factor: float = 1.0,
        n_group: Optional[int] = 1,
        topk_group: Optional[int] = 1,
        first_k_dense_replace: int = 0,
        norm_topk_prob: bool = True,
        attention_bias: bool = False,
        use_qk_norm: bool = False,
        use_rope: bool = False,
        gqa_interval: int = 4,
        gqa_layers: Optional[List[int]] = None,
        use_gqa_gate: bool = True,
        use_gqa_gate_bias: bool = False,
        linear_attn_config: Optional[dict] = None,
        kda_use_full_proj: bool = False,
        kda_gate_lower_bound: float = -5.0,
        kda_allow_neg_eigval: bool = True,
        layer_types: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        # HF's ``from_dict`` replays the whole config.json as kwargs, so every
        # key we pass to ``super().__init__`` explicitly must be removed here or
        # it arrives twice ("got multiple values for keyword argument").
        for _dup in (
            "model_type",
            "num_experts",
            "num_experts_per_token",
            "num_shared_experts",
            "moe_renormalize",
            "moe_router_activation_func",
            "moe_layer_freq",
            "use_grouped_topk",
            "num_expert_group",
            "topk_group",
            "topk_method",
        ):
            kwargs.pop(_dup, None)

        if gqa_interval <= 0:
            raise ValueError("gqa_interval must be greater than zero")

        # ---- resolve which layers are full attention (0-based) --------------
        # HF instantiates the class with no kwargs in a few places (diff-dict
        # defaults); those throwaway instances fall through to the interval
        # fallback, so the gate log records whether the split was explicit.
        explicit_split = gqa_layers is not None or layer_types is not None
        if gqa_layers is not None:
            gqa_layers = [int(x) for x in gqa_layers]
            if len(set(gqa_layers)) != len(gqa_layers) or any(
                i < 0 or i >= num_hidden_layers for i in gqa_layers
            ):
                raise ValueError(
                    "gqa_layers must contain unique layer indices within the model, "
                    f"got {gqa_layers!r} for num_hidden_layers={num_hidden_layers}"
                )
        elif layer_types is not None:
            gqa_layers = [
                i for i, t in enumerate(layer_types) if t == "full_attention"
            ]
        else:
            # gqa_layers takes priority over gqa_interval; only used as fallback.
            gqa_layers = [
                i for i in range(num_hidden_layers) if (i + 1) % gqa_interval == 0
            ]
        self._solar_gqa_layers = sorted(gqa_layers)
        kda_layer_ids_0b = [
            i for i in range(num_hidden_layers) if i not in set(self._solar_gqa_layers)
        ]

        # ---- bridge to KimiLinearConfig's linear_attn_config (1-BASED) ------
        base_linear = dict(linear_attn_config or {})
        base_linear.setdefault("short_conv_kernel_size", 4)
        base_linear.setdefault("head_dim", head_dim)
        base_linear.setdefault("num_heads", num_attention_heads)
        base_linear.setdefault("num_kv_heads", None)
        base_linear["kda_layers"] = [i + 1 for i in kda_layer_ids_0b]
        base_linear["full_attn_layers"] = [i + 1 for i in self._solar_gqa_layers]

        # ---- widen the compressed-tensors ignore list -----------------------
        self._augment_quantization_ignore(kwargs, kda_layer_ids_0b)

        super().__init__(
            model_type="solar_open2",
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            tie_word_embeddings=tie_word_embeddings,
            max_position_embeddings=max_position_embeddings,
            # MoE (Solar spelling -> Kimi spelling)
            moe_intermediate_size=moe_intermediate_size,
            moe_renormalize=norm_topk_prob,
            moe_router_activation_func="sigmoid",
            num_experts=n_routed_experts,
            num_experts_per_token=num_experts_per_tok,
            num_shared_experts=n_shared_experts,
            routed_scaling_factor=routed_scaling_factor,
            first_k_dense_replace=first_k_dense_replace,
            moe_layer_freq=1,
            use_grouped_topk=True,
            num_expert_group=1 if n_group is None else int(n_group),
            topk_group=1 if topk_group is None else int(topk_group),
            topk_method="noaux_tc",
            # MLA fields stay None -> KimiLinearConfig.is_mla == False.
            linear_attn_config=base_linear,
            **kwargs,
        )

        # Solar-specific knobs consumed by srt/models/solar_open2.py
        self.attention_bias = attention_bias
        self.use_qk_norm = use_qk_norm
        self.use_rope = use_rope
        self.gqa_interval = gqa_interval
        self.gqa_layers = self._solar_gqa_layers
        self.use_gqa_gate = use_gqa_gate
        self.use_gqa_gate_bias = use_gqa_gate_bias
        self.kda_use_full_proj = kda_use_full_proj
        self.kda_gate_lower_bound = kda_gate_lower_bound
        self.kda_allow_neg_eigval = kda_allow_neg_eigval
        self.layer_types = [
            "full_attention" if i in set(self._solar_gqa_layers) else "linear_attention"
            for i in range(num_hidden_layers)
        ]

        if kda_use_full_proj:
            raise NotImplementedError(
                "SolarOpen2: kda_use_full_proj=True (single f_proj/g_proj instead of "
                "the factored f_a/f_b + g_a/g_b pair) is not wired in the SGLang port."
            )

        # [SOLAR-GATE] boot-log the layer split - risk #1 in the porting delta
        # (Kimi is 1-based, Solar is 0-based; a shifted split keeps every tensor
        # shape valid and only shows up as quality loss).
        logger.info(
            "[SOLAR-GATE] explicit_split=%s full_attention_layer_ids=%s (n=%d) "
            "n_kda=%d",
            explicit_split,
            self.full_attention_layer_ids,
            len(self.full_attention_layer_ids),
            len(kda_layer_ids_0b),
        )

    @staticmethod
    def _augment_quantization_ignore(
        kwargs: Dict[str, Any], kda_layer_ids_0b: List[int]
    ) -> None:
        """Add the KDA layers' q/k/v/o_proj to compressed-tensors' ignore list.

        They carry no ``weight_scale`` on disk (bf16), and
        ``compressed_tensors/utils.py find_matched_target`` raises for a layer
        that matches neither a target nor the ignore list.
        """
        qc = kwargs.get("quantization_config")
        if not isinstance(qc, dict):
            return
        if qc.get("quant_method") != "compressed-tensors":
            return
        ignore = list(qc.get("ignore") or [])
        layer_alt = "|".join(str(i) for i in kda_layer_ids_0b)
        proj_alt = "|".join(_KDA_UNQUANTIZED_PROJ)
        rule = rf"re:model\.layers\.({layer_alt})\.self_attn\.({proj_alt})"
        if rule not in ignore:
            ignore.append(rule)
        qc["ignore"] = ignore
        kwargs["quantization_config"] = qc
        logger.info(
            "[SOLAR-GATE] compressed-tensors ignore widened with %d KDA layers "
            "x %s", len(kda_layer_ids_0b), list(_KDA_UNQUANTIZED_PROJ)
        )

    # ---- convention overrides ------------------------------------------------
    def is_kda_layer(self, layer_idx: int) -> bool:
        """0-based membership test (KimiLinearConfig's base impl is 1-based)."""
        return layer_idx not in set(self._solar_gqa_layers)

    @property
    def is_linear_attn(self) -> bool:
        return True


__all__ = ["SolarOpen2Config"]


# ---------------------------------------------------------------------------
# Register with the hybrid linear-attention model registry. Imported eagerly
# from srt/utils/hf_transformers/common.py (see solar_patch.py), so this runs in
# every process that loads a config - including spawned TP workers.
# ---------------------------------------------------------------------------
def _register() -> None:
    from sglang.srt.configs.linear_attn_model_registry import (
        LinearAttnModelSpec,
        get_linear_attn_spec_by_arch,
        register_linear_attn_model,
    )

    if get_linear_attn_spec_by_arch("SolarOpen2ForCausalLM") is not None:
        return
    register_linear_attn_model(
        LinearAttnModelSpec(
            config_class=SolarOpen2Config,
            backend_class_name=(
                "sglang.srt.layers.attention.linear.kda_backend.KDAAttnBackend"
            ),
            arch_names=["SolarOpen2ForCausalLM"],
            uses_mamba_radix_cache=True,
            support_mamba_cache=True,
            # Consulted by
            # arg_groups.overrides.supports_mamba_cache_extra_buffer, which is
            # what resolves the mamba radix cache strategy. Solar-Open2 reuses
            # KimiDeltaAttention / KDAAttnBackend and so performs the same
            # track-snapshot writes KimiLinearForCausalLM does. With this
            # False the strategy resolves to no_buffer, which force-disables
            # the overlap scheduler and makes any page_size > 1 fail a startup
            # assertion.
            support_mamba_cache_extra_buffer=True,
        )
    )


_register()
