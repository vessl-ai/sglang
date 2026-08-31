# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference-only GLM5-Next Speculative Decoding."""

import copy
import logging

from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.models.utils import WeightsMapper

logger = logging.getLogger(__name__)


class Glm5NextForConditionalGenerationNextN(DeepseekV3ForCausalLMNextN):
    @classmethod
    def get_hf_to_sglang_mapper(cls, config) -> WeightsMapper:
        text_config = getattr(config, "text_config", config)
        return WeightsMapper(
            orig_to_new_substr={
                f"model.layers.{text_config.num_hidden_layers}": "model.decoder",
            },
        )

    def _resolve_nextn_quant_config(self, config, quant_config):
        """Keep a checkpoint-declared BF16 NextN block unquantized.

        The Hybrid FP8+NVFP4 checkpoint advertises a global FP8 quantization
        method, but its full ``model.layers.<num_hidden_layers>.*`` NextN block
        is listed in ``quantization_config.ignore`` and stored as BF16. Passing
        the target quant config into that block allocates FP8 parameters and
        loads the BF16 payload without matching scales, corrupting the first
        QKV projection.
        """
        raw_quant_config = getattr(config, "quantization_config", None) or {}
        if hasattr(raw_quant_config, "to_dict"):
            raw_quant_config = raw_quant_config.to_dict()
        ignored = (
            raw_quant_config.get("ignore", [])
            if isinstance(raw_quant_config, dict)
            else []
        )
        nextn_layer_pattern = f"model.layers.{config.num_hidden_layers}.*"
        if nextn_layer_pattern in ignored:
            logger.warning(
                "GLM5 NextN layer %s is checkpoint-declared unquantized; "
                "using BF16 draft modules",
                nextn_layer_pattern,
            )
            return None
        resolved = super()._resolve_nextn_quant_config(config, quant_config)
        return self._remap_ignored_layers_for_nextn(config, resolved)

    @classmethod
    def _remap_ignored_layers_for_nextn(cls, config, quant_config):
        """Name the checkpoint's un-quantized MTP modules the way the draft does.

        The draft renames the checkpoint's MTP block from
        ``model.layers.<num_hidden_layers>.*`` to ``model.decoder.*``
        (``get_hf_to_sglang_mapper`` above, mirroring ``DeepseekModelNextN``'s
        ``layer_name = "decoder"``), but ``quantization_config``'s
        ``modules_to_not_convert`` lists them under the checkpoint name. Nothing
        translates between the two, so ``is_layer_skipped()`` never matches
        inside the draft: every module the release deliberately kept at bf16 --
        for GLM-5.3-Flash-W4AFP8 that is ``mlp.gate``, the DSA
        ``self_attn.indexer.*`` family, ``eh_proj``/``enorm``/``hnorm`` and the
        layernorms, 18 entries -- is given a quantized method instead, and the
        first one to load dies with::

            ValueError: Downcasting not allowed:
              target.dtype=torch.float8_e4m3fn, loaded_weight.dtype=torch.bfloat16

        Adding the mapped names restores exactly the checkpoint's own intent, so
        accuracy is unchanged: the MTP routed experts stay quantized (they are
        not in the list), and the target model's own resolution is untouched --
        the config is copied rather than mutated, since the target shares it.
        """
        if quant_config is None:
            return None
        ignored = list(getattr(quant_config, "ignored_layers", None) or [])
        if not ignored:
            return quant_config

        mapper = cls.get_hf_to_sglang_mapper(config)
        extra = []
        for name in ignored:
            # Callers normalise each entry to both "x" and "model.x"; the mapper
            # only matches the "model."-prefixed form, so map that one and emit
            # both forms back.
            candidate = name if name.startswith("model.") else f"model.{name}"
            mapped = mapper._map_name(candidate)
            if mapped != candidate:
                extra.append(mapped)
                extra.append(mapped.removeprefix("model."))
        if not extra:
            return quant_config

        seen = set(ignored)
        remapped = ignored + [e for e in dict.fromkeys(extra) if e not in seen]
        logger.info(
            "GLM5 NextN: mapped %d checkpoint-named ignored-layer entries onto "
            "the draft's model.decoder.* prefix (%d -> %d)",
            len(remapped) - len(ignored),
            len(ignored),
            len(remapped),
        )
        resolved = copy.copy(quant_config)
        resolved.ignored_layers = remapped
        return resolved

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(
            getattr(config, "text_config", config),
            quant_config=quant_config,
            prefix=prefix,
        )

    def load_weights(self, weights):
        if not hasattr(self, "fuse_qkv_a_proj"):
            self.fuse_qkv_a_proj = getattr(self.config, "q_lora_rank", None) is not None
        layer_id = self.config.num_hidden_layers
        layer_prefixes = (
            f"model.layers.{layer_id}.",
            f"model.language_model.layers.{layer_id}.",
        )
        nextn_weights = (
            (name, weight)
            for name, weight in weights
            if name.startswith(layer_prefixes)
        )
        return Glm5NextForConditionalGeneration.load_weights(
            self, nextn_weights, is_nextn=True
        )


EntryClass = [Glm5NextForConditionalGenerationNextN]
