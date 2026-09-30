"""Unit tests for SolarOpen2ForCausalLM.load_weights on float-quantized Linears.

A full-FP8 checkpoint quantizes Linears this model builds unquantized
(``g_proj``, the KDA ``f_b_proj``/``g_b_proj``). Loading their FP8 weight
without its ``weight_scale`` boots and generates garbage, so these pin the
dequantize-at-load path and the gates that refuse a half-loaded pair.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoeWeightScaleSupported
from sglang.srt.models import solar_open2
from sglang.srt.models.solar_open2 import (
    SolarOpen2ForCausalLM,
    _FloatQuantDequantizer,
)

_GQA = "model.layers.0.self_attn"
_KDA = "model.layers.1.self_attn"


class _RecordingParam:
    """Holds what each weight_loader call received, keyed by shard id."""

    def __init__(self, dtype=torch.bfloat16):
        self.dtype = dtype
        self.loads = {}

    def weight_loader(self, param, loaded_weight, shard_id=None, **kwargs):
        self.loads[shard_id] = loaded_weight


def _fp8_weight_and_scale(rows=4, cols=3):
    weight = (
        torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols) / 4
    ).to(torch.float8_e4m3fn)
    scale = torch.linspace(0.5, 2.0, rows, dtype=torch.float32).reshape(rows, 1)
    return weight, scale


def _expected(weight, scale, dtype=torch.bfloat16):
    return (weight.to(torch.float32) * scale).to(dtype)


def _make_model(named_parameters):
    model = object.__new__(SolarOpen2ForCausalLM)
    model.config = SimpleNamespace(num_experts=2)
    model.pp_group = SimpleNamespace(world_size=1)
    model.named_parameters = lambda: iter(named_parameters.items())
    model._log_load_gates = lambda params_dict, loaded_params: None
    return model


class TestFloatQuantDequantizer(unittest.TestCase):
    def _run(self, weights, owned=()):
        dequantizer = _FloatQuantDequantizer(
            lambda prefix: None if prefix in owned else torch.bfloat16
        )
        return list(dequantizer.wrap(weights)), dequantizer

    def test_scale_before_and_after_weight_dequantize_identically(self):
        weight, scale = _fp8_weight_and_scale()
        name = f"{_GQA}.g_proj"

        scale_first, _ = self._run(
            [(f"{name}.weight_scale", scale), (f"{name}.weight", weight)]
        )
        weight_first, _ = self._run(
            [(f"{name}.weight", weight), (f"{name}.weight_scale", scale)]
        )

        self.assertEqual([n for n, _ in scale_first], [f"{name}.weight"])
        self.assertTrue(torch.equal(scale_first[0][1], weight_first[0][1]))
        self.assertTrue(torch.equal(scale_first[0][1], _expected(weight, scale)))

    def test_module_owning_its_scale_passes_both_tensors_through(self):
        weight, scale = _fp8_weight_and_scale()
        name = f"{_GQA}.q_proj"
        weights = [(f"{name}.weight", weight), (f"{name}.weight_scale", scale)]

        emitted, dequantizer = self._run(weights, owned={name})

        self.assertEqual(emitted, weights)
        self.assertEqual(dequantizer.dequantized, 0)

    def test_scale_whose_weight_is_not_fp8_stays_unpaired(self):
        _, scale = _fp8_weight_and_scale()
        name = f"{_GQA}.g_proj"
        bf16_weight = torch.ones(4, 3, dtype=torch.bfloat16)

        emitted, dequantizer = self._run(
            [(f"{name}.weight_scale", scale), (f"{name}.weight", bf16_weight)]
        )

        self.assertEqual([n for n, _ in emitted], [f"{name}.weight"])
        self.assertEqual(dequantizer.unpaired(), [f"{name}.weight_scale"])


class TestSolarOpen2FloatQuantLoad(unittest.TestCase):
    def test_fused_kda_places_f_b_and_g_b_in_their_shards(self):
        fused = _RecordingParam()
        model = _make_model({f"{_KDA}.fused_fg_b_proj.weight": fused})
        f_weight, f_scale = _fp8_weight_and_scale()
        g_weight, g_scale = _fp8_weight_and_scale()
        g_scale = g_scale * 3

        with mock.patch.object(solar_open2, "_FUSE_KDA", True):
            model.load_weights(
                [
                    (f"{_KDA}.g_b_proj.weight_scale", g_scale),
                    (f"{_KDA}.f_b_proj.weight", f_weight),
                    (f"{_KDA}.g_b_proj.weight", g_weight),
                    (f"{_KDA}.f_b_proj.weight_scale", f_scale),
                ]
            )

        self.assertEqual(set(fused.loads), {0, 1})
        self.assertTrue(torch.equal(fused.loads[0], _expected(f_weight, f_scale)))
        self.assertTrue(torch.equal(fused.loads[1], _expected(g_weight, g_scale)))

    def test_unfused_kda_and_g_proj_load_dequantized(self):
        params = {
            f"{_KDA}.f_b_proj.weight": _RecordingParam(),
            f"{_GQA}.g_proj.weight": _RecordingParam(torch.float16),
        }
        model = _make_model(params)
        weight, scale = _fp8_weight_and_scale()

        with mock.patch.object(solar_open2, "_FUSE_KDA", False):
            model.load_weights(
                [
                    (f"{_KDA}.f_b_proj.weight", weight),
                    (f"{_GQA}.g_proj.weight_scale", scale),
                    (f"{_KDA}.f_b_proj.weight_scale", scale),
                    (f"{_GQA}.g_proj.weight", weight),
                ]
            )

        f_b = params[f"{_KDA}.f_b_proj.weight"].loads[None]
        g = params[f"{_GQA}.g_proj.weight"].loads[None]
        self.assertTrue(torch.equal(f_b, _expected(weight, scale)))
        self.assertEqual(g.dtype, torch.float16)
        self.assertTrue(torch.equal(g, _expected(weight, scale, torch.float16)))

    def test_quantized_module_keeps_its_fp8_weight_and_scale(self):
        weight_param = _RecordingParam(torch.float8_e4m3fn)
        scale_param = _RecordingParam(torch.float32)
        model = _make_model(
            {
                f"{_GQA}.qkv_proj.weight": weight_param,
                f"{_GQA}.qkv_proj.weight_scale": scale_param,
            }
        )
        weight, scale = _fp8_weight_and_scale()

        model.load_weights(
            [(f"{_GQA}.q_proj.weight", weight), (f"{_GQA}.q_proj.weight_scale", scale)]
        )

        self.assertIs(weight_param.loads["q"], weight)
        self.assertIs(scale_param.loads["q"], scale)

    def test_scale_with_no_parameter_or_weight_raises(self):
        model = _make_model({f"{_GQA}.g_proj.weight": _RecordingParam()})
        _, scale = _fp8_weight_and_scale()

        with self.assertRaisesRegex(ValueError, r"o_proj\.weight_scale"):
            model.load_weights([(f"{_GQA}.o_proj.weight_scale", scale)])

    def test_fp8_weight_whose_scale_never_arrives_raises(self):
        model = _make_model({f"{_GQA}.g_proj.weight": _RecordingParam()})
        weight, _ = _fp8_weight_and_scale()

        with self.assertRaisesRegex(ValueError, r"g_proj\.weight'"):
            model.load_weights([(f"{_GQA}.g_proj.weight", weight)])

    def test_bf16_checkpoint_loads_as_before_and_q_scale_is_skipped(self):
        g_proj = _RecordingParam()
        model = _make_model({f"{_GQA}.g_proj.weight": g_proj})
        bf16_weight = torch.ones(4, 3, dtype=torch.bfloat16)

        model.load_weights(
            [
                (f"{_GQA}.g_proj.weight", bf16_weight),
                (f"{_GQA}.q_scale", torch.ones(1)),
            ]
        )

        self.assertIs(g_proj.loads[None], bf16_weight)


class TestSolarOpen2MoeScaleGate(unittest.TestCase):
    def _gate(self, scale, quant_method):
        scale.quant_method = quant_method
        model = object.__new__(SolarOpen2ForCausalLM)
        model.config = SimpleNamespace(
            full_attention_layer_ids=[], moe_intermediate_size=1280
        )
        experts = SimpleNamespace(w2_weight_scale=scale)
        model.model = SimpleNamespace(
            start_layer=0,
            layers=[SimpleNamespace(mlp=SimpleNamespace(experts=experts))],
        )
        model._log_load_gates({}, set())

    def test_per_channel_fp8_scale_passes(self):
        self._gate(torch.ones(128, 2048, 1), FusedMoeWeightScaleSupported.CHANNEL.value)

    def test_group_scale_floored_by_tp_sharding_raises(self):
        with self.assertRaisesRegex(ValueError, "expert parallelism"):
            self._gate(
                torch.ones(128, 2048, 1), FusedMoeWeightScaleSupported.GROUP.value
            )


if __name__ == "__main__":
    unittest.main()
