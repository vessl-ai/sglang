"""Tests for serving block-FP8 linears through HALOps (SGLANG_HQ_HALOPS_BLOCK_FP8).

They run without the optional ``halops`` wheel and without a GPU: HALOps is
mocked, so these check the scale layout, which calls reach HALOps, and that the
switch-off path is the original one. The numbers HALOps computes are checked
only on a GPU, by the self-check mode.
"""

import logging
import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization import fp8, halops_block_fp8
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend
from sglang.srt.layers.quantization.halops_block_fp8 import (
    HALOPS_MODE,
    halops_linear,
    halops_weight_scale,
    swizzle_scale_to_sfatom,
)
from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cutlass_128x4_reference(scale: torch.Tensor) -> torch.Tensor:
    """The CUTLASS 128x4 scale-factor atom, element by element: row r, column c
    of a [M, K // 32] scale lands at
    ((r // 128) * (cols // 4) + c // 4) * 512 + (r % 32) * 16 + ((r // 32) % 4) * 4 + c % 4.
    FlashInfer's block_scale_interleave writes this layout."""
    rows, cols = scale.shape
    out = torch.empty(rows * cols, dtype=scale.dtype)
    for r in range(rows):
        for c in range(cols):
            off = (
                ((r // 128) * (cols // 4) + c // 4) * 512
                + (r % 32) * 16
                + ((r // 32) % 4) * 4
                + c % 4
            )
            out[off] = scale[r, c]
    return out.view(rows, cols)


def _scale(n: int, k: int) -> torch.Tensor:
    return torch.randint(100, 140, (n, k // 32), dtype=torch.uint8)


class _ResetModuleState(CustomTestCase):
    def setUp(self):
        halops_block_fp8._used_shapes.clear()
        halops_block_fp8._selfcheck_left = None


class TestSfAtomLayout(CustomTestCase):
    def test_matches_the_cutlass_atom(self):
        for n, k in ((128, 128), (256, 512), (384, 256)):
            with self.subTest(n=n, k=k):
                scale = _scale(n, k)
                self.assertTrue(
                    torch.equal(
                        swizzle_scale_to_sfatom(scale), _cutlass_128x4_reference(scale)
                    )
                )


@mock.patch.object(halops_block_fp8, "load_halops_gemm")
class TestHalopsWeightScale(CustomTestCase):
    def test_unaligned_shapes_stay_on_flashinfer(self, _load):
        for n, k in ((192, 256), (256, 160)):
            with self.subTest(n=n, k=k):
                weight = torch.zeros(n, k)
                self.assertIsNone(halops_weight_scale(weight, _scale(n, k), None))

    def test_takes_the_scale_flashinfer_uses(self, _load):
        weight = torch.zeros(256, 512)
        scale = _scale(256, 512)
        flashinfer_scale = _cutlass_128x4_reference(scale).reshape(-1)
        out = halops_weight_scale(weight, scale, flashinfer_scale)
        self.assertTrue(torch.equal(out.reshape(-1), flashinfer_scale))

    def test_a_differing_flashinfer_scale_keeps_the_layer_on_flashinfer(self, _load):
        weight = torch.zeros(256, 512)
        scale = _scale(256, 512)
        flashinfer_scale = _cutlass_128x4_reference(scale).clone().reshape(-1)
        flashinfer_scale[7] += 1
        self.assertIsNone(halops_weight_scale(weight, scale, flashinfer_scale))


class _FakeHalops:
    def __init__(self, value: float = 1.0):
        self.value = value
        self.quant_calls = []
        self.gemm_calls = []

    def group_wise_quant(self, x, mode, use_swizzle):
        self.quant_calls.append((x, mode, use_swizzle))
        return x.to(torch.float8_e4m3fn), torch.zeros(1, dtype=torch.uint8)

    def dense_gemm(self, **kwargs):
        self.gemm_calls.append(kwargs)
        kwargs["output"].fill_(self.value)

    def patch(self):
        return mock.patch.object(
            halops_block_fp8,
            "load_halops_gemm",
            return_value=(self.dense_gemm, self.group_wise_quant),
        )


class TestHalopsLinear(_ResetModuleState):
    def _run(self, fake, x, n=256, bias=None, fallback=None):
        weight = torch.zeros(n, x.shape[-1], dtype=torch.float8_e4m3fn)
        sfb = torch.zeros(1, dtype=torch.uint8)
        fallback = fallback or mock.Mock(side_effect=AssertionError("fallback"))
        with fake.patch():
            out = halops_linear(
                input=x, weight=weight, weight_scale=sfb, bias=bias, fallback=fallback
            )
        return out, weight, sfb

    def test_forwards_operands_to_dense_gemm(self):
        fake = _FakeHalops()
        x = torch.ones(2, 3, 128, dtype=torch.float16)
        bias = torch.zeros(256, dtype=torch.float16)
        fake.value = 2.0
        with mock.patch.object(halops_block_fp8, "HALOPS_BIAS_M_ALIGN", 6):
            out, weight, sfb = self._run(fake, x, bias=bias)
        self.assertEqual(out.shape, (2, 3, 256))
        self.assertEqual(out.dtype, torch.float16)
        self.assertTrue(torch.all(out == 2.0))
        (q_in, mode, use_swizzle) = fake.quant_calls[0]
        self.assertEqual(q_in.shape, (6, 128))
        self.assertEqual((mode, use_swizzle), (HALOPS_MODE, True))
        call = fake.gemm_calls[0]
        self.assertIs(call["weight"], weight)
        self.assertIs(call["sfb"], sfb)
        self.assertEqual(call["output"].shape, (6, 256))
        self.assertEqual(call["output"].dtype, torch.bfloat16)
        self.assertEqual(call["bias"].dtype, torch.bfloat16)
        self.assertEqual(call["mode"], HALOPS_MODE)

    def test_shapes_no_tile_takes_run_the_original_kernel(self):
        cases = {
            "no rows": (torch.zeros(0, 128, dtype=torch.bfloat16), None),
            "fp32 input": (torch.ones(4, 128, dtype=torch.float32), None),
            "bias with M not a multiple of 256": (
                torch.ones(4, 128, dtype=torch.bfloat16),
                torch.zeros(256, dtype=torch.bfloat16),
            ),
        }
        for name, (x, bias) in cases.items():
            with self.subTest(name):
                fake = _FakeHalops()
                sentinel = torch.empty(0)
                out, _, _ = self._run(
                    fake, x, bias=bias, fallback=mock.Mock(return_value=sentinel)
                )
                self.assertIs(out, sentinel)
                self.assertEqual(fake.gemm_calls, [])

    def test_logs_each_shape_on_first_use(self):
        fake = _FakeHalops()
        x = torch.ones(4, 128, dtype=torch.bfloat16)
        with self.assertLogs(halops_block_fp8.logger, logging.INFO) as logs:
            self._run(fake, x, n=256)
            self._run(fake, x, n=256)
            self._run(fake, x, n=384)
        lines = [r for r in logs.output if "dense GEMM in use" in r]
        self.assertEqual(len(lines), 2)
        self.assertIn("N=256 K=128 first M=4", lines[0])

    def test_self_check_compares_the_first_nonzero_calls(self):
        fake = _FakeHalops(value=1.0)
        x = torch.ones(4, 128, dtype=torch.bfloat16)
        zeros = mock.Mock(return_value=torch.zeros(4, 256))
        ref = mock.Mock(return_value=torch.full((4, 256), 0.5))
        with (
            mock.patch.object(
                halops_block_fp8.envs.SGLANG_HQ_HALOPS_BLOCK_FP8_SELFCHECK,
                "get",
                return_value=2,
            ),
            self.assertLogs(halops_block_fp8.logger, logging.INFO) as logs,
        ):
            self._run(fake, x, fallback=zeros)
            for _ in range(3):
                self._run(fake, x, fallback=ref)
        self.assertEqual(zeros.call_count, 1)
        self.assertEqual(ref.call_count, 2)
        checks = [r for r in logs.output if "self-check" in r]
        self.assertEqual(len(checks), 2)
        self.assertIn("max_abs_diff=0.5", checks[0])

    def test_self_check_is_off_by_default(self):
        fake = _FakeHalops()
        x = torch.ones(4, 128, dtype=torch.bfloat16)
        self._run(fake, x)
        self.assertEqual(len(fake.gemm_calls), 1)


def _method(halops_on: bool) -> Fp8LinearMethod:
    method = Fp8LinearMethod.__new__(Fp8LinearMethod)
    method.use_marlin = False
    method.use_mxfp8 = False
    method.block_quant = True
    method.block_fp8_as_mxfp8 = True
    method.weight_block_size = [32, 32]
    method.mxfp8_dense_backend = Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL
    method.mxfp8_prefill_autotune_min_tokens = None
    method.w8a8_mxfp8_linear = mock.Mock(return_value="flashinfer")
    method.halops_block_fp8 = halops_on
    return method


def _layer(n=256, k=512):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros(n, k, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale_inv = torch.nn.Parameter(
        torch.full((n // 32, k // 32), 2.0**-3), requires_grad=False
    )
    return layer


def _process_scale(self, layer, scale_u8):
    layer.weight_scale_inv_swizzled = _cutlass_128x4_reference(scale_u8).reshape(-1)


@mock.patch.object(
    Fp8LinearMethod, "_process_mxfp8_linear_weight_scale", _process_scale
)
class TestFp8LinearMethodWiring(_ResetModuleState):
    def test_switch_off_prepares_and_runs_the_original_path(self):
        method, layer = _method(False), _layer()
        with (
            mock.patch.object(fp8, "halops_weight_scale") as weight_scale,
            mock.patch.object(fp8, "halops_linear") as linear,
        ):
            method._prepare_block_fp8_as_mxfp8(layer)
            out = method.apply(layer, torch.ones(4, 512, dtype=torch.bfloat16))
        weight_scale.assert_not_called()
        linear.assert_not_called()
        self.assertFalse(hasattr(layer, "weight_scale_inv_halops"))
        self.assertEqual(out, "flashinfer")
        self.assertIs(
            method.w8a8_mxfp8_linear.call_args.kwargs["weight_scale"],
            layer.weight_scale_inv_swizzled,
        )

    def test_switch_on_runs_bf16_input_on_halops(self):
        method, layer = _method(True), _layer()
        fake = _FakeHalops()
        x = torch.ones(4, 512, dtype=torch.bfloat16)
        with fake.patch():
            method._prepare_block_fp8_as_mxfp8(layer)
            out = method.apply(layer, x)
        self.assertTrue(
            torch.equal(
                layer.weight_scale_inv_halops.reshape(-1),
                layer.weight_scale_inv_swizzled,
            )
        )
        self.assertIs(fake.gemm_calls[0]["sfb"], layer.weight_scale_inv_halops)
        self.assertEqual(out.shape, (4, 256))
        method.w8a8_mxfp8_linear.assert_not_called()

    def test_switch_on_fallback_is_the_original_call(self):
        method, layer = _method(True), _layer()
        x = torch.ones(4, 512, dtype=torch.bfloat16)
        with _FakeHalops().patch():
            method._prepare_block_fp8_as_mxfp8(layer)
        with mock.patch.object(fp8, "halops_linear") as linear:
            method.apply(layer, x)
        linear.call_args.kwargs["fallback"]()
        self.assertEqual(
            method.w8a8_mxfp8_linear.call_args.kwargs,
            dict(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale_inv_swizzled,
                input_scale=None,
                bias=None,
            ),
        )

    def test_switch_on_keeps_prequantized_input_on_flashinfer(self):
        method, layer = _method(True), _layer()
        with _FakeHalops().patch():
            method._prepare_block_fp8_as_mxfp8(layer)
        x = Mxfp8SwizzledInput(
            torch.zeros(4, 512, dtype=torch.float8_e4m3fn),
            torch.zeros(1, dtype=torch.uint8),
        )
        with mock.patch.object(fp8, "halops_linear") as linear:
            out = method.apply(layer, x)
        linear.assert_not_called()
        self.assertEqual(out, "flashinfer")


if __name__ == "__main__":
    unittest.main()
