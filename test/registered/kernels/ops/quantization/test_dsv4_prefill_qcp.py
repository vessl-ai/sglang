"""INF-681 k-qcp quant/copy fusions for the DeepSeek-V4.1 Hopper prefill
(layers/quantization/dsv4_prefill_qcp.py), each against the unfused path it replaces,
bit for bit:

* SGLANG_OPT_DSV4_WO_B_FUSED_ACT_QUANT: the 1x128 quant read straight from wo_a's grouped
  einsum output equals flatten(1) + ue8m0_group_quant_fp8_mn, and wo_b's output equals
  block128_fp8_linear's (CP4 rank shapes: 16,302 / 1,728 / 128 rows, 8 groups of 1024);
* SGLANG_OPT_DSV4_MOE_SHARED_ADD_IN_TOPK_SUM: the top-k sum with the shared add folded in
  equals the top-k sum followed by shared.add_(routed, alpha) (alpha 1 and the model's 1.5,
  masked slots), standalone and through the W4A8 prefill experts' handshake (sorted-sum
  and small-step paths);
* SGLANG_OPT_DSV4_MOE_INPUT_DUAL_QUANT: one pass gives the MoE input's 1x128 quant (GRAN128
  shared gate_up) and per-row quant (W4A8 routed experts), equal to the two quantizers;
  a GRAN128 linear takes the pre-quantized activation; the experts take the row quant of
  exactly their input, once.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 (DeepGEMM SF_GRAN 128 dense path)",
)


def _grouped(rows: int, groups: int, seed: int):
    # wo_a's real layout: einsum("tgd,grd->tgr") returns the token-major view of [G, T, R].
    g = torch.Generator(device="cuda").manual_seed(seed)
    o = torch.randn(rows, groups, 512, device="cuda", generator=g).bfloat16()
    w = (torch.randn(groups, 1024, 512, device="cuda", generator=g) * 0.05).bfloat16()
    y = torch.einsum("tgd,grd->tgr", o, w)
    y[: rows // 7] *= 1e-3  # small rows: other exponents
    return y


@pytest.mark.parametrize("rows", [0, 1, 128, 1728, 16302])
@pytest.mark.parametrize("groups", [8, 2])
def test_grouped_quant_matches_flatten_then_quant(rows, groups):
    from sglang.srt.layers.quantization.dsv4_prefill_qcp import (
        grouped_ue8m0_quant_fp8_mn,
    )
    from sglang.srt.layers.quantization.fp8_block32_gran128 import (
        ue8m0_group_quant_fp8_mn,
    )

    y = _grouped(rows, groups, rows + groups)
    if rows > 1:
        assert not y.is_contiguous(), "the einsum layout this path exists for"
    ref_q, ref_s = ue8m0_group_quant_fp8_mn(y.flatten(1).contiguous(), 128)
    q, s = grouped_ue8m0_quant_fp8_mn(y)
    assert q.shape == ref_q.shape and s.shape == ref_s.shape
    assert s.stride() == ref_s.stride()
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(s, ref_s)


@pytest.mark.parametrize("rows", [1, 128, 1728, 16302])
def test_wo_b_from_grouped_matches_block128_linear(rows):
    from sglang.srt.layers.quantization.dsv4_prefill_qcp import wo_b_from_grouped
    from sglang.srt.layers.quantization.fp8_block32_gran128 import (
        block128_fp8_linear,
    )

    g = torch.Generator(device="cuda").manual_seed(7)
    n, k = 5120, 8192
    weight = (torch.randn(n, k, device="cuda", generator=g) * 100).clamp(-448, 448).to(
        torch.float8_e4m3fn
    )
    scale = torch.exp2(
        torch.randint(-12, -6, (n // 128, k // 128), device="cuda", generator=g).float()
    )
    layer = SimpleNamespace(
        weight=weight, weight_scale_inv=scale, dense_gran128=True, bias=None, tp_size=1
    )
    y = _grouped(rows, 8, 100 + rows)
    ref = block128_fp8_linear(y.flatten(1), weight, scale)
    got = wo_b_from_grouped(y, layer)
    assert got is not None
    assert torch.equal(got, ref)
    # A TP-reduced wo_b keeps the regular path.
    assert wo_b_from_grouped(y, SimpleNamespace(**{**vars(layer), "tp_size": 4})) is None


def _sum_inputs(tokens, top_k, hidden, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = tokens * top_k
    down = torch.randn(n, hidden, device="cuda", generator=g).bfloat16()
    combine = torch.randperm(n, device="cuda", generator=g).int()
    shared = torch.randn(tokens, hidden, device="cuda", generator=g).bfloat16()
    return down, combine, shared


@pytest.mark.parametrize("tokens", [1, 300, 4096])
@pytest.mark.parametrize("alpha", [1.0, 1.5])
def test_topk_sum_add_matches_sum_then_add(tokens, alpha):
    from sglang.srt.layers.quantization.dsv4_prefill_qcp import (
        gather_topk_sum_add,
        topk_sum_by_pos_add,
    )
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import (
        _gather_topk_sum,
        _topk_sum_by_pos,
    )

    top_k, hidden = 6, 5120
    down, combine, shared = _sum_inputs(tokens, top_k, hidden, tokens)
    routed = _gather_topk_sum(down, combine, torch.empty_like(shared), top_k)
    ref = shared.clone().add_(routed, alpha=alpha)
    got = gather_topk_sum_add(down, combine, shared.clone(), top_k, alpha)
    assert torch.equal(got, ref)
    pos = torch.randint(-1, down.shape[0], (tokens * top_k,), device="cuda").int()
    routed = _topk_sum_by_pos(down, pos, torch.empty_like(shared), top_k)
    ref = shared.clone().add_(routed, alpha=alpha)
    assert torch.equal(topk_sum_by_pos_add(down, pos, shared.clone(), top_k, alpha), ref)


def _experts_and_method(small_step_rows: int):
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location(
        "w4a8_test",
        os.path.join(
            os.path.dirname(__file__), "..", "moe", "test_mxfp4_w4a8_prefill.py"
        ),
    )
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import (
        Mxfp4W4A8PrefillExperts,
    )

    runner = Mxfp4W4A8PrefillExperts(
        *helpers._weights(90),
        helpers.LIMIT,
        prefetch=False,
        sorted_sum=True,
        small_step_rows=small_step_rows,
    )
    method = object.__new__(Mxfp4MarlinMoEMethod)
    method._use_w4a8 = True
    method._w4a8 = runner
    return helpers, runner, method


@pytest.mark.parametrize("small_step_rows,rows", [(0, 300), (512, 300), (0, 2048)])
@pytest.mark.parametrize("prescaled", [True, False])
def test_experts_fold_the_shared_add(small_step_rows, rows, prescaled):
    from sglang.srt.layers.quantization import dsv4_prefill_qcp as qcp

    helpers, runner, method = _experts_and_method(small_step_rows)
    experts = SimpleNamespace(
        quant_method=method,
        reduce_results=False,
        should_fuse_routed_scaling_factor_in_topk=prescaled,
    )
    x, ids, wts = helpers._inputs(rows, 3)
    ids[0, -1] = -1
    shared = torch.randn(rows, helpers.H, device="cuda").bfloat16()
    alpha = 1.0 if prescaled else 1.5
    ref = shared.clone().add_(runner(x, ids, wts), alpha=alpha)

    got = shared.clone()
    assert qcp.arm_shared_add(experts, got, 1.5)
    out = runner(x, ids, wts)
    assert qcp.disarm_shared_add(experts)
    assert out.data_ptr() == got.data_ptr()
    assert torch.equal(got, ref)
    # Disarmed: the experts are back to the plain routed sum.
    assert torch.equal(shared.clone().add_(runner(x, ids, wts), alpha=alpha), ref)
    assert not qcp.disarm_shared_add(experts)


@pytest.mark.parametrize("rows", [1, 300, 6912, 65208])
def test_dual_quant_matches_both_quantizers(rows):
    from sglang.srt.layers.quantization.dsv4_prefill_qcp import dual_quant_fp8
    from sglang.srt.layers.quantization.fp8_block32_gran128 import (
        ue8m0_group_quant_fp8_mn,
    )
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import _row_quant

    g = torch.Generator(device="cuda").manual_seed(rows)
    x = (torch.randn(rows, 5120, device="cuda", generator=g) * 4).bfloat16()
    x[: rows // 5] *= 1e-4
    g128, (q2, s2) = dual_quant_fp8(x)
    q1_ref, s1_ref = ue8m0_group_quant_fp8_mn(x, 128)
    q2_ref, s2_ref = _row_quant(x)
    assert torch.equal(g128.q.view(torch.uint8), q1_ref.view(torch.uint8))
    assert g128.s.stride() == s1_ref.stride() and torch.equal(g128.s, s1_ref)
    assert torch.equal(q2.view(torch.uint8), q2_ref.view(torch.uint8))
    assert torch.equal(s2, s2_ref)


@pytest.mark.parametrize("rows", [1, 300, 6912])
def test_gran128_linear_takes_prequantized_activation(rows):
    from sglang.srt.layers.quantization import fp8_block32_gran128
    from sglang.srt.layers.quantization.dsv4_prefill_qcp import dual_quant_fp8

    g = torch.Generator(device="cuda").manual_seed(11)
    n, k = 1152, 5120
    weight = (torch.randn(n, k, device="cuda", generator=g) * 100).clamp(-448, 448).to(
        torch.float8_e4m3fn
    )
    scale = torch.exp2(
        torch.randint(-12, -6, (n // 128, k // 128), device="cuda", generator=g).float()
    )
    layer = SimpleNamespace(weight=weight, weight_scale_inv=scale)
    x = torch.randn(rows, k, device="cuda", generator=g).bfloat16()
    ref = fp8_block32_gran128.apply(layer, x, None)
    got = fp8_block32_gran128.apply(layer, dual_quant_fp8(x)[0], None)
    assert torch.equal(got, ref)
    with pytest.raises(RuntimeError):
        fp8_block32_gran128.apply(layer, (x, x), None)


@pytest.mark.parametrize("small_step_rows,rows,offered", [(0, 300, True), (512, 300, False)])
def test_experts_take_the_offered_row_quant(small_step_rows, rows, offered):
    from sglang.srt.layers.quantization import dsv4_prefill_qcp as qcp

    helpers, runner, method = _experts_and_method(small_step_rows)
    x, ids, wts = helpers._inputs(rows, 4)
    ref = runner(x, ids, wts)
    moe = SimpleNamespace(
        shared_experts=SimpleNamespace(
            gate_up_proj=SimpleNamespace(
                dense_gran128=True, weight=torch.empty(2 * 128, helpers.H, device="cuda")
            )
        ),
        experts=SimpleNamespace(quant_method=method),
    )
    g128 = qcp.offer_dual_quant(moe, x)
    assert (g128 is not None) == offered  # the small-step path quantizes its own layout
    if not offered:
        return
    assert runner.qcp_row_quant is not None
    assert torch.equal(runner(x, ids, wts), ref)
    assert runner.qcp_row_quant is None  # taken once
    # An offer for another tensor is not taken (and the output stays the same).
    qcp.offer_dual_quant(moe, x.clone())
    assert torch.equal(runner(x, ids, wts), ref)
    qcp.withdraw_row_quant(moe)
    assert runner.qcp_row_quant is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
