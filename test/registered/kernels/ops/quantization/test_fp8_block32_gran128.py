"""SGLANG_OPT_DSV4_DENSE_GEMM_GRAN128: 32x32 UE8M0 block-FP8 dense GEMMs on SM90 at SF_GRAN 128.

Checks the one-time weight requantization (32x32 -> 128x128 power-of-two blocks), the 1x128
activation quantizer (values against the CUDA per-token-group quantizer, MN-major scale layout),
the GEMM against an FP32 reference and against the engine's 1x32 / SF_GRAN 32 path, and the
Fp8LinearMethod hook (process_weights_after_loading converts, apply routes).
"""

import os
import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
    pytest.skip("SM90 CUDA is required", allow_module_level=True)

from sglang.kernels.ops.quantization.fp8_kernel import (  # noqa: E402
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.quantization import fp8_block32_gran128  # noqa: E402
from sglang.srt.layers.quantization.fp8_block32_gran128 import (  # noqa: E402
    block128_fp8_linear,
    expand_scale_128_to_32,
    requant_block32_to_block128,
    ue8m0_group_quant_fp8_g128,
    ue8m0_group_quant_fp8_mn,
)

torch.backends.cuda.matmul.allow_tf32 = False


def _expand(s, gran, n, k):
    return s.repeat_interleave(gran, 0).repeat_interleave(gran, 1)[:n, :k]


def _random_block32_weight(n, k, exp_lo=-9, exp_hi=-3, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(n, k, device="cuda", generator=g) * 120).clamp(-448, 448)
    q = q.to(torch.float8_e4m3fn)
    e = torch.randint(
        exp_lo, exp_hi, ((n + 31) // 32, (k + 31) // 32), device="cuda", generator=g
    )
    return q, torch.exp2(e.float())


@pytest.mark.parametrize("n,k", [(1792, 5120), (1088, 576), (8192, 1280)])
def test_requant_same_exponent_is_bit_exact(n, k):
    # Every 32x32 sub-block of a 128x128 block shares the exponent -> nothing may move.
    q, _ = _random_block32_weight(n, k, seed=n + k)
    e = torch.randint(-9, -3, ((n + 127) // 128, (k + 127) // 128), device="cuda")
    s32 = torch.exp2(e.float()).repeat_interleave(4, 0).repeat_interleave(4, 1)
    s32 = s32[: (n + 31) // 32, : (k + 31) // 32].contiguous()
    # make sure every 128x128 block holds its max magnitude so the UE8M0 rule keeps 2^e
    qf = q.float()
    qf[::128, ::128] = 448.0
    q = qf.to(torch.float8_e4m3fn)
    q128, s128 = requant_block32_to_block128(q, s32)
    w_ref = q.float() * _expand(s32, 32, n, k)
    w_new = q128.float() * _expand(s128, 128, n, k)
    assert s128.shape == ((n + 127) // 128, (k + 127) // 128)
    assert torch.equal(w_new, w_ref)


@pytest.mark.parametrize("n,k", [(1792, 5120), (1088, 576), (4096, 1280)])
def test_requant_mixed_exponents_only_moves_subnormals(n, k):
    q, s32 = _random_block32_weight(n, k, seed=3 * n + k)
    q128, s128 = requant_block32_to_block128(q, s32, rows_per_chunk=256)
    w_ref = q.float() * _expand(s32, 32, n, k)
    w_new = q128.float() * _expand(s128, 128, n, k)
    assert torch.equal(
        torch.frexp(s128)[0], torch.full_like(s128, 0.5)
    )  # powers of two
    changed = w_new != w_ref
    # an element may only move when its new e4m3 code is subnormal (|q| < 2^-6)
    assert torch.all(q128.float().abs()[changed] <= 2.0**-6)
    # and by at most half a subnormal step (2^-10) of the new scale
    err = (w_new - w_ref).abs() / _expand(s128, 128, n, k)
    assert err.max().item() <= 2.0**-10 + 1e-12
    assert ((w_new - w_ref).norm() / w_ref.norm()).item() < 1e-3


@pytest.mark.parametrize("m", [1, 7, 130, 4096])
@pytest.mark.parametrize("k", [1280, 2048, 5120, 6144])
def test_act_quant_matches_cuda_group128_bitwise(m, k):
    torch.manual_seed(m * 13 + k)
    x = torch.randn(m, k, device="cuda") * torch.exp(
        torch.randn(m, 1, device="cuda") * 3
    )
    x[:, :128] = 0  # an all-zero group takes the eps floor
    x = x.bfloat16()
    q_ref, s_ref = sglang_per_token_group_quant_fp8(x, 128, scale_ue8m0=True)
    q, s = ue8m0_group_quant_fp8_g128(x)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(s, s_ref)
    # MN-major, M stride aligned to 4 floats (DeepGEMM's TMA layout, no transpose needed)
    assert s.stride(0) == 1 and s.stride(1) == (m + 3) // 4 * 4


@pytest.mark.parametrize("m", [1, 130, 750])
@pytest.mark.parametrize("k", [576, 2048, 5120])
def test_act_quant_group32_matches_cuda_bitwise(m, k):
    torch.manual_seed(m * 5 + k)
    x = torch.randn(m, k, device="cuda") * torch.exp(
        torch.randn(m, 1, device="cuda") * 3
    )
    x[:, :32] = 0
    x = x.bfloat16()
    q_ref, s_ref = sglang_per_token_group_quant_fp8(x, 32, scale_ue8m0=True)
    q, s = ue8m0_group_quant_fp8_mn(x, 32)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(s, s_ref)
    assert s.stride(0) == 1 and s.stride(1) == (m + 3) // 4 * 4


@pytest.mark.parametrize("m,k", [(5, 576), (300, 576), (64, 640 + 64)])
def test_act_quant_partial_last_group(m, k):
    torch.manual_seed(m + k)
    x = (torch.randn(m, k, device="cuda") * 4).bfloat16()
    q, s = ue8m0_group_quant_fp8_g128(x)
    groups = (k + 127) // 128
    xp = torch.zeros(m, groups * 128, device="cuda")
    xp[:, :k] = x.float()
    amax = xp.view(m, groups, 128).abs().amax(-1).clamp(min=1e-10)
    s_ref = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    assert torch.equal(s, s_ref)
    q_ref = (x.float() / s_ref.repeat_interleave(128, 1)[:, :k]).to(torch.float8_e4m3fn)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))


def _engine_gran32(x, q32, s32):
    os.environ["SGLANG_SM90_TRITON_UE8M0_ACT_QUANT"] = "1"
    from sglang.srt.layers.quantization.fp8_utils import (
        deepgemm_w8a8_block_fp8_linear_with_fallback,
    )

    return deepgemm_w8a8_block_fp8_linear_with_fallback(
        x, q32, [32, 32], s32, act_scale_ue8m0=True
    )


@pytest.mark.parametrize(
    "m,n,k",
    [
        (32728, 8192, 1280),
        (768, 5120, 2048),
        (750, 1792, 5120),
        (528, 5120, 576),
        (3, 1152, 5120),
    ],
)
def test_linear_matches_reference_and_gran32(m, n, k):
    torch.manual_seed(m + n + k)
    q32, s32 = _random_block32_weight(n, k, seed=m + n)
    x = (
        torch.randn(m, k, device="cuda") * torch.exp(torch.randn(1, k, device="cuda"))
    ).bfloat16()
    q128, s128 = requant_block32_to_block128(q32, s32)
    y = block128_fp8_linear(x, q128, s128)
    # FP32 product of the quantized operands this path uses
    xq, xs = ue8m0_group_quant_fp8_g128(x)
    x_deq = xq.float() * xs.repeat_interleave(128, 1)[:, :k]
    ref = x_deq @ (q128.float() * _expand(s128, 128, n, k)).t()
    assert ((y.float() - ref).norm() / ref.norm()).item() < 3e-3
    y32 = _engine_gran32(x, q32, s32)
    assert ((y.float() - y32.float()).norm() / y32.float().norm()).item() < 3e-3


@pytest.mark.parametrize(
    "m,n,k", [(750, 5120, 2048), (704, 5120, 576), (512, 8192, 1280)]
)
def test_gran32_window_fallback_matches_gran128(m, n, k, monkeypatch):
    """Inside a slow-heuristic window the same requantized weight runs SF_GRAN 32 with the
    expanded scales; its output must match the SF_GRAN 128 kernel's."""
    q32, s32 = _random_block32_weight(n, k, seed=m * 3 + n)
    q128, s128 = requant_block32_to_block128(q32, s32)
    s32x = expand_scale_128_to_32(s128, n, k)
    assert torch.equal(_expand(s32x, 32, n, k), _expand(s128, 128, n, k))
    x = (torch.randn(m, k, device="cuda") * 3).bfloat16()
    monkeypatch.setattr(fp8_block32_gran128, "_GRAN32_WINDOWS", {})
    y128 = block128_fp8_linear(x, q128, s128, None, s32x)
    monkeypatch.setattr(fp8_block32_gran128, "_GRAN32_WINDOWS", {(n, k): [(m, m)]})
    assert fp8_block32_gran128.prefer_gran32_kernel(m, n, k)
    y32x = block128_fp8_linear(x, q128, s128, None, s32x)
    assert not torch.equal(y128, y32x)  # really a different kernel
    assert ((y32x.float() - y128.float()).norm() / y128.float().norm()).item() < 2e-3


def test_fp8_linear_method_hook(monkeypatch):
    """process_weights_after_loading converts an eligible layer and apply routes it here."""
    monkeypatch.setenv("SGLANG_OPT_DSV4_DENSE_GEMM_GRAN128", "1")
    from torch.nn import Module, Parameter

    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

    cfg = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[32, 32],
        scale_fmt="ue8m0",
    )
    method = Fp8LinearMethod(cfg)
    n, k = 1792, 5120
    q32, s32 = _random_block32_weight(n, k, seed=11)
    layer = Module()
    layer.weight = Parameter(q32.clone(), requires_grad=False)
    layer.weight_scale_inv = Parameter(s32.clone(), requires_grad=False)
    layer.orig_dtype = torch.bfloat16
    x = torch.randn(257, k, device="cuda").bfloat16()
    y32 = _engine_gran32(x, q32, s32)
    method.process_weights_after_loading(layer)
    assert getattr(layer, "dense_gran128", False)
    assert layer.weight_scale_inv.shape == (n // 128, k // 128)
    y = method.apply(layer, x)
    assert ((y.float() - y32.float()).norm() / y32.float().norm()).item() < 3e-3


_CKPT = os.environ.get(
    "SGLANG_TEST_DSV41_CKPT", "/opt/lv/data/models/DeepSeek-V4.1-Flash-1017"
)


def _draft_wkv_weights():
    """The DSpark draft stages' wkv weights (checkpoint mtp.{0,1,2}.attn.wkv, [512, 5120]
    FP8 with 32x32 UE8M0 scales); random weights of that shape without the checkpoint."""
    index = os.path.join(_CKPT, "model.safetensors.index.json")
    if not os.path.exists(index):
        return [_random_block32_weight(512, 5120, seed=40 + i) for i in range(3)], False
    import json

    from safetensors import safe_open

    weight_map = json.load(open(index))["weight_map"]
    out = []
    for i in range(3):
        tensors = []
        for suffix in ("weight", "scale"):
            name = f"mtp.{i}.attn.wkv.{suffix}"
            with safe_open(
                os.path.join(_CKPT, weight_map[name]), "pt", device="cpu"
            ) as f:
                tensors.append(f.get_tensor(name).cuda())
        out.append((tensors[0], tensors[1].float()))
    return out, True


class _Fp8Linear(torch.nn.Module):
    """Just enough of ReplicatedLinear: forward = quant_method.apply, returns (y, None)."""

    def forward(self, x):
        return self.quant_method.apply(self, x), None


def _draft_wkv_layers(switch_on, monkeypatch):
    from torch.nn import Parameter

    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

    monkeypatch.setenv("SGLANG_OPT_DSV4_DENSE_GEMM_GRAN128", "1" if switch_on else "0")
    cfg = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[32, 32],
        scale_fmt="ue8m0",
    )
    weights, real = _draft_wkv_weights()
    layers = []
    for q, s in weights:
        layer = _Fp8Linear()
        layer.weight = Parameter(q.clone(), requires_grad=False)
        layer.weight_scale_inv = Parameter(s.clone(), requires_grad=False)
        layer.orig_dtype = torch.bfloat16
        layer.quant_method = Fp8LinearMethod(cfg)
        layer.quant_method.process_weights_after_loading(layer)
        layers.append(layer)
    return layers, real


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


# 1: draft step, 750: decode verify (bs 125 x 6), 2560: inside the (1536, 5120) gran-32
# window, 32728: a full prefill chunk
@pytest.mark.parametrize("m", [1, 750, 2560, 32728])
def test_dspark_commit_kv_proj_fused_gran128(m, monkeypatch):
    """The DSpark draft stacks its stages' wkv weights and calls the linear kernel itself
    (prefill KV injection and the decode verify epilogue). With the switch on, the stack
    must use the 128x128 scales the conversion left and match the switch-off output."""
    from sglang.kernels.ops.speculative.dspark import dspark_draft_model as ddm

    torch.manual_seed(m)
    x = torch.randn(m, 5120, device="cuda").bfloat16()
    ddm._STACKED_WEIGHT_CACHE.clear()
    off, _ = _draft_wkv_layers(False, monkeypatch)
    assert not any(getattr(layer, "dense_gran128", False) for layer in off)
    y_off = ddm.CommitKvProj.execute(main_x=x, wkv_linears=off)
    assert not getattr(ddm._stacked_wkv_weight(wkv_linears=off), "gran128", False)
    ddm._STACKED_WEIGHT_CACHE.clear()
    on, real = _draft_wkv_layers(True, monkeypatch)
    assert all(getattr(layer, "dense_gran128", False) for layer in on)
    assert ddm._fused_commit_kv_proj_supported(wkv_linears=on)
    y_on = ddm.CommitKvProj.execute(main_x=x, wkv_linears=on)
    stacked = ddm._stacked_wkv_weight(wkv_linears=on)
    assert stacked.gran128 and stacked.fp8_scale.shape == (1536 // 128, 5120 // 128)
    # (1536, 5120) has a gran-32 window, so the stack carries the expanded scale
    assert stacked.fp8_scale_32x is not None
    assert fp8_block32_gran128.prefer_gran32_kernel(m, 1536, 5120) == (
        2305 <= m <= 2816
    )
    # the per-linear path (Fp8LinearMethod.apply) on the same converted layers
    y_lin = ddm.commit_kv_proj(main_x=x, wkv_linears=on)
    assert len(y_on) == len(y_off) == 3
    for a, b, c in zip(y_on, y_off, y_lin):
        assert a.shape == b.shape == (m, 512)
        assert _rel(a, b) < 3e-3, (m, real, _rel(a, b))
        assert _rel(a, c) < 3e-3, (m, real, _rel(a, c))
    ddm._STACKED_WEIGHT_CACHE.clear()


@pytest.mark.parametrize("m", [528, 750])
def test_dspark_commit_kv_proj_fused_gran128_cuda_graph(m, monkeypatch):
    """The decoder runs the commit inside its verify CUDA graph (dspark_verify
    _static_epilogue): capture it and replay on new input."""
    from sglang.kernels.ops.speculative.dspark import dspark_draft_model as ddm

    ddm._STACKED_WEIGHT_CACHE.clear()
    on, _ = _draft_wkv_layers(True, monkeypatch)
    static_x = torch.randn(m, 5120, device="cuda").bfloat16()
    for _ in range(2):  # build the stack and JIT-compile outside the capture
        ddm.commit_kv_proj_fused(main_x=static_x, wkv_linears=on)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_static = ddm.commit_kv_proj_fused(main_x=static_x, wkv_linears=on)
    new_x = torch.randn(m, 5120, device="cuda").bfloat16()
    static_x.copy_(new_x)
    graph.replay()
    torch.cuda.synchronize()
    y_eager = ddm.commit_kv_proj_fused(main_x=new_x, wkv_linears=on)
    for a, b in zip(y_static, y_eager):
        assert torch.equal(a, b)
    ddm._STACKED_WEIGHT_CACHE.clear()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
