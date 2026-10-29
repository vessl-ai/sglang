"""INF-681 k-qcp: SGLANG_OPT_DSV4_CP_FUSED_MHC_POST_COMBINE lets prefill context
parallelism take the fused mHC post + next combine + RMSNorm kernel.

The gate in deepseek_v4_mhc._post_fusion used to exclude CP, so a CP4 prefill ran
mhc_post and the combine + norm separately even with SGLANG_DSV41_SM90_FUSED_MHC_POST_COMBINE
set. Checked at the CP4 rank row counts of the DeepSeek-V4.1 prefill (65,208-token step:
16,302 rows per rank; 4,096 the kernel's lower bound; a 1-request step: 1,728 rows, which
stays unfused):

* switch on, CP: the fused kernel runs, its output equals the non-CP fused path bit for bit,
  the rebuilt residual equals the unfused path bit for bit and the normed rows are within
  1 bf16 ulp of it (the kernel's documented rounding);
* switch off, CP: the unfused path, as before.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] not in (9, 10),
    reason="SM90 or Blackwell fused mHC post + combine",
)

HIDDEN, MULT = 5120, 4


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    exponent = torch.frexp(x.float().abs().clamp_min(2.0**-126))[1]
    return torch.ldexp(torch.ones_like(x, dtype=torch.float32), exponent - 8)


def _inputs(m: int):
    g = torch.Generator(device="cuda").manual_seed(1000 + m)
    residual = (torch.randn(m, MULT, HIDDEN, device="cuda", generator=g) * 3).bfloat16()
    y = (torch.randn(m, HIDDEN, device="cuda", generator=g) * 0.5).bfloat16()
    post = torch.rand(m, MULT, device="cuda", generator=g) * 2
    comb = torch.softmax(torch.randn(m, MULT, MULT, device="cuda", generator=g), -1)
    pre = torch.rand(m, MULT, device="cuda", generator=g)
    weight = (1 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=g)).bfloat16()
    return y, residual, (pre, post, comb), weight


def _layer(cp_prefill: bool, weight: torch.Tensor):
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models import deepseek_v4_mhc as model_mhc

    cfg = model_mhc.HcConfig(
        mult=MULT,
        sinkhorn_iters=20,
        eps=1e-6,
        rms_eps=1e-20,
        hidden=HIDDEN,
        pre_from_prev=True,
        cp_prefill=cp_prefill,
    )
    norm = RMSNorm(HIDDEN, eps=1e-20).cuda()
    norm.weight.data = weight.clone()
    unused = torch.empty(0, device="cuda")
    next_hc = model_mhc.HcSubLayer(cfg, unused, unused, unused, norm)
    boundary = model_mhc.make_boundary(norm, accepts_mxfp8=False, hc=next_hc)
    assert boundary.norm_fusable
    return model_mhc.HcSubLayer(cfg, unused, unused, unused, norm), boundary, next_hc


def _run(monkeypatch, m: int, cp_prefill: bool, switch: bool):
    import sglang.kernels.ops.layernorm.mhc as kernel_mhc
    from sglang.srt.environ import envs
    from sglang.srt.models import deepseek_v4_mhc as model_mhc

    monkeypatch.setattr(kernel_mhc, "is_dsa_prefill_cp_interleave", lambda: cp_prefill)
    monkeypatch.setattr(
        model_mhc, "get_parallel", lambda: SimpleNamespace(attn_dp_size=1)
    )
    y, residual, coefficients, weight = _inputs(m)
    hc, boundary, next_hc = _layer(cp_prefill, weight)
    with (
        envs.SGLANG_DSV41_SM90_FUSED_MHC_POST_COMBINE.override(True),
        envs.SGLANG_OPT_DSV4_CP_FUSED_MHC_POST_COMBINE.override(switch),
    ):
        state = model_mhc._post_fusion(hc, y, residual, coefficients, boundary)
    fused = isinstance(state.input, model_mhc.HcNormed)
    normed = model_mhc.combine(next_hc, state)
    # The unfused reference: the plain post, then the next hyper-connection's combine.
    pre, post, comb = coefficients
    ref_state = model_mhc.HcState(model_mhc.post(hc.cfg, y, residual, post, comb), pre)
    ref_normed = model_mhc.combine(next_hc, ref_state)
    return fused, state.residual, normed, ref_state.residual, ref_normed


@pytest.mark.parametrize("m", [4096, 16302])
def test_cp_takes_fused_kernel_and_matches(monkeypatch, m):
    fused, residual, normed, ref_residual, ref_normed = _run(monkeypatch, m, True, True)
    assert fused, "CP did not take the fused post + combine + norm kernel"
    assert torch.equal(residual, ref_residual)
    diff = (normed.float() - ref_normed.float()).abs()
    assert (diff <= _bf16_ulp(ref_normed)).all(), diff.max().item()
    assert (normed != ref_normed).float().mean().item() < 1e-4
    # CP and non-CP run the same kernel on the same rows: bit-identical outputs.
    fused_tp, residual_tp, normed_tp, _, _ = _run(monkeypatch, m, False, False)
    assert fused_tp
    assert torch.equal(residual, residual_tp) and torch.equal(normed, normed_tp)


@pytest.mark.parametrize("m", [1728, 16302])
def test_cp_without_switch_or_below_4096_stays_unfused(monkeypatch, m):
    switch = m < 4096  # below the kernel's range even the switch keeps the unfused path
    fused, residual, normed, ref_residual, ref_normed = _run(monkeypatch, m, True, switch)
    assert not fused
    assert torch.equal(residual, ref_residual) and torch.equal(normed, ref_normed)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
