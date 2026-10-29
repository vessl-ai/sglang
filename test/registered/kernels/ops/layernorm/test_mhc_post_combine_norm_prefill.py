"""The fused mHC post + next combine + RMSNorm prefill kernel against the unfused path
it replaces (mhc_post, then hc_combine_norm): the rebuilt residual is identical and the
normed output within 1 bf16 ulp. On SM90 this is the opt-in
SGLANG_DSV41_SM90_FUSED_MHC_POST_COMBINE path; its dispatch gate keeps the unfused
path in batch-invariant mode."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

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


@pytest.mark.parametrize("m", [4096, 9441, 16384])
def test_fused_matches_unfused(m, monkeypatch):
    import sglang.kernels.ops.layernorm.mhc as mhc
    from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm
    from sglang.kernels.ops.layernorm.mhc_post_combine_norm_prefill import (
        mhc_post_combine_norm_prefill,
    )

    monkeypatch.setattr(mhc, "is_dsa_prefill_cp_interleave", lambda: False)
    g = torch.Generator(device="cuda").manual_seed(m)
    residual = (torch.randn(m, MULT, HIDDEN, device="cuda", generator=g) * 3).bfloat16()
    x = (torch.randn(m, HIDDEN, device="cuda", generator=g) * 0.5).bfloat16()
    post = torch.rand(m, MULT, device="cuda", generator=g) * 2
    comb = torch.softmax(torch.randn(m, MULT, MULT, device="cuda", generator=g), -1)
    pre = torch.rand(m, MULT, device="cuda", generator=g)
    weight = (1 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=g)).bfloat16()
    eps = 1e-20

    ref_residual = mhc.mhc_post(x, residual, post, comb)
    ref_normed = hc_combine_norm(ref_residual.flatten(1), pre, weight, eps)
    got_residual, got_normed = mhc_post_combine_norm_prefill(
        x, residual, post, comb, pre, weight, eps
    )
    assert torch.equal(got_residual, ref_residual)
    diff = (got_normed.float() - ref_normed.float()).abs()
    assert (diff <= _bf16_ulp(ref_normed)).all(), diff.max().item()
    mismatch = (got_normed != ref_normed).float().mean().item()
    assert mismatch < 1e-4, mismatch


def test_sm90_gate_respects_switch_and_batch_invariant_mode(monkeypatch):
    from sglang.srt.environ import envs
    from sglang.srt.models import deepseek_v4_mhc

    for sm90, switch, invariant, expected in (
        (True, True, False, True),
        (True, True, True, False),
        (True, False, False, False),
        (False, True, False, False),
    ):
        with (
            patch.object(
                deepseek_v4_mhc,
                "get_platform",
                return_value=SimpleNamespace(is_sm90=sm90),
            ),
            patch(
                "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
                return_value=invariant,
            ),
            envs.SGLANG_DSV41_SM90_FUSED_MHC_POST_COMBINE.override(switch),
        ):
            assert deepseek_v4_mhc._sm90_fused_post_combine() is expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
