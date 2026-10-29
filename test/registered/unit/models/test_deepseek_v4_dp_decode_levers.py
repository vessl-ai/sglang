"""INF-681 k-dnm switches for the DeepSeek-V4.1 DP-attention decoder (all default off).

- SGLANG_OPT_DSV4_WO_A_STRIDED_BMM: the token-major strided bmm is bit-identical to the
  einsum it replaces, for the DP4 group count (8) and the TP4 one (2).
- SGLANG_OPT_DSV4_DP_DECODE_MHC_FUSED: at DP-attention decode rows the split-H post is
  bit-identical to the TileLang post, and the fused combine + RMSNorm stays within 1 bf16
  ulp of combine then RMSNorm; the gates do not move without the switch, without
  attention DP, or above the row bounds.
- SGLANG_OPT_DSV4_DP_DECODE_STATS_STREAM: the stats-stream gate opens for DP verify rows
  only with the switch on.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90 DP-attention decode gates",
)

HIDDEN, MULT = 5120, 4


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    exponent = torch.frexp(x.float().abs().clamp_min(2.0**-126))[1]
    return torch.ldexp(torch.ones_like(x, dtype=torch.float32), exponent - 8)


@pytest.fixture
def dp4(monkeypatch):
    """attention DP 4 for the mHC gates, without a process group."""
    import sglang.srt.models.deepseek_v4_mhc as hc

    monkeypatch.setattr(hc, "get_parallel", lambda: SimpleNamespace(attn_dp_size=4))
    return hc


@pytest.mark.parametrize("groups", [8, 2])
@pytest.mark.parametrize("tokens", [7, 152, 192, 256])
def test_wo_a_strided_bmm_bit_identical(groups, tokens, monkeypatch):
    from sglang.srt.environ import envs
    from sglang.srt.models.deepseek_v4 import _apply_wo_a_bf16_matmul

    g = torch.Generator(device="cuda").manual_seed(tokens * 31 + groups)
    o = torch.randn(tokens, groups * 4096, device="cuda", generator=g).bfloat16()
    o = o.view(tokens, groups, 4096)
    wo_a = (torch.randn(groups, 1024, 4096, device="cuda", generator=g) * 0.02).bfloat16()
    kwargs = dict(is_decode=False, is_target_verify=True, fast_path=True)
    with envs.SGLANG_OPT_DSV4_WO_A_STRIDED_BMM.override(False):
        ref = _apply_wo_a_bf16_matmul(o, wo_a, **kwargs)
    with envs.SGLANG_OPT_DSV4_WO_A_STRIDED_BMM.override(True):
        out = _apply_wo_a_bf16_matmul(o, wo_a, **kwargs)
    assert out.shape == (tokens, groups, 1024)
    assert out.is_contiguous()
    assert torch.equal(out, ref)
    # wo_b reads the flattened rows; the strided result needs no copy for that.
    assert out.flatten(1).data_ptr() == out.data_ptr()


@pytest.mark.parametrize("m", [65, 128, 192, 200])
def test_split_h_post_bit_identical_at_dp_rows(m, dp4, monkeypatch):
    import sglang.kernels.ops.layernorm.mhc as mhc
    from sglang.srt.environ import envs

    monkeypatch.setattr(mhc, "is_dsa_prefill_cp_interleave", lambda: False)
    cfg = dp4.HcConfig(
        mult=MULT, sinkhorn_iters=20, eps=1e-6, rms_eps=1e-20, hidden=HIDDEN,
        pre_from_prev=True, cp_prefill=False,
    )
    g = torch.Generator(device="cuda").manual_seed(m)
    residual = (torch.randn(m, MULT, HIDDEN, device="cuda", generator=g) * 3).bfloat16()
    x = (torch.randn(m, HIDDEN, device="cuda", generator=g) * 0.5).bfloat16()
    post = torch.rand(m, MULT, device="cuda", generator=g) * 2
    comb = torch.softmax(torch.randn(m, MULT, MULT, device="cuda", generator=g), -1)
    calls = []
    real = dp4.mhc_post_split_h
    monkeypatch.setattr(dp4, "mhc_post_split_h", lambda *a, **k: calls.append(1) or real(*a, **k))
    with envs.SGLANG_OPT_DSV4_DP_DECODE_MHC_FUSED.override(False):
        ref = dp4.post(cfg, x, residual, post, comb)
    assert not calls
    with envs.SGLANG_OPT_DSV4_DP_DECODE_MHC_FUSED.override(True):
        out = dp4.post(cfg, x, residual, post, comb)
    assert calls == [1]
    assert torch.equal(out, ref)


def test_split_h_post_gate_bounds(dp4, monkeypatch):
    import sglang.kernels.ops.layernorm.mhc as mhc
    from sglang.srt.environ import envs

    monkeypatch.setattr(mhc, "is_dsa_prefill_cp_interleave", lambda: False)
    cfg = dp4.HcConfig(
        mult=MULT, sinkhorn_iters=20, eps=1e-6, rms_eps=1e-20, hidden=HIDDEN,
        pre_from_prev=True, cp_prefill=False,
    )
    calls = []
    monkeypatch.setattr(dp4, "mhc_post_split_h", lambda *a, **k: calls.append(1))
    m = dp4._DP_DECODE_SPLIT_H_POST_MAX_ROWS + 1
    residual = torch.zeros(m, MULT, HIDDEN, device="cuda", dtype=torch.bfloat16)
    x = torch.zeros(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
    post = torch.zeros(m, MULT, device="cuda")
    comb = torch.zeros(m, MULT, MULT, device="cuda")
    with envs.SGLANG_OPT_DSV4_DP_DECODE_MHC_FUSED.override(True):
        dp4.post(cfg, x, residual, post, comb)
    assert not calls
    # without attention DP the gate stays closed at DP-like row counts
    monkeypatch.setattr(dp4, "get_parallel", lambda: SimpleNamespace(attn_dp_size=1))
    with envs.SGLANG_OPT_DSV4_DP_DECODE_MHC_FUSED.override(True):
        dp4.post(cfg, x[:128], residual[:128], post[:128], comb[:128])
    assert not calls


@pytest.mark.parametrize("m", [97, 152, 192, 512])
def test_fused_combine_norm_at_dp_rows(m, dp4, monkeypatch):
    from sglang.srt.environ import envs
    from sglang.srt.layers.layernorm import RMSNorm

    g = torch.Generator(device="cuda").manual_seed(m)
    norm = RMSNorm(HIDDEN, eps=1e-20).cuda()
    norm.weight.data = (1 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=g)).bfloat16()
    hc = SimpleNamespace(cfg=SimpleNamespace(mult=MULT, hidden=HIDDEN), norm=norm)
    residual = (torch.randn(m, MULT, HIDDEN, device="cuda", generator=g) * 3).bfloat16()
    pre = torch.rand(m, MULT, device="cuda", generator=g)
    state = SimpleNamespace(residual=residual, pre=pre)
    fused = []
    import sglang.kernels.ops.layernorm.hc_combine_norm as hcn

    real = hcn.hc_combine_norm
    monkeypatch.setattr(hcn, "hc_combine_norm", lambda *a, **k: fused.append(1) or real(*a, **k))
    with envs.SGLANG_OPT_DSV4_DP_DECODE_MHC_FUSED.override(False):
        ref = dp4._combine(hc, state, None)
    assert not fused
    with envs.SGLANG_OPT_DSV4_DP_DECODE_MHC_FUSED.override(True):
        out = dp4._combine(hc, state, None)
    assert fused == [1]
    diff = (out.float() - ref.float()).abs()
    assert bool((diff <= _bf16_ulp(ref)).all()), float((diff / _bf16_ulp(ref)).max())


def test_stats_stream_gate(dp4):
    from sglang.srt.environ import envs

    cfg = SimpleNamespace(pre_from_prev=True)
    verify = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: False, is_target_verify=lambda: True))
    extend = SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: False, is_target_verify=lambda: False))
    x = torch.empty(192, MULT, HIDDEN, device="cuda", dtype=torch.bfloat16)
    with envs.SGLANG_OPT_DSV4_DP_DECODE_STATS_STREAM.override(False):
        assert not dp4.use_stats_stream(cfg, verify, x)
    with envs.SGLANG_OPT_DSV4_DP_DECODE_STATS_STREAM.override(True):
        assert dp4.use_stats_stream(cfg, verify, x)
        assert not dp4.use_stats_stream(cfg, extend, x)
        big = torch.empty(513, MULT, HIDDEN, device="cuda", dtype=torch.bfloat16)
        assert not dp4.use_stats_stream(cfg, verify, big)
