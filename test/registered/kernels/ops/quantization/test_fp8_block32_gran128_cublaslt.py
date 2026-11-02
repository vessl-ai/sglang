"""INF-681 SGLANG_OPT_DSV4_GRAN128_CUBLAS_ABOVE_ROWS: above the row threshold (and up to each shape's last winning row
count) the listed 128x128-requantized linears run cuBLASLt's blockwise FP8 kernel and give exactly DeepGEMM's bits;
outside that window, and for shapes not in the list, DeepGEMM keeps running."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90 DeepGEMM / cuBLASLt blockwise FP8",
)


def _weight(n, k, seed):
    from sglang.srt.layers.quantization import fp8_block32_gran128 as g128

    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
    exp = torch.randint(-12, -6, (n // 32, k // 32), device="cuda", generator=g)
    return g128.requant_block32_to_block128(w, torch.exp2(exp.float()))


@pytest.mark.parametrize("n,k", [(5120, 8192), (4608, 5120)])
@pytest.mark.parametrize("m", [260, 288, 320])
def test_cublaslt_matches_deepgemm_above_threshold(n, k, m):
    import deep_gemm

    from sglang.srt.environ import envs
    from sglang.srt.layers.quantization import fp8_block32_gran128 as g128

    if deep_gemm.get_num_sms() != 132:
        pytest.skip("rows/shape table measured on 132-SM H200")
    w, s = _weight(n, k, m)
    x = torch.randn(m, k, device="cuda").bfloat16()
    with envs.SGLANG_OPT_DSV4_GRAN128_CUBLAS_ABOVE_ROWS.override(0):
        assert not g128.prefer_cublaslt_blockwise(m, n, k)
        ref = g128.block128_fp8_linear(x, w, s)
    with envs.SGLANG_OPT_DSV4_GRAN128_CUBLAS_ABOVE_ROWS.override(256):
        assert g128.prefer_cublaslt_blockwise(m, n, k)
        out = g128.block128_fp8_linear(x, w, s)
    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    assert torch.equal(out, ref)


def test_gate_keeps_deepgemm():
    from sglang.srt.environ import envs
    from sglang.srt.layers.quantization import fp8_block32_gran128 as g128

    with envs.SGLANG_OPT_DSV4_GRAN128_CUBLAS_ABOVE_ROWS.override(256):
        assert not g128.prefer_cublaslt_blockwise(256, 5120, 8192)  # at the threshold
        assert not g128.prefer_cublaslt_blockwise(257, 5120, 8192)  # M % 4 != 0
        assert not g128.prefer_cublaslt_blockwise(288, 32768, 1280)  # wq_b: not listed
        assert not g128.prefer_cublaslt_blockwise(288, 1792, 5120)  # wqkv_a: not listed


def test_gate_upper_bound_keeps_deepgemm():
    import deep_gemm

    from sglang.srt.environ import envs
    from sglang.srt.layers.quantization import fp8_block32_gran128 as g128

    if deep_gemm.get_num_sms() != 132:
        pytest.skip("rows/shape table measured on 132-SM H200")
    with envs.SGLANG_OPT_DSV4_GRAN128_CUBLAS_ABOVE_ROWS.override(256):
        assert g128.prefer_cublaslt_blockwise(320, 5120, 8192)  # wo_b: last winning M
        assert not g128.prefer_cublaslt_blockwise(324, 5120, 8192)  # DeepGEMM fast again
        assert not g128.prefer_cublaslt_blockwise(400, 5120, 8192)  # cuBLASLt slower
        assert g128.prefer_cublaslt_blockwise(384, 4608, 5120)  # gate_up: last winning M
        assert not g128.prefer_cublaslt_blockwise(388, 4608, 5120)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
