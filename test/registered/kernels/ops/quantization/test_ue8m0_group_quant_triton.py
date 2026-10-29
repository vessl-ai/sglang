import itertools

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from sglang.kernels.ops.quantization.fp8_kernel import (  # noqa: E402
    sglang_per_token_group_quant_fp8,
)
from sglang.kernels.ops.quantization.ue8m0_group_quant_triton import (  # noqa: E402
    ue8m0_group_quant_fp8,
)


@pytest.mark.parametrize(
    "m,k", list(itertools.product([1, 7, 130, 4096], [576, 1280, 2048, 5120, 6144]))
)
def test_matches_cuda_v2_bitwise(m, k):
    torch.manual_seed(m * 7 + k)
    x = torch.randn(m, k, device="cuda") * torch.exp(
        torch.randn(m, 1, device="cuda") * 3
    )
    x[:, :32] = 0  # an all-zero group takes the eps floor
    x[0, 40] = 3e4  # the largest group amax (scale 2^7; ceil scaling never clamps it)
    x = x.bfloat16()
    q_ref, s_ref = sglang_per_token_group_quant_fp8(x, 32, scale_ue8m0=True)
    q, s = ue8m0_group_quant_fp8(x, 32)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(s, s_ref)


@pytest.mark.parametrize("group,k", [(128, 1280), (128, 5120), (32, 2048)])
def test_matches_cuda_v2_group_sizes_and_strided_rows(group, k):
    # The switch applies to every UE8M0 model: group 128 (DeepSeek-V3.x blocks) too.
    # The Triton kernel also takes a row-strided view; v2 needs contiguous input.
    torch.manual_seed(group + k)
    wide = torch.randn(130, k + 256, device="cuda").bfloat16()
    x = wide[:, :k]
    q_ref, s_ref = sglang_per_token_group_quant_fp8(
        x.contiguous(), group, scale_ue8m0=True
    )
    q, s = ue8m0_group_quant_fp8(x, group)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(s, s_ref)
