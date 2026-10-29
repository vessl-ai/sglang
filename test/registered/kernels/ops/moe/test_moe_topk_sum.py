"""moe_topk_sum: out[M, K] = in[M, topk, K].sum(dim=1), including M > 65535.

The kernel used to launch gridDim.y = M and failed with "invalid argument" from
65,536 rows (a 65536-row prefill chunk); rows now stride over a capped grid.
"""

import pytest
import torch

from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference(x: torch.Tensor) -> torch.Tensor:
    # The kernel accumulates the topk rows in fp32 in order j = 0..topk-1.
    acc = torch.zeros(x.shape[0], x.shape[2], dtype=torch.float32, device=x.device)
    for j in range(x.shape[1]):
        acc += x[:, j, :].float()
    return acc.to(torch.bfloat16)


@pytest.mark.parametrize(
    "m,topk,k",
    [
        (1, 6, 5120),
        (7, 3, 3584),
        (32728, 6, 5120),
        (65535, 6, 5120),
        (65536, 6, 5120),
        (65537, 2, 512),
        (131073, 6, 1024),
    ],
)
def test_topk_sum_matches_reference(m, topk, k):
    torch.manual_seed(m)
    x = torch.randn(m, topk, k, dtype=torch.bfloat16, device="cuda")
    out = torch.full((m, k), float("nan"), dtype=torch.bfloat16, device="cuda")
    moe_topk_sum(x, out)
    torch.cuda.synchronize()
    ref = _reference(x)
    # Same accumulation order in fp32, so the bf16 results are bit-identical.
    assert torch.equal(out, ref)


def test_topk_sum_empty():
    x = torch.empty(0, 6, 5120, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(0, 5120, dtype=torch.bfloat16, device="cuda")
    moe_topk_sum(x, out)
