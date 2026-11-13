"""K=128-tile w4a8 grouped GEMM for GLM-5.3's TP-4 MoE shapes on H200.

K=128 tiles read scales packed 1-way where the sgl_kernel wheel's K=512 tiles
read them 4-way, so the weight loader and this GEMM must agree on which shapes
take this path: both ask `uses_k128_tiles`.

The extension is JIT-compiled with ``torch.utils.cpp_extension`` on first use
and cached under ``TORCH_EXTENSIONS_DIR``.
"""

import importlib.util
import os
import re

import torch

from sglang.kernels.jit.utils import cache_once

_SHAPES = {(1024, 4096), (4096, 512)}


def uses_k128_tiles(n: int, k: int, device: torch.device) -> bool:
    """Whether an [E, N, K] w4a8 weight runs here, and so needs 1-way scales."""
    return (n, k) in _SHAPES and bool(
        re.search(r"(?:^| )H200(?: |$)", torch.cuda.get_device_name(device))
    )


@cache_once
def _ext():
    from sglang.srt.utils.cpp_extension_loader import load_extension_with_recovery

    kernels = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    aot = os.path.join(kernels, "aot")
    # The w4a8 collective builders need the CUTLASS 4.2 API; flashinfer's
    # bundled 4.5 headers fail to build them, DeepGEMM's 4.2.1 ones build.
    cutlass = os.path.join(
        importlib.util.find_spec("deep_gemm").submodule_search_locations[0], "include"
    )
    includes = [
        f"-I{p}"
        for p in (
            os.path.join(aot, "csrc"),
            os.path.join(aot, "csrc", "moe", "cutlass_moe", "w4a8"),
            os.path.join(aot, "include"),
            cutlass,
        )
    ]
    return load_extension_with_recovery(
        name="w4a8_h200_k128",
        sources=[os.path.join(kernels, "jit", "csrc", "moe", "w4a8_h200_k128.cu")],
        extra_cflags=["-O3", *includes],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-gencode=arch=compute_90a,code=sm_90a",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "-DNDEBUG",
            "--use_fast_math",
            *includes,
        ],
    )


def cutlass_w4a8_moe_mm(
    d: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    experts_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    a_strides: torch.Tensor,
    b_strides: torch.Tensor,
    d_strides: torch.Tensor,
    s_strides: torch.Tensor,
    chunk_size: int = 128,
    topk: int = 8,
) -> None:
    """`sgl_kernel.cutlass_w4a8_moe_mm`, sending the shapes `uses_k128_tiles` names here."""
    if uses_k128_tiles(d.size(1), a.size(1), d.device):
        _ext().w4a8_moe_mm_k128(
            d, a, b, a_scales, b_scales, experts_offsets, problem_sizes,
            a_strides, b_strides, d_strides, s_strides, chunk_size, topk,
        )
        return
    from sgl_kernel import cutlass_w4a8_moe_mm as _wheel_mm

    _wheel_mm(
        d, a, b, a_scales, b_scales, experts_offsets, problem_sizes,
        a_strides, b_strides, d_strides, s_strides, chunk_size, topk,
    )
