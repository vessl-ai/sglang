# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    marlin_make_workspace,
    marlin_permute_scales,
    should_use_atomic_add_reduce,
)
from sglang.srt.layers.quantization.utils import get_scalar_types
from sglang.srt.utils import is_cuda
from sglang.srt.utils.custom_op import register_custom_op

_is_cuda = is_cuda()
if _is_cuda:
    from sglang.kernels.ops.gemm.gptq_marlin import gptq_marlin_gemm
    from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack

ScalarType, scalar_types = get_scalar_types()

logger = logging.getLogger(__name__)


def fp8_fused_exponent_bias_into_scales(scales):
    fp8_exponent = 4
    if scales.dtype == torch.half:
        target_exponent = 5
    elif scales.dtype == torch.bfloat16:
        target_exponent = 8
    # exponent_bias_fp16 = 2 ** 4 - 2 ** 3 = 8
    # exponent_bias_bf16 = 2 ** 7 - 2 ** 3 = 120
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp8_exponent - 1)
    s = torch.ones_like(scales) * 2
    s = s**exponent_bias
    return scales * s


def fake_apply_fp8_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor],
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    out_shape = input.shape[:-1] + (size_n,)
    fake_output = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    return fake_output


@register_custom_op(fake_impl=fake_apply_fp8_marlin_linear)
def apply_fp8_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: Optional[torch.Tensor],
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
) -> torch.Tensor:
    # For GPUs that lack FP8 hardware support, we can leverage the
    # Marlin kernel for fast weight-only FP8 quantization

    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n,)

    use_atomic_add = should_use_atomic_add_reduce(
        m=reshaped_x.size(0), n=size_n, k=size_k, device=input.device, dtype=input.dtype
    )

    output = gptq_marlin_gemm(
        a=reshaped_x,
        c=None,
        b_q_weight=weight,
        b_scales=weight_scale,
        global_scale=None,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=workspace,
        b_q_type=scalar_types.float8_e4m3fn,
        size_m=reshaped_x.size(0),
        size_n=size_n,
        size_k=size_k,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
    )

    if bias is not None:
        output.add_(bias)

    return output.reshape(out_shape)


def _regroup_fp8_block32_to_128(layer: torch.nn.Module) -> bool:
    """[32,32] ue8m0 블록 스케일 → [32,128] 그룹 (SM80 Marlin FP8 은 group 32 커널이 없다)."""
    wbs = getattr(layer, "weight_block_size", None)
    if wbs is None or wbs[1] != 32:
        return False
    sname = "weight_scale" if "weight_scale" in dir(layer) else "weight_scale_inv"
    w = layer.weight.data
    n, k = w.shape
    sf = getattr(layer, sname).data.to(torch.float32)
    nb, kb = sf.shape
    assert n == nb * wbs[0] and k == kb * 32, (w.shape, sf.shape, wbs)
    import os
    mode = os.environ.get("SGLANG_SM80_FP8_DENSE", "regroup")  # regroup | g32 | bf16
    if mode == "g32" and k % 64 == 0 and n % 64 == 0:
        return False  # [32,32] 그대로: gptq_marlin.cuh 에 kFE4M3fn group32 인스턴스 추가(아래 CUH 패치)
    # DSpark draft main_proj: FP8(재그룹 포함)이면 accept≈1 (yaleyoou 선례) → bf16 dequant 강제. 토글 SGLANG_SM80_DSPARK_MAIN_PROJ_BF16(기본 0)
    force_bf16 = getattr(layer, "_sm80_force_bf16", False) and os.environ.get("SGLANG_SM80_DSPARK_MAIN_PROJ_BF16", "0") == "1"
    if mode in ("g32", "bf16") or force_bf16 or kb % 4 != 0 or n % 64 != 0:
        # Marlin 불가 형상(예: shared expert w2 K=288) 또는 bf16 arm: bf16 로 dequant(2의 거듭제곱 스케일이라 정확) 후 F.linear.
        wf = w.to(torch.float32).view(nb, wbs[0], kb, 32) * sf[:, None, :, None]
        layer.weight.data = wf.reshape(n, k).to(torch.bfloat16)
        layer._sm80_bf16_dequant = True
        return True
    # [C1 hybrid] keep an exact bf16 copy (dequantised with the ORIGINAL [32,32] scales, 2^k so lossless in bf16) next to the
    # Marlin-regrouped weight; Fp8LinearMethod.apply uses it for rows >= SGLANG_R2_DENSE_BF16_MIN_TOKENS (prefill chunks).
    # Cost: +2 B/param of every Marlin-served dense layer. Round-1 arm-BF measured fp8->bf16 replacement = +1.2 GB/GPU
    # (50.11 -> 51.33), so keeping BOTH copies is about +2.4 GB/GPU; at ~1.6 KB KV per token per GPU (7.16M tokens in
    # 11.5 GB, sm80-triage.md:71) the KV pool shrinks by roughly 2.4 GB / 1.6 KB = 1.5M tokens (-14% of 10.8M).
    # Lossless vs the checkpoint: the current regroup path is already within 1e-7 rel. (c/c1_regroup_stats.py), so this is a speed lever.
    _min_tok = int(os.environ.get("SGLANG_R2_DENSE_BF16_MIN_TOKENS", "0"))
    if _min_tok > 0:
        layer._r2_w_bf16 = (
            (w.to(torch.float32).view(nb, wbs[0], kb, 32) * sf[:, None, :, None]).reshape(n, k).to(torch.bfloat16)
        )
        layer._r2_bf16_min_tokens = _min_tok
    s4 = sf.view(nb, kb // 4, 4)
    smax = s4.max(dim=-1, keepdim=True).values
    ratio = s4 / smax
    out = torch.empty_like(w)
    step = max(1, (1 << 24) // max(1, wbs[0] * k))
    for r in range(0, nb, step):
        e = min(nb, r + step)
        wf = w[r * wbs[0] : e * wbs[0]].to(torch.float32).view(e - r, wbs[0], kb // 4, 4, 32)
        wf = wf * ratio[r:e, None, :, :, None]
        out[r * wbs[0] : e * wbs[0]] = wf.reshape(-1, k).to(w.dtype)
    layer.weight.data = out
    setattr(layer, sname, torch.nn.Parameter(smax.squeeze(-1).contiguous(), requires_grad=False))
    layer.weight_block_size = [wbs[0], 128]
    return False


def prepare_fp8_layer_for_marlin(
    layer: torch.nn.Module, size_k_first: bool = True
) -> None:
    logger.warning_once(
        "Your GPU does not have native support for FP8 computation but "
        "FP8 quantization is being used. Weight-only FP8 compression will "
        "be used leveraging the Marlin kernel. This may degrade "
        "performance for compute-heavy workloads."
    )

    if _regroup_fp8_block32_to_128(layer):
        return
    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    weight_block_size = getattr(layer, "weight_block_size", None)

    if size_k_first:
        assert layer.weight.shape == (part_size_k, part_size_n)
    else:
        assert layer.weight.shape == (part_size_n, part_size_k)

    device = layer.weight.device

    # WORKSPACE
    layer.workspace = marlin_make_workspace(device)

    # WEIGHT
    # Repack weights to marlin format
    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = pack_fp8_to_int32(layer.weight, size_k_first)
    if not size_k_first:
        qweight = qweight.T.contiguous()

    marlin_qweight = gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=8,
    )
    layer.weight = torch.nn.Parameter(marlin_qweight, requires_grad=False)

    # WEIGHT SCALES
    # Permute scales
    if "weight_scale" in dir(layer):
        scales = layer.weight_scale.to(layer.orig_dtype)
    elif "weight_scale_inv" in dir(layer):
        scales = layer.weight_scale_inv.to(layer.orig_dtype)
        del layer.weight_scale_inv

    group_size = -1 if weight_block_size is None else weight_block_size[1]

    # marlin kernel only support channel-wise and group-wise quantization
    # we need to convert the scales
    if weight_block_size is None:
        if scales.nelement() == 1:
            # tensor-wise quantization -> channel-wise quantization
            # (1, 1) =>(repeat)=> (1, size_n)
            scales = scales.view(1, 1).repeat_interleave(part_size_n, 1)
        elif scales.nelement() > 1 and scales.nelement() != part_size_n:
            assert part_size_n % scales.nelement() == 0
            s_size = scales.nelement()
            # tensor-wise quantization (for gate-up proj)
            #     -> channel-wise quantization
            # (1, s_size) =>(repeat)=> (1, size_n)
            scales = scales.view(1, s_size)
            scales = scales.repeat_interleave(part_size_n // s_size, 1)
        else:
            # channel-wise quantization
            # (1, size_n)
            scales = scales.view(1, part_size_n)
    else:
        # block-wise quantization -> group-wise quantization
        # (size_k // block_size[1], ceil(size_n / block_size[0]))
        #  =>(repeat)=> (size_k // block_size[1], size_n)
        if not size_k_first:
            scales = scales.T.contiguous()
        block_n = weight_block_size[0]
        scales = scales.repeat_interleave(block_n, 1)
        # size_n may not divisible by block_size[0]
        scales = scales[:, :part_size_n]

    marlin_scales = marlin_permute_scales(
        s=scales, size_k=part_size_k, size_n=part_size_n, group_size=group_size
    )
    marlin_scales = fp8_fused_exponent_bias_into_scales(marlin_scales)
    layer.weight_scale = torch.nn.Parameter(marlin_scales, requires_grad=False)

    # The dense FP8 Marlin wrapper adds bias after the kernel returns, so the
    # bias must remain in logical output-channel order. Only scales need the
    # Marlin tile permutation.
    if hasattr(layer, "bias") and layer.bias is not None:
        assert layer.bias.shape == (part_size_n,)
        layer.bias = torch.nn.Parameter(layer.bias.detach(), requires_grad=False)


def pack_fp8_to_int32(
    fp8_tensor: torch.Tensor, size_k_first: bool = True
) -> torch.Tensor:
    """
    Repack FP8 weights to gptq format (packed int32 elements)
    """
    assert fp8_tensor.dtype == torch.float8_e4m3fn
    assert fp8_tensor.ndim == 2

    fp8_tensor = fp8_tensor.T if size_k_first else fp8_tensor
    fp8_tensor = fp8_tensor.contiguous()
    # fp8_tensor is contiguous and have shape (N, K) now
    # with `.view(torch.int32)`, it become (N, K // 4)
    int32_tensor = fp8_tensor.view(torch.int32)
    return int32_tensor.T.contiguous() if size_k_first else int32_tensor
