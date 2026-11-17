"""Fused RoPE and two-stage fp4 packing for low-ratio index keys and queries.

Keys add RMSNorm and a 68-byte cache store and use the group's first position;
queries have neither and use each token's own position.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

from .fp4_indexer import INDEX_K_SLOT_BYTES
from .utils import make_name

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

# `IndexQKernel::kBlockKernelMinRows` in fp4_indexer_rope.cuh: smaller calls keep
# the warp kernel, so the self-check skips them.
_BLOCK_KERNEL_MIN_ROWS = 8192
_block_checks_left = None


@functools.cache
def _index_q_block_path() -> bool:
    from sglang.srt.environ import envs

    return envs.SGLANG_HQ_INDEX_Q_BLOCK_KERNEL.get()


@cache_once
def _jit_index_k_module(
    head_dim: int, rope_dim: int, page_size: int, ratio: int
) -> Module:
    args = make_cpp_args(head_dim, rope_dim, page_size, ratio, is_arch_support_pdl())
    return load_jit(
        make_name("index_k_rope_pack"),
        *args,
        cuda_files=["deepseek_v4/fp4_indexer_rope.cuh"],
        cuda_wrappers=[("index_k", f"IndexKKernel<{args}>::run_index_k")],
    )


@cache_once
def _jit_index_q_module(head_dim: int, rope_dim: int, block_path: bool) -> Module:
    # Off, the template arguments and so the module are the original ones.
    extra = (True,) if block_path else ()
    args = make_cpp_args(head_dim, rope_dim, is_arch_support_pdl(), *extra)
    if block_path:
        logger.info(
            "hq INF-845 index-Q block kernel on: SM100 calls of >= %d rows run "
            "one thread per fp4 block",
            _BLOCK_KERNEL_MIN_ROWS,
        )
    return load_jit(
        make_name("index_q_rope_pack"),
        *args,
        cuda_files=["deepseek_v4/fp4_indexer_rope.cuh"],
        cuda_wrappers=[
            ("index_q_weights", f"IndexQKernel<{args}>::run_index_q_weights"),
        ],
    )


def index_k_norm_rope_pack_store(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    loc: torch.Tensor,
    cache: torch.Tensor,
    *,
    ratio: int,
) -> None:
    """Normalize, rotate, quantize twice and store one index-K slot per token.

    :param input: ``[num_tokens, index_head_dim]`` bf16 -- ``wk(latent)``,
                  *before* ``k_norm``.
    :param norm_weight: ``[index_head_dim]`` bf16, ``k_norm.weight``.
    :param eps: ``k_norm.eps``.
    :param freqs_cis: ``[max_pos, rope_head_dim]`` fp32, real/imag interleaved --
                      ``torch.view_as_real(freqs).flatten(-2)``. Indexed
                      in-kernel, so pass the whole table rather than a gather.
    :param positions: ``[num_tokens]`` int32 or int64, the token position. The
                      group position is masked out of it in-kernel.
    :param loc: ``[num_tokens]`` int64, the index-K slot. ``0`` is the reserved
                dummy; those rows publish nothing.
    :param cache: the layer's index-K buffer, ``[npages, page_size * 68]`` uint8.
    :param ratio: the layer's compress ratio. A power of two.

    .. note:: Two quantization stages, not one. The fake-quant's amax floor is
       ``6 * 2**-126`` and the packer's is ``1e-4``, applied on opposite sides
       of the divide by 6, so the packer can recover an exponent the fake-quant
       gave away and collapsing them is not equivalent.
    """
    head_dim = input.shape[-1]
    _jit_index_k_module(
        head_dim, freqs_cis.shape[-1], cache.shape[1] // INDEX_K_SLOT_BYTES, ratio
    ).index_k(input, norm_weight, freqs_cis, positions, loc, cache, float(eps))


def index_q_rope_pack_weights(
    input: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    head_weights: torch.Tensor,
    weight_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate, quantize twice and pack one indexer query per (token, head), plus
    the indexer's head weights, one launch.

    Head weights match ``head_weights(x).float()``: multiply in fp32,
    round to nearest-even bf16, then widen to fp32.

    :param head_weights: ``[num_tokens, heads]`` bf16, the raw ``weights_proj``
                         output (before the scale).
    :param weight_scale: ``softmax_scale * heads**-0.5``; rounded to fp32 in the
                         kernel exactly as torch rounds a Python scalar for a
                         bf16 tensor multiply.
    :return: ``(payload, scale, weights)`` -- ``[num_tokens * heads, index_head_dim // 2]``
             int8, ``[num_tokens * heads]`` int32 (the four ue8m0 block exponents
             packed little-endian) and ``[num_tokens, heads]`` fp32.
    """
    num_tokens, heads, head_dim = input.shape
    rows = num_tokens * heads
    payload = input.new_empty((rows, head_dim // 2), dtype=torch.int8)
    scale = input.new_empty((rows,), dtype=torch.int32)
    weights = input.new_empty((num_tokens, heads), dtype=torch.float32)

    block_path = _index_q_block_path()
    _jit_index_q_module(head_dim, freqs_cis.shape[-1], block_path).index_q_weights(
        input,
        freqs_cis,
        positions,
        payload,
        scale,
        head_weights,
        weights,
        float(weight_scale),
    )
    if block_path and rows >= _BLOCK_KERNEL_MIN_ROWS:
        _self_check_block_kernel(
            (input, freqs_cis, positions, head_weights, weight_scale),
            (payload, scale, weights),
        )
    return payload, scale, weights


def _self_check_block_kernel(args, outputs) -> None:
    """Rerun a block-kernel call on the warp kernel and log whether the bytes match,
    for the first ``SGLANG_HQ_INDEX_Q_BLOCK_KERNEL_CHECK`` calls."""
    global _block_checks_left
    if _block_checks_left is None:
        from sglang.srt.environ import envs

        _block_checks_left = envs.SGLANG_HQ_INDEX_Q_BLOCK_KERNEL_CHECK.get()
    if _block_checks_left <= 0 or torch.cuda.is_current_stream_capturing():
        return
    _block_checks_left -= 1
    input, freqs_cis, positions, head_weights, weight_scale = args
    reference = tuple(torch.empty_like(t) for t in outputs)
    _jit_index_q_module(input.shape[-1], freqs_cis.shape[-1], False).index_q_weights(
        input,
        freqs_cis,
        positions,
        reference[0],
        reference[1],
        head_weights,
        reference[2],
        float(weight_scale),
    )
    same = {
        name: torch.equal(got, want)
        for name, got, want in zip(("payload", "scale", "weights"), outputs, reference)
    }
    logger.info(
        "hq INF-845 index-Q block kernel self-check rows=%d %s: %s",
        outputs[1].numel(),
        "match" if all(same.values()) else "MISMATCH",
        same,
    )
