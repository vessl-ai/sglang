"""SM8x (A100 family): the predicate and the settings of the TileLang sparse MLA kernels.

Pure Python on purpose: the settings are validated at boot (arg resolution) and unit
tested without tilelang.
"""

import functools
import logging
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def is_sm80_family(device_index=None) -> bool:
    """True on CUDA GPUs with compute capability 8.0 to 8.8 (A100, A30, RTX 30xx).

    They have no fp8 MMA, no Triton fp8e4nv convert and no DeepGEMM build. SM89 and
    newer keep the upstream code paths.
    """
    if torch.version.hip is not None or not torch.cuda.is_available():
        return False
    return (8, 0) <= torch.cuda.get_device_capability(device_index) < (8, 9)


@functools.cache
def sm80_v1_cfg() -> Optional[Tuple[int, int, int]]:
    """(block_I, num_stages, threads) for the V1 prefill kernel from
    SGLANG_SM80_SPARSE_MLA_V1_CFG, or None to keep the kernel defaults."""
    field = envs.SGLANG_SM80_SPARSE_MLA_V1_CFG
    raw = field.get_strict()
    if not raw:
        return None
    try:
        cfg = tuple(int(x) for x in raw.split(","))
    except ValueError:
        cfg = ()
    if len(cfg) != 3 or min(cfg) <= 0:
        raise ValueError(
            f"{field.name} must be three positive integers "
            f"'block_I,num_stages,threads', got {raw!r}"
        )
    logger.info(
        "SM8x sparse MLA V1 prefill tile (block_I, num_stages, threads): %s", cfg
    )
    return cfg


@functools.cache
def decode_split_enabled() -> bool:
    """SGLANG_SM80_SPARSE_MLA_DECODE_SPLIT, parsed strictly."""
    return envs.SGLANG_SM80_SPARSE_MLA_DECODE_SPLIT.get_strict()


def validate_sm80_settings() -> None:
    """Boot-time check of the SM8x sparse MLA settings."""
    sm80_v1_cfg()
    decode_split_enabled()


# The V1 tile used for fp8 KV when SGLANG_SM80_SPARSE_MLA_V1_CFG is unset.
FP8KV_DEFAULT_V1_CFG = (32, 1, 128)


@functools.cache
def validate_sm80_fp8kv() -> None:
    """Boot-time check of the SM8x fp8 KV settings (the kernel factories check again).

    The packed converter writes shared memory outside the pipeline's sync, so its V1
    tile must not be pipelined.
    """
    configured = sm80_v1_cfg()
    cfg = configured or FP8KV_DEFAULT_V1_CFG
    if cfg[1] != 1:
        raise ValueError(
            f"fp8 KV on SM8x needs num_stages == 1 in "
            f"{envs.SGLANG_SM80_SPARSE_MLA_V1_CFG.name}, got {cfg}"
        )
    logger.info(
        "SM8x fp8 KV is decoded to bf16 in-kernel; V1 tile (block_I, num_stages, "
        "threads) = %s (%s)",
        cfg,
        envs.SGLANG_SM80_SPARSE_MLA_V1_CFG.name if configured else "fp8 KV default",
    )
