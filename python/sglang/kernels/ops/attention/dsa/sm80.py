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
