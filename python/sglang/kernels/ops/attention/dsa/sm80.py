"""SM8x (A100 family) predicate shared by the k-pool indexer and the TileLang kernels."""

from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def is_sm80_family(device_index=None) -> bool:
    """True on CUDA GPUs with compute capability 8.0 to 8.8 (A100, A30, RTX 30xx).

    They have no fp8 MMA, no Triton fp8e4nv convert and no DeepGEMM build. SM89 and
    newer keep the upstream code paths.
    """
    if torch.version.hip is not None or not torch.cuda.is_available():
        return False
    return (8, 0) <= torch.cuda.get_device_capability(device_index) < (8, 9)
