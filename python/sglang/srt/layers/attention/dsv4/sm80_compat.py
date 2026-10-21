"""SM80(A100) 판별 — FP8 텐서코어/Triton fp8e4nv 가 없는 Ampere(8.0~8.8)에서 DSv4 를 SM120 폴백 경로로 보내는 게이트."""
import functools

import torch


@functools.cache
def is_sm80_dsv4() -> bool:
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    return (8, 0) <= torch.cuda.get_device_capability() < (8, 9)
