"""SM80(A100) 판별 — FP8 텐서코어/Triton fp8e4nv 가 없는 Ampere(8.0~8.8)에서 DSv4 를 SM120 폴백 경로로 보내는 게이트."""
import functools
import os

import torch


@functools.cache
def is_sm80_dsv4() -> bool:
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    return (8, 0) <= torch.cuda.get_device_capability() < (8, 9)


@functools.cache
def sm80_sparse_prefill_ws() -> bool:
    """[0013] A100 에서 upstream sparse-prefill(bf16 워크스페이스, KV 1회 dequant) 경로 사용 — env SGLANG_SM80_SPARSE_PREFILL_WS=1, 기본 OFF."""
    return is_sm80_dsv4() and os.environ.get("SGLANG_SM80_SPARSE_PREFILL_WS", "0").lower() in ("1", "true")


@functools.cache
def sm80_sparse_prefill_union_g() -> int:
    """[0013] 인접 토큰 gather 공유 그룹 크기(1=끔, 2|4) — env SGLANG_SM80_SPARSE_PREFILL_UNION. 정렬+dedup 오버헤드가 있어 기본 1."""
    return int(os.environ.get("SGLANG_SM80_SPARSE_PREFILL_UNION", "1"))
