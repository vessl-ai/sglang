"""SM80 용 e4m3fn 소프트웨어 변환 (Triton). SW_FP8 이 False(SM89+)이면 호출되지 않는다."""
import torch
import triton
import triton.language as tl


def _sw_needed() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability() < (8, 9)
    )


SW_FP8 = tl.constexpr(_sw_needed())


@triton.jit
def e4m3fn_to_f32(b):
    """uint8/int e4m3fn 바이트 -> float32 (NaN 코드 0x7F/0xFF 는 정상값으로 디코드되지 않음 — ckpt/KV 에 없음)."""
    b = b.to(tl.int32) & 0xFF
    e = (b >> 3) & 15
    m = (b & 7).to(tl.float32)
    pow2 = ((e + 120) << 23).to(tl.float32, bitcast=True)  # 2**(e-7)
    v = tl.where(e == 0, m * 0.001953125, (1.0 + m * 0.125) * pow2)
    return tl.where((b & 0x80) != 0, -v, v)


@triton.jit
def f32_to_e4m3fn(x):
    """float32 -> e4m3fn 바이트(uint8), RNE, |x|>448 은 448 로 포화 (torch .to(float8_e4m3fn) 와 동일)."""
    sign = (x.to(tl.int32, bitcast=True) >> 31) & 1
    ax = tl.minimum(tl.abs(x), 448.0)
    bits = ax.to(tl.int32, bitcast=True)
    r = bits + 0x7FFFF + ((bits >> 20) & 1)
    e = (r >> 23) - 120
    normal = (e << 3) | ((r >> 20) & 7)
    sub = ((ax * 512.0 + 12582912.0) - 12582912.0).to(tl.int32)
    code = tl.where(e >= 1, normal, sub)
    return ((sign << 7) | code).to(tl.uint8)
