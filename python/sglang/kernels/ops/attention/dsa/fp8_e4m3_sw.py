"""Software e4m3fn codec for Triton kernels on CUDA GPUs without fp8 (SM80 family).

Triton rejects the ``fp8e4nv`` type itself below SM89 (``type fp8e4nv not supported
in this architecture`` at make_ir), so on those GPUs no fp8 tensor may reach a Triton
kernel: callers pass ``uint8`` views and kernels store the bytes from
:func:`fp8_store_value`. Elsewhere ``SW_FP8`` is False and the kernel stores the fp32
value exactly as before (hardware RNE satfinite convert).
"""

import triton
import triton.language as tl


@triton.jit
def f32_to_e4m3fn_u8(x):
    """fp32 -> e4m3fn byte, RNE, saturating to +-448, NaN -> 0x7F|sign.

    Bit-exact with torch ``.to(torch.float8_e4m3fn)`` (and the SM89+ convert) on
    [-448, 448], the only range the k-pool quantizers store.
    """
    fbits = x.to(tl.float32).to(tl.uint32, bitcast=True)
    sign8 = ((fbits >> 24) & 0x80).to(tl.uint8)
    mag = fbits & 0x7FFFFFFF
    is_nan = mag > 0x7F800000
    exp = (mag >> 23).to(tl.int32) - 127
    mant = (mag & 0x7FFFFF) | 0x800000
    # Normals keep 3 mantissa bits (drop 20); e4m3 subnormals (exp < -6) drop more.
    shift = tl.minimum(20 + tl.maximum(-6 - exp, 0), 30)
    keep = (mant >> shift).to(tl.uint32)
    rest = mant & ((1 << shift) - 1)
    half = (1 << (shift - 1)).to(tl.uint32)
    keep += ((rest > half) | ((rest == half) & ((keep & 1) == 1))).to(tl.uint32)
    # keep carries the implicit bit for normals, so the add-carry also handles
    # mantissa overflow into the exponent and subnormal -> min-normal promotion.
    val = tl.where(exp >= -6, ((exp + 6) << 3).to(tl.uint32) + keep, keep)
    val = tl.minimum(val, 0x7E)
    val = tl.where(is_nan, 0x7F, val)
    return val.to(tl.uint8) | sign8


@triton.jit
def fp8_store_value(x, SW_FP8: tl.constexpr):
    """Value to ``tl.store`` into an e4m3fn cache: bytes on SW_FP8, else x unchanged."""
    if SW_FP8:
        return f32_to_e4m3fn_u8(x)
    else:
        return x
