import torch
import triton
import triton.language as tl

_GROUP = 128
_FP8_MAX = 448.0


@triton.jit
def _pack_rows_fp8_kernel(
    x_ptr,
    packed_ptr,
    stride_x,
    stride_packed,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    groups = tl.arange(0, K // GROUP)
    cols = tl.arange(0, GROUP)
    offs = groups[:, None] * GROUP + cols[None, :]
    x = tl.load(x_ptr + row * stride_x + offs).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10)
    # Power-of-two scales, so a block-128 UE8M0 consumer requantizes the
    # unpacked rows without further rounding.
    scale = tl.exp2(tl.ceil(tl.log2(amax / FP8_MAX)))
    q = (x / scale[:, None]).to(tl.float8e4nv)
    row_ptr = packed_ptr + row * stride_packed
    tl.store(row_ptr + offs, q.to(tl.uint8, bitcast=True))
    tl.store((row_ptr + K).to(tl.pointer_type(tl.float32)) + groups, scale)


@triton.jit
def _unpack_rows_fp8_kernel(
    packed_ptr,
    out_ptr,
    stride_packed,
    stride_out,
    K: tl.constexpr,
    GROUP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    groups = tl.arange(0, K // GROUP)
    cols = tl.arange(0, GROUP)
    offs = groups[:, None] * GROUP + cols[None, :]
    row_ptr = packed_ptr + row * stride_packed
    q = tl.load(row_ptr + offs).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    scale = tl.load((row_ptr + K).to(tl.pointer_type(tl.float32)) + groups)
    out = q * scale[:, None]
    tl.store(out_ptr + row * stride_out + offs, out.to(out_ptr.dtype.element_ty))


def pack_rows_fp8(x: torch.Tensor) -> torch.Tensor:
    """[rows, K] -> [rows, K + 4 * K // 128] uint8: e4m3 values, then one
    fp32 scale per 128 columns."""
    rows, k = x.shape
    assert k % _GROUP == 0 and x.stride(-1) == 1
    packed = torch.empty(
        (rows, k + 4 * (k // _GROUP)), dtype=torch.uint8, device=x.device
    )
    if rows:
        _pack_rows_fp8_kernel[(rows,)](
            x,
            packed,
            x.stride(0),
            packed.stride(0),
            K=k,
            GROUP=_GROUP,
            FP8_MAX=_FP8_MAX,
        )
    return packed


def unpack_rows_fp8(packed: torch.Tensor, out: torch.Tensor) -> None:
    rows, k = out.shape
    assert packed.shape == (rows, k + 4 * (k // _GROUP)) and out.stride(-1) == 1
    if rows:
        _unpack_rows_fp8_kernel[(rows,)](
            packed, out, packed.stride(0), out.stride(0), K=k, GROUP=_GROUP
        )
