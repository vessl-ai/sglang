"""INF-874: SGLANG_OPT_DSV4_MXFP4_CONVERT_I32, the MXFP4 -> FP8 word conversion kernel with
int32 index arithmetic. It must store the same bytes at the same addresses as the int64
kernel: plain and gate/up-interleaved (FUSED_SWIGLU) layouts, one or several launches,
odd sizes, a multi-launch size with many programs; and the experts built with the switch
must give the same output bits as without it."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 only",
)
pytest.importorskip("triton_kernels")

H, I, TOPK, LIMIT = 1024, 192, 6, 10.0


def _packed(e, n, k, spread, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(-128, 128, (e, n, k // 2), generator=g, dtype=torch.int8)
    w.view(-1)[:256] = torch.arange(-128, 128, dtype=torch.int8)
    s = torch.randint(
        122 - spread, 123, (e, n, k // 32), generator=g, dtype=torch.uint8
    )
    return w.cuda(), s.view(torch.float8_e8m0fnu).cuda()


# (experts, rows, K, exponent spread): the unit-test shapes, odd sizes, and one with
# 16.8M words (16,384 programs) so a 16-launch split has full-sized slices.
SHAPES = [
    (16, 2 * I, H, 11),
    (16, H, I, 4),
    (3, 6, 64, 0),
    (5, 10, 96, 11),
    (32, 1024, 4096, 6),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("num_launches", [1, 3, 16])
@pytest.mark.parametrize("interleave", [False, True])
def test_int32_index_kernel_matches_int64(shape, num_launches, interleave):
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import _fold, _mxfp4_to_fp8

    e, n, k, spread = shape
    if interleave and n % 2:
        pytest.skip("interleave needs an even row count")
    w, s = _packed(e, n, k, spread, seed=e * n + k)
    smax, _ = _fold(s, "w")
    half = n // 2 if interleave else 0
    ref = torch.full((e, n, k), 0x7F, device="cuda", dtype=torch.uint8)
    got = torch.full_like(ref, 0x7F)
    _mxfp4_to_fp8(
        w, s, smax, ref.view(torch.float8_e4m3fn), num_launches, interleave_half=half
    )
    _mxfp4_to_fp8(
        w,
        s,
        smax,
        got.view(torch.float8_e4m3fn),
        num_launches,
        interleave_half=half,
        index_i32=True,
    )
    assert torch.equal(got, ref)


def test_int32_index_kernel_matches_byte_kernel():
    """Against the INF-674 byte kernel (the reference of the word kernel) on the plain layout."""
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import _fold, _mxfp4_to_fp8

    w, s = _packed(16, 2 * I, H, 11, seed=7)
    smax, _ = _fold(s, "w")
    ref = torch.empty(16, 2 * I, H, device="cuda", dtype=torch.float8_e4m3fn)
    got = torch.empty_like(ref)
    _mxfp4_to_fp8(w, s, smax, ref, 0)
    _mxfp4_to_fp8(w, s, smax, got, 4, index_i32=True)
    assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8))


@pytest.mark.parametrize("rows", [37, 2048, 9000])
@pytest.mark.parametrize("prefetch_chunks", [0, 16])
def test_experts_with_int32_conversion_match(rows, prefetch_chunks):
    """The i10 prefill switch set (fused SwiGLU, counting route, two-CTA W2, sorted sum), with
    and without the sliced prefetch, three layers sharing one scratch: same output bits."""
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import (
        Mxfp4W4A8PrefillExperts,
    )

    def weights(seed):
        return (*_packed(16, 2 * I, H, 4, seed), *_packed(16, H, I, 4, seed + 1))

    kw = dict(
        prefetch=prefetch_chunks > 0,
        prefetch_chunks=prefetch_chunks,
        sorted_sum=True,
        fused_swiglu=True,
        counting_route=True,
        w2_two_cta_min_rows=1,
    )
    layers = [weights(60 + 2 * i) for i in range(3)]
    base = [Mxfp4W4A8PrefillExperts(*w, LIMIT, **kw) for w in layers]
    i32 = [
        Mxfp4W4A8PrefillExperts(*w, LIMIT, convert_index_i32=True, **kw) for w in layers
    ]
    g = torch.Generator(device="cuda").manual_seed(rows)
    x = torch.randn(rows, H, device="cuda", generator=g).bfloat16()
    score = torch.rand(rows, 16, device="cuda", generator=g)
    wts, ids = torch.topk(score, TOPK, -1)
    ids, wts = ids.int(), (wts / wts.sum(-1, keepdim=True)).float()
    for i in [0, 1, 2, 0, 2, 1]:
        assert torch.equal(i32[i](x, ids, wts), base[i](x, ids, wts)), f"layer {i}"
