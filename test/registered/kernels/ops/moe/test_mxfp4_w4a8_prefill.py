"""SM90 W4A8 MXFP4 experts (SGLANG_OPT_DSV4_W4A8_PREFILL_MOE): exact FP8 weights, output
against fp32 references, and the side-stream prefetch against in-line conversion."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 only",
)

E, H, I, TOPK = 16, 1024, 192, 6
LIMIT = 10.0
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _weights(seed, spread=4):
    g = torch.Generator().manual_seed(seed)

    def one(n, k):
        w = torch.randint(-128, 128, (E, n, k // 2), generator=g, dtype=torch.int8)
        s = torch.randint(
            122 - spread, 123, (E, n, k // 32), generator=g, dtype=torch.uint8
        )
        return w.cuda(), s.view(torch.float8_e8m0fnu).cuda()

    return (*one(2 * I, H), *one(H, I))


def _dequant(w, s):
    lut = torch.tensor(E2M1 + [-v for v in E2M1], device=w.device)
    b = w.view(torch.uint8).long()
    v = torch.stack([lut[b & 0xF], lut[b >> 4]], -1).flatten(-2)
    return v * torch.exp2(s.view(torch.uint8).float() - 127).repeat_interleave(32, -1)


def _act_quant(x):
    # The model's own act_quant: FP8 e4m3 per 1x32 block with a power-of-2 scale.
    xf = x.float().unflatten(-1, (-1, 32))
    sc = torch.exp2(
        torch.ceil(torch.log2(xf.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448))
    )
    return ((xf / sc).to(torch.float8_e4m3fn).float() * sc).flatten(-2)


def _reference(w13, s13, w2, s2, x, ids, wts, native_fp8):
    out = torch.zeros(x.shape[0], H, device=x.device)
    xin = _act_quant(x) if native_fp8 else x.float()
    for e in range(E):
        rows, slot = (ids == e).nonzero(as_tuple=True)
        gu = (xin[rows] @ _dequant(w13[e], s13[e]).T).bfloat16().float()
        gate, up = gu[:, :I].clamp(max=LIMIT), gu[:, I:].clamp(-LIMIT, LIMIT)
        a = (torch.nn.functional.silu(gate) * up).bfloat16().float()
        if native_fp8:
            a = _act_quant(a)
        out.index_add_(0, rows, (a @ _dequant(w2[e], s2[e]).T) * wts[rows, slot, None])
    return out


def _inputs(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, H, device="cuda", generator=g).bfloat16()
    w, ids = torch.topk(torch.rand(rows, E, device="cuda", generator=g), TOPK, -1)
    return x, ids.int(), (w / w.sum(-1, keepdim=True)).float()


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_fp8_weights_are_exact():
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import _fold, _mxfp4_to_fp8

    for spread in (0, 4, 11):
        w13, s13, w2, s2 = _weights(spread, spread)
        for w, s in ((w13, s13), (w2, s2)):
            smax, channel_scale = _fold(s, "w")
            out = torch.empty(
                *w.shape[:-1], w.shape[-1] * 2, device="cuda", dtype=torch.float8_e4m3fn
            )
            _mxfp4_to_fp8(w, s, smax, out)
            assert torch.equal(out.float() * channel_scale[..., None], _dequant(w, s))


def test_fold_rejects_wide_exponent_spread():
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import _fold

    _, s13, _, _ = _weights(0, spread=12)
    with pytest.raises(ValueError):
        _fold(s13, "w13")


@pytest.mark.parametrize("rows", [1, 37, 2048])
def test_output_matches_references(rows):
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import (
        Mxfp4W4A8PrefillExperts,
    )

    w13, s13, w2, s2 = _weights(1)
    experts = Mxfp4W4A8PrefillExperts(w13, s13, w2, s2, LIMIT, prefetch=False)
    x, ids, wts = _inputs(rows, 2)
    ids[0, -1] = -1  # masked slot
    y = experts(x, ids, wts)
    native = _reference(w13, s13, w2, s2, x, ids, wts, native_fp8=True)
    bf16 = _reference(w13, s13, w2, s2, x, ids, wts, native_fp8=False)
    assert torch.isfinite(y).all()
    # Same numerics as the model's own W4A8 reference up to FP8 tensor-core accumulation.
    assert _rel(y, native) < 0.025
    # The FP8 activation quantization itself (both GEMM inputs) is ~4-5% vs BF16.
    assert _rel(y, bf16) < 0.08
    assert experts(x[:0], ids[:0], wts[:0]).shape == (0, H)


def test_prefetch_matches_inline_conversion():
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import (
        Mxfp4W4A8PrefillExperts,
    )

    layers = [_weights(10 + i) for i in range(3)]
    inline = [Mxfp4W4A8PrefillExperts(*w, LIMIT, prefetch=False) for w in layers]
    fetched = [Mxfp4W4A8PrefillExperts(*w, LIMIT, prefetch=True) for w in layers]
    x, ids, wts = _inputs(512, 3)
    expected = [inline[i](x, ids, wts) for i in range(3)]
    # In order twice (prefetch hits), then out of order (misses fall back to in-line).
    order = [0, 1, 2, 0, 1, 2, 2, 0, 0, 1]
    hits = 0
    for i in order:
        hits += fetched[i].scratch.holder is fetched[i]
        out = fetched[i](x, ids, wts)
        assert torch.equal(out, expected[i]), f"layer {i}"
    assert hits >= 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
