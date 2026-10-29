"""SM90 W4A8 MXFP4 experts (SGLANG_OPT_DSV4_W4A8_PREFILL_MOE): exact FP8 weights, output
against fp32 references, the side-stream prefetch against in-line conversion (also with
either stream delayed, so a missing event wait would corrupt the output), the refusal
under CUDA graph capture, and the switch precedence in Mxfp4MarlinMoEMethod.
INF-681: the word conversion kernel (bit-identical to the byte kernel, also sliced), the
sliced prefetch after the top-k sum (SGLANG_DSV4_W4A8_PREFETCH_CHUNKS) and the late-layer
W4A16 selection (SGLANG_OPT_DSV4_W4A8_LATE_W4A16_FROM_LAYER)."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=150, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 only",
)
pytest.importorskip("triton_kernels")

# ~25 ms on H200: far longer than one layer's conversion or GEMMs at these shapes.
_DELAY_CYCLES = 50_000_000

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


def _run_prefetch_sequence(fetched, inline, x, ids, wts):
    expected = [inline[i](x, ids, wts) for i in range(len(inline))]
    order = [0, 1, 2, 0, 1, 2, 2, 0, 0, 1]
    for i in order:
        out = fetched[i](x, ids, wts)
        assert torch.equal(out, expected[i]), f"layer {i}"


@pytest.mark.parametrize("delayed", ["side_stream", "current_stream"])
def test_prefetch_orders_streams_under_delay(delayed, monkeypatch):
    """side_stream: every side-stream conversion starts late, so a GEMM that did not
    wait on `ready` would read the previous layer's weights. current_stream: every
    layer's GEMMs start late, so a prefetch that did not wait on `free` would overwrite
    the buffers they are about to read."""
    from sglang.srt.layers.quantization import mxfp4_w4a8_prefill as mod

    layers = [_weights(20 + i) for i in range(3)]
    inline = [mod.Mxfp4W4A8PrefillExperts(*w, LIMIT, prefetch=False) for w in layers]
    fetched = [mod.Mxfp4W4A8PrefillExperts(*w, LIMIT, prefetch=True) for w in layers]
    x, ids, wts = _inputs(512, 4)
    if delayed == "side_stream":
        convert = mod.Mxfp4W4A8PrefillExperts._convert_into_scratch

        def slow_convert(self, *args):
            if torch.cuda.current_stream() == self.scratch.stream:
                torch.cuda._sleep(_DELAY_CYCLES)
            convert(self, *args)

        monkeypatch.setattr(
            mod.Mxfp4W4A8PrefillExperts, "_convert_into_scratch", slow_convert
        )
    else:
        row_quant = mod._row_quant

        def slow_row_quant(x):
            out = row_quant(x)
            torch.cuda._sleep(_DELAY_CYCLES)
            return out

        monkeypatch.setattr(mod, "_row_quant", slow_row_quant)
    _run_prefetch_sequence(fetched, inline, x, ids, wts)


@pytest.mark.parametrize("prefetch", [False, True])
def test_refuses_cuda_graph_capture(prefetch):
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import (
        Mxfp4W4A8PrefillExperts,
    )

    experts = Mxfp4W4A8PrefillExperts(*_weights(30), LIMIT, prefetch=prefetch)
    x, ids, wts = _inputs(64, 5)
    experts(x, ids, wts)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="does not support CUDA graphs"):
        with torch.cuda.graph(graph):
            experts(x, ids, wts)


@pytest.mark.parametrize(
    "w4a8,triton,decode,expected",
    [
        ("1", "1", "1", (True, False, False)),
        ("1", "0", "0", (True, False, False)),
        ("0", "1", "0", (False, True, False)),
        ("0", "0", "1", (False, True, True)),
        ("0", "0", "0", (False, False, False)),
    ],
)
def test_switch_precedence(w4a8, triton, decode, expected, monkeypatch):
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    monkeypatch.setenv("SGLANG_OPT_DSV4_W4A8_PREFILL_MOE", w4a8)
    monkeypatch.setenv("SGLANG_OPT_DSV4_TRITON_KERNELS_MOE", triton)
    monkeypatch.setenv("SGLANG_OPT_DSV4_TRITON_KERNELS_MOE_DECODE", decode)
    method = Mxfp4MarlinMoEMethod(
        SimpleNamespace(process_weights_after_loading=lambda layer: None), "test"
    )
    got = (
        method._use_w4a8,
        method._use_triton_kernels,
        method._triton_kernels_decode,
    )
    assert got == expected


def test_rejects_non_dsv4_experts(monkeypatch):
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    monkeypatch.setenv("SGLANG_OPT_DSV4_W4A8_PREFILL_MOE", "1")
    method = Mxfp4MarlinMoEMethod(
        SimpleNamespace(process_weights_after_loading=lambda layer: None), "test"
    )
    method.hidden_pad = 0
    method.runner = SimpleNamespace(
        config=SimpleNamespace(
            activation="silu", is_gated=True, gemm1_alpha=None, swiglu_limit=None
        )
    )
    with pytest.raises(ValueError, match="clamped SwiGLU"):
        method._prepare_w4a8_weights(torch.nn.Module())


@pytest.mark.parametrize("num_launches", [1, 3, 7])
def test_word_kernel_matches_byte_kernel(num_launches):
    """Every packed byte value, every exponent shift d in [-5, 6], odd tensor sizes."""
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import _fold, _mxfp4_to_fp8

    g = torch.Generator().manual_seed(40 + num_launches)
    for n, k, spread in ((2 * I, H, 11), (H, I, 4), (3, 64, 0), (5, 96, 11)):
        e = 3
        w = torch.randint(-128, 128, (e, n, k // 2), generator=g, dtype=torch.int8)
        w.view(-1)[:256] = torch.arange(-128, 128, dtype=torch.int8)
        s = torch.randint(
            122 - spread, 123, (e, n, k // 32), generator=g, dtype=torch.uint8
        )
        w, s = w.cuda(), s.view(torch.float8_e8m0fnu).cuda()
        smax, _ = _fold(s, "w")
        ref = torch.empty(e, n, k, device="cuda", dtype=torch.float8_e4m3fn)
        got = torch.full_like(ref.view(torch.uint8), 0x7F).view(torch.float8_e4m3fn)
        _mxfp4_to_fp8(w, s, smax, ref, 0)
        _mxfp4_to_fp8(w, s, smax, got, num_launches)
        assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8)), (n, k)


@pytest.mark.parametrize("rows", [512, 8192])
def test_sliced_prefetch_matches_inline_conversion(rows):
    """rows 8192: the prefetch is sliced and issued after the top-k sum; rows 512: one
    word-kernel launch before it. Both must give the in-line byte-kernel output."""
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import (
        _OVERLAP_MIN_ROWS,
        Mxfp4W4A8PrefillExperts,
    )

    assert (rows >= _OVERLAP_MIN_ROWS) == (rows == 8192)
    layers = [_weights(50 + i) for i in range(3)]
    inline = [Mxfp4W4A8PrefillExperts(*w, LIMIT, prefetch=False) for w in layers]
    fetched = [
        Mxfp4W4A8PrefillExperts(*w, LIMIT, prefetch=True, prefetch_chunks=4)
        for w in layers
    ]
    x, ids, wts = _inputs(rows, 6)
    expected = [inline[i](x, ids, wts) for i in range(3)]
    hits = 0
    for i in [0, 1, 2, 0, 1, 2, 2, 0, 0, 1]:
        hits += fetched[i].scratch.holder is fetched[i]
        assert torch.equal(fetched[i](x, ids, wts), expected[i]), f"layer {i}"
    assert hits >= 4


@pytest.mark.parametrize("delayed", ["side_stream", "current_stream"])
def test_sliced_prefetch_orders_streams_under_delay(delayed, monkeypatch):
    """As test_prefetch_orders_streams_under_delay, for the sliced prefetch issued after
    the top-k sum (the `free` event is then recorded after it)."""
    from sglang.srt.layers.quantization import mxfp4_w4a8_prefill as mod

    layers = [_weights(60 + i) for i in range(3)]
    inline = [mod.Mxfp4W4A8PrefillExperts(*w, LIMIT, prefetch=False) for w in layers]
    fetched = [
        mod.Mxfp4W4A8PrefillExperts(*w, LIMIT, prefetch=True, prefetch_chunks=4)
        for w in layers
    ]
    x, ids, wts = _inputs(8192, 7)
    if delayed == "side_stream":
        convert = mod.Mxfp4W4A8PrefillExperts._convert_into_scratch

        def slow_convert(self, *args):
            if torch.cuda.current_stream() == self.scratch.stream:
                torch.cuda._sleep(_DELAY_CYCLES)
            convert(self, *args)

        monkeypatch.setattr(
            mod.Mxfp4W4A8PrefillExperts, "_convert_into_scratch", slow_convert
        )
    else:
        row_quant = mod._row_quant

        def slow_row_quant(x):
            out = row_quant(x)
            torch.cuda._sleep(_DELAY_CYCLES)
            return out

        monkeypatch.setattr(mod, "_row_quant", slow_row_quant)
    _run_prefetch_sequence(fetched, inline, x, ids, wts)


@pytest.mark.parametrize("rows", [1, 37, 2048])
def test_sorted_sum_matches_scattered_sum(rows):
    """SGLANG_OPT_DSV4_W4A8_W2_SORTED_SUM: same output bit for bit, masked slots too."""
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import (
        Mxfp4W4A8PrefillExperts,
    )

    w = _weights(70)
    ref = Mxfp4W4A8PrefillExperts(*w, LIMIT, prefetch=False)
    got = Mxfp4W4A8PrefillExperts(*w, LIMIT, prefetch=False, sorted_sum=True)
    x, ids, wts = _inputs(rows, 8)
    ids[0, -1] = -1
    assert torch.equal(got(x, ids, wts), ref(x, ids, wts))


@pytest.mark.parametrize(
    "start,prefix,bounded,expected",
    [
        (21, "model.layers.25.mlp.experts", True, True),
        (21, "model.layers.21.mlp.experts", True, True),
        (21, "model.layers.20.mlp.experts", True, False),
        (21, "model.stages.0.mlp.experts", True, False),  # DSpark draft stage
        (21, "model.layers.25.mlp.experts", False, False),  # no bounded replay
        (-1, "model.layers.25.mlp.experts", True, False),  # switch off
    ],
)
def test_late_layer_w4a16_selection(start, prefix, bounded, expected, monkeypatch):
    import sglang.srt.runtime_context as runtime_context
    from sglang.srt.layers.quantization import mxfp4_marlin_moe as mod

    monkeypatch.setenv("SGLANG_OPT_DSV4_W4A8_LATE_W4A16_FROM_LAYER", str(start))
    features = SimpleNamespace(enable_decoder_swa_bounded_replay=bounded)
    monkeypatch.setattr(
        runtime_context, "get_exec", lambda: SimpleNamespace(features=features)
    )
    assert mod._is_late_w4a16_layer(prefix) == expected


@pytest.mark.parametrize(
    "backend,expected",
    [
        ("triton", (False, True, True, False)),
        ("marlin", (False, False, False, False)),
        ("sm90", (False, False, False, True)),
    ],
)
def test_late_layer_w4a16_backend(backend, expected, monkeypatch):
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    monkeypatch.setenv("SGLANG_OPT_DSV4_W4A8_PREFILL_MOE", "1")
    monkeypatch.setenv("SGLANG_OPT_DSV4_W4A8_LATE_W4A16_BACKEND", backend)
    method = Mxfp4MarlinMoEMethod(
        SimpleNamespace(process_weights_after_loading=lambda layer: None),
        "model.layers.25.mlp.experts",
    )
    assert method._use_w4a8
    method._use_late_w4a16()
    got = (
        method._use_w4a8,
        method._use_triton_kernels,
        method._triton_kernels_decode,
        method._sm90_decode_kernel,
    )
    assert got == expected
    monkeypatch.setenv("SGLANG_OPT_DSV4_W4A8_LATE_W4A16_BACKEND", "other")
    with pytest.raises(ValueError, match="'triton', 'marlin' or 'sm90'"):
        method._use_late_w4a16()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
