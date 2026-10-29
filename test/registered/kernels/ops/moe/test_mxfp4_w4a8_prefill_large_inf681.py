"""INF-681 k-pmoe: large-step switches of the SM90 W4A8 MXFP4 prefill experts. Every switch must give the
same bits as the reference path (SGLANG_OPT_DSV4_W4A8_W2_SORTED_SUM on, the others off):
- SGLANG_OPT_DSV4_W4A8_FUSED_SWIGLU: interleaved gate/up FP8 copy + SwiGLU in the gate/up GEMM epilogue;
- SGLANG_OPT_DSV4_W4A8_COUNTING_ROUTE: counting-sort routing;
- SGLANG_OPT_DSV4_W4A8_W2_TWO_CTA_MIN_ROWS: down-projection with 128x128 tiles, 2 CTAs per SM;
- SGLANG_OPT_DSV4_W4A8_FUSED_SHARED_ADD: shared expert output added inside the top-k sum, equal to a
  separate shared.add_(routed); plus the hand-over API of Mxfp4MarlinMoEMethod."""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 only",
)
pytest.importorskip("triton_kernels")

E, H, I, TOPK = 16, 1024, 192, 6
LIMIT = 10.0

SWITCHES = {
    "swiglu": dict(fused_swiglu=True),
    "count": dict(counting_route=True),
    # w2cta applies from 128 rows per expert on average (rows 2048 and 9000 here)
    "w2cta": dict(w2_two_cta_min_rows=1),
    "add": dict(fused_shared_add=True),
    "all": dict(
        fused_swiglu=True,
        counting_route=True,
        w2_two_cta_min_rows=1,
        fused_shared_add=True,
    ),
}


def _weights(seed, spread=4):
    g = torch.Generator().manual_seed(seed)

    def one(n, k):
        w = torch.randint(-128, 128, (E, n, k // 2), generator=g, dtype=torch.int8)
        s = torch.randint(
            122 - spread, 123, (E, n, k // 32), generator=g, dtype=torch.uint8
        )
        return w.cuda(), s.view(torch.float8_e8m0fnu).cuda()

    return (*one(2 * I, H), *one(H, I))


def _inputs(rows, seed, skew=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, H, device="cuda", generator=g).bfloat16()
    score = torch.rand(rows, E, device="cuda", generator=g)
    if skew:  # a few hot experts, some cold ones
        score = score * torch.linspace(4.0, 0.1, E, device="cuda")
    w, ids = torch.topk(score, TOPK, -1)
    return x, ids.int(), (w / w.sum(-1, keepdim=True)).float()


def _experts(w, **kw):
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import (
        Mxfp4W4A8PrefillExperts,
    )

    return Mxfp4W4A8PrefillExperts(*w, LIMIT, prefetch=False, sorted_sum=True, **kw)


def test_interleaved_conversion_is_a_row_permutation():
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import _fold, _mxfp4_to_fp8

    w13, s13, _, _ = _weights(3)
    smax, _ = _fold(s13, "w13")
    plain = torch.empty(E, 2 * I, H, device="cuda", dtype=torch.float8_e4m3fn)
    il = torch.empty_like(plain)
    _mxfp4_to_fp8(w13, s13, smax, plain, 1)
    for launches in (1, 3):
        _mxfp4_to_fp8(w13, s13, smax, il, launches, interleave_half=I)
        want = torch.stack([plain[:, :I], plain[:, I:]], 2).reshape(plain.shape)
        assert torch.equal(il.view(torch.uint8), want.view(torch.uint8))


@pytest.mark.parametrize("name", list(SWITCHES))
@pytest.mark.parametrize("rows", [1, 37, 2048, 9000])
@pytest.mark.parametrize("skew", [False, True])
def test_switch_matches_reference_bit_for_bit(name, rows, skew):
    w = _weights(11)
    x, ids, wts = _inputs(rows, 12, skew)
    ref = _experts(w)(x, ids, wts)
    got = _experts(w, **SWITCHES[name])(x, ids, wts)
    assert torch.equal(got, ref)


@pytest.mark.parametrize("name", ["base"] + list(SWITCHES))
@pytest.mark.parametrize("rows", [37, 2048])
def test_add_to_equals_separate_add(name, rows):
    w = _weights(21)
    x, ids, wts = _inputs(rows, 22)
    shared = (torch.randn(rows, H, device="cuda") * 0.05).bfloat16()
    want = shared.clone().add_(_experts(w)(x, ids, wts))
    target = shared.clone()
    got = _experts(w, **SWITCHES.get(name, {}))(x, ids, wts, add_to=target)
    assert got.data_ptr() == target.data_ptr()
    assert torch.equal(got, want)


def test_add_to_on_the_direct_and_scatter_paths():
    """The direct small-step path fuses the add too; the scattered-sum path adds separately.
    Both return shared + routed with the bits of a separate shared.add_(routed)."""
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import (
        Mxfp4W4A8PrefillExperts,
    )

    w = _weights(31)
    x, ids, wts = _inputs(512, 32)
    shared = (torch.randn(512, H, device="cuda") * 0.05).bfloat16()
    for kw in (dict(small_step_rows=1024), dict(sorted_sum=False)):
        ex = Mxfp4W4A8PrefillExperts(
            *w, LIMIT, prefetch=False, fused_shared_add=True, **kw
        )
        want = shared.clone().add_(ex(x, ids, wts))
        assert torch.equal(ex(x, ids, wts, add_to=shared.clone()), want)


def test_masked_slots_counting_route():
    """Masked slots (id -1) get no row in the counting sort; the reference routes them to expert 0
    with weight 0. The sums agree (a zero contribution either way)."""
    w = _weights(41)
    x, ids, wts = _inputs(300, 42)
    ids[::7, -1] = -1
    ref = _experts(w)(x, ids, wts)
    got = _experts(w, counting_route=True)(x, ids, wts)
    assert torch.equal(got, ref)


def test_count_route_layout():
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import _count_route

    _, ids, wts = _inputs(1000, 51, skew=True)
    r = _count_route(ids, wts, E)
    flat = ids.flatten().long()
    assert torch.equal(r.expert_counts.long(), torch.bincount(flat, minlength=E))
    # every slot has one row; rows of expert e are contiguous and in expert order
    assert torch.equal(
        torch.sort(r.pos.long()).values, torch.arange(flat.numel(), device="cuda")
    )
    row_expert = torch.empty_like(flat)
    row_expert[r.pos.long()] = flat
    assert torch.all(row_expert[1:] >= row_expert[:-1])
    slot = torch.arange(flat.numel(), device="cuda")
    assert torch.equal(r.gather_indx.long()[r.pos.long()], slot // TOPK)
    assert torch.equal(r.sorted_weights[r.pos.long()], wts.flatten())


def test_method_shared_add_handover(monkeypatch):
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    monkeypatch.setenv("SGLANG_OPT_DSV4_W4A8_PREFILL_MOE", "1")
    method = Mxfp4MarlinMoEMethod(
        SimpleNamespace(process_weights_after_loading=lambda layer: None), "test"
    )
    experts = SimpleNamespace(should_fuse_routed_scaling_factor_in_topk=True)
    assert not method.accepts_shared_add(experts)  # no W4A8 experts prepared yet
    method._w4a8 = SimpleNamespace(fused_shared_add=True)
    assert method.accepts_shared_add(experts)
    assert not method.accepts_shared_add(
        SimpleNamespace(should_fuse_routed_scaling_factor_in_topk=False)
    )
    method._w4a8 = SimpleNamespace(fused_shared_add=False)
    assert not method.accepts_shared_add(experts)
    t = torch.zeros(1)
    method.set_pending_shared_add(t)
    assert method.take_unconsumed_shared_add() is t
    assert method.take_unconsumed_shared_add() is None


def test_late_layer_sm90_w4a8_backend(monkeypatch):
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    monkeypatch.setenv("SGLANG_OPT_DSV4_W4A8_PREFILL_MOE", "1")
    monkeypatch.setenv("SGLANG_OPT_DSV4_W4A8_LATE_W4A16_BACKEND", "sm90_w4a8")
    method = Mxfp4MarlinMoEMethod(
        SimpleNamespace(process_weights_after_loading=lambda layer: None),
        "model.layers.25.mlp.experts",
    )
    assert method._use_w4a8 and not method._sm90_decode_w4a8
    method._use_late_w4a16()
    got = (
        method._use_w4a8,
        method._use_triton_kernels,
        method._triton_kernels_decode,
        method._sm90_decode_kernel,
        method._sm90_decode_w4a8,
    )
    assert got == (False, False, False, False, True)
