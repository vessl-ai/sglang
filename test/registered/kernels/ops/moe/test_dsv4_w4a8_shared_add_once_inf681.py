"""INF-681 int4: the two shared-expert-add fusions of the W4A8 prefill experts add the shared output once.

SGLANG_OPT_DSV4_MOE_SHARED_ADD_IN_TOPK_SUM (k-qcp, vessl-ai/sglang#111) and SGLANG_OPT_DSV4_W4A8_FUSED_SHARED_ADD
(k-pmoe, #112) fold the same `shared + routed` into the top-k sum. With both switches set the k-qcp one runs and the
k-pmoe hand-over stays off; if both were armed on one call, the experts refuse instead of adding twice. k-qcp's fold
must also work under k-pmoe's counting route, whose routing has no combine_indx."""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 only",
)
pytest.importorskip("triton_kernels")

E, H, I, TOPK = 16, 1024, 192, 6
LIMIT = 10.0


def _weights(seed):
    g = torch.Generator().manual_seed(seed)

    def one(n, k):
        w = torch.randint(-128, 128, (E, n, k // 2), generator=g, dtype=torch.int8)
        s = torch.randint(118, 123, (E, n, k // 32), generator=g, dtype=torch.uint8)
        return w.cuda(), s.view(torch.float8_e8m0fnu).cuda()

    return (*one(2 * I, H), *one(H, I))


def _inputs(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, H, device="cuda", generator=g).bfloat16()
    score = torch.rand(rows, E, device="cuda", generator=g)
    w, ids = torch.topk(score, TOPK, -1)
    shared = (torch.randn(rows, H, device="cuda", generator=g) * 0.05).bfloat16()
    return x, ids.int(), (w / w.sum(-1, keepdim=True)).float(), shared


def _experts(w, **kw):
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import (
        Mxfp4W4A8PrefillExperts,
    )

    return Mxfp4W4A8PrefillExperts(*w, LIMIT, prefetch=False, sorted_sum=True, **kw)


def _arm_qcp(runner, shared, alpha=1.0):
    # what dsv4_prefill_qcp.arm_shared_add leaves on the runner
    runner.qcp_shared_add = (shared, alpha)
    runner.qcp_shared_add_taken = False


@pytest.mark.parametrize(
    "kw",
    [
        dict(),
        dict(counting_route=True),
        dict(counting_route=True, fused_swiglu=True, w2_two_cta_min_rows=1),
        dict(small_step_rows=4096),
    ],
    ids=["sorted", "count", "count-all", "direct"],
)
@pytest.mark.parametrize("rows", [37, 2048])
def test_qcp_fold_equals_one_separate_add(kw, rows):
    w = _weights(61)
    x, ids, wts, shared = _inputs(rows, 62)
    want = shared.clone().add_(_experts(w)(x, ids, wts))
    ex = _experts(w, **kw)
    target = shared.clone()
    _arm_qcp(ex, target)
    got = ex(x, ids, wts)
    assert ex.qcp_shared_add_taken
    assert got.data_ptr() == target.data_ptr()
    assert torch.equal(got, want)


@pytest.mark.parametrize("kw", [dict(), dict(counting_route=True), dict(small_step_rows=4096)])
def test_both_fusions_on_one_call_refuse(kw):
    w = _weights(71)
    x, ids, wts, shared = _inputs(256, 72)
    ex = _experts(w, fused_shared_add=True, **kw)
    _arm_qcp(ex, shared.clone())
    with pytest.raises(AssertionError, match="twice"):
        ex(x, ids, wts, add_to=shared.clone())


def test_method_hands_over_only_without_qcp(monkeypatch):
    from sglang.srt.layers.quantization import mxfp4_marlin_moe as m

    monkeypatch.setenv("SGLANG_OPT_DSV4_W4A8_PREFILL_MOE", "1")
    monkeypatch.setenv("SGLANG_OPT_DSV4_W4A8_FUSED_SHARED_ADD", "1")
    method = m.Mxfp4MarlinMoEMethod(
        SimpleNamespace(process_weights_after_loading=lambda layer: None), "test"
    )
    method._w4a8 = SimpleNamespace(fused_shared_add=True)
    experts = SimpleNamespace(should_fuse_routed_scaling_factor_in_topk=True)
    monkeypatch.setenv("SGLANG_OPT_DSV4_MOE_SHARED_ADD_IN_TOPK_SUM", "0")
    assert m._w4a8_fused_shared_add()
    assert method.accepts_shared_add(experts)
    monkeypatch.setenv("SGLANG_OPT_DSV4_MOE_SHARED_ADD_IN_TOPK_SUM", "1")
    assert not m._w4a8_fused_shared_add()  # the runner is built without k-pmoe's add
    assert not method.accepts_shared_add(experts)  # and the model never hands it over
