"""INF-874: SGLANG_OPT_DSV4_W4A8_CTA_ROUTE, the W4A8 prefill experts' counting sort with CTA-local
atomics (w4a8_count_route.cuh). It must give _count_route's layout (same counts, rows of an expert
contiguous and in expert order, gather / sorted weights / pos consistent, masked slots without a row)
and the same MoE output bits."""

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


def _weights(seed, spread=4):
    g = torch.Generator().manual_seed(seed)

    def one(n, k):
        w = torch.randint(-128, 128, (E, n, k // 2), generator=g, dtype=torch.int8)
        s = torch.randint(
            122 - spread, 123, (E, n, k // 32), generator=g, dtype=torch.uint8
        )
        return w.cuda(), s.view(torch.float8_e8m0fnu).cuda()

    return (*one(2 * I, H), *one(H, I))


def _routing(rows, seed, experts=E, skew=False, mask_every=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    score = torch.rand(rows, experts, device="cuda", generator=g)
    if skew:  # a few hot experts, some cold ones
        score = score * torch.linspace(4.0, 0.1, experts, device="cuda")
    w, ids = torch.topk(score, TOPK, -1)
    ids = ids.int()
    if mask_every:
        ids[::mask_every, -1] = -1
    return ids, (w / w.sum(-1, keepdim=True)).float()


def _check_layout(r, ids, wts, experts):
    flat = ids.flatten().long()
    valid = flat >= 0
    assert torch.equal(
        r.expert_counts.long(), torch.bincount(flat[valid], minlength=experts)
    )
    pos = r.pos.long()
    assert torch.all(pos[~valid] == -1)
    n = int(valid.sum())
    # every valid slot has its own row, rows 0..n-1
    assert torch.equal(torch.sort(pos[valid]).values, torch.arange(n, device="cuda"))
    row_expert = torch.empty(n, dtype=torch.long, device="cuda")
    row_expert[pos[valid]] = flat[valid]
    assert torch.all(row_expert[1:] >= row_expert[:-1])
    slot = torch.arange(flat.numel(), device="cuda")
    assert torch.equal(r.gather_indx.long()[pos[valid]], slot[valid] // TOPK)
    assert torch.equal(r.sorted_weights[pos[valid]], wts.flatten()[valid])


@pytest.mark.parametrize("rows", [1, 37, 1000, 9000])
@pytest.mark.parametrize("experts", [16, 384])
@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("chunk", [32, 4096])
def test_cta_route_layout(rows, experts, skew, chunk):
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import (
        _count_route,
        _count_route_cta,
    )

    ids, wts = _routing(rows, 51, experts, skew, mask_every=7)
    r = _count_route_cta(ids, wts, experts, chunk)
    _check_layout(r, ids, wts, experts)
    ref = _count_route(ids, wts, experts)
    assert torch.equal(r.expert_counts, ref.expert_counts)


@pytest.mark.parametrize("rows", [1, 37, 2048, 9000])
@pytest.mark.parametrize("skew", [False, True])
def test_cta_route_moe_output_bit_for_bit(rows, skew, monkeypatch):
    from sglang.srt.layers.quantization.mxfp4_w4a8_prefill import (
        Mxfp4W4A8PrefillExperts,
    )

    w = _weights(11)
    g = torch.Generator(device="cuda").manual_seed(12)
    x = torch.randn(rows, H, device="cuda", generator=g).bfloat16()
    ids, wts = _routing(rows, 13, E, skew, mask_every=5)

    def run():
        ex = Mxfp4W4A8PrefillExperts(
            *w,
            LIMIT,
            prefetch=False,
            sorted_sum=True,
            counting_route=True,
            fused_swiglu=True,
            w2_two_cta_min_rows=1,
        )
        return ex(x, ids, wts)

    monkeypatch.setenv("SGLANG_OPT_DSV4_W4A8_CTA_ROUTE", "0")
    ref = run()
    monkeypatch.setenv("SGLANG_OPT_DSV4_W4A8_CTA_ROUTE", "1")
    for _ in range(3):
        assert torch.equal(run(), ref)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
