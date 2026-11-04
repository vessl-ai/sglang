"""INF-730 SGLANG_OPT_DSV4_DP_PAD_MASK_REAL_COUNTS: DP MAX_LEN pad rows dropped from the decode MoE.

Under attention DP + MAX_LEN padding the MoE input is [W * max_len] rows, rank r's real rows first in its chunk. The
existing SGLANG_OPT_MASK_DP_PAD_MOE mask reads global_num_tokens_gpu, which holds the padded count by the time a
MAX_LEN forward runs, so it masks nothing; the switch makes it read the real per-rank counts. Checked here on one
GPU (the W ranks' chunks are just rows of one tensor):
  - the mask with real counts sets exactly the pad rows' top-k to -1, with padded counts (today) none;
  - captured in a CUDA graph with the static count buffer, every replay masks by the counts written before it;
  - the v2 W4A8 decode MoE gives the real rows the same bits whether the pad rows are routed (today) or masked,
    for the target (384 experts, top 6, 4 verify rows) and draft (128, top 3, 3 rows) shapes, skew 0 and 1.0.
"""

import pytest
import torch

from sglang.srt.utils.common import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm90_supported(),
    reason="SM90 wgmma W4A8 decode MoE",
)

HIDDEN, INTER = 5120, 576
SWIGLU_LIMIT = 10.0
W = 4
SHAPES = {"target": (384, 6, 4), "draft": (128, 3, 3)}


@pytest.fixture
def switch_on(monkeypatch):
    from sglang.srt.layers import dp_attention

    monkeypatch.setenv("SGLANG_OPT_MASK_DP_PAD_MOE", "1")
    monkeypatch.setenv("SGLANG_OPT_DSV4_DP_PAD_MASK_REAL_COUNTS", "1")
    dp_attention.dp_pad_mask_real_counts_enabled.cache_clear()
    yield dp_attention
    dp_attention.dp_pad_mask_real_counts_enabled.cache_clear()
    dp_attention.set_dp_buffer_len(0, 0, False)


def _ids(rows, num_experts, top_k, seed, skew=0.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pop = torch.zeros(num_experts, device="cuda")
    if skew:
        pop = (-skew * torch.log(torch.arange(1, num_experts + 1, device="cuda").float()))[
            torch.randperm(num_experts, device="cuda", generator=g)
        ]
    logits = pop + torch.randn(rows, num_experts, device="cuda", generator=g)
    w, ids = torch.topk(torch.softmax(logits, -1), top_k, dim=-1)
    return ids.int().contiguous(), (w / w.sum(-1, keepdim=True)).float().contiguous()


def _expected_mask(counts, max_len, top_k):
    pos = torch.arange(W * max_len, device="cuda") % max_len
    rank = torch.arange(W * max_len, device="cuda") // max_len
    return (pos >= torch.tensor(counts, device="cuda")[rank]).unsqueeze(1).expand(-1, top_k)


@pytest.mark.parametrize("counts", [[16, 16, 16, 16], [13, 16, 9, 0], [1, 2, 3, 4]])
def test_mask_reads_real_counts(switch_on, counts):
    dpa = switch_on
    max_len, top_k = 16, 6
    ids, _ = _ids(W * max_len, 384, top_k, seed=1)
    ref = ids.clone()
    padded = torch.full((W,), max_len, dtype=torch.int64, device="cuda")
    real = torch.tensor(counts, dtype=torch.int32, device="cuda")
    # Today's state under MAX_LEN: only the padded counts are published -> nothing is masked.
    dpa.set_dp_buffer_len(W * max_len, max_len, True, [max_len] * W, padded)
    dpa.mask_dp_pad_moe_topk_ids(ids)
    assert torch.equal(ids, ref)
    # With the real counts: exactly the pad rows become -1.
    dpa.set_dp_buffer_len(W * max_len, max_len, True, [max_len] * W, padded, real)
    dpa.mask_dp_pad_moe_topk_ids(ids)
    m = _expected_mask(counts, max_len, top_k)
    assert torch.equal(ids[m], torch.full_like(ids[m], -1))
    assert torch.equal(ids[~m], ref[~m])


def test_mask_in_cuda_graph_follows_replay_counts(switch_on):
    dpa = switch_on
    max_len, top_k = 32, 6
    buf = dpa.dp_real_num_tokens_buffer(W, "cuda")
    src, _ = _ids(W * max_len, 384, top_k, seed=2)
    ids = src.clone()
    dpa.set_dp_buffer_len(W * max_len, max_len, True, [max_len] * W, None, buf)
    buf.fill_(1 << 30)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        dpa.mask_dp_pad_moe_topk_ids(ids)  # warmup (all valid)
    torch.cuda.current_stream().wait_stream(s)
    assert torch.equal(ids, src)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        dpa.mask_dp_pad_moe_topk_ids(ids)
    for counts in ([32, 32, 32, 32], [30, 25, 32, 7], [0, 1, 31, 32]):
        ids.copy_(src)
        buf.copy_(torch.tensor(counts, dtype=torch.int32))
        g.replay()
        torch.cuda.synchronize()
        m = _expected_mask(counts, max_len, top_k)
        assert torch.equal(ids[m], torch.full_like(ids[m], -1)), counts
        assert torch.equal(ids[~m], src[~m]), counts


def _synthetic_packed(num_experts, seed=0):
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

    g = torch.Generator().manual_seed(seed)
    nib = lambda *s: torch.randint(-128, 128, s, generator=g, dtype=torch.int8)  # noqa: E731
    sc = lambda *s: torch.randint(119, 123, s, generator=g, dtype=torch.uint8).view(torch.float8_e8m0fnu)  # noqa: E731
    return ws.pack_experts(
        nib(num_experts, 2 * INTER, HIDDEN // 2).cuda(),
        sc(num_experts, 2 * INTER, HIDDEN // 32).cuda(),
        nib(num_experts, HIDDEN, INTER // 2).cuda(),
        sc(num_experts, HIDDEN, INTER // 32).cuda(),
    )


@pytest.mark.parametrize("shape", ["target", "draft"])
@pytest.mark.parametrize("skew", [0.0, 1.0])
@pytest.mark.parametrize(
    "reqs,bucket", [((94, 96, 96, 95), 96), ((104, 106, 108, 110), 112), ((8, 1, 0, 5), 8)]
)
def test_v2_real_rows_unchanged_when_pads_masked(switch_on, shape, skew, reqs, bucket):
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2 as v2

    dpa = switch_on
    num_experts, top_k, rpr = SHAPES[shape]
    max_len = bucket * rpr
    counts = [r * rpr for r in reqs]
    if W * max_len * top_k > v2.PREP_MAX_SLOTS:
        pytest.skip("over the v2 prologue's slot limit")
    packed = _synthetic_packed(num_experts, seed=3)
    g = torch.Generator(device="cuda").manual_seed(4)
    x = (torch.randn(W * max_len, HIDDEN, device="cuda", generator=g) * 0.7).to(torch.bfloat16)
    ids, w = _ids(W * max_len, num_experts, top_k, seed=5, skew=skew)
    m = _expected_mask(counts, max_len, top_k)[:, 0]
    x[m] = (torch.randn(int(m.sum()), HIDDEN, device="cuda", generator=g) * 1e3).to(torch.bfloat16)  # stale pads
    out_routed = v2.decode_moe(x, ids.clone(), w, packed, SWIGLU_LIMIT)
    masked = ids.clone()
    dpa.set_dp_buffer_len(
        W * max_len, max_len, True, [max_len] * W, None, torch.tensor(counts, dtype=torch.int32, device="cuda")
    )
    dpa.mask_dp_pad_moe_topk_ids(masked)
    assert int((masked[m] != -1).sum()) == 0 and torch.equal(masked[~m], ids[~m])
    out_masked = v2.decode_moe(x, masked, w, packed, SWIGLU_LIMIT)
    assert torch.equal(out_masked[~m], out_routed[~m])


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
