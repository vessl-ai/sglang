"""INF-874 R3P: EP over the prefill-CP group for the W4A8 early-layer experts (mxfp4_w4a8_ep.py).

The four ranks are emulated in one process (one GPU): each rank's plan and experts are built
from the same full weights. Checks: every routed slot is computed on exactly one rank, the
ranks' row counts are what the water-fill reports, the loads are balanced when the hot experts
carry enough slots, the sum of the four partial outputs is the full computation (up to the
bf16 rounding of the partials), the TP-slice exchange (gloo, CPU) rebuilds the full experts."""

import os

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="SM90 only",
)
pytest.importorskip("triton_kernels")

E, H, I_FULL, TOPK, R = 24, 5120, 2304, 6, 4
LIMIT = 10.0
KW = dict(
    sorted_sum=True,
    fused_swiglu=True,
    counting_route=True,
    w2_two_cta_min_rows=16384,
    prefetch=False,
    prefetch_chunks=16,
    convert_index_i32=True,
)


def _full_weights(seed, i_full=I_FULL, spread=4):
    g = torch.Generator().manual_seed(seed)

    def one(n, k):
        w = torch.randint(-128, 128, (E, n, k // 2), generator=g, dtype=torch.int8)
        s = torch.randint(122 - spread, 123, (E, n, k // 32), generator=g, dtype=torch.uint8)
        return w, s.view(torch.float8_e8m0fnu)

    return (*one(2 * i_full, H), *one(H, i_full))


def _routing(rows, seed, skew=3.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    score = torch.rand(rows, E, device="cuda", generator=g) * torch.linspace(skew, 1.0, E, device="cuda")
    w, ids = torch.topk(score, TOPK, -1)
    return ids.int(), (w / w.sum(-1, keepdim=True)).float()


def _table(hot):
    # contiguous owners (E / R per rank) for the non-hot experts
    return [e * R // E for e in range(E)], list(hot)


@pytest.mark.parametrize("hot", [[], [0, 1], [0, 1, 2, 3]])
@pytest.mark.parametrize("rows", [1, 777, 20000])
def test_assign_each_slot_once(hot, rows):
    from sglang.srt.layers.quantization.mxfp4_w4a8_ep import EpPlan

    owner, hot = _table(hot)
    ids, _ = _routing(rows, 1)
    taken = torch.zeros(ids.numel(), dtype=torch.int32, device="cuda")
    nv_all = []
    loads = []
    for r in range(R):
        plan = EpPlan(owner, hot, r, "cuda")
        loc, nv = plan.assign(ids)
        loc = loc.flatten()
        taken += (loc >= 0).int()
        nv_all.append(nv.clone())
        assert int(nv[r]) == int((loc >= 0).sum())
        # local ids point at the right global expert
        experts = torch.tensor(plan.experts, device="cuda")
        m = loc >= 0
        assert torch.equal(experts[loc[m].long()], ids.flatten()[m].long())
        loads.append(int(nv[r]))
    assert torch.all(taken == 1)
    for nv in nv_all[1:]:
        assert torch.equal(nv, nv_all[0])  # every rank computes the same split
    assert sum(loads) == ids.numel()
    # water-fill: the largest load is the larger of the heaviest rank's owned slots and the mean (+ rounding)
    counts = torch.bincount(ids.flatten().long(), minlength=E).cpu()
    owned = [sum(int(counts[e]) for e in range(E) if owner[e] == r and e not in hot) for r in range(R)]
    bound = max(max(owned), -(-ids.numel() // R)) + len(hot)
    assert max(loads) <= bound, (loads, owned)


def _ep_rank_experts(full, owner, hot, r, i_full=I_FULL):
    from sglang.srt.layers.quantization.mxfp4_w4a8_ep import EpPlan, Mxfp4W4A8EPExperts

    plan = EpPlan(owner, hot, r, "cuda")
    idx = torch.tensor(plan.experts)
    w13, s13, w2, s2 = (t[idx].contiguous().cuda() for t in full)
    return Mxfp4W4A8EPExperts(w13, s13, w2, s2, LIMIT, plan=plan, **KW)


def test_balanced_when_hot_mass_suffices():
    from sglang.srt.layers.quantization.mxfp4_w4a8_ep import EpPlan

    owner = [e % R for e in range(E)]  # interleaved owners: owned loads nearly equal
    hot = [0, 1, 2, 3]
    ids, _ = _routing(20000, 7)
    loads = [int(EpPlan(owner, hot, r, "cuda").assign(ids)[1][r]) for r in range(R)]
    assert max(loads) - min(loads) <= len(hot), loads


@pytest.mark.parametrize("pre_quant", [False, True])
def test_partials_sum_to_full(pre_quant):
    from sglang.srt.layers.quantization import mxfp4_w4a8_prefill as wp

    full = _full_weights(2)
    ref = wp.Mxfp4W4A8PrefillExperts(*(t.cuda() for t in full), LIMIT, **KW)
    rows = 20000
    ids, wts = _routing(rows, 3)
    g = torch.Generator(device="cuda").manual_seed(4)
    x = torch.randn(rows, H, device="cuda", generator=g).bfloat16()
    y_ref = ref(x, ids, wts).float()
    owner, hot = _table([0, 1, 2, 3])
    total = torch.zeros_like(y_ref)
    for r in range(R):
        ep = _ep_rank_experts(full, owner, hot, r)
        rq = wp.W4A8RowQuant(*wp._row_quant(x)) if pre_quant else None
        part = ep(x, ids, wts, row_quant=rq)
        assert part.shape == x.shape and torch.isfinite(part.float()).all()
        total += part.float()
        del ep
    rel = ((total - y_ref).norm() / y_ref.norm()).item()
    assert rel < 1e-2, rel


def _gloo_worker(rank, port, full, owner, hot, out_dir):
    import torch.distributed as dist

    from sglang.srt.layers.quantization.mxfp4_w4a8_ep import (
        assemble_full,
        exchange_to_full,
        local_experts,
    )

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=R)

    class G:
        rank_in_group = rank
        device_group = dist.group.WORLD

    i = I_FULL // R
    w13, s13, w2, s2 = full
    # this rank's TP slice, the loader's layout: [w1 slice; w3 slice] rows, w2 columns
    sl13 = torch.cat([w13[:, rank * i:(rank + 1) * i], w13[:, I_FULL + rank * i:I_FULL + (rank + 1) * i]], 1)
    sl13s = torch.cat([s13[:, rank * i:(rank + 1) * i], s13[:, I_FULL + rank * i:I_FULL + (rank + 1) * i]], 1)
    sl2 = w2[:, :, rank * i // 2:(rank + 1) * i // 2].contiguous()
    sl2s = s2[:, :, rank * i // 32:(rank + 1) * i // 32].contiguous()
    lists = [local_experts(owner, hot, r) for r in range(R)]
    got = assemble_full(*exchange_to_full((sl13, sl13s, sl2, sl2s), lists, G))
    idx = torch.tensor(lists[rank])
    want = [t[idx].contiguous().view(torch.uint8) for t in full]
    ok = all(torch.equal(a.view(torch.uint8), b) for a, b in zip(got, want))
    open(os.path.join(out_dir, f"r{rank}"), "w").write("ok" if ok else "bad")
    dist.destroy_process_group()


def test_exchange_rebuilds_full_experts(tmp_path):
    import random

    import torch.multiprocessing as mp

    full = _full_weights(5, i_full=256)  # small intermediate for the CPU exchange
    global I_FULL
    saved, I_FULL = I_FULL, 256
    try:
        owner, hot = _table([5, 7])
        port = random.randint(20000, 40000)
        mp.spawn(_gloo_worker_entry, args=(port, full, owner, hot, str(tmp_path)), nprocs=R, join=True)
    finally:
        I_FULL = saved
    assert [open(tmp_path / f"r{r}").read() for r in range(R)] == ["ok"] * R


def _gloo_worker_entry(rank, port, full, owner, hot, out_dir):
    global I_FULL
    I_FULL = 256
    _gloo_worker(rank, port, full, owner, hot, out_dir)
