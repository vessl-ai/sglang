"""INF-874 PCOLL P2 (SGLANG_OPT_DSV4_ENGRAM_PREFETCH_SHARED_LOCAL): with a shared host
table every rank copies its own engram rows (prefetch mode "local", and the in-layer
lookup through the same raw gather + dequant). The result must be the bits of the
per_rank layout's lookup (bf16 gather per shard + bf16 all-reduce, -0 as +0) under
prefill CP and without CP; with the measurement toggle off the shared table must also
serve the replaced reduce-scatter path (each rank's row range only) with the same bits.
Ranks run one after another; the collectives are emulated by summing every rank's
buffer, which is what NCCL computes when only one shard is nonzero."""

import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from sglang.kernels.ops.embeddings.engram_gather import engram_gather
from sglang.srt.layers import engram as E
from sglang.srt.layers.cp import pcoll
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DIM, BLK, COLS = 256, 32, 24
ENV = {
    "SGLANG_OPT_DSV4_ENGRAM_PREFETCH": "1",
    "SGLANG_OPT_DSV4_ENGRAM_PREFETCH_SHARED_LOCAL": "1",
}


def _random_table(n_rows: int, seed: int, host: bool):
    """fp8 payload with -0, +0 and the extremes; e8m0 scales incl. 2**-127, 2**127."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, 256, (n_rows, DIM), generator=g, dtype=torch.int32)
    w = torch.where((w & 0x7F) == 0x7F, w ^ 0x01, w)  # e4m3fn NaN -> neighbour
    w[:, :8] = torch.tensor([0x00, 0x80, 0x01, 0x81, 0x7E, 0xFE, 0x08, 0x88])
    s = torch.randint(110, 140, (n_rows, DIM // BLK), generator=g, dtype=torch.int32)
    s[0, 0], s[1, 1], s[2, 2] = 0, 254, 1
    w8 = w.to(torch.uint8).view(torch.float8_e4m3fn)
    s8 = s.to(torch.uint8).view(torch.float8_e8m0fnu)
    if host:
        return w8.pin_memory(), s8.pin_memory()
    return w8.cuda(), s8.cuda()


def _bits(t):
    return t.contiguous().view(torch.int16)


def _shared_embed(w, s, n_rows, rank, tp):
    """An EngramEmbedding whose shared host table is (w, s): every row, rank's range."""
    m = object.__new__(E.EngramEmbedding)
    nn.Module.__init__(m)
    m.dim, m.tp_size, m.num_embeddings = DIM, tp, n_rows
    m.row_start = n_rows * rank // tp
    m.rows = n_rows * (rank + 1) // tp - m.row_start
    m.host_table = SimpleNamespace(layout="shared")
    m._host_table_ptrs = (w.data_ptr(), s.data_ptr())
    m.weight = nn.Parameter(w, requires_grad=False)
    m.scale = nn.Parameter(s, requires_grad=False)
    return m


def _reference(w, s, n_rows, tp, flat):
    """per_rank layout: bf16 gather of each shard's rows, bf16 all-reduce."""
    acc = torch.zeros(flat.numel(), DIM, dtype=torch.float32, device="cuda")
    for r in range(tp):
        lo, hi = n_rows * r // tp, n_rows * (r + 1) // tp
        part = torch.empty(flat.numel(), DIM, dtype=torch.bfloat16, device="cuda")
        engram_gather(
            w[lo:].data_ptr(), s[lo:].data_ptr(), flat, part, DIM, BLK,
            row_lo=lo, row_hi=hi,
        )  # fmt: skip
        acc = acc + part.float()
    return acc.to(torch.bfloat16)


def _parallel(rank, tp, cp, world):
    attn_tp = tp // cp

    class _CpGroup:
        rank_in_group = rank // attn_tp

        def reduce_scatter_tensor(self, out, inp):
            out.copy_(world.total(inp).view(cp, -1)[self.rank_in_group])

    return SimpleNamespace(
        attn_dp_size=1, tp_size=tp, attn_cp_size=cp, attn_cp_rank=rank // attn_tp,
        attn_cp_group=_CpGroup(), tp_group=SimpleNamespace(unique_name="tp"),
    )  # fmt: skip


class _World:
    def __init__(self):
        self.bufs = {}  # data_ptr -> (layer, rank)
        self.raw = {}

    def total(self, inp):
        layer, _ = self.bufs[inp.data_ptr()]
        return sum(v.view(torch.int32) for (l, _r), v in self.raw.items() if l == layer)


def _fb():
    return SimpleNamespace(forward_mode=SimpleNamespace(
        is_extend_without_speculative=lambda: True))  # fmt: skip


@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize("toggle", [True, False])
@pytest.mark.parametrize("tp,cp,total", [(4, 4, 1), (4, 4, 4099), (4, 1, 777)])
def test_prefetch_shared_matches_per_rank_lookup(host, toggle, tp, cp, total):
    n_rows, layers = 9001, (1, 14)
    tables = {l: _random_table(n_rows, seed=100 + l, host=host) for l in layers}
    g = torch.Generator(device="cuda").manual_seed(total)
    full = torch.randint(0, n_rows, (total, 2, COLS), generator=g, device="cuda")
    full[:3, 0, 0] = torch.tensor([0, 1, 2], device="cuda")  # the special rows
    local = -(-total // cp) if cp > 1 else total
    world = _World()
    prefetches = {}
    with mock.patch.dict(os.environ, ENV), mock.patch.object(
        pcoll, "enabled", lambda name: toggle
    ):
        for r in range(tp):
            engrams = [
                (l, SimpleNamespace(embed=_shared_embed(*tables[l], n_rows, r, tp),
                                    layer_hash_index=k))  # fmt: skip
                for k, l in enumerate(layers)
            ]
            with mock.patch.object(E, "get_parallel", return_value=_parallel(r, tp, cp, world)):
                pf = E.EngramPrefetch.maybe_start(
                    engrams, full, _fb(), cp=cp > 1, local_rows=local, late_layer_start=21,
                )  # fmt: skip
            assert pf is not None
            want = "local" if toggle else {4: "rs", 1: "ar"}[cp]
            assert pf.mode == want
            for l, entry in pf.layers.items():
                world.raw[l, r] = entry.buf
                world.bufs[entry.buf.data_ptr()] = (l, r)
            prefetches[r] = pf
        torch.cuda.synchronize()
        world.raw = {k: v.clone() for k, v in world.raw.items()}
        got = {}
        for r in range(tp):

            def ar(t, group_name):
                t.copy_(world.total(t))

            with mock.patch.object(
                E, "get_parallel", return_value=_parallel(r, tp, cp, world)
            ), mock.patch.object(E, "inplace_all_reduce", ar):
                for i in range(22):
                    prefetches[r].reduce_due(i)
                    if i in layers:
                        got[i, r] = prefetches[r].take(i)
            assert not prefetches[r].layers

    for k, l in enumerate(layers):
        if cp > 1:
            parts = []
            for c in range(cp):
                mine = full[c::cp, k, :]
                parts.append(torch.cat([mine, mine.new_zeros(local - mine.shape[0], COLS)]))
            rows = torch.cat(parts)
        else:
            rows = full[:, k, :]
        ref = _reference(*tables[l], n_rows, tp, rows.reshape(-1).contiguous())
        ref = ref.view(rows.shape[0], COLS, DIM)
        for r in range(tp):
            c = r // (tp // cp) if cp > 1 else 0
            want = ref[c * local : (c + 1) * local] if cp > 1 else ref
            assert torch.equal(_bits(got[l, r]), _bits(want)), (l, r)


@pytest.mark.parametrize("host", [False, True])
def test_in_layer_shared_lookup_matches_per_rank(host):
    n_rows, tp = 6151, 4
    w, s = _random_table(n_rows, seed=3, host=host)
    g = torch.Generator(device="cuda").manual_seed(3)
    ids = torch.randint(0, n_rows, (1031, COLS), generator=g, device="cuda")
    ids[:3, 0] = torch.tensor([0, 1, 2], device="cuda")
    ref = _reference(w, s, n_rows, tp, ids.reshape(-1).contiguous()).view(1031, COLS, DIM)
    m = _shared_embed(w, s, n_rows, 2, tp)
    with mock.patch.dict(os.environ, ENV):
        assert m._shared_local and m.can_prefetch()
        out = m(ids, None, cp_all_tokens=True)
    assert torch.equal(_bits(out), _bits(ref))
    # The switch off keeps the existing shared lookup and no prefetch.
    with mock.patch.dict(os.environ, {"SGLANG_OPT_DSV4_ENGRAM_PREFETCH_SHARED_LOCAL": "0"}):
        assert not m._shared_local and not m.can_prefetch()


def test_off_by_default():
    assert not E.envs.SGLANG_OPT_DSV4_ENGRAM_PREFETCH_SHARED_LOCAL.get()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
