"""SGLANG_OPT_DSV4_ENGRAM_PREFETCH: the prefetched engram lookup (raw fp8 + scale
gather per shard, integer sum over the shards, dequant) must give the bits of the
in-layer lookup (bf16 gather per shard + bf16 all-reduce), for the reduce-scatter
under prefill CP, the all-reduce without CP and the all-reduce + CP slice under
attention TP. The collectives are emulated on one GPU by summing every shard's
buffer, which is what NCCL computes when only one shard is nonzero."""

import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from sglang.kernels.ops.embeddings.engram_gather import (
    engram_dequant_raw,
    engram_gather,
    engram_gather_raw,
    engram_raw_chunk_bytes,
)
from sglang.srt.layers import engram as E
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DIM, BLK, COLS = 256, 32, 24


def _random_table(n_rows: int, seed: int, host: bool):
    """fp8 e4m3 payload with -0, +0 and the extremes; e8m0 scales incl. 2**-127 and
    2**127 (no NaN codes: real tables have none)."""
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


def _shard_ranges(n_rows: int, tp: int):
    return [(n_rows * r // tp, n_rows * (r + 1) // tp) for r in range(tp)]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _ar_bf16(parts):
    """bf16 all-reduce of zero-padded shards: x + 0 + ... in fp32, rounded back."""
    acc = torch.zeros_like(parts[0], dtype=torch.float32)
    for p in parts:
        acc = acc + p.float()
    return acc.to(torch.bfloat16)


@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize("cp,tokens", [(1, 1), (1, 37), (4, 4), (4, 333), (4, 1031)])
@pytest.mark.parametrize("programs,block_rows", [(0, 16), (3, 8), (64, 32)])
@pytest.mark.parametrize("ordered", [False, True])
def test_raw_gather_sum_dequant_is_lookup_allreduce(
    host, cp, tokens, programs, block_rows, ordered
):
    tp, n_rows = 4, 6151
    shards = []
    w, s = _random_table(n_rows, seed=tokens, host=host)
    for lo, hi in _shard_ranges(n_rows, tp):
        # Row slices of the (pinned) table: one shard per rank, as the per_rank layout.
        shards.append((w[lo:hi], s[lo:hi], lo, hi))
    local = -(-tokens // cp)
    g = torch.Generator(device="cuda").manual_seed(7)
    ids = torch.randint(0, n_rows, (cp * local, COLS), generator=g, device="cuda")
    ids[:, 0] = torch.arange(cp * local, device="cuda") % 3  # the special rows
    flat = ids.reshape(-1).contiguous()

    old = _ar_bf16(
        [
            engram_gather(
                ws.data_ptr(), ss.data_ptr(), flat,
                torch.empty(flat.numel(), DIM, dtype=torch.bfloat16, device="cuda"),
                DIM, BLK, row_lo=lo, row_hi=hi,
            )
            for ws, ss, lo, hi in shards
        ]
    )  # fmt: skip

    rows_per_chunk = local * COLS
    chunk_bytes = engram_raw_chunk_bytes(rows_per_chunk, DIM, BLK)
    acc = torch.zeros(cp * chunk_bytes // 4, dtype=torch.int32, device="cuda")
    for ws, ss, lo, hi in shards:
        buf = torch.full((cp * chunk_bytes,), 0xA5, dtype=torch.uint8, device="cuda")
        engram_gather_raw(
            ws.data_ptr(), ss.data_ptr(), flat, buf, DIM, BLK, rows_per_chunk,
            row_lo=lo, row_hi=hi, num_programs=programs, block_rows=block_rows,
            order=torch.argsort(flat) if ordered else None,
        )  # fmt: skip
        acc += buf.view(torch.int32)
    raw = acc.view(torch.uint8).view(cp, chunk_bytes)
    for k in range(cp):
        out = torch.empty(rows_per_chunk, DIM, dtype=torch.bfloat16, device="cuda")
        engram_dequant_raw(raw[k].contiguous(), out, DIM, BLK)
        ref = old[k * rows_per_chunk : (k + 1) * rows_per_chunk]
        assert torch.equal(_bits(out), _bits(ref)), f"chunk {k}"
    # The -0 payload byte really occurs, so the canonicalization is exercised.
    assert (w.view(torch.uint8)[:, 1] == 0x80).all()


def test_cp_rank_major_rows_is_allgather_of_padded_slices():
    for total, cp in [(1, 4), (4, 4), (5, 4), (4097, 4), (10, 2)]:
        local = -(-total // cp) + (1 if total % 3 == 0 else 0)
        full = torch.arange(total * 2 * 3).view(total, 2, 3) + 1
        old = []
        for r in range(cp):
            mine = full[r::cp]
            pad = local - mine.shape[0]
            old.append(torch.cat([mine, mine.new_zeros(pad, 2, 3)]))
        new = E.cp_rank_major_rows(full, cp, local).reshape(-1, 2, 3)
        assert torch.equal(new, torch.cat(old))


# ---- EngramPrefetch on emulated ranks ------------------------------------------


def _embed(w, s, lo, hi, tp):
    m = object.__new__(E.EngramEmbedding)
    nn.Module.__init__(m)
    m.dim, m.tp_size, m.row_start, m.rows, m.host_table = DIM, tp, lo, hi - lo, None
    m.weight = nn.Parameter(w, requires_grad=False)
    m.scale = nn.Parameter(s, requires_grad=False)
    return m


class _World:
    """tp ranks run one after another; a collective sees every rank's buffer."""

    def __init__(self, tp, cp, n_rows, layers=(1, 14)):
        self.tp, self.cp = tp, cp
        self.layers = layers
        self.tables = {}
        self.embeds = {}  # (layer, rank) -> embed
        for li, layer in enumerate(layers):
            w, s = _random_table(n_rows, seed=100 + layer, host=False)
            for r, (lo, hi) in enumerate(_shard_ranges(n_rows, tp)):
                self.embeds[layer, r] = _embed(
                    w[lo:hi].contiguous(), s[lo:hi].contiguous(), lo, hi, tp
                )
        self.bufs = {}  # data_ptr of a rank's buffer -> (layer, rank)
        self.rank = 0

    def parallel(self, rank):
        attn_tp = self.tp // self.cp
        world = self

        class _CpGroup:
            rank_in_group = rank // attn_tp

            def reduce_scatter_tensor(self, out, inp):
                layer, _ = world.bufs[inp.data_ptr()]
                total = world.total(layer)
                out.copy_(total.view(world.cp, -1)[self.rank_in_group])

        return SimpleNamespace(
            attn_dp_size=1, tp_size=self.tp, attn_cp_size=self.cp,
            attn_cp_rank=rank // attn_tp, attn_cp_group=_CpGroup(),
            tp_group=SimpleNamespace(unique_name="tp"),
        )  # fmt: skip

    def total(self, layer):
        return sum(self.raw[layer, s].view(torch.int32) for s in range(self.tp))

    def inplace_all_reduce(self, t, group_name):
        layer, _ = self.bufs[t.data_ptr()]
        t.copy_(self.total(layer))


def _fb():
    return SimpleNamespace(forward_mode=SimpleNamespace(
        is_extend_without_speculative=lambda: True))  # fmt: skip


@pytest.mark.parametrize(
    "tp,cp,total", [(4, 4, 1), (4, 4, 4099), (4, 1, 777), (4, 2, 1000)]
)
def test_prefetch_matches_in_layer_lookup(tp, cp, total):
    n_rows = 9001
    world = _World(tp, cp, n_rows)
    g = torch.Generator(device="cuda").manual_seed(total)
    full = torch.randint(0, n_rows, (total, 2, COLS), generator=g, device="cuda")
    local = -(-total // cp) if cp > 1 else total
    engrams_of = lambda r: [  # noqa: E731
        (layer, SimpleNamespace(embed=world.embeds[layer, r], layer_hash_index=k))
        for k, layer in enumerate(world.layers)
    ]
    prefetches = {}
    world.raw = {}
    with mock.patch.dict(os.environ, {"SGLANG_OPT_DSV4_ENGRAM_PREFETCH": "1"}):
        for r in range(tp):
            with mock.patch.object(E, "get_parallel", return_value=world.parallel(r)):
                pf = E.EngramPrefetch.maybe_start(
                    engrams_of(r), full, _fb(), cp=cp > 1, local_rows=local,
                    late_layer_start=21,
                )  # fmt: skip
            assert pf is not None
            assert pf.mode == {4: "rs", 1: "ar", 2: "ar_slice"}[cp]
            for layer, entry in pf.layers.items():
                world.raw[layer, r] = entry.buf
                world.bufs[entry.buf.data_ptr()] = (layer, r)
            prefetches[r] = pf
        torch.cuda.synchronize()
        # Snapshot the side-stream results before any rank's collective rewrites them.
        world.raw = {k: v.clone() for k, v in world.raw.items()}
        got = {}
        for r in range(tp):
            ar = world.inplace_all_reduce
            with mock.patch.object(
                E, "get_parallel", return_value=world.parallel(r)
            ), mock.patch.object(E, "inplace_all_reduce", ar):
                for i in range(0, 22):
                    prefetches[r].reduce_due(i)
                    if i in world.layers:
                        got[i, r] = prefetches[r].take(i)
            assert not prefetches[r].layers

    for k, layer in enumerate(world.layers):
        if cp > 1:
            # The in-layer path: each CP rank's interleaved, zero-padded rows,
            # all-gathered in rank order.
            parts = []
            for c in range(cp):
                mine = full[c::cp, k, :]
                pad = mine.new_zeros(local - mine.shape[0], COLS)
                parts.append(torch.cat([mine, pad]))
            rows = torch.cat(parts)
        else:
            rows = full[:, k, :]
        flat = rows.reshape(-1).contiguous()
        old = _ar_bf16([world.embeds[layer, s]._owned_rows(flat) for s in range(tp)])
        old = old.view(rows.shape[0], COLS, DIM)
        for r in range(tp):
            attn_tp = tp // cp if cp > 1 else 1
            if cp > 1:
                c = r // attn_tp
                ref = old[c * local : (c + 1) * local]
            else:
                ref = old
            assert got[layer, r].shape == ref.shape
            assert torch.equal(_bits(got[layer, r]), _bits(ref)), (layer, r)


def test_prefetch_off_by_default():
    assert not E.envs.SGLANG_OPT_DSV4_ENGRAM_PREFETCH.get()
    assert E.EngramPrefetch.maybe_start(
        [(1, None)], torch.zeros(1, 2, COLS, dtype=torch.int64, device="cuda"), _fb(),
        cp=True, local_rows=1, late_layer_start=None,
    ) is None  # fmt: skip
