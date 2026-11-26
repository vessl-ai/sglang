"""INF-874 R2IDX: the prefill indexer switches keep their outputs bit for bit.

SGLANG_OPT_DSV4_PREFILL_CANDIDATE_TAIL_ONLY: the candidate source publishes block ids
for the late-layer tail rows only. Those rows' block ids, and every row's selection,
must equal the all-rows publish (the selection as a set: its order is not
deterministic even between two all-rows runs; several score tiles, requests with no visible
position, a tail longer than a request's rows, different kept block counts).

SGLANG_OPT_DSV4_PREFILL_INDEX_Q_FP8: the SM90 index queries in one launch must equal
index_q_rope_pack_weights -> fp4_index_rows_to_fp8 -> weights * row scale, byte for
byte, on rows that hit every rounding branch, with int32 and int64 positions.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEV = "cuda"


def _freqs():
    from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis

    f = precompute_freqs_cis(
        dim=64, seqlen=1 << 17, original_seq_len=65536, base=160000,
        factor=16, beta_fast=32, beta_slow=1,
    )
    return torch.view_as_real(f.to(DEV)).flatten(-2)


class TestCandidateTailOnly(CustomTestCase):
    def _case(self, extend, seq, tail, budget=None, seed=0):
        import sglang.srt.layers.attention.dsv4.v41_indexer.scoring as scoring
        from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
            _publish_prefill_blocks,
        )
        from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import (
            DeepGEMMPrefillData,
        )

        g = torch.Generator(device=DEV).manual_seed(seed)
        rows_per_request, pos = [], []
        for e, s in zip(extend, seq):
            p = torch.arange(s - e, s, 4)  # one CP4 rank's rows
            rows_per_request.append(p.shape[0])
            pos.append(p)
        pos = torch.cat(pos).to(DEV)
        T = pos.shape[0]
        lens = list(seq)
        starts = torch.tensor([0] + torch.tensor(lens).cumsum(0)[:-1].tolist(), device=DEV)
        req = torch.repeat_interleave(
            torch.arange(len(seq), device=DEV), torch.tensor(rows_per_request, device=DEV)
        )
        cols = sum(lens)
        data = DeepGEMMPrefillData(
            k_slots=torch.arange(cols, device=DEV),
            request_starts=starts[req].to(torch.int32),
            lens_per_request=lens,
            rows_per_request=rows_per_request,
            compress_lens=(pos + 1).to(torch.int32),
            q=(torch.randn(T, 32, 128, device=DEV, generator=g) * 2).to(torch.float8_e4m3fn),
            q_sf=None,
            weights=torch.randn(T, 32, device=DEV, generator=g) * 0.05,
        )
        kv = (
            (torch.randn(cols, 128, device=DEV, generator=g) * 2).to(torch.float8_e4m3fn),
            torch.rand(cols, device=DEV, generator=g) + 0.5,
        )
        kw = dict(data=data, kv=kv, topk=64, topk_blocks=128, block_size=8, batched=True)
        ctx = (
            patch.object(scoring, "_DEEP_GEMM_SCORE_BUDGET_BYTES", budget)
            if budget
            else contextlib.nullcontext()
        )
        with ctx:
            sel_a, blk_a = _publish_prefill_blocks(**kw)
            sel_b, blk_b = _publish_prefill_blocks(**kw, tail_rows_per_request=tail)
        rows, start = [], 0
        for n, t in zip(rows_per_request, tail):
            rows.extend(range(start + n - min(t, n), start + n))
            start += n
        rows = torch.tensor(rows, dtype=torch.int64, device=DEV)
        # topk_transform_ragged_v2 leaves each row's selection unordered (the order
        # differs between two runs of the same path), so compare the sets
        self.assertTrue(torch.equal(sel_a.sort(dim=1).values, sel_b.sort(dim=1).values))
        self.assertTrue(torch.equal(blk_a[rows], blk_b[rows]))
        self.assertGreater(int((blk_a[rows] >= 0).sum()), 0)
        other = torch.ones(T, dtype=torch.bool, device=DEV)
        other[rows] = False
        self.assertTrue(bool((blk_b[other] == -1).all()))

    def test_one_tile(self):
        self._case([600, 37, 1, 300, 9], [3000, 37, 5, 1200, 2000], [32, 32, 32, 5, 1])

    def test_many_tiles(self):
        # a tiny score budget cuts the rows into several tiles, tails straddle tile edges
        self._case([600, 37, 1, 300, 9], [3000, 37, 5, 1200, 2000], [40, 3, 1, 32, 3],
                   budget=64 * 3000 * 4, seed=1)

    def test_kept_counts_differ(self):
        # contexts shorter than topk_blocks * block_size keep fewer blocks
        self._case([64, 64, 64, 64], [80, 700, 2000, 90], [16, 16, 16, 16], budget=40 * 2000 * 4, seed=2)

    def test_backend_gate(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend as B
        from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
            DenseBlocksBackend,
        )

        dense = DenseBlocksBackend.__new__(DenseBlocksBackend)
        full = SimpleNamespace()
        tail = SimpleNamespace(
            late_layer_tail=SimpleNamespace(
                cp_metadata=object(), local_lens_cpu=[3, 4], extend_seq_lens_cpu=[12, 16]
            )
        )
        b = SimpleNamespace(prefill_candidates=dense, forward_metadata=full, tail_forward_metadata=tail)
        f = B._candidate_tail_rows_per_request
        with envs.SGLANG_OPT_DSV4_PREFILL_CANDIDATE_TAIL_ONLY.override(False):
            self.assertIsNone(f(b))
        with envs.SGLANG_OPT_DSV4_PREFILL_CANDIDATE_TAIL_ONLY.override(True):
            self.assertEqual(f(b), [3, 4])
            tail.late_layer_tail.cp_metadata = None
            self.assertEqual(f(b), [12, 16])
            b.tail_forward_metadata = full  # no separate tail: every row
            self.assertIsNone(f(b))


class TestIndexQFp8(CustomTestCase):
    def _case(self, n, pos_dtype, seed=0):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_rows_to_fp8
        from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope import (
            index_q_rope_pack_weights,
        )
        from sglang.kernels.ops.attention.dsv4.index_q_fp8 import (
            index_q_rope_fp8_weights,
        )

        H, D = 32, 128
        g = torch.Generator(device=DEV).manual_seed(seed)
        q = torch.randn(n, H, D, device=DEV, generator=g) * 1.5
        k = max(n // 8, 1)
        q[:k] = 0
        q[k : 2 * k] *= 1e-30
        q[2 * k : 3 * k] *= 1e4
        grid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device=DEV)
        q[3 * k : 4 * k] = grid[torch.randint(0, 8, (k, H, D), device=DEV, generator=g)] * (
            2.0 ** torch.randint(-3, 4, (k, H, 1), device=DEV, generator=g)
        )
        q[4 * k : 5 * k] = -torch.rand(k, H, D, device=DEV, generator=g) * 1e-3
        q = q.to(torch.bfloat16)
        hw = (torch.randn(n, H, device=DEV, generator=g) * 3).to(torch.bfloat16)
        pos = torch.randint(0, 100000, (n,), device=DEV, generator=g).to(pos_dtype)
        ws = 128**-0.5 * H**-0.5
        freqs = _freqs()
        q4, sf, w = index_q_rope_pack_weights(q, freqs, pos, hw, ws)
        a8, s = fp4_index_rows_to_fp8(q4.view(n, H, 64), sf.view(n, H))
        aw = w * s
        b8, bw = index_q_rope_fp8_weights(q, freqs, pos, hw, ws)
        self.assertTrue(torch.equal(a8.view(torch.uint8), b8.view(torch.uint8)))
        self.assertTrue(torch.equal(aw.view(torch.int32), bw.view(torch.int32)))

    def test_rows(self):
        for n in (1, 3, 17, 1000):
            with self.subTest(n=n):
                self._case(n, torch.int64, seed=n)

    def test_int32_positions(self):
        self._case(257, torch.int32, seed=5)


if __name__ == "__main__":
    unittest.main()
