"""SGLANG_OPT_DSV4_BATCHED_CANDIDATE_BLOCKS: the batched candidate source / consumer
of the DeepGEMM prefill indexer must equal the per-request path bit for bit, ties,
signed zeros and the order of every selection included."""

import itertools
import unittest
import zlib
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    candidate_block_keys,
    select_candidate_block_ids,
)
from sglang.srt.layers.attention.dsv4.v41_indexer import dense_blocks, scoring
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_ARCH = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def _rows_layout(requests, *, cp_rank=None, cp_size=4, tail=None):
    """Per row: request, compress_len, request start column. ``requests`` are
    (extend rows, context lc). Rows are each request's newest positions (ratio 1);
    ``tail`` keeps a request's last ``tail`` rows (bounded replay), ``cp_rank`` the
    rows whose global index is congruent to it (CP interleave)."""
    rows_per_request, lens, starts, start, global_row = [], [], [], 0, 0
    for extend, lc in requests:
        keep = list(range(extend))
        if tail is not None:
            keep = keep[max(0, extend - tail) :]
        count = 0
        for i in keep:
            g_index = global_row + i
            if cp_rank is not None and g_index % cp_size != cp_rank:
                continue
            lens.append(lc - extend + 1 + i if lc > 0 else 0)
            starts.append(start)
            count += 1
        rows_per_request.append(count)
        global_row += extend
        start += lc
    return rows_per_request, lens, starts


def _data(requests, rows_per_request, lens, starts):
    rows = len(lens)
    dev = "cuda"
    return scoring.DeepGEMMPrefillData(
        k_slots=torch.zeros(1, dtype=torch.int64, device=dev),
        request_starts=torch.tensor(starts, dtype=torch.int32, device=dev),
        lens_per_request=[lc for _, lc in requests],
        rows_per_request=rows_per_request,
        compress_lens=torch.tensor(lens, dtype=torch.int32, device=dev),
        q=torch.zeros((rows, 1, 1), device=dev),
        q_sf=None,
        weights=torch.zeros((rows, 1), device=dev),
    )


def _logits(rows, width, lens, mode, seed):
    """fp32 [rows, width] scores; past each row's length garbage (NaN / +-inf /
    huge), which neither path may read."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    if mode == "random":
        x = torch.randn((rows, width), device="cuda", generator=g)
    elif mode == "levels":  # heavy ties among a few values
        x = torch.randint(0, 3, (rows, width), device="cuda", generator=g).float()
    elif mode == "zeros":  # +0 / -0 ties, the sign decided by the reduction order
        x = torch.where(
            torch.rand((rows, width), device="cuda", generator=g) < 0.5, -0.0, 0.0
        )
        x = torch.where(
            torch.rand((rows, width), device="cuda", generator=g) < 0.2, -1.0, x
        )
    elif mode == "nan":  # a few NaN / +-inf inside the visible scores
        x = torch.randint(0, 4, (rows, width), device="cuda", generator=g).float()
        u = torch.rand((rows, width), device="cuda", generator=g)
        x = torch.where(u < 0.002, torch.nan, x)
        x = torch.where((u >= 0.002) & (u < 0.004), -torch.inf, x)
        x = torch.where((u >= 0.004) & (u < 0.005), torch.inf, x)
    else:
        raise ValueError(mode)
    lens_t = torch.tensor(lens, dtype=torch.int64, device="cuda")
    past = torch.arange(width, device="cuda")[None, :] >= lens_t[:, None]
    garbage = torch.full_like(x, torch.nan)
    garbage[:, 1::3] = torch.inf
    garbage[:, 2::3] = 3e38
    return torch.where(past, garbage, x)


# (extend rows, lc) per request; lc past topk_blocks * 8 keeps topk_blocks blocks
_CASES = {
    "1 request": [(37, 2000)],
    "2 requests": [(40, 1500), (33, 2300)],
    "10 requests mixed": [
        (9, 300),
        (40, 2100),
        (1, 1),
        (17, 900),
        (64, 2048),
        (5, 515),
        (33, 1999),
        (40, 40),
        (21, 2600),
        (12, 77),
    ],
    "16 requests long": [(8 + i, 1100 + 97 * i) for i in range(16)],
    "16 requests one short": [(8, 1300)] * 7 + [(3, 60)] + [(8, 1400)] * 8,
    "empty request": [(0, 0), (1, 1), (5, 600), (0, 0), (9, 1200)],
}


@unittest.skipUnless(_ARCH in (9, 10), "requires SM90 or SM100")
class TestBatchedCandidateBlocks(CustomTestCase):
    topk_blocks = (
        64  # 2048 in the model; scaled with the context so the runs stay small
    )
    block = 8

    def _layouts(self, requests):
        yield "rows", _rows_layout(requests)
        yield "tail", _rows_layout(requests, tail=16)
        for r in range(4):
            yield f"cp{r}", _rows_layout(requests, cp_rank=r)
        yield "cp1 tail", _rows_layout(requests, cp_rank=1, tail=16)

    def test_block_keys_match_torch_amax(self):
        for rows, width, mode in itertools.product(
            (1, 7, 300), (8, 13, 1000, 4097), ("random", "levels", "zeros", "nan")
        ):
            g = torch.Generator().manual_seed(rows * width)
            lens = torch.randint(0, width + 1, (rows,), generator=g).tolist()
            logits = _logits(rows, width + 5, lens, mode, seed=rows + width)[:, :width]
            num_blocks = -(-width // 8)
            keys = candidate_block_keys(
                logits, torch.tensor(lens, dtype=torch.int32, device="cuda"), num_blocks
            )
            # the per-request path's keys, built with its own ops
            lens_t = torch.tensor(lens, dtype=torch.int32, device="cuda")
            s = logits.clone()
            s.masked_fill_(
                torch.arange(width, device="cuda")[None, :] >= lens_t[:, None],
                -torch.inf,
            )
            s = torch.nn.functional.pad(s, (0, -width % 8), value=-torch.inf)
            ref = s.unflatten(-1, (-1, 8)).amax(dim=-1)
            last = (lens_t[:, None] - 1) // 8
            ref = ref.masked_fill(
                torch.arange(num_blocks, device="cuda") == last, torch.inf
            )
            with self.subTest(rows=rows, width=width, mode=mode):
                self.assertTrue(torch.equal(_bits(keys), _bits(ref)))

    def _publish(self, data, logits, batched):
        blocks = torch.full(
            (data.num_rows, min(self.topk_blocks, -(-max(data.lens_per_request) // 8))),
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        fn = (
            dense_blocks._publish_tile_blocks_batched
            if batched
            else dense_blocks._publish_tile_blocks
        )
        work = logits.clone()  # the per-request path masks it in place
        fn(
            data=data,
            tile=slice(0, data.num_rows),
            logits=work,
            blocks=blocks,
            topk_blocks=self.topk_blocks,
            block_size=self.block,
        )
        return blocks

    def _consume(self, data, logits, blocks, batched, topk):
        out = data.empty_selection(topk)
        fn = (
            dense_blocks._consume_tile_blocks_batched
            if batched
            else dense_blocks._consume_tile_blocks
        )
        fn(
            data=data,
            tile=slice(0, data.num_rows),
            logits=logits,
            blocks=blocks,
            block_size=self.block,
            out=out,
        )
        return out

    def test_tile_source_and_consumer_bit_identical(self):
        checked = 0
        for name, requests in _CASES.items():
            for layout, (rows_per_request, lens, starts) in self._layouts(requests):
                if not lens:
                    continue
                data = _data(requests, rows_per_request, lens, starts)
                width = -(-max(lc for _, lc in requests) // 8) * 8
                for mode in ("random", "levels", "zeros", "nan"):
                    seed = zlib.crc32(f"{name}/{layout}/{mode}".encode()) % 10_000
                    src = _logits(data.num_rows, width, lens, mode, seed)
                    a = self._publish(data, src, batched=False)
                    b = self._publish(data, src, batched=True)
                    with self.subTest(
                        case=name, layout=layout, mode=mode, part="source"
                    ):
                        self.assertTrue(torch.equal(a, b))
                    con = _logits(data.num_rows, width, lens, mode, seed + 1)
                    for topk in (5, 37):
                        x = self._consume(data, con, a, batched=False, topk=topk)
                        y = self._consume(data, con, a, batched=True, topk=topk)
                        with self.subTest(
                            case=name,
                            layout=layout,
                            mode=mode,
                            topk=topk,
                            part="consumer",
                        ):
                            self.assertTrue(torch.equal(x, y))
                    checked += 1
        self.assertGreater(checked, 100)

    def test_published_ids_past_a_request_are_dropped_alike(self):
        # Ids at or past a request's blocks never come from a source; the consumer
        # still treats them alike in both paths (the causal mask drops them).
        requests = [(20, 300), (20, 900)]
        rows_per_request, lens, starts = _rows_layout(requests)
        data = _data(requests, rows_per_request, lens, starts)
        width = 904
        g = torch.Generator(device="cuda").manual_seed(5)
        blocks = torch.randint(-1, 113, (data.num_rows, 40), device="cuda", generator=g)
        blocks = blocks.to(torch.int32)
        con = _logits(data.num_rows, width, lens, "levels", 9)
        x = self._consume(data, con, blocks, batched=False, topk=31)
        y = self._consume(data, con, blocks, batched=True, topk=31)
        self.assertTrue(torch.equal(x, y))

    def test_model_shape_block_topk(self):
        # Layer 20 at the verdict shape on one CP4 rank, row count cut: 2048 kept
        # blocks of 8 over ~75k compressed positions, one request short of 2048 blocks.
        self.topk_blocks = 2048
        requests = [(160, 75400), (160, 74100), (40, 9000), (160, 75500)]
        for layout, (rows_per_request, lens, starts) in [
            ("cp2", _rows_layout(requests, cp_rank=2)),
            ("cp0 tail", _rows_layout(requests, cp_rank=0, tail=128)),
        ]:
            data = _data(requests, rows_per_request, lens, starts)
            width = -(-75500 // 4) * 4
            for mode in ("random", "levels", "zeros"):
                src = _logits(data.num_rows, width, lens, mode, 11)
                a = self._publish(data, src, batched=False)
                b = self._publish(data, src, batched=True)
                with self.subTest(layout=layout, mode=mode, part="source"):
                    self.assertTrue(torch.equal(a, b))
                con = _logits(data.num_rows, width + 4, lens, mode, 12)
                x = self._consume(data, con, a, batched=False, topk=512)
                y = self._consume(data, con, a, batched=True, topk=512)
                with self.subTest(layout=layout, mode=mode, part="consumer"):
                    self.assertTrue(torch.equal(x, y))


def _make_deep_gemm_inputs(request_lengths, *, zero_queries, signed_weights, seed):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
        fp4_index_rows_to_fp8,
        quantize_fp4_indexer_tensor,
    )

    torch.manual_seed(seed)
    rows = sum(q for q, _ in request_lengths)
    q = torch.randn((rows, 32, 128), dtype=torch.bfloat16, device="cuda")
    if zero_queries:
        q.zero_()
    packed, scales = quantize_fp4_indexer_tensor(q.flatten(0, 1), rne=True)
    kv = quantize_fp4_indexer_tensor(
        torch.randn(
            (sum(n for _, n in request_lengths), 128),
            dtype=torch.bfloat16,
            device="cuda",
        ),
        rne=True,
    )
    starts, lengths, start = [], [], 0
    for queries, context in request_lengths:
        starts.extend([start] * queries)
        lengths.extend(range(context - queries + 1, context + 1))
        start += context
    q = (packed.view(rows, 32, 64), scales.view(rows, 32))
    weights = torch.rand((rows, 32), dtype=torch.float32, device="cuda")
    if signed_weights:
        weights = weights * torch.where(torch.rand_like(weights) < 0.5, -1.0, 1.0)
        weights[::3] = -weights[::3].abs()  # rows whose zero scores are -0.0
    if _ARCH == 9:
        q_fp8, q_scale = fp4_index_rows_to_fp8(*q)
        q, weights, kv = (q_fp8, None), weights * q_scale, fp4_index_rows_to_fp8(*kv)
    data = scoring.DeepGEMMPrefillData(
        k_slots=torch.arange(kv[0].shape[0], device="cuda"),
        request_starts=torch.tensor(starts, dtype=torch.int32, device="cuda"),
        lens_per_request=[n for _, n in request_lengths],
        rows_per_request=[q_ for q_, _ in request_lengths],
        compress_lens=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        q=q[0],
        q_sf=q[1],
        weights=weights,
    )
    return data, kv


def _picked_scores(data, kv, selected):
    """Each row's selected scores, sorted (bits), -inf where nothing is selected."""
    with patch.object(scoring, "_DEEP_GEMM_SCORE_BUDGET_BYTES", 2 << 30):
        (tile, logits), *rest = list(scoring.score_tiles(data, kv, width_align=4))
    assert not rest and tile == slice(0, data.num_rows)
    columns = (selected - data.request_starts[:, None]).long()
    valid = selected >= 0
    assert bool(((columns < data.compress_lens[:, None]) | ~valid).all())
    picked = logits.gather(1, columns.clamp(0, logits.shape[1] - 1))
    return _bits(picked.masked_fill(~valid, -torch.inf).sort(-1).values)


@unittest.skipUnless(_ARCH in (9, 10), "requires SM90 or SM100")
class TestBatchedCandidateBlocksDeepGEMM(CustomTestCase):
    """Through the real score tiles (DeepGEMM logits), several tiles per call."""

    def test_publish_and_consume_through_score_tiles(self):
        cases = [
            [(1, 1), (33, 511), (257, 4097)],
            [(64, 5000), (64, 5100)],
            [(16, 700 + 300 * i) for i in range(10)],
            [(8, 3000)] * 16,
        ]
        for n, request_lengths in enumerate(cases):
            for zero_queries, signed in ((False, False), (True, False), (True, True)):
                data, kv = _make_deep_gemm_inputs(
                    request_lengths,
                    zero_queries=zero_queries,
                    signed_weights=signed,
                    seed=n,
                )
                consumer, _ = _make_deep_gemm_inputs(
                    request_lengths,
                    zero_queries=zero_queries,
                    signed_weights=signed,
                    seed=100 + n,
                )
                for budget in (2 << 30, 256 << 10):
                    with patch.object(scoring, "_DEEP_GEMM_SCORE_BUDGET_BYTES", budget):
                        ref = dense_blocks._publish_prefill_blocks(
                            data=data, kv=kv, topk=512, topk_blocks=128, block_size=8
                        )
                        got = dense_blocks._publish_prefill_blocks(
                            data=data,
                            kv=kv,
                            topk=512,
                            topk_blocks=128,
                            block_size=8,
                            batched=True,
                        )
                        sub = dict(
                            case=n, zero=zero_queries, signed=signed, budget=budget
                        )
                        # The source's own top-k (topk_transform_ragged_v2, untouched
                        # here) varies run to run in its order, and among tied scores
                        # in its set, with or without this switch: compare the scores
                        # it picks.
                        with self.subTest(**sub, part="source selection"):
                            self.assertTrue(
                                torch.equal(
                                    _picked_scores(data, kv, ref[0]),
                                    _picked_scores(data, kv, got[0]),
                                )
                            )
                        with self.subTest(**sub, part="source blocks"):
                            self.assertTrue(torch.equal(ref[1], got[1]))
                        sel_ref = dense_blocks._consume_prefill_blocks(
                            data=consumer, kv=kv, topk=512, blocks=ref[1], block_size=8
                        )
                        sel_got = dense_blocks._consume_prefill_blocks(
                            data=consumer,
                            kv=kv,
                            topk=512,
                            blocks=ref[1],
                            block_size=8,
                            batched=True,
                        )
                        with self.subTest(**sub, part="consumer"):
                            self.assertTrue(torch.equal(sel_ref, sel_got))


if __name__ == "__main__":
    unittest.main()
