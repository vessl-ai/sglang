import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    fp4_index_rows_to_fp8,
    quantize_fp4_indexer_tensor,
)
from sglang.srt.layers.attention.dsv4.v41_indexer import dense_blocks, scoring
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_ARCH = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
# SM90 scores e4m3-widened operands with fp8_mqa_logits, SM100 packed fp4 ones
# with fp8_fp4_mqa_logits; the selection logic under test is the same.
_FP8 = _ARCH == 9
_MQA_LOGITS = "fp8_mqa_logits" if _FP8 else "fp8_fp4_mqa_logits"


def make_inputs(request_lengths, ratio=1, zero_queries=False, seed=17):
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
    starts, lengths = [], []
    start = 0
    for queries, context in request_lengths:
        starts.extend([start] * queries)
        lengths.extend(
            (position + 1) // ratio
            for position in range(context * ratio - queries, context * ratio)
        )
        start += context
    q = (packed.view(rows, 32, 64), scales.view(rows, 32))
    weights = torch.rand((rows, 32), dtype=torch.float32, device="cuda")
    if _FP8:
        q_fp8, q_scale = fp4_index_rows_to_fp8(*q)
        q, weights, kv = (q_fp8, None), weights * q_scale, fp4_index_rows_to_fp8(*kv)
    return dict(
        q=q,
        kv=kv,
        weights=weights,
        starts=torch.tensor(starts, dtype=torch.int32, device="cuda"),
        lengths=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        request_lengths=request_lengths,
        topk=512,
        candidate_topk_blocks=2,
        candidate_block_size=8,
    )


def run_dense(inputs, *, publish_candidates, candidates):
    """The plain top-k, or the CP source's own top-k plus its candidate blocks
    (`publish_candidates`), or a CP consumer's top-k among `candidates`."""
    data = scoring.DeepGEMMPrefillData(
        k_slots=torch.arange(inputs["kv"][0].shape[0], device="cuda"),
        request_starts=inputs["starts"],
        lens_per_request=[n for _, n in inputs["request_lengths"]],
        rows_per_request=[q for q, _ in inputs["request_lengths"]],
        compress_lens=inputs["lengths"],
        q=inputs["q"][0],
        q_sf=inputs["q"][1],
        weights=inputs["weights"],
    )
    if publish_candidates:
        selected, blocks = dense_blocks._publish_prefill_blocks(
            data=data,
            kv=inputs["kv"],
            topk=inputs["topk"],
            topk_blocks=inputs["candidate_topk_blocks"],
            block_size=inputs["candidate_block_size"],
        )
        return selected, dense_blocks.BlockIds(
            blocks=blocks, rows_per_request=data.rows_per_request
        )
    if candidates is not None:
        selected = dense_blocks._consume_prefill_blocks(
            data=data,
            kv=inputs["kv"],
            topk=inputs["topk"],
            blocks=candidates.blocks,
            block_size=inputs["candidate_block_size"],
        )
        return selected, None
    # The plain top-k writes request-local positions into a -1 filled buffer;
    # the checks compare flattened-K columns, as the CP paths return.
    local = torch.full(
        (data.compress_lens.shape[0], inputs["topk"]),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    scoring.dense_prefill_topk(data, inputs["kv"], out=local)
    starts = data.request_starts[:, None]
    return torch.where(local >= 0, local + starts, local), None


def request_blocks(inputs, candidates):
    """Each request's rows of the published block ids, cut to its kept count
    ``min(topk_blocks, ceil(context / block))``."""
    out, row = [], 0
    block = inputs["candidate_block_size"]
    for queries, context in inputs["request_lengths"]:
        kept = min(inputs["candidate_topk_blocks"], -(-context // block))
        out.append(candidates.blocks[row : row + queries, :kept])
        row += queries
    return out


def dense_scores(inputs):
    import deep_gemm

    width = (max(n for _, n in inputs["request_lengths"]) + 3) // 4 * 4
    scores = getattr(deep_gemm, _MQA_LOGITS)(
        inputs["q"][0] if _FP8 else inputs["q"],
        inputs["kv"],
        inputs["weights"],
        inputs["starts"],
        inputs["starts"] + inputs["lengths"],
        False,
        width,
    )
    return scores.masked_fill_(
        torch.arange(width, device="cuda")[None, :] >= inputs["lengths"][:, None],
        -torch.inf,
    )


@unittest.skipUnless(_ARCH in (9, 10), "requires SM90 or SM100")
class TestDensePrefillIndexer(CustomTestCase):
    def assert_topk(self, inputs, selected, scores):
        columns = (selected - inputs["starts"][:, None]).long()
        valid = selected >= 0
        expected_count = torch.isfinite(scores).sum(-1).clamp_max(inputs["topk"])
        torch.testing.assert_close(valid.sum(-1), expected_count)
        self.assertTrue(
            (~valid | ((columns >= 0) & (columns < inputs["lengths"][:, None]))).all()
        )
        actual = scores.gather(1, columns.clamp(0, scores.shape[1] - 1)).masked_fill(
            ~valid, -torch.inf
        )
        expected = scores.topk(min(inputs["topk"], scores.shape[1]), dim=-1).values
        expected = torch.nn.functional.pad(
            expected, (0, inputs["topk"] - expected.shape[1]), value=-torch.inf
        )
        torch.testing.assert_close(
            actual.sort(descending=True).values, expected, rtol=1e-5, atol=1e-5
        )
        ordered = (
            columns.masked_fill(~valid, torch.iinfo(torch.int64).max).sort().values
        )
        self.assertTrue(
            (
                (ordered[:, 1:] != ordered[:, :-1])
                | (ordered[:, 1:] == torch.iinfo(torch.int64).max)
            ).all()
        )

    def test_ragged_source_consumer_and_replay(self):
        for ratio in (1, 2):
            for zero_queries in (False, True):
                with self.subTest(ratio=ratio, zero_queries=zero_queries):
                    inputs = make_inputs(
                        [(0, 0), (1, 1), (33, 511), (257, 4097)],
                        ratio=ratio,
                        zero_queries=zero_queries,
                    )
                    inputs["candidate_topk_blocks"] = 128
                    scores = dense_scores(inputs)
                    consumer_inputs = make_inputs(
                        inputs["request_lengths"],
                        ratio=ratio,
                        zero_queries=zero_queries,
                        seed=29,
                    )
                    consumer_inputs["kv"] = inputs["kv"]
                    consumer_inputs["candidate_topk_blocks"] = 128
                    consumer_scores = dense_scores(consumer_inputs)
                    with patch.object(
                        scoring, "_DEEP_GEMM_SCORE_BUDGET_BYTES", 128 << 10
                    ):
                        selected, candidates = run_dense(
                            inputs, publish_candidates=True, candidates=None
                        )
                        self.assert_topk(inputs, selected, scores)
                        row = 0
                        for (queries, context), blocks in zip(
                            inputs["request_lengths"],
                            request_blocks(inputs, candidates),
                        ):
                            local = scores[row : row + queries, :context]
                            if queries and context:
                                padded = torch.nn.functional.pad(
                                    local, (0, -context % 8), value=-torch.inf
                                )
                                block_scores = padded.unflatten(-1, (-1, 8)).amax(-1)
                                last = (inputs["lengths"][row : row + queries] - 1) // 8
                                block_scores.masked_fill_(
                                    torch.arange(block_scores.shape[1], device="cuda")[
                                        None, :
                                    ]
                                    == last[:, None],
                                    torch.inf,
                                )
                                chosen_scores = block_scores.gather(
                                    1, blocks.long().clamp_min(0)
                                ).masked_fill(blocks < 0, -torch.inf)
                                torch.testing.assert_close(
                                    chosen_scores.sort(descending=True).values,
                                    block_scores.topk(blocks.shape[1]).values,
                                    rtol=1e-5,
                                    atol=1e-5,
                                )
                                columns = torch.arange(context, device="cuda")
                                member = (
                                    columns[None, :, None] // 8 == blocks[:, None, :]
                                ).any(-1)
                                consumer_scores[
                                    row : row + queries, :context
                                ].masked_fill_(
                                    ~member,
                                    -torch.inf,
                                )
                            row += queries
                        self.assertGreater(
                            torch.isfinite(consumer_scores[-1]).sum().item(),
                            inputs["topk"],
                        )
                        self.assertLess(
                            torch.isfinite(consumer_scores[-1]).sum().item(),
                            inputs["lengths"][-1].item(),
                        )
                        selected, published = run_dense(
                            consumer_inputs,
                            publish_candidates=False,
                            candidates=candidates,
                        )
                        self.assertIsNone(published)
                        self.assert_topk(consumer_inputs, selected, consumer_scores)
                        tail_lengths = [0, 0, 7, 31]
                        rows, row = [], 0
                        for (queries, _), tail in zip(
                            inputs["request_lengths"], tail_lengths
                        ):
                            rows.extend(range(row + queries - tail, row + queries))
                            row += queries
                        rows = torch.tensor(rows, dtype=torch.int64, device="cuda")
                        tail_inputs = dict(
                            consumer_inputs,
                            q=tuple(
                                None if t is None else t[rows]
                                for t in consumer_inputs["q"]
                            ),
                            weights=consumer_inputs["weights"][rows],
                            starts=inputs["starts"][rows],
                            lengths=inputs["lengths"][rows],
                            request_lengths=list(
                                zip(
                                    tail_lengths,
                                    [n for _, n in inputs["request_lengths"]],
                                )
                            ),
                        )
                        selected, _ = run_dense(
                            tail_inputs,
                            publish_candidates=False,
                            candidates=candidates.tail(tail_lengths),
                        )
                        self.assert_topk(tail_inputs, selected, consumer_scores[rows])

    def test_row_shards_merge_to_the_whole(self):
        # What row-sharded ranks compute (deep_gemm_row_shard): each scores its own
        # row slice; the max-merge of the -1 padded parts is the unsharded result.
        inputs = make_inputs([(5, 300), (40, 1031), (90, 4097)], ratio=2)
        inputs["candidate_topk_blocks"] = 64
        data = scoring.DeepGEMMPrefillData(
            k_slots=torch.arange(inputs["kv"][0].shape[0], device="cuda"),
            request_starts=inputs["starts"],
            lens_per_request=[n for _, n in inputs["request_lengths"]],
            rows_per_request=[q for q, _ in inputs["request_lengths"]],
            compress_lens=inputs["lengths"],
            q=inputs["q"][0],
            q_sf=inputs["q"][1],
            weights=inputs["weights"],
        )
        rows, topk = data.num_rows, inputs["topk"]
        whole = torch.full((rows, topk), -1, dtype=torch.int32, device="cuda")
        scoring.dense_prefill_topk(data, inputs["kv"], out=whole)
        whole_sel, whole_blocks = dense_blocks._publish_prefill_blocks(
            data=data, kv=inputs["kv"], topk=topk, topk_blocks=64, block_size=8
        )
        # A candidate consumer layer reads the merged (whole) published blocks and,
        # sharded, the block rows of its own row slice.
        whole_consumed = dense_blocks._consume_prefill_blocks(
            data=data, kv=inputs["kv"], topk=topk, blocks=whole_blocks, block_size=8
        )
        self.assertTrue(bool((whole_consumed >= 0).any()))
        for size in (2, 3, 4):
            merged = torch.full_like(whole, -1)
            merged_sel, merged_blocks = (
                torch.full_like(whole_sel, -1),
                torch.full_like(whole_blocks, -1),
            )
            merged_consumed = torch.full_like(whole_consumed, -1)
            for rank in range(size):
                shard = slice(rows * rank // size, rows * (rank + 1) // size)
                part = data.row_slice(shard)
                self.assertEqual(sum(part.rows_per_request), shard.stop - shard.start)
                out = torch.full_like(whole, -1)
                scoring.dense_prefill_topk(part, inputs["kv"], out=out[shard])
                merged = torch.maximum(merged, out)
                sel, blocks = dense_blocks._publish_prefill_blocks(
                    data=part, kv=inputs["kv"], topk=topk, topk_blocks=64, block_size=8
                )
                merged_sel = torch.maximum(
                    merged_sel, dense_blocks._unshard_rows(sel, shard, rows)
                )
                merged_blocks = torch.maximum(
                    merged_blocks, dense_blocks._unshard_rows(blocks, shard, rows)
                )
                consumed = dense_blocks._consume_prefill_blocks(
                    data=part,
                    kv=inputs["kv"],
                    topk=topk,
                    blocks=whole_blocks[shard],
                    block_size=8,
                )
                merged_consumed = torch.maximum(
                    merged_consumed, dense_blocks._unshard_rows(consumed, shard, rows)
                )
            for actual, expected in (
                (merged, whole),
                (merged_sel, whole_sel),
                (merged_blocks, whole_blocks),
                (merged_consumed, whole_consumed),
            ):
                torch.testing.assert_close(
                    actual.sort(-1).values, expected.sort(-1).values, rtol=0, atol=0
                )

    def test_cp_interleaved_rows_match_the_whole_batch(self):
        # What a prefill-CP rank scores: the batch's rows whose flattened index is
        # congruent to its rank, still grouped by request, against the same
        # flattened K. Its top-k, published blocks and candidate top-k must be the
        # whole batch's on those rows.
        from sglang.srt.layers.cp.interleave import interleave_rows_per_request

        for ratio in (1, 2):
            inputs = make_inputs(
                [(5, 300), (2, 7), (41, 1031), (90, 4097)], ratio=ratio
            )
            inputs["candidate_topk_blocks"] = 64
            queries = [q for q, _ in inputs["request_lengths"]]
            rows, topk = sum(queries), inputs["topk"]

            def data_for(index, rows_per_request):
                return scoring.DeepGEMMPrefillData(
                    k_slots=torch.arange(inputs["kv"][0].shape[0], device="cuda"),
                    request_starts=inputs["starts"][index],
                    lens_per_request=[n for _, n in inputs["request_lengths"]],
                    rows_per_request=rows_per_request,
                    compress_lens=inputs["lengths"][index],
                    q=inputs["q"][0][index],
                    q_sf=None if inputs["q"][1] is None else inputs["q"][1][index],
                    weights=inputs["weights"][index],
                )

            whole = data_for(torch.arange(rows, device="cuda"), queries)
            whole_topk = torch.full((rows, topk), -1, dtype=torch.int32, device="cuda")
            scoring.dense_prefill_topk(whole, inputs["kv"], out=whole_topk)
            whole_sel, whole_blocks = dense_blocks._publish_prefill_blocks(
                data=whole, kv=inputs["kv"], topk=topk, topk_blocks=64, block_size=8
            )
            whole_consumed = dense_blocks._consume_prefill_blocks(
                data=whole,
                kv=inputs["kv"],
                topk=topk,
                blocks=whole_blocks,
                block_size=8,
            )
            for cp_size in (2, 4):
                for rank in range(cp_size):
                    with self.subTest(ratio=ratio, cp_size=cp_size, rank=rank):
                        index = torch.arange(rank, rows, cp_size, device="cuda")
                        counts = interleave_rows_per_request(queries, rank, cp_size)
                        self.assertEqual(sum(counts), index.numel())
                        part = data_for(index, counts)
                        part_topk = torch.full(
                            (index.numel(), topk), -1, dtype=torch.int32, device="cuda"
                        )
                        scoring.dense_prefill_topk(part, inputs["kv"], out=part_topk)
                        sel, blocks = dense_blocks._publish_prefill_blocks(
                            data=part,
                            kv=inputs["kv"],
                            topk=topk,
                            topk_blocks=64,
                            block_size=8,
                        )
                        consumed = dense_blocks._consume_prefill_blocks(
                            data=part,
                            kv=inputs["kv"],
                            topk=topk,
                            blocks=blocks,
                            block_size=8,
                        )
                        for actual, expected in (
                            (part_topk, whole_topk[index]),
                            (sel, whole_sel[index]),
                            (blocks, whole_blocks[index]),
                            (consumed, whole_consumed[index]),
                        ):
                            torch.testing.assert_close(
                                actual.sort(-1).values,
                                expected.sort(-1).values,
                                rtol=0,
                                atol=0,
                            )

    def test_unfiltered_and_zero_length_requests(self):
        for request_lengths in ([(257, 8192)], [(1, 0), (1, 1), (0, 7)]):
            for publish in (False, True):
                with self.subTest(request_lengths=request_lengths, publish=publish):
                    inputs = make_inputs(request_lengths)
                    selected, candidates = run_dense(
                        inputs, publish_candidates=publish, candidates=None
                    )
                    self.assert_topk(inputs, selected, dense_scores(inputs))
                    if publish:
                        self.assertEqual(
                            [b.shape[0] for b in request_blocks(inputs, candidates)],
                            [q for q, _ in request_lengths],
                        )

    def test_empty_queries_or_context(self):
        # block ids are [rows, min(topk_blocks, ceil(max context / 8))]
        for request_lengths, shape, block_shape in (
            ([], (0, 512), (0, 0)),
            ([(0, 0)], (0, 512), (0, 0)),
            ([(0, 0), (0, 17)], (0, 512), (0, 2)),
            ([(1, 0)], (1, 512), (1, 0)),
        ):
            with self.subTest(request_lengths=request_lengths):
                inputs = make_inputs(request_lengths)
                selected, candidates = run_dense(
                    inputs, publish_candidates=True, candidates=None
                )
                torch.testing.assert_close(
                    selected, torch.full(shape, -1, dtype=torch.int32, device="cuda")
                )
                self.assertEqual(tuple(candidates.blocks.shape), block_shape)

    def test_score_budget_includes_allocation_padding(self):
        import deep_gemm

        mqa_logits = getattr(deep_gemm, _MQA_LOGITS)
        inputs = make_inputs([(13, 257)])
        expected = dense_scores(inputs)
        for budget in (32 << 10, 28 << 10, 8 << 10):
            with self.subTest(budget=budget):
                allocations = []

                def checked_logits(*args, **kwargs):
                    before = torch.cuda.memory_allocated()
                    logits = mqa_logits(*args, **kwargs)
                    allocations.append(torch.cuda.memory_allocated() - before)
                    return logits

                with (
                    patch.object(scoring, "_DEEP_GEMM_SCORE_BUDGET_BYTES", budget),
                    patch(f"deep_gemm.{_MQA_LOGITS}", new=checked_logits),
                ):
                    selected, _ = run_dense(
                        inputs, publish_candidates=True, candidates=None
                    )
                self.assertTrue(allocations)
                self.assertLessEqual(max(allocations), budget)
                self.assert_topk(inputs, selected, expected)

    def test_score_memory_is_bounded(self):
        for context, limit_gib in ((65536, 3), (65535, 5)):
            with self.subTest(context=context):
                inputs = make_inputs([(16384, context)])
                inputs["candidate_topk_blocks"] = 2048
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()
                selected, candidates = run_dense(
                    inputs, publish_candidates=True, candidates=None
                )
                torch.cuda.synchronize()
                self.assertLess(
                    torch.cuda.max_memory_allocated() - baseline, limit_gib << 30
                )
                self.assertEqual(tuple(selected.shape), (16384, 512))
                self.assertEqual(tuple(candidates.blocks.shape), (16384, 2048))
                del selected
                selected, published = run_dense(
                    inputs, publish_candidates=False, candidates=candidates
                )
                self.assertIsNone(published)
                del selected
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                for _ in range(3):
                    torch.cuda.reset_peak_memory_stats()
                    selected, published = run_dense(
                        inputs, publish_candidates=False, candidates=candidates
                    )
                    torch.cuda.synchronize()
                    self.assertIsNone(published)
                    self.assertLess(
                        torch.cuda.max_memory_allocated() - baseline, 4 << 30
                    )
                    self.assertEqual(tuple(selected.shape), (16384, 512))
                    del selected
                    torch.cuda.synchronize()
                    self.assertEqual(torch.cuda.memory_allocated(), baseline)
                del inputs, candidates


if __name__ == "__main__":
    unittest.main()
