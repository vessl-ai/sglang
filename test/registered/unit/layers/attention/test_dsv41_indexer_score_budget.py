"""CPU checks of SGLANG_DSV41_INDEXER_SCORE_BUDGET_MB: the budget the DeepGEMM
prefill indexer's logits tiles are scored under, the engine-start log line, and
the self-check against the built-in budget. DeepGEMM is replaced by a fake that
tiles a fixed score matrix by its budget."""

import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4 import v41_indexer
from sglang.srt.layers.attention.dsv4.v41_indexer import scoring
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROWS, WIDTH = 17, 8
BUILT_IN = scoring._DEEP_GEMM_SCORE_BUDGET_BYTES
# Rows per fake tile: 7 under the built-in budget and 3 under 1 MB, so the two
# tilings share no boundary but the first and the last.
ROWS_PER_TILE = {BUILT_IN: 7, 1 << 20: 3}


@contextmanager
def budget_env(budget_mb=None, self_check=None):
    with patch.dict(os.environ), patch.object(scoring, "_score_self_checks", 0):
        envs.SGLANG_DSV41_INDEXER_SCORE_BUDGET_MB.clear()
        envs.SGLANG_DSV41_INDEXER_SCORE_BUDGET_SELFCHECK.clear()
        if budget_mb is not None:
            envs.SGLANG_DSV41_INDEXER_SCORE_BUDGET_MB.set(budget_mb)
        if self_check is not None:
            envs.SGLANG_DSV41_INDEXER_SCORE_BUDGET_SELFCHECK.set(self_check)
        yield


class FakeTiles:
    """Stands in for flat_index_logits_tiles: tiles ``scores`` by the budget it is
    given, with budget-dependent garbage past each row's length."""

    def __init__(self, lengths, corrupt_rows=()):
        self.lengths = lengths
        self.scores = torch.arange(ROWS * WIDTH, dtype=torch.float32).view(ROWS, WIDTH)
        self.corrupt_rows = corrupt_rows
        self.budgets = []

    def __call__(self, *, lengths, budget_bytes, **_):
        self.budgets.append(budget_bytes)
        scores = self.scores.clone()
        past = torch.arange(WIDTH)[None, :] >= lengths[:, None]
        scores[past] = float(budget_bytes)
        if budget_bytes == BUILT_IN:
            for row in self.corrupt_rows:
                scores[row, 0] += 1
        step = ROWS_PER_TILE.get(budget_bytes, ROWS)
        for start in range(0, ROWS, step):
            tile = slice(start, min(start + step, ROWS))
            yield tile, scores[tile]


def make_data():
    lengths = (torch.arange(ROWS) % WIDTH) + 1
    data = SimpleNamespace(
        q_fp4=torch.zeros(ROWS, 2, 64, dtype=torch.int8),
        q_sf=torch.zeros(ROWS, 2, dtype=torch.int32),
        weights=torch.ones(ROWS, 2),
        request_starts=torch.zeros(ROWS, dtype=torch.int32),
        compress_lens=lengths,
        lens_per_request=[WIDTH],
    )
    return data, lengths


class TestScoreTileBudget(CustomTestCase):
    def run_tiles(self, fake, data):
        with patch.object(scoring, "flat_index_logits_tiles", new=fake):
            return list(scoring.score_tiles(data, kv=None))

    def test_unset_scores_under_the_built_in_budget_only(self):
        data, lengths = make_data()
        fake = FakeTiles(lengths)
        # The self-check belongs to the switch: alone it must not score anything.
        with budget_env(self_check=3):
            self.run_tiles(fake, data)
        self.assertEqual(fake.budgets, [BUILT_IN])

    def test_set_scores_under_the_given_megabytes(self):
        data, lengths = make_data()
        fake = FakeTiles(lengths)
        with budget_env(budget_mb=512):
            self.run_tiles(fake, data)
        self.assertEqual(fake.budgets, [512 << 20])

    def test_non_positive_budget_is_refused(self):
        with budget_env(budget_mb=0), self.assertRaises(ValueError):
            scoring.log_score_tile_budget()

    def test_self_check_passes_the_tiles_through_and_counts_no_difference(self):
        data, lengths = make_data()
        fake = FakeTiles(lengths)
        with budget_env(budget_mb=1, self_check=1):
            with self.assertLogs(scoring.logger, level="INFO") as logs:
                tiles = self.run_tiles(fake, data)
            self.assertIn("0 of 17 rows in 6 tiles differ", logs.output[0])
            self.assertEqual(
                [tile for tile, _ in tiles],
                [slice(s, min(s + 3, ROWS)) for s in range(0, ROWS, 3)],
            )
            valid = torch.arange(WIDTH)[None, :] < lengths[:, None]
            scored = torch.cat([logits for _, logits in tiles])
            self.assertTrue(torch.equal(scored[valid], fake.scores[valid]))
            # Only the first call is checked.
            fake.budgets.clear()
            self.run_tiles(fake, data)
            self.assertEqual(fake.budgets, [1 << 20])

    def test_self_check_reports_rows_whose_logits_differ(self):
        data, lengths = make_data()
        # One row in each of two built-in tiles, away from the tile boundaries.
        fake = FakeTiles(lengths, corrupt_rows=(4, 12))
        with budget_env(budget_mb=1, self_check=1):
            with self.assertLogs(scoring.logger, level="WARNING") as logs:
                self.run_tiles(fake, data)
        self.assertIn("2 of 17 rows", logs.output[0])


class TestEngineStartLine(CustomTestCase):
    """The line is a promise to the readback that proves the switch is on in a
    running engine (a log regex held outside this repository)."""

    def make_indexer(self, use_deep_gemm_prefill):
        with (
            patch.object(
                v41_indexer,
                "_use_deep_gemm_prefill",
                return_value=use_deep_gemm_prefill,
            ),
            patch.object(v41_indexer, "is_sm100_or_newer", return_value=False),
        ):
            v41_indexer.make_full_topk_indexer(token_to_kv_pool=None, req_to_token=None)

    def test_line_names_the_budget_when_the_deep_gemm_prefill_indexer_is_built(self):
        with budget_env(budget_mb=512):
            with self.assertLogs(scoring.logger, level="INFO") as logs:
                self.make_indexer(use_deep_gemm_prefill=True)
        self.assertRegex(
            logs.output[0], r"DSV4\.1 prefill indexer: logits tile budget 512 MB"
        )

    def test_no_line_when_unset_or_without_the_deep_gemm_prefill_indexer(self):
        for budget_mb, deep_gemm in ((None, True), (512, False)):
            with self.subTest(budget_mb=budget_mb, deep_gemm=deep_gemm):
                with budget_env(budget_mb=budget_mb):
                    with self.assertNoLogs(scoring.logger, level="INFO"):
                        self.make_indexer(use_deep_gemm_prefill=deep_gemm)


if __name__ == "__main__":
    unittest.main()
