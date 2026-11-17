"""CPU checks of SGLANG_DSV41_INDEXER_WAVE_TILES: the DeepGEMM prefill indexer's
logits tiles rounded down to whole SM waves, the engine-start log line, and the
self-check against the unrounded tiles. DeepGEMM is replaced by a fake whose
logits depend only on each row, and the SM count is fixed."""

import os
import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4 import index_logits
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HEADS = 64  # row alignment 128 // 64 = 2
NUM_SMS = 3  # one wave = 2 * 3 = 6 rows
ROWS, WIDTH = 40, 64  # one row of fp32 logits = 256 columns * 4 bytes
ROW_BYTES = 1024
# 15 rows fit the budget: 14 rows per tile at the row alignment, 12 in whole waves.
BUDGET = 15 * ROW_BYTES


@contextmanager
def wave_env(on=None, self_check=None):
    with (
        patch.dict(os.environ),
        patch.object(index_logits, "_wave_tile_self_checks", 0),
        patch.object(index_logits, "_num_sms", lambda: NUM_SMS),
        patch.object(torch.version, "cuda", "12.9"),
    ):
        envs.SGLANG_DSV41_INDEXER_WAVE_TILES.clear()
        envs.SGLANG_DSV41_INDEXER_WAVE_TILES_SELFCHECK.clear()
        if on is not None:
            envs.SGLANG_DSV41_INDEXER_WAVE_TILES.set(on)
        if self_check is not None:
            envs.SGLANG_DSV41_INDEXER_WAVE_TILES_SELFCHECK.set(self_check)
        yield


class FakeDeepGEMM:
    """``fp8_fp4_mqa_logits`` whose row ``i`` is ``1000 * i + column``, ``i`` read
    from the query; ``tile_dependent`` adds the tile's row count, so two tilings
    of the same rows disagree."""

    def __init__(self, tile_dependent=False):
        self.tile_dependent = tile_dependent
        self.tile_rows = []

    def fp8_fp4_mqa_logits(self, q, kv, weights, starts, ends, _, width, **__):
        rows = q[0].shape[0]
        self.tile_rows.append(rows)
        row_ids = q[0][:, 0, 0].to(torch.float32)
        logits = row_ids[:, None] * 1000 + torch.arange(width, dtype=torch.float32)
        if self.tile_dependent:
            logits = logits + rows
        return logits


def run_tiles(fake):
    q_fp4 = torch.zeros(ROWS, HEADS, 1)
    q_fp4[:, 0, 0] = torch.arange(ROWS, dtype=torch.float32)
    lengths = torch.full((ROWS,), WIDTH, dtype=torch.int32)
    with patch.dict(
        sys.modules,
        {"deep_gemm": SimpleNamespace(fp8_fp4_mqa_logits=fake.fp8_fp4_mqa_logits)},
    ):
        tiles = list(
            index_logits.flat_index_logits_tiles(
                q=(q_fp4, torch.zeros(ROWS, HEADS)),
                kv=(torch.zeros(1), torch.zeros(1)),
                weights=torch.ones(ROWS, HEADS),
                starts=torch.zeros(ROWS, dtype=torch.int32),
                lengths=lengths,
                context_lengths=[WIDTH],
                budget_bytes=BUDGET,
            )
        )
    return tiles


class TestRowsPerTile(CustomTestCase):
    def rows_per_tile(self, budget_rows):
        return index_logits.flat_index_logits_rows_per_tile(
            ROWS, WIDTH, heads=HEADS, budget_bytes=budget_rows * ROW_BYTES
        )

    def test_off_keeps_the_row_aligned_tile(self):
        with wave_env():
            self.assertEqual(self.rows_per_tile(15), 14)

    def test_on_rounds_a_tile_past_one_wave_down_to_whole_waves(self):
        with wave_env(on=True):
            self.assertEqual(self.rows_per_tile(15), 12)
            self.assertEqual(self.rows_per_tile(24), 24)

    def test_on_keeps_a_tile_within_one_wave(self):
        with wave_env(on=True):
            self.assertEqual(self.rows_per_tile(5), 4)

    def test_on_keeps_one_tile_when_all_rows_fit(self):
        with wave_env(on=True):
            self.assertEqual(self.rows_per_tile(ROWS), ROWS)


class TestTiles(CustomTestCase):
    def assert_logits_of_every_row(self, tiles):
        full = torch.cat([logits for _, logits in tiles])
        expected = torch.arange(ROWS, dtype=torch.float32)[
            :, None
        ] * 1000 + torch.arange(WIDTH, dtype=torch.float32)
        self.assertTrue(torch.equal(full, expected))

    def test_off_tiles_by_the_row_aligned_tile_without_self_check(self):
        fake = FakeDeepGEMM()
        with wave_env(self_check=5):
            self.assert_logits_of_every_row(run_tiles(fake))
        self.assertEqual(fake.tile_rows, [14, 14, 12])

    def test_on_tiles_in_whole_waves(self):
        fake = FakeDeepGEMM()
        with wave_env(on=True):
            tiles = run_tiles(fake)
        self.assertEqual([t.stop - t.start for t, _ in tiles], [12, 12, 12, 4])
        self.assert_logits_of_every_row(tiles)

    def test_self_check_logs_no_difference_for_row_wise_logits(self):
        fake = FakeDeepGEMM()
        with (
            wave_env(on=True, self_check=1),
            self.assertLogs(index_logits.logger, "INFO") as logs,
        ):
            self.assert_logits_of_every_row(run_tiles(fake))
        self.assertIn("0 of 40 rows in 4 tiles differ", logs.output[-1])
        self.assertEqual(sorted(fake.tile_rows), sorted([12, 12, 12, 4, 14, 14, 12]))

    def test_self_check_warns_when_the_tiling_changes_the_logits(self):
        with (
            wave_env(on=True, self_check=1),
            self.assertLogs(index_logits.logger, "WARNING") as logs,
        ):
            run_tiles(FakeDeepGEMM(tile_dependent=True))
        self.assertIn("rows in 4 tiles differ", logs.output[-1])
        self.assertNotIn(" 0 of 40", logs.output[-1])

    def test_self_check_stops_after_its_count(self):
        fake = FakeDeepGEMM()
        with wave_env(on=True, self_check=1):
            run_tiles(fake)
            fake.tile_rows.clear()
            run_tiles(fake)
        self.assertEqual(fake.tile_rows, [12, 12, 12, 4])


class TestStartLog(CustomTestCase):
    def test_logs_when_on(self):
        with wave_env(on=True), self.assertLogs(index_logits.logger, "INFO") as logs:
            index_logits.log_wave_tiles()
        self.assertIn("logits tiles rounded to whole SM waves", logs.output[0])

    def test_silent_when_off(self):
        with wave_env(), self.assertNoLogs(index_logits.logger):
            index_logits.log_wave_tiles()


if __name__ == "__main__":
    unittest.main()
