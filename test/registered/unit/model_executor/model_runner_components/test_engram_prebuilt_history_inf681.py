"""PD decode admission: the Engram history rows for a PREBUILT batch (INF-681 k-dhost).

Upstream #44787 changed `NgramEmbeddingManager._prepare_engram_history` for PREBUILT batches in two ways:
the rows come from the tail of the prompt instead of a full `origin_input_ids + output_ids` copy, and both
host-to-device copies go through `async_h2d` (pinned, non_blocking). The second matters on the PD decoder:
the scheduler has just made its schedule stream wait for the in-flight forward (the prebuilt drain), so a
pageable `torch.tensor(..., device=cuda)` blocks the host until that forward ends and leaves the GPU idle
while the rest of the admission runs.

These tests run on CPU: the rows must equal the old full-copy computation for every prompt/output length,
and the PREBUILT branch must stage both copies through `async_h2d`.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.model_executor.model_runner_components import (  # noqa: E402
    ngram_embedding_manager as nem,
)

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _old_rows(reqs, n1):
    rows = []
    for req in reqs:
        fill_ids = req.origin_input_ids + req.output_ids
        end = len(fill_ids) - 1
        ids = fill_ids[max(0, end - n1) : end]
        rows.append([0] * (n1 - len(ids)) + list(ids))
    return rows


def _req(prompt_len, out_len, slot, base):
    return SimpleNamespace(
        origin_input_ids=list(range(base, base + prompt_len)),
        output_ids=list(range(base + 10**6, base + 10**6 + out_len)),
        kv=SimpleNamespace(req_pool_idx=slot),
    )


class _Hasher:
    def __init__(self, max_ngram_size, slots):
        self.max_ngram_size = max_ngram_size
        self.history = torch.full((slots, max_ngram_size - 1), -1, dtype=torch.int64)


def _manager(hasher):
    return nem.NgramEmbeddingManager(
        enabled=False, table=None, n=0, engram_hasher=hasher
    )


class TestEngramPrebuiltHistory(CustomTestCase):
    def test_rows_equal_full_copy(self):
        for max_ngram in (2, 3, 4):
            n1 = max_ngram - 1
            reqs = []
            slot = 0
            for prompt_len in (1, 2, 3, 4, 5, 75_000):
                for out_len in (0, 1, 2, 5):
                    if prompt_len + out_len == 0:
                        continue
                    reqs.append(_req(prompt_len, out_len, slot, base=slot * 100_000))
                    slot += 1
            hasher = _Hasher(max_ngram, slot + 3)
            batch = SimpleNamespace(forward_mode=ForwardMode.PREBUILT, reqs=reqs)
            _manager(hasher)._prepare_engram_history(batch)
            want = torch.tensor(_old_rows(reqs, n1), dtype=torch.int64)
            got = hasher.history[[r.kv.req_pool_idx for r in reqs]]
            self.assertTrue(torch.equal(got, want), f"max_ngram={max_ngram}")
            # rows not in the batch stay untouched
            self.assertTrue(
                torch.equal(
                    hasher.history[slot:], torch.full((3, n1), -1, dtype=torch.int64)
                )
            )

    def test_prebuilt_copies_are_staged(self):
        hasher = _Hasher(3, 8)
        reqs = [_req(75_000, 1, 2, 0), _req(10, 1, 5, 7)]
        batch = SimpleNamespace(forward_mode=ForwardMode.PREBUILT, reqs=reqs)
        with mock.patch.object(nem, "async_h2d", wraps=nem.async_h2d) as staged:
            _manager(hasher)._prepare_engram_history(batch)
        self.assertEqual(staged.call_count, 2)
        self.assertTrue(
            torch.equal(hasher.history[[2, 5]], torch.tensor(_old_rows(reqs, 2)))
        )


if __name__ == "__main__":
    unittest.main()
