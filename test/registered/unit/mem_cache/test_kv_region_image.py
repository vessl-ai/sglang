"""KV region image: a released and resumed KV region gets its startup bytes back.

The pools are built on CPU and each tensor stands in for one memory-saver
allocation. "Resume" is simulated by filling every allocation with 0xFF, which
is what fresh physical pages may hold (and NaN for every float dtype).
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache import kv_region_image as kri
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# These tests build CPU tensors without naming the device. Another test in the
# same process may have left torch's default device set to cuda; pin it to CPU
# for this module and give the previous setting back afterwards.
_cpu_default_device = contextlib.ExitStack()


def setUpModule():
    _cpu_default_device.enter_context(torch.device("cpu"))


def tearDownModule():
    _cpu_default_device.close()


def _pools():
    c4 = CompressStatePool(
        size=4096,
        ring_size=8,
        overlap=True,
        head_dim=64,
        dtype=torch.float32,
        device="cpu",
        enable_memory_saver=False,
        ratio=4,
        online=False,
        request_scoped=False,
        swa_page_size=8,
    )
    pair = CompressStatePool(
        size=64 * 2,
        ring_size=2,
        overlap=False,
        head_dim=64,
        dtype=torch.float32,
        device="cpu",
        enable_memory_saver=False,
        ratio=2,
        request_scoped=True,
        online=False,
    )
    req = ReqToTokenPool(
        size=32, max_context_len=4096, device="cpu", enable_memory_saver=False
    )
    kv = torch.zeros(512, 64 * 1024, dtype=torch.uint8)  # 32 MiB of KV pages
    return c4, pair, req, kv


def _allocations(c4, pair, req, kv):
    tensors = [
        c4.kv_score_buffer.kv_score,
        pair.kv_score_buffer.kv_score,
        req.req_to_token,
        kv,
    ]
    # The fresh pool's torch.empty rows are arbitrary; make them deterministic so
    # "same bytes as at startup" is checkable (on GPU they are whatever they are).
    c4.kv_score_buffer.kv_score[:-1].fill_(0)
    return {(t.data_ptr(), t.nbytes, 0): t.view(-1).view(torch.uint8) for t in tensors}


class TestKVRegionImage(unittest.TestCase):
    def _region(self):
        c4, pair, req, kv = _pools()
        allocs = _allocations(c4, pair, req, kv)
        # Warmup wrote a few KV pages and one request row.
        kv[3].fill_(7)
        kv[200, :100].fill_(9)
        req.req_to_token[1, :10] = torch.arange(1, 11, dtype=torch.int32)
        view = lambda a, s, d: allocs[(a, s, d)]  # noqa: E731
        return (c4, pair, req, kv), allocs, view

    def test_resume_then_apply_gives_the_startup_bytes_back(self):
        pools, allocs, view = self._region()
        before = {seg: t.clone() for seg, t in allocs.items()}
        image = kri.KVRegionImage.capture(list(allocs), view=view)
        # Mostly zero: only the non-zero chunks are kept.
        self.assertLess(image.nbytes, sum(s[1] for s in allocs) // 10)
        for t in allocs.values():  # memory saver resume: fresh pages
            t.fill_(0xFF)
        image.apply(list(allocs), view=view)
        for seg, t in allocs.items():
            self.assertTrue(torch.equal(t, before[seg]), seg)
        c4, pair, req, kv = pools
        self.assertTrue(torch.all(c4.kv_score_buffer[-1].score == float("-inf")))
        self.assertTrue(torch.all(pair.kv_score_buffer.score == float("-inf")))

    def test_matches_a_freshly_constructed_region_byte_for_byte(self):
        # Without warmup writes the restored region equals a new one exactly.
        c4, pair, req, kv = _pools()
        allocs = _allocations(c4, pair, req, kv)
        view = lambda a, s, d: allocs[(a, s, d)]  # noqa: E731
        image = kri.KVRegionImage.capture(list(allocs), view=view)
        for t in allocs.values():
            t.fill_(0xFF)
        image.apply(list(allocs), view=view)
        fresh = _allocations(*_pools())
        for got, want in zip(allocs.values(), fresh.values()):
            self.assertTrue(torch.equal(got, want))

    def test_tail_smaller_than_a_chunk(self):
        t = torch.zeros(kri.CHUNK_BYTES * 2 + 24, dtype=torch.uint8)
        t[-3] = 5
        seg = (t.data_ptr(), t.nbytes, 0)
        image = kri.KVRegionImage.capture([seg], view=lambda *a: t)
        t.fill_(0xFF)
        image.apply([seg], view=lambda *a: t)
        self.assertEqual(int(t.sum()), 5)

    def test_refuses_a_changed_layout_and_an_oversized_image(self):
        _, allocs, view = self._region()
        image = kri.KVRegionImage.capture(list(allocs), view=view)
        with self.assertRaisesRegex(RuntimeError, "differ"):
            image.apply(list(allocs)[:-1], view=view)
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            kri.KVRegionImage.capture(list(allocs), view=view, max_bytes=1)

    def test_segments_come_from_the_tags_mem_pools(self):
        pool = MagicMock()
        pool.snapshot.return_value = [
            {"address": 0x2000, "total_size": 4096, "device": 0},
            {"address": 0x1000, "total_size": 2048, "device": 0},
        ]
        other = MagicMock()
        other.snapshot.return_value = [
            {"address": 0x9000, "total_size": 1, "device": 0}
        ]
        impl = SimpleNamespace(
            _mem_pools={
                ("kv_cache", False, False, "", 0): pool,
                ("weights", False, False, "", 0): other,
            }
        )
        tms = SimpleNamespace(torch_memory_saver=SimpleNamespace(_impl=impl))
        with patch.dict("sys.modules", {"torch_memory_saver": tms}):
            segs = kri.memory_saver_segments("kv_cache")
        self.assertEqual(segs, [(0x1000, 2048, 0), (0x2000, 4096, 0)])


if __name__ == "__main__":
    unittest.main()
