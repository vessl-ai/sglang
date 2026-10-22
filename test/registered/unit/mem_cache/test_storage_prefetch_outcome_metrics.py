"""L3 prefetch outcome counters: where each one is counted; no GPU kernels."""

import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock, call

import torch

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    PrefetchOperation,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _OngoingPrefetch,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_REQ = CacheRequestHandle("r", 0)
_PAGE = 64


def _cache(policy="timeout", collector=None):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.tree_core = SimpleNamespace(page_size=_PAGE)  # backs cache.page_size
    cache.prefetch_stop_policy = policy
    cache.enable_storage_metrics = collector is not None
    cache.storage_metrics_collector = collector
    cache._prefetch_outcome_stats = defaultdict(int)
    cache.ongoing_prefetch = {}
    cache.storage_prefetch_retries = StoragePrefetchRetries()
    cache.revoke_pending_prefetch = Mock()
    cache._handle_prefetch_result = Mock()
    cache._can_terminate_prefetch = Mock(return_value=True)
    cache.cache_controller = SimpleNamespace(
        pp_prefetch_decisions={},
        terminate_prefetch=Mock(return_value=(0, [])),
        append_host_mem_release=Mock(),
        prefetch_tokens_occupied=0,
    )
    return cache


def _operation(requested=8 * _PAGE):
    op = PrefetchOperation(_REQ, list(range(requested)))
    op.stats_requested_tokens = requested
    return op


def _track(cache, op, host_indices=None, comp_xfers=None):
    cache.ongoing_prefetch[_REQ] = _OngoingPrefetch(
        0,
        RadixKey(list(range(op.stats_requested_tokens))),
        host_indices,
        op,
        None,
        comp_xfers or {},
    )


class TestPrefetchOutcomeCounter(unittest.TestCase):
    def test_forwards_to_the_collector_only_when_metrics_are_on(self):
        collector = Mock()
        cache = _cache(collector=collector)
        cache._count_prefetch_outcome("redundant_tokens", 128)
        cache._count_prefetch_outcome("issued")
        self.assertEqual(cache._prefetch_outcome_stats["redundant_tokens"], 128)
        self.assertEqual(
            collector.log_prefetch_outcome.call_args_list,
            [call("redundant_tokens", 128), call("issued", 1)],
        )

        off = _cache()
        off._count_prefetch_outcome("issued")
        self.assertEqual(off._prefetch_outcome_stats["issued"], 1)

        # A collector without storage metrics enabled is never touched.
        muted = _cache()
        muted.storage_metrics_collector = Mock()
        muted._count_prefetch_outcome("issued")
        muted.storage_metrics_collector.log_prefetch_outcome.assert_not_called()


class TestPrefetchDeadline(unittest.TestCase):
    def test_deadline_before_the_lookup_answers_loses_the_requested_span(self):
        collector = Mock()
        cache = _cache(collector=collector)
        op = _operation(requested=10 * _PAGE)
        _track(cache, op)

        self.assertTrue(cache.check_prefetch_progress(_REQ))

        stats = cache._prefetch_outcome_stats
        self.assertEqual(stats["revoked_deadline"], 1)
        self.assertEqual(stats["deadline_tokens"], 10 * _PAGE)
        self.assertEqual(stats["revoked_stop_policy"], 0)
        cache.revoke_pending_prefetch.assert_called_once_with(_REQ)
        cache._handle_prefetch_result.assert_not_called()
        cache.cache_controller.terminate_prefetch.assert_called_once_with(op)
        self.assertEqual(
            collector.log_prefetch_outcome.call_args_list,
            [call("revoked_deadline", 1), call("deadline_tokens", 10 * _PAGE)],
        )

    def test_best_effort_stop_is_not_a_deadline(self):
        cache = _cache(policy="best_effort")
        _track(cache, _operation())

        cache.check_prefetch_progress(_REQ)

        stats = cache._prefetch_outcome_stats
        self.assertEqual(stats["revoked_stop_policy"], 1)
        self.assertEqual(stats["revoked_deadline"], 0)

    def test_not_yet_due_counts_nothing(self):
        cache = _cache()
        cache._can_terminate_prefetch.return_value = False
        _track(cache, _operation())

        self.assertFalse(cache.check_prefetch_progress(_REQ))
        self.assertEqual(sum(cache._prefetch_outcome_stats.values()), 0)
        cache.revoke_pending_prefetch.assert_not_called()

    def test_controller_side_termination_is_not_a_cut(self):
        # A miss or an abort marks the operation terminated first; the hit
        # drain owns that accounting (revoked_full_miss / insufficient).
        cache = _cache()
        op = _operation()
        op.mark_terminate()
        _track(cache, op)

        cache.check_prefetch_progress(_REQ)

        self.assertEqual(sum(cache._prefetch_outcome_stats.values()), 0)
        cache._can_terminate_prefetch.assert_not_called()
        cache.revoke_pending_prefetch.assert_called_once_with(_REQ)

    def test_cut_of_a_parked_hit_loses_only_the_known_hit(self):
        # Buffer mode: the lookup answered (requested and miss already
        # counted) but the hit was parked for staging when the cut came.
        cache = _cache()
        op = _operation(requested=10 * _PAGE)
        op.storage_hit_count = 6 * _PAGE
        cache._account_prefetch_outcome(op, revoked=False)
        _track(cache, op)

        cache.check_prefetch_progress(_REQ)

        stats = cache._prefetch_outcome_stats
        self.assertEqual(stats["l1l2_miss_tokens"], 10 * _PAGE)
        self.assertEqual(stats["l3_miss_tokens"], 4 * _PAGE)
        self.assertEqual(stats["revoked_deadline"], 1)
        self.assertEqual(stats["deadline_tokens"], 6 * _PAGE)
        cache.revoke_pending_prefetch.assert_called_once_with(_REQ)

    def test_cut_mid_transfer_is_handed_to_the_result_handler(self):
        cache = _cache()
        op = _operation()
        op.host_indices = torch.arange(4 * _PAGE)
        _track(cache, op, host_indices=op.host_indices)

        cache.check_prefetch_progress(_REQ)

        self.assertEqual(op.stats_cut_reason, "deadline")
        cache._handle_prefetch_result.assert_called_once_with(op)
        # The handler decides what was lost; nothing is counted here.
        self.assertEqual(sum(cache._prefetch_outcome_stats.values()), 0)


class TestHybridAllOrNothing(unittest.TestCase):
    def _hybrid_fixture(self, completed_pages, mamba_hit_pages, cut_reason=None):
        cache = _cache()
        cache.storage_existence_cache = Mock()
        cache._finish_storage_prefetch = Mock()
        cache.dec_host_lock_ref = Mock()
        cache.buffer_pipeline = None
        cache._prefetch_occupied_span = Mock(return_value=0)
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.prefetch_loaded_storage_start_by_reqid = {}
        hash_value = [f"h{i}" for i in range(4)]
        op = _operation(requested=4 * _PAGE)
        op.hash_value = hash_value
        op.completed_tokens = completed_pages * _PAGE
        op.stats_cut_reason = cut_reason
        op.pool_transfers = [PoolTransfer(name=PoolName.MAMBA, keys=["h3"])]
        op.pool_storage_result = SimpleNamespace(
            extra_pool_hit_pages={PoolName.MAMBA: mamba_hit_pages}
        )
        host_indices = torch.arange(4 * _PAGE)
        _track(cache, op, host_indices=host_indices)
        return cache, op, hash_value, host_indices

    def _check(self, cache, op, hash_value, host_indices):
        return cache._check_hybrid_prefetch_result(
            _REQ,
            op,
            op.completed_tokens,
            hash_value,
            host_indices,
            0,
            None,
            cache.ongoing_prefetch[_REQ].prefetch_key,
        )

    def test_short_aux_pool_discards_the_whole_hit(self):
        cache, op, hv, hi = self._hybrid_fixture(completed_pages=4, mamba_hit_pages=0)

        self.assertFalse(self._check(cache, op, hv, hi))

        stats = cache._prefetch_outcome_stats
        self.assertEqual(stats["revoked_all_or_nothing"], 1)
        self.assertEqual(stats["all_or_nothing_tokens"], 4 * _PAGE)
        self.assertEqual(stats["revoked_deadline"], 0)
        self.assertNotIn(_REQ, cache.ongoing_prefetch)

    def test_deadline_cut_is_counted_as_deadline_not_all_or_nothing(self):
        cache, op, hv, hi = self._hybrid_fixture(
            completed_pages=1, mamba_hit_pages=1, cut_reason="deadline"
        )

        self.assertFalse(self._check(cache, op, hv, hi))

        stats = cache._prefetch_outcome_stats
        self.assertEqual(stats["revoked_deadline"], 1)
        self.assertEqual(stats["deadline_tokens"], 4 * _PAGE)
        self.assertEqual(stats["revoked_all_or_nothing"], 0)
        self.assertEqual(stats["all_or_nothing_tokens"], 0)

    def test_complete_hybrid_result_counts_nothing(self):
        cache, op, hv, hi = self._hybrid_fixture(completed_pages=4, mamba_hit_pages=1)

        self.assertTrue(self._check(cache, op, hv, hi))
        self.assertEqual(sum(cache._prefetch_outcome_stats.values()), 0)


class TestRedundantPrefetch(unittest.TestCase):
    def _result_fixture(self, prefix_len, completed_pages=4, cut_reason=None):
        cache = _cache()
        cache._handle_prefetch_result = (
            UnifiedRadixCache._handle_prefetch_result.__get__(cache)
        )
        cache._check_hybrid_prefetch_result = Mock(return_value=True)
        cache._resolve_storage_prefetch_tokens = Mock()
        cache._finish_storage_prefetch = Mock()
        cache._apply_cache_actions = Mock()
        cache.dec_host_lock_ref = Mock()
        cache.buffer_pipeline = None
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.prefetch_loaded_storage_start_by_reqid = {}
        cache.cache_controller.mem_pool_host = SimpleNamespace(free=Mock())
        cache.tree_core = SimpleNamespace(
            page_size=_PAGE,
            insert_host=Mock(
                return_value=SimpleNamespace(
                    prefix_len=prefix_len,
                    host_insert_dropped=False,
                    cache_actions=[],
                )
            ),
            commit_hicache_transfers=Mock(),
        )
        op = _operation(requested=4 * _PAGE)
        op.hash_value = [f"h{i}" for i in range(4)]
        op.completed_tokens = completed_pages * _PAGE
        op.stats_cut_reason = cut_reason
        _track(cache, op, host_indices=torch.arange(4 * _PAGE))
        return cache, op

    def test_pages_already_in_host_count_as_redundant(self):
        cache, op = self._result_fixture(prefix_len=3 * _PAGE)

        cache._handle_prefetch_result(op)

        stats = cache._prefetch_outcome_stats
        self.assertEqual(stats["redundant_tokens"], 3 * _PAGE)
        # Only the new page is credited to L3 at admission.
        self.assertEqual(cache.prefetch_loaded_tokens_by_reqid[_REQ], _PAGE)

    def test_fresh_insert_counts_no_redundant_tokens(self):
        cache, op = self._result_fixture(prefix_len=0)

        cache._handle_prefetch_result(op)

        self.assertEqual(cache._prefetch_outcome_stats["redundant_tokens"], 0)
        self.assertEqual(cache.prefetch_loaded_tokens_by_reqid[_REQ], 4 * _PAGE)

    def test_cut_tail_on_a_non_hybrid_model_counts_deadline_tokens(self):
        cache, op = self._result_fixture(
            prefix_len=0, completed_pages=1, cut_reason="deadline"
        )

        cache._handle_prefetch_result(op)

        stats = cache._prefetch_outcome_stats
        self.assertEqual(stats["deadline_tokens"], 3 * _PAGE)
        # The completed prefix is kept, so the prefetch is not revoked.
        self.assertEqual(stats["revoked_deadline"], 0)

    def test_cut_with_nothing_transferred_is_revoked(self):
        cache, op = self._result_fixture(
            prefix_len=0, completed_pages=0, cut_reason="deadline"
        )

        cache._handle_prefetch_result(op)

        stats = cache._prefetch_outcome_stats
        self.assertEqual(stats["revoked_deadline"], 1)
        self.assertEqual(stats["deadline_tokens"], 4 * _PAGE)


if __name__ == "__main__":
    unittest.main()
