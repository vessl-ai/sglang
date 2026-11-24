"""Unit tests for srt/disaggregation/decode -- the prefill parallel-info fetch
runs off the decode scheduler thread, so an unanswering prefill bootstrap
server cannot stall the scheduler."""

import socket
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, wait
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ALIVE = "10.0.0.1:8998"
# Refuses the connection: the fetch fails at once.
DEAD = "10.0.0.2:8998"
# Accepts but never answers until the test opens its gate, as an unanswering
# peer does until the HTTP timeout.
HUNG = "10.0.0.3:8998"


class _Prefills:
    """Stands in for CommonKVManager.try_ensure_parallel_info and the prefill
    bootstrap servers behind it."""

    def __init__(self, prefill_info_table):
        self.prefill_info_table = prefill_info_table
        self.alive = {ALIVE}
        self.mismatched = set()
        self.gate = threading.Event()
        self.hung_entered = threading.Event()
        self.callers = []
        self.lock = threading.Lock()

    def calls(self, addr):
        with self.lock:
            return sum(1 for called, _ in self.callers if called == addr)

    def try_ensure_parallel_info(self, addr):
        with self.lock:
            self.callers.append((addr, threading.current_thread()))
        # On the caller's thread, never wait: a blocking caller must fail the
        # thread check, not hang the test.
        if addr == HUNG and threading.current_thread() is not threading.main_thread():
            self.hung_entered.set()
            self.gate.wait()
        if addr in self.mismatched:
            raise RuntimeError(f"Page size mismatch with {addr}")
        if addr not in self.alive:
            return False
        self.prefill_info_table[addr] = SimpleNamespace(
            dp_size=2, follow_bootstrap_room=False
        )
        return True


class TestDecodePrefillInfoFetch(CustomTestCase):
    def setUp(self):
        self.prefill_info_table = {}
        self.prefills = _Prefills(self.prefill_info_table)
        self.executor = ThreadPoolExecutor(max_workers=4)
        # The KV manager's executor, shared by DP-rank queries and rebootstrap
        # calls: fetches must not use it.
        self.shared_executor = ThreadPoolExecutor(max_workers=1)
        self.shared_executor_users = []
        self.addCleanup(self._shutdown)

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.kv_manager = SimpleNamespace(
            prefill_info_table=self.prefill_info_table,
            try_ensure_parallel_info=self.prefills.try_ensure_parallel_info,
            _ensure_prefill_recompute_executor=self._shared_executor,
        )
        queue._prefill_dp_rank_queries = {}
        queue._prefill_info_fetches = {}
        queue._prefill_info_executor = self.executor
        queue._ensure_retry_count = {}
        queue._ensure_last_attempt_time = {}
        queue._max_ensure_retries = 15
        queue._ensure_retry_interval = 0
        self.queue = queue

    def _shared_executor(self):
        self.shared_executor_users.append(threading.current_thread())
        return self.shared_executor

    def _shutdown(self):
        self.prefills.gate.set()
        self.queue._prefill_info_executor.shutdown(wait=True)
        self.executor.shutdown(wait=True)
        self.shared_executor.shutdown(wait=True)

    def _reqs(self, n=1, addr=ALIVE):
        host, port = addr.split(":")
        return [
            SimpleNamespace(
                req=SimpleNamespace(
                    rid=str(i),
                    bootstrap_host=host,
                    bootstrap_port=int(port),
                    bootstrap_room=7 + i,
                    disagg_prefill_dp_rank=None,
                ),
                kv_receiver=MagicMock(),
                awaiting_prefill_dp_rank=True,
            )
            for i in range(n)
        ]

    def _ensure(self, addr_to_reqs):
        """One scheduler pass over the pending addrs; returns the ready ones."""
        return self.queue._ensure_prefill_info(addr_to_reqs)

    def _set_pending(self, reqs):
        self.queue._reqs_awaiting_prefill_dp_rank = lambda: reqs

    def _release_prerequisites(self):
        self.queue._pending_publish = []

    def _settle(self, addr):
        """Wait until the fetch in flight for addr has finished."""
        done, _ = wait([self.queue._prefill_info_fetches[addr]], timeout=10)
        self.assertTrue(done, f"the fetch for {addr} never ran")

    def _hang(self):
        """Start a fetch for HUNG and wait until a worker is inside it."""
        self._ensure({HUNG: self._reqs()})
        self.assertTrue(self.prefills.hung_entered.wait(timeout=10))

    def test_fetch_never_runs_on_the_scheduler_thread(self):
        self._hang()

        self.assertEqual(self._ensure({HUNG: self._reqs()}), {})
        for _, thread in self.prefills.callers:
            self.assertIsNot(thread, threading.main_thread())
        self.assertEqual(self.shared_executor_users, [])

    def test_healthy_addr_is_ready_and_then_cached(self):
        reqs = self._reqs()

        ready = self._ensure({ALIVE: reqs})
        if not ready:
            # At most one pass later than a fetch on the scheduler thread.
            self._settle(ALIVE)
            ready = self._ensure({ALIVE: reqs})
        self.assertEqual(ready, {ALIVE: reqs})

        more = self._reqs()
        self.assertEqual(self._ensure({ALIVE: more}), {ALIVE: more})
        self.assertEqual(self.prefills.calls(ALIVE), 1)
        self.assertEqual(self.queue._prefill_info_fetches, {})

    def test_a_hung_addr_does_not_hold_up_a_live_one(self):
        self._hang()
        alive_reqs = self._reqs()
        addr_to_reqs = {HUNG: self._reqs(), ALIVE: alive_reqs}

        self._ensure(addr_to_reqs)
        self._settle(ALIVE)

        self.assertEqual(self._ensure(addr_to_reqs), {ALIVE: alive_reqs})

    def test_waiting_on_a_running_fetch_neither_resubmits_nor_spends_a_retry(self):
        self._hang()
        reqs = self._reqs(50)

        for _ in range(20):
            self.assertEqual(self._ensure({HUNG: reqs}), {})

        self.assertEqual(self.prefills.calls(HUNG), 1)
        self.assertNotIn(HUNG, self.queue._ensure_retry_count)
        for decode_req in reqs:
            decode_req.kv_receiver.abort.assert_not_called()

    def test_a_running_fetch_is_reused_when_its_requests_come_back(self):
        self._hang()

        # Requests leave (e.g. aborted by the client) and come back.
        for _ in range(5):
            self._ensure({})
            self._ensure({HUNG: self._reqs()})

        self.assertEqual(self.prefills.calls(HUNG), 1)
        self.assertEqual(list(self.queue._prefill_info_fetches), [HUNG])

    def test_a_finished_fetch_nobody_waits_for_is_discarded(self):
        self._ensure({DEAD: self._reqs()})
        self._settle(DEAD)

        self._ensure({})
        self.assertEqual(self.queue._prefill_info_fetches, {})

        # The prefill is back: a new request probes again.
        self.prefills.alive.add(DEAD)
        reqs = self._reqs()
        self._ensure({DEAD: reqs})
        self._settle(DEAD)
        self.assertEqual(self._ensure({DEAD: reqs}), {DEAD: reqs})
        self.assertEqual(self.prefills.calls(DEAD), 2)

    def test_a_queued_fetch_nobody_waits_for_is_cancelled(self):
        self.queue._prefill_info_executor = ThreadPoolExecutor(max_workers=1)
        self._hang()
        self._ensure({HUNG: self._reqs(), DEAD: self._reqs()})
        queued = self.queue._prefill_info_fetches[DEAD]

        self._ensure({HUNG: self._reqs()})

        self.assertTrue(queued.cancelled())
        self.assertEqual(list(self.queue._prefill_info_fetches), [HUNG])

    def test_retry_interval_paces_new_attempts(self):
        self.queue._ensure_retry_interval = 1.0
        clock = SimpleNamespace(now=100.0)
        reqs = self._reqs()
        with patch(
            "sglang.srt.disaggregation.decode.time",
            SimpleNamespace(monotonic=lambda: clock.now),
        ):
            self._ensure({DEAD: reqs})
            self._settle(DEAD)
            self._ensure({DEAD: reqs})
            self.assertEqual(self.queue._ensure_retry_count[DEAD], 1)

            clock.now = 100.9
            self._ensure({DEAD: reqs})
            self.assertEqual(self.prefills.calls(DEAD), 1)
            self.assertNotIn(DEAD, self.queue._prefill_info_fetches)

            clock.now = 101.0
            self._ensure({DEAD: reqs})
            self.assertIn(DEAD, self.queue._prefill_info_fetches)
            self._settle(DEAD)
            self.assertEqual(self.prefills.calls(DEAD), 2)

    def test_dead_then_alive_addr_becomes_ready(self):
        reqs = self._reqs()
        self._ensure({DEAD: reqs})
        self._settle(DEAD)
        self._ensure({DEAD: reqs})
        self.assertEqual(self.queue._ensure_retry_count[DEAD], 1)

        self.prefills.alive.add(DEAD)
        self._ensure({DEAD: reqs})
        self._settle(DEAD)

        self.assertEqual(self._ensure({DEAD: reqs}), {DEAD: reqs})
        self.assertNotIn(DEAD, self.queue._ensure_retry_count)

    def test_requests_fail_once_the_retry_budget_is_spent(self):
        self.queue._max_ensure_retries = 2
        reqs = self._reqs(3)

        for _ in range(2):
            self._ensure({DEAD: reqs})
            self._settle(DEAD)
            self._ensure({DEAD: reqs})

        self.assertEqual(self.prefills.calls(DEAD), 2)
        for decode_req in reqs:
            decode_req.kv_receiver.abort.assert_called_once()
        self.assertNotIn(DEAD, self.queue._ensure_retry_count)

    def test_a_config_mismatch_raises_on_the_scheduler_thread(self):
        self.prefills.alive.add(DEAD)
        self.prefills.mismatched.add(DEAD)
        reqs = self._reqs()

        self._ensure({DEAD: reqs})
        self._settle(DEAD)

        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            self._ensure({DEAD: reqs})

    def test_resolve_queries_the_dp_rank_once_the_info_is_in(self):
        reqs = self._reqs()
        self._set_pending(reqs)
        self.queue._init_receiver = MagicMock()

        with patch(
            "sglang.srt.disaggregation.decode.CommonKVReceiver.query_prefill_dp_ranks",
            return_value={"7": 1},
        ) as query:
            self.queue._resolve_pending_reqs()
            if not self.queue._init_receiver.called:
                self._settle(ALIVE)
                self.queue._resolve_pending_reqs()

        query.assert_called_once_with(ALIVE, [7])
        self.queue._init_receiver.assert_called_once_with(reqs[0], 1)

    def test_a_fetch_whose_info_the_heartbeat_dropped_is_not_ready(self):
        reqs = self._reqs()
        self._set_pending(reqs)
        self.queue._init_receiver = MagicMock()
        self._ensure({ALIVE: reqs})
        if ALIVE in self.queue._prefill_info_fetches:
            self._settle(ALIVE)
        # The heartbeat finds the prefill gone and drops its info.
        self.prefill_info_table.pop(ALIVE, None)
        self.prefills.alive.discard(ALIVE)

        with patch(
            "sglang.srt.disaggregation.decode.CommonKVReceiver.query_prefill_dp_ranks"
        ) as query:
            self.queue._resolve_pending_reqs()

        query.assert_not_called()
        self.queue._init_receiver.assert_not_called()

    def test_fetches_are_dropped_when_nothing_is_pending(self):
        self._hang()
        self._ensure({HUNG: self._reqs(), DEAD: self._reqs()})
        self._settle(DEAD)
        self._set_pending([])

        self.queue._resolve_pending_reqs()

        self.assertEqual(list(self.queue._prefill_info_fetches), [HUNG])

    def test_release_memory_occupation_drops_fetches(self):
        self._hang()
        self._ensure({HUNG: self._reqs(), DEAD: self._reqs()})
        self._settle(DEAD)
        self._release_prerequisites()

        self.queue.release_memory_occupation()

        self.assertEqual(list(self.queue._prefill_info_fetches), [HUNG])

    def test_real_fetch_to_an_unanswering_socket(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        self.addCleanup(listener.close)
        addr = f"127.0.0.1:{listener.getsockname()[1]}"
        manager = SimpleNamespace(prefill_info_table={})
        self.queue.kv_manager.prefill_info_table = manager.prefill_info_table
        self.queue.kv_manager.try_ensure_parallel_info = (
            CommonKVManager.try_ensure_parallel_info.__get__(manager)
        )
        get_threads = []
        real_get = requests.get

        def short_get(url, timeout):
            get_threads.append(threading.current_thread())
            return real_get(url, timeout=0.5)

        reqs = self._reqs(addr=addr)
        with patch("sglang.srt.disaggregation.common.conn.requests.get", short_get):
            self.assertEqual(self._ensure({addr: reqs}), {})
            self._settle(addr)
            self.assertEqual(self._ensure({addr: reqs}), {})

        self.assertEqual(len(get_threads), 1)
        self.assertIsNot(get_threads[0], threading.main_thread())
        self.assertEqual(self.queue._ensure_retry_count[addr], 1)


if __name__ == "__main__":
    unittest.main()
