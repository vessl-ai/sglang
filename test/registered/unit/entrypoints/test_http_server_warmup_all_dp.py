"""INF-834 SGLANG_OPT_INF834_WARMUP_ALL_DP: rank pinning and failure handling."""

import asyncio
import os
import threading
import time
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import aiohttp
from sglang.srt.entrypoints import http_server
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

M = "sglang.srt.entrypoints.http_server"
URL = "http://127.0.0.1:30000"


class FakeHTTP:
    """Stands in for requests.get/post and aiohttp.ClientSession.

    `rank_status(rank)` gives the status of the pinned request to that DP rank;
    it may raise, or block on `release` (an Event) to model a slow rank."""

    def __init__(self, num_dp_ranks, usual_status=200, rank_status=None):
        self.num_dp_ranks = num_dp_ranks
        self.usual_status = usual_status
        self.rank_status = rank_status or (lambda rank: 200)
        self.usual_posts = []
        self.async_posts = []
        self.sessions = []
        self.lock = threading.Lock()
        self.all_started = None

    def get(self, url, **kwargs):
        assert url == URL + "/model_info"
        return SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"is_generation": True, "has_image_understanding": False},
        )

    def post(self, url, **kwargs):
        self.usual_posts.append((url, kwargs))
        return SimpleNamespace(status_code=self.usual_status, text="usual warmup")

    def session_class(self):
        fake = self

        class Response:
            def __init__(self, rank):
                self.rank = rank
                self.status = None

            async def __aenter__(self):
                if fake.all_started is None:
                    fake.all_started = asyncio.Event()
                if len(fake.async_posts) == fake.num_dp_ranks:
                    fake.all_started.set()
                # Every request is in flight before any returns: they are parallel.
                await asyncio.wait_for(fake.all_started.wait(), timeout=5)
                self.status = await asyncio.get_running_loop().run_in_executor(
                    None, fake.rank_status, self.rank
                )
                return self

            async def __aexit__(self, *args):
                pass

            async def read(self):
                return b""

        class Session:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                fake.sessions.append(self)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def post(self, url, **kwargs):
                with fake.lock:
                    fake.async_posts.append((url, kwargs))
                rank = kwargs["json"].get("routed_dp_rank")
                return Response(rank)

        return Session


class TestWarmupAllDp(unittest.TestCase):
    def run_warmup(
        self,
        fake,
        switch_on=True,
        num_dp_ranks=8,
        disaggregation_mode="null",
        skip_tokenizer_init=False,
        fast_poll=True,
    ):
        tokenizer_manager = SimpleNamespace(server_status=None, served_model_name="m")
        kills = []
        env = {
            "SGLANG_OPT_INF834_WARMUP_ALL_DP": "1" if switch_on else "0",
            "SGLANG_OPT_INF834_FAST_WARMUP_POLL": "1" if fast_poll else "0",
        }
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, env))
            stack.enter_context(patch(f"{M}.requests.get", fake.get))
            stack.enter_context(patch(f"{M}.requests.post", fake.post))
            stack.enter_context(
                patch(f"{M}.aiohttp.ClientSession", fake.session_class())
            )
            stack.enter_context(
                patch(f"{M}.kill_process_tree", lambda pid: kills.append(pid))
            )
            stack.enter_context(patch(f"{M}.time.sleep", lambda s: None))
            stack.enter_context(
                patch(
                    f"{M}._global_state",
                    SimpleNamespace(tokenizer_manager=tokenizer_manager),
                )
            )
            stack.enter_context(patch(f"{M}.ssl_verify_of", lambda args: False))
            stack.enter_context(patch(f"{M}.is_mps", lambda: False))
            stack.enter_context(
                patch(
                    f"{M}.get_serving",
                    lambda: SimpleNamespace(
                        api_key=None, skip_tokenizer_init=skip_tokenizer_init
                    ),
                )
            )
            stack.enter_context(
                patch(
                    f"{M}.get_disagg",
                    lambda: SimpleNamespace(
                        language_only=False,
                        language_model_only=False,
                        disaggregation_mode=disaggregation_mode,
                    ),
                )
            )
            stack.enter_context(
                patch(
                    f"{M}.get_exec",
                    lambda: SimpleNamespace(
                        features=SimpleNamespace(
                            enable_encoder_swa_bounded_replay=False
                        )
                    ),
                )
            )
            stack.enter_context(
                patch(
                    f"{M}.get_parallel",
                    lambda: SimpleNamespace(num_dp_ranks=num_dp_ranks),
                )
            )
            stack.enter_context(
                patch(
                    f"{M}.get_observability",
                    lambda: SimpleNamespace(debug_tensor_dump_input_file=None),
                )
            )
            server_args = SimpleNamespace(url=lambda: URL)
            with self.assertLogs(http_server.logger, level="INFO") as logs:
                http_server.logger.info("start")
                ok = http_server._execute_server_warmup(server_args)
        return ok, tokenizer_manager.server_status, kills, "\n".join(logs.output)

    def pinned(self, fake):
        return {
            kwargs["json"]["routed_dp_rank"]: (url, kwargs)
            for url, kwargs in fake.async_posts
        }

    def test_one_parallel_request_pinned_to_each_dp_rank(self):
        fake = FakeHTTP(num_dp_ranks=8)
        ok, status, kills, log = self.run_warmup(fake)

        self.assertTrue(ok)
        self.assertEqual(status, http_server.ServerStatus.Up)
        self.assertEqual(kills, [])
        # The usual warmup request still goes out once, unchanged.
        self.assertEqual(len(fake.usual_posts), 1)
        self.assertEqual(fake.usual_posts[0][0], URL + "/generate")
        self.assertEqual(len(fake.usual_posts[0][1]["json"]["text"]), 8)
        # Exactly one request per DP rank, all on /generate, none for PD.
        self.assertEqual(len(fake.async_posts), 8)
        pinned = self.pinned(fake)
        self.assertEqual(set(pinned), set(range(8)))
        for url, kwargs in pinned.values():
            self.assertEqual(url, URL + "/generate")
            self.assertEqual(kwargs["json"]["text"], "The capital city of France is")
            self.assertNotIn("bootstrap_host", kwargs["json"])
            self.assertEqual(kwargs["json"]["sampling_params"]["max_new_tokens"], 8)
        self.assertEqual(len(fake.sessions), 1)
        self.assertEqual(fake.sessions[0].kwargs["timeout"].total, 600)
        self.assertIn("8 DP ranks returned 200", log)

    def test_skip_tokenizer_init_sends_input_ids(self):
        fake = FakeHTTP(num_dp_ranks=4)
        ok, _, _, _ = self.run_warmup(fake, num_dp_ranks=4, skip_tokenizer_init=True)
        self.assertTrue(ok)
        for _, kwargs in self.pinned(fake).values():
            self.assertEqual(kwargs["json"]["input_ids"], [10, 11, 12])
            self.assertNotIn("text", kwargs["json"])

    def test_ready_waits_for_the_slowest_rank(self):
        release = threading.Event()

        def rank_status(rank):
            if rank == 5:
                release.wait(timeout=5)
            return 200

        fake = FakeHTTP(num_dp_ranks=8, rank_status=rank_status)
        threading.Timer(0.3, release.set).start()
        start = time.perf_counter()
        ok, status, _, _ = self.run_warmup(fake)
        self.assertTrue(ok)
        self.assertEqual(status, http_server.ServerStatus.Up)
        self.assertGreaterEqual(time.perf_counter() - start, 0.25)

    def test_one_rank_non_200_is_the_usual_warmup_failure(self):
        fake = FakeHTTP(num_dp_ranks=8, rank_status=lambda r: 500 if r == 3 else 200)
        ok, status, kills, log = self.run_warmup(fake)
        self.assertFalse(ok)
        self.assertNotEqual(status, http_server.ServerStatus.Up)
        self.assertEqual(kills, [os.getpid()])
        self.assertIn("Initialization failed. warmup error", log)
        self.assertIn("failed DP ranks {3: 500}", log)

    def test_one_rank_exception_is_the_usual_warmup_failure(self):
        def rank_status(rank):
            if rank == 6:
                raise aiohttp.ClientConnectionError("rank 6 gone")
            return 200

        fake = FakeHTTP(num_dp_ranks=8, rank_status=rank_status)
        ok, status, kills, log = self.run_warmup(fake)
        self.assertFalse(ok)
        self.assertNotEqual(status, http_server.ServerStatus.Up)
        self.assertEqual(kills, [os.getpid()])
        self.assertIn("Initialization failed. warmup error", log)
        self.assertIn("rank 6 gone", log)

    def test_usual_request_failure_does_not_wait_for_pinned_ranks(self):
        never = threading.Event()

        def rank_status(rank):
            never.wait(timeout=3)
            return 200

        # Today's failure, switch off: killed and False.
        fake_off = FakeHTTP(num_dp_ranks=8, usual_status=500)
        ok_off, status_off, kills_off, log_off = self.run_warmup(
            fake_off, switch_on=False
        )
        # Same failure with the switch on, while every pinned request hangs.
        fake_on = FakeHTTP(num_dp_ranks=8, usual_status=500, rank_status=rank_status)
        start = time.perf_counter()
        ok_on, status_on, kills_on, log_on = self.run_warmup(fake_on)
        elapsed = time.perf_counter() - start
        never.set()

        self.assertEqual((ok_on, kills_on), (ok_off, kills_off))
        self.assertEqual((ok_on, kills_on), (False, [os.getpid()]))
        self.assertEqual(status_on, status_off)
        self.assertIn("Initialization failed. warmup error", log_on)
        self.assertLess(elapsed, 2.0)

    def test_switch_off_sends_no_pinned_request(self):
        fake = FakeHTTP(num_dp_ranks=8)
        ok, status, _, log = self.run_warmup(fake, switch_on=False)
        self.assertTrue(ok)
        self.assertEqual(status, http_server.ServerStatus.Up)
        self.assertEqual(fake.async_posts, [])
        self.assertNotIn("INF834 warmup all DP", log)

    def test_single_dp_rank_is_not_applied(self):
        fake = FakeHTTP(num_dp_ranks=1)
        ok, _, _, log = self.run_warmup(fake, num_dp_ranks=1)
        self.assertTrue(ok)
        self.assertEqual(fake.async_posts, [])
        self.assertIn("INF834 warmup all DP: not applied", log)

    def test_pd_mode_keeps_the_disaggregation_warmup_only(self):
        fake = FakeHTTP(num_dp_ranks=4)
        ok, status, _, log = self.run_warmup(
            fake, num_dp_ranks=4, disaggregation_mode="decode"
        )
        self.assertTrue(ok)
        self.assertEqual(status, http_server.ServerStatus.Up)
        self.assertEqual(fake.usual_posts, [])
        self.assertEqual(len(fake.async_posts), 4)
        for _, kwargs in fake.async_posts:
            self.assertIn("bootstrap_host", kwargs["json"])
        self.assertNotIn("INF834 warmup all DP", log)

    def test_slow_model_info_poll_still_works_with_switch(self):
        fake = FakeHTTP(num_dp_ranks=8)
        ok, status, _, _ = self.run_warmup(fake, fast_poll=False)
        self.assertTrue(ok)
        self.assertEqual(status, http_server.ServerStatus.Up)
        self.assertEqual(len(fake.async_posts), 8)


if __name__ == "__main__":
    unittest.main()
