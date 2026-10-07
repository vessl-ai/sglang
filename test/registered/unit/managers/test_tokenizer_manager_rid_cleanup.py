"""
Unit tests for rid_to_state cleanup in TokenizerManager.

Verifies that request IDs are properly removed from rid_to_state after
completion or abort, allowing resubmission with the same rid without
triggering "Duplicate request ID detected" errors.

Covers:
  - _handle_abort_req cleans up rid_to_state
  - _handle_batch_output cleans up rid_to_state on finished requests
  - _init_req_state rejects duplicate rids
  - Resubmission succeeds after cleanup
  - Output for a rid with no state aborts that request on the scheduler,
    once per window, and not at all when doing so would take a live rid
  - Handler failures clean up pending and dispatched requests
"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import msgspec

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import tokenizer_manager  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    AbortReq,
    BatchStrOutput,
    GenerateReqInput,
)
from sglang.srt.managers.tokenizer_manager import (  # noqa: E402
    _MAX_TRACKED_ORPHAN_RIDS,
    HEALTH_CHECK_RID_PREFIX,
    ReqState,
    TokenizerManager,
)
from sglang.srt.observability.metrics_collector import (  # noqa: E402
    TokenizerMetricsCollector,
)
from sglang.srt.observability.req_time_stats import (  # noqa: E402
    APIServerReqTimeStats,
)
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


_NOT_FINISHED = object()  # Sentinel: request has not finished yet

# ---------------------------------------------------------------------------
# Per-request field defaults for BatchStrOutput construction.
# Categorised by value shape so that _make_batch_str_output can assign
# type-appropriate defaults without hardcoding every field name.
# When a field is renamed upstream, the old name simply won't appear in
# msgspec.structs.fields() and the new name will fall through to the
# pattern-matching or safe fallback — no test breakage.
# ---------------------------------------------------------------------------

_PER_REQUEST_INT_FIELDS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "retraction_counts",
        # Speculative-decoding int-scalar fields (current and historical names)
        "spec_verify_ct",
        "spec_accepted_drafts",
        "spec_num_correct_drafts",
    }
)

_PER_REQUEST_FLOAT_FIELDS = frozenset(
    {
        "output_token_entropy_val",
    }
)

_PER_REQUEST_NESTED_LIST_FIELDS = frozenset(
    {
        "output_ids",
        # Logprob fields
        "input_token_logprobs_val",
        "input_token_logprobs_idx",
        "output_token_logprobs_val",
        "output_token_logprobs_idx",
        "input_top_logprobs_val",
        "input_top_logprobs_idx",
        "output_top_logprobs_val",
        "output_top_logprobs_idx",
        "input_token_ids_logprobs_val",
        "input_token_ids_logprobs_idx",
        "output_token_ids_logprobs_val",
        "output_token_ids_logprobs_idx",
        # Speculative-decoding histogram fields (current and historical names)
        "spec_acceptance_histogram",
        "spec_correct_drafts_histogram",
    }
)

_PER_REQUEST_OPTIONAL_FIELDS = frozenset(
    {
        "output_hidden_states",
        "routed_experts",
        "indexer_topk",
        "placeholder_tokens_idx",
        "placeholder_tokens_val",
    }
)


def _make_tokenizer_manager(case) -> TokenizerManager:
    """Create a TokenizerManager with mocked dependencies, bypassing __init__.

    The config it reads comes from the bags, so the stand-in needs a published
    config rather than attributes on a mock.
    """
    override = get_context().override_server_args(speculative_algorithm=None)
    override.install()
    case.addCleanup(override.restore)
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = MagicMock()
    tm._config_updates = []
    tm.server_args.enable_trace = False
    tm.server_args.enable_metrics = False
    tm.server_args.enable_lora = False
    tm.server_args.speculative_algorithm = None
    tm.server_args.incremental_streaming_output = False
    tm.server_args.skip_tokenizer_init = False
    tm.server_args.batch_notify_size = 1
    tm.server_args.weight_version = "1"
    tm.server_args.crash_dump_folder = ""
    tm.server_args.dp_size = 1
    tm.disaggregation_mode = "none"
    tm.init_running_status()
    tm.tokenizer_ipc_name = None
    tm.enable_metrics = False
    tm.enable_trace = False
    tm.enable_lora = False
    tm.incremental_streaming_output = False
    tm.allow_auto_truncate = False
    tm.skip_tokenizer_init = False
    tm.dump_requests_folder = ""
    tm.crash_dump_folder = ""
    tm.send_to_scheduler = MagicMock()
    return tm


def _make_req_state(rid: str = "test_rid") -> ReqState:
    """Create a minimal ReqState for testing."""
    obj = Mock(spec=GenerateReqInput)
    obj.rid = rid
    obj.stream = False
    obj.return_logprob = False
    obj.lora_path = None
    obj.log_metrics = False
    return ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=obj,
        time_stats=APIServerReqTimeStats(),
    )


def _make_abort_req(rid: str, abort_message: str = "Aborted") -> AbortReq:
    """Create an AbortReq for testing."""
    return AbortReq(
        rid=rid,
        abort_all=False,
        finished_reason={"type": "abort", "message": abort_message},
        abort_message=abort_message,
    )


def _make_batch_str_output(rid, finished_reason=None) -> BatchStrOutput:
    """Create a minimal BatchStrOutput. ``rid`` is one rid or a list.

    Uses struct field introspection so that new or renamed fields in
    BatchStrOutput don't break this test.  Only the fields that matter for
    test logic (rids, finished_reasons, output_strs) are set explicitly;
    all others receive type-appropriate defaults based on naming patterns.
    Fields with class-level defaults are left alone automatically.
    """
    rids = [rid] if isinstance(rid, str) else list(rid)
    n = len(rids)
    if finished_reason is _NOT_FINISHED:
        fr = None
    elif finished_reason is None:
        fr = {"type": "length"}
    else:
        fr = finished_reason

    kwargs = {}
    for f in msgspec.structs.fields(BatchStrOutput):
        if f.name == "rids":
            kwargs[f.name] = rids
        elif f.name == "finished_reasons":
            kwargs[f.name] = [fr] * n
        elif f.name == "output_strs":
            kwargs[f.name] = ["hello"] * n
        elif f.name in _PER_REQUEST_INT_FIELDS:
            kwargs[f.name] = [0] * n
        elif f.name in _PER_REQUEST_FLOAT_FIELDS:
            kwargs[f.name] = [0.0] * n
        elif f.name in _PER_REQUEST_NESTED_LIST_FIELDS:
            kwargs[f.name] = [[]] * n
        elif f.name in _PER_REQUEST_OPTIONAL_FIELDS:
            kwargs[f.name] = [None] * n
        elif f.name == "output_token_sampling_mask":
            kwargs[f.name] = None
        # Fields with class defaults — skip, let the default be used
        elif (
            f.default is not msgspec.NODEFAULT
            or f.default_factory is not msgspec.NODEFAULT
        ):
            continue
        # Unknown required field — provide a safe per-request default.
        # Most BatchStrOutput fields are per-request lists; [[]] works for
        # List[List[...]] and is unlikely to crash on [i] indexing for
        # List[int] either (the inner [] just means "no data").
        else:
            kwargs[f.name] = [[]] * n

    return BatchStrOutput(**kwargs)


class TestEngineResponseWait(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tm = _make_tokenizer_manager(self)
        self.tm.incremental_streaming_output = True
        self.tm.request_logger = Mock()
        self.tm.request_metrics_exporter_manager = Mock()
        self.tm.request_metrics_exporter_manager.exporter_enabled.return_value = False
        self.state = _make_req_state("engine_wait")
        self.state.obj.stream = True
        self.state.obj.background = False
        self.tm.rid_to_state[self.state.obj.rid] = self.state

    async def test_available_and_later_outputs_keep_order(self):
        stream = self.tm._wait_one_response(self.state.obj)
        first = {"output_ids": [1], "meta_info": {"finish_reason": None}}
        final = {"output_ids": [2], "meta_info": {"finish_reason": {"type": "length"}}}
        self.state.out_list.append(first)
        self.state.event.set()
        with patch(
            "asyncio.wait_for",
            side_effect=AssertionError("Engine installed an HTTP timeout"),
        ):
            self.assertEqual((await anext(stream))["output_ids"], [1])
            pending = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            self.assertFalse(pending.done())
            self.state.out_list.append(final)
            self.state.finished = True
            self.state.event.set()
            self.assertEqual((await pending)["output_ids"], [2])
            with self.assertRaises(StopAsyncIteration):
                await anext(stream)

    async def test_cancelling_pending_wait_removes_event_waiter(self):
        stream = self.tm._wait_one_response(self.state.obj)
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        self.assertTrue(self.state.event._waiters)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertFalse(self.state.event._waiters)
        await stream.aclose()

    async def test_abort_and_shutdown_errors_wake_engine_iterator(self):
        for status in (400, 500, 503):
            with self.subTest(status=status):
                state = _make_req_state(f"engine_abort_{status}")
                state.obj.stream = True
                self.tm.rid_to_state[state.obj.rid] = state
                stream = self.tm._wait_one_response(state.obj)
                pending = asyncio.create_task(anext(stream))
                await asyncio.sleep(0)
                self.tm._handle_abort_req(
                    AbortReq(
                        rid=state.obj.rid,
                        finished_reason={
                            "type": "abort",
                            "status_code": status,
                            "message": "test",
                        },
                    )
                )
                result = await pending
                self.assertEqual(
                    result["meta_info"]["finish_reason"]["status_code"], status
                )
                self.assertNotIn(state.obj.rid, self.tm.rid_to_state)
                with self.assertRaises(StopAsyncIteration):
                    await anext(stream)

    async def test_http_timeout_still_checks_disconnection(self):
        request = Mock()
        request.is_disconnected = AsyncMock(return_value=True)
        self.tm.abort_request = Mock()
        stream = self.tm._wait_one_response(self.state.obj, request)
        with patch(
            "sglang.srt.managers.tokenizer_manager._REQUEST_STATE_WAIT_TIMEOUT", 0.001
        ):
            with self.assertRaisesRegex(ValueError, "disconnected"):
                await anext(stream)
        request.is_disconnected.assert_awaited_once()
        self.tm.abort_request.assert_called_once_with(self.state.obj.rid)


class TestRidToStateCleanupOnAbort(CustomTestCase):
    """Test that _handle_abort_req removes rid from rid_to_state."""

    def test_abort_removes_rid_from_state(self):
        """After _handle_abort_req, rid should be removed from rid_to_state."""
        tm = _make_tokenizer_manager(self)
        rid = "abort_test_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        abort_req = _make_abort_req(rid)
        tm._handle_abort_req(abort_req)

        self.assertNotIn(rid, tm.rid_to_state)

    def test_abort_allows_resubmit_same_rid(self):
        """After abort, _init_req_state should accept the same rid again."""
        tm = _make_tokenizer_manager(self)
        rid = "resubmit_after_abort_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        abort_req = _make_abort_req(rid)
        tm._handle_abort_req(abort_req)

        # Resubmit with the same rid — should not raise
        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None
        tm._init_req_state(obj)

        self.assertIn(rid, tm.rid_to_state)

    def test_abort_sets_finished_and_notifies(self):
        """_handle_abort_req should mark state as finished and set the event."""
        tm = _make_tokenizer_manager(self)
        rid = "abort_notify_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        abort_req = _make_abort_req(rid)
        tm._handle_abort_req(abort_req)

        self.assertTrue(state.finished)
        self.assertTrue(state.event.is_set())
        self.assertEqual(len(state.out_list), 1)
        self.assertEqual(
            state.out_list[0]["meta_info"]["finish_reason"]["type"], "abort"
        )


class TestAbortOutputPayload(CustomTestCase):
    """An abort chunk is often the only thing a client sees;
    it must carry the same optional fields as a normal finish chunk."""

    def test_abort_includes_prompt_token_ids_only_when_requested(self):
        """The abort chunk carries prompt_token_ids captured at tokenization,
        and omits the field when the request did not ask for them."""
        tm = _make_tokenizer_manager(self)
        with_ids = _make_req_state("abort_prompt_ids_rid")
        with_ids.prompt_token_ids = [1, 2, 3]
        without_ids = _make_req_state("abort_no_prompt_ids_rid")

        for state in (with_ids, without_ids):
            tm.rid_to_state[state.obj.rid] = state
            tm._handle_abort_req(_make_abort_req(state.obj.rid))

        self.assertEqual(with_ids.out_list[0]["prompt_token_ids"], [1, 2, 3])
        self.assertNotIn("prompt_token_ids", without_ids.out_list[0])

    def test_abort_output_ids_match_the_streaming_mode(self):
        """Only incremental streaming collapses the abort chunk to the last
        token; cumulative chunks supersede, so they carry the whole generation.
        """
        cases = [
            ("incremental stream", True, True, [7]),
            ("cumulative stream", True, False, [5, 6, 7]),
            ("non-stream", False, False, [5, 6, 7]),
        ]
        for name, is_stream, incremental, expected in cases:
            with self.subTest(name):
                tm = _make_tokenizer_manager(self)
                tm.incremental_streaming_output = incremental
                rid = f"abort_output_ids_{name}"
                state = _make_req_state(rid)
                state.obj.stream = is_stream
                state.output_ids = [5, 6, 7]
                tm.rid_to_state[rid] = state

                tm._handle_abort_req(_make_abort_req(rid))

                out = state.out_list[0]
                self.assertEqual(out["output_ids"], expected)
                self.assertEqual(out["meta_info"]["completion_tokens"], 3)


class TestRidToStateCleanupOnBatchOutput(CustomTestCase):
    """Test that _handle_batch_output removes rid from rid_to_state on completion."""

    def test_batch_output_removes_rid_on_finish(self):
        """When a request finishes in _handle_batch_output, rid should be removed."""
        tm = _make_tokenizer_manager(self)
        rid = "batch_finish_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        batch_output = _make_batch_str_output(rid)
        asyncio.run(tm._handle_batch_output(batch_output))

        self.assertNotIn(rid, tm.rid_to_state)

    def test_batch_output_allows_resubmit_after_finish(self):
        """After a request finishes, the same rid can be resubmitted."""
        tm = _make_tokenizer_manager(self)
        rid = "batch_resubmit_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        batch_output = _make_batch_str_output(rid)
        asyncio.run(tm._handle_batch_output(batch_output))

        # Resubmit with the same rid — should not raise
        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None
        tm._init_req_state(obj)

        self.assertIn(rid, tm.rid_to_state)

    def test_batch_output_keeps_rid_when_not_finished(self):
        """When a request is not yet finished, rid should remain in rid_to_state."""
        tm = _make_tokenizer_manager(self)
        rid = "batch_ongoing_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        # finished_reason=_NOT_FINISHED means the request is still ongoing
        batch_output = _make_batch_str_output(rid, finished_reason=_NOT_FINISHED)
        asyncio.run(tm._handle_batch_output(batch_output))

        self.assertIn(rid, tm.rid_to_state)


class TestOrphanedOutputAborts(CustomTestCase):
    """Output for a rid with no state means the scheduler is still generating
    for a request nobody is reading: it must be aborted -- once per window,
    and not at all when the abort would take a live request with it."""

    def _sent_aborts(self, tm):
        return [
            c.args[0]
            for c in tm._dispatch_to_scheduler.call_args_list
            if isinstance(c.args[0], AbortReq)
        ]

    def _capture_dispatch(self, tm):
        # Deliberately bypasses stamp_http_worker_ipc;
        # test_orphan_abort_is_stamped_for_the_owning_worker covers that.
        tm._dispatch_to_scheduler = Mock()

    def test_orphaned_output_aborts_on_the_scheduler(self):
        tm = _make_tokenizer_manager(self)
        self._capture_dispatch(tm)
        rid = "orphan_rid"
        # No rid_to_state entry: the state was deleted while the scheduler ran on.
        asyncio.run(tm._handle_batch_output(_make_batch_str_output(rid)))

        aborts = self._sent_aborts(tm)
        self.assertEqual(len(aborts), 1)
        self.assertEqual(aborts[0].rid, rid)
        self.assertFalse(aborts[0].abort_all)

    def test_abort_is_sent_once_per_rid(self):
        """Output keeps arriving until the scheduler acts on the abort."""
        tm = _make_tokenizer_manager(self)
        self._capture_dispatch(tm)
        rid = "orphan_repeat_rid"
        for _ in range(5):
            asyncio.run(tm._handle_batch_output(_make_batch_str_output(rid)))

        self.assertEqual(len(self._sent_aborts(tm)), 1)

    def test_abort_is_retried_once_the_window_passes(self):
        """Nothing here can see whether the abort landed, so it goes again."""
        tm = _make_tokenizer_manager(self)
        self._capture_dispatch(tm)
        rid = "orphan_retry_rid"
        asyncio.run(tm._handle_batch_output(_make_batch_str_output(rid)))
        with patch.object(tokenizer_manager, "_ORPHAN_ABORT_RETRY_S", 0.0):
            with self.assertLogs(tokenizer_manager.logger, level="ERROR") as logs:
                asyncio.run(tm._handle_batch_output(_make_batch_str_output(rid)))

        self.assertEqual([a.rid for a in self._sent_aborts(tm)], [rid, rid])
        self.assertTrue(any("again" in line for line in logs.output))

    def test_resubmitted_rid_is_aborted_again_when_orphaned_again(self):
        """A rid that came back to life is no longer a known orphan."""
        tm = _make_tokenizer_manager(self)
        self._capture_dispatch(tm)
        rid = "orphan_then_resubmit"
        asyncio.run(tm._handle_batch_output(_make_batch_str_output(rid)))

        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None
        tm._init_req_state(obj)
        del tm.rid_to_state[rid]  # its stream breaks again

        asyncio.run(tm._handle_batch_output(_make_batch_str_output(rid)))
        self.assertEqual(len(self._sent_aborts(tm)), 2)

    def test_empty_rid_does_not_abort_everything(self):
        """The scheduler matches by startswith: an empty rid matches all."""
        tm = _make_tokenizer_manager(self)
        self._capture_dispatch(tm)
        # Load-bearing: every live rid startswith(""), so a non-empty
        # rid_to_state would let the prefix guard absorb this case and the
        # empty-rid guard would go untested.
        self.assertEqual(tm.rid_to_state, {})
        asyncio.run(tm._handle_batch_output(_make_batch_str_output("")))

        self.assertEqual(self._sent_aborts(tm), [])

    def test_live_sibling_prefix_is_not_aborted_and_says_so(self):
        """A batch expands one rid into <rid>_0..<rid>_N; aborting an orphaned
        X_1 would prefix-match a live X_10 on the scheduler. That leaves a
        real leak running, so it must not be silent."""
        tm = _make_tokenizer_manager(self)
        self._capture_dispatch(tm)
        tm.rid_to_state["X_10"] = _make_req_state("X_10")
        with self.assertLogs(tokenizer_manager.logger, level="ERROR") as logs:
            asyncio.run(tm._handle_batch_output(_make_batch_str_output("X_1")))

        self.assertEqual(self._sent_aborts(tm), [])
        self.assertTrue(any("X_10" in line for line in logs.output))

    def test_a_blocked_orphan_is_logged_once_per_window(self):
        """The collision scan is O(live requests) and output keeps arriving;
        the stamp is what keeps both the scan and the log off every message."""
        tm = _make_tokenizer_manager(self)
        self._capture_dispatch(tm)
        tm.rid_to_state["X_10"] = _make_req_state("X_10")
        with self.assertLogs(tokenizer_manager.logger, level="ERROR") as logs:
            for _ in range(5):
                asyncio.run(tm._handle_batch_output(_make_batch_str_output("X_1")))

        self.assertEqual(len(logs.output), 1)
        self.assertEqual(self._sent_aborts(tm), [])

    def test_orphaned_longer_rid_is_aborted_despite_a_live_prefix(self):
        """The guard is one-directional on purpose: the scheduler matches the
        rid we send by startswith, so aborting X_10 cannot reach a live X_1."""
        tm = _make_tokenizer_manager(self)
        self._capture_dispatch(tm)
        tm.rid_to_state["X_1"] = _make_req_state("X_1")
        asyncio.run(tm._handle_batch_output(_make_batch_str_output("X_10")))

        self.assertEqual([a.rid for a in self._sent_aborts(tm)], ["X_10"])
        self.assertIn("X_1", tm.rid_to_state)

    def test_dispatch_failure_does_not_propagate_and_is_throttled(self):
        """handle_loop has no handler; print_exception_wrapper kills the tree.
        These sends fail permanently, so the retry waits for the window."""
        tm = _make_tokenizer_manager(self)
        tm._dispatch_to_scheduler = Mock(side_effect=RuntimeError("zmq down"))
        rid = "orphan_dispatch_boom"
        with self.assertLogs(tokenizer_manager.logger, level="ERROR") as logs:
            for _ in range(3):
                asyncio.run(tm._handle_batch_output(_make_batch_str_output(rid)))

        self.assertEqual(tm._dispatch_to_scheduler.call_count, 1)
        # Only the failure line: no "aborted it" claim for a send that failed.
        self.assertEqual(len(logs.output), 1)
        self.assertIn(rid, tm._aborted_orphan_rids)
        # Recorded as not sent, so the next attempt does not claim "again".
        self.assertFalse(tm._aborted_orphan_rids[rid][1])
        self.assertTrue(any("Failed to abort orphaned" in line for line in logs.output))

    def test_health_check_rid_is_not_aborted(self):
        """The /health_generate race pops its own rid; that is not an orphan."""
        tm = _make_tokenizer_manager(self)
        self._capture_dispatch(tm)
        asyncio.run(
            tm._handle_batch_output(
                _make_batch_str_output(HEALTH_CHECK_RID_PREFIX + "x")
            )
        )

        self.assertEqual(self._sent_aborts(tm), [])

    def test_known_rid_is_not_aborted(self):
        tm = _make_tokenizer_manager(self)
        self._capture_dispatch(tm)
        rid = "live_rid"
        tm.rid_to_state[rid] = _make_req_state(rid)
        asyncio.run(tm._handle_batch_output(_make_batch_str_output(rid)))

        self.assertEqual(self._sent_aborts(tm), [])

    def test_live_siblings_in_the_batch_still_get_their_output(self):
        """The orphan branch continues the loop -- an orphan sits next to live
        siblings and must not cost the ones after it their output."""
        tm = _make_tokenizer_manager(self)
        self._capture_dispatch(tm)
        before, after = "sibling_a", "sibling_c"
        states = {r: _make_req_state(r) for r in (before, after)}
        tm.rid_to_state.update(states)

        asyncio.run(
            tm._handle_batch_output(
                _make_batch_str_output([before, "sibling_orphan_b", after])
            )
        )

        self.assertEqual([a.rid for a in self._sent_aborts(tm)], ["sibling_orphan_b"])
        for rid, state in states.items():
            self.assertTrue(state.finished, f"{rid} was never marked finished")
            self.assertEqual(state.out_list[-1]["meta_info"]["id"], rid)
            self.assertNotIn(rid, tm.rid_to_state)

    def test_cap_holds_and_every_orphan_is_still_aborted(self):
        tm = _make_tokenizer_manager(self)
        self._capture_dispatch(tm)
        n = _MAX_TRACKED_ORPHAN_RIDS + 10
        for i in range(n):
            asyncio.run(tm._handle_batch_output(_make_batch_str_output(f"orphan_{i}")))

        self.assertEqual(len(self._sent_aborts(tm)), n)
        # Bounded, and not pruned to nothing: a prune that empties the map
        # turns every later output message back into a duplicate abort.
        self.assertEqual(len(tm._aborted_orphan_rids), _MAX_TRACKED_ORPHAN_RIDS)

    def test_eviction_drops_the_oldest_attempt_first(self):
        """Re-inserting moves a rid to the end, so the oldest entry is the one
        nearest the end of its retry window."""
        tm = _make_tokenizer_manager(self)
        now = time.monotonic()
        tm._aborted_orphan_rids = {
            f"old_{i}": (now, True) for i in range(_MAX_TRACKED_ORPHAN_RIDS)
        }
        tm._remember_orphan_abort("newcomer", now, sent=True)

        self.assertEqual(len(tm._aborted_orphan_rids), _MAX_TRACKED_ORPHAN_RIDS)
        self.assertNotIn("old_0", tm._aborted_orphan_rids)
        self.assertIn("newcomer", tm._aborted_orphan_rids)

    def test_blocked_orphan_is_aborted_once_the_colliding_rid_ends(self):
        """The prefix guard defers the abort, it does not cancel it: once
        X_10 is gone nothing extends X_1 and the leak must finally be killed."""
        tm = _make_tokenizer_manager(self)
        self._capture_dispatch(tm)
        tm.rid_to_state["X_10"] = _make_req_state("X_10")
        asyncio.run(tm._handle_batch_output(_make_batch_str_output("X_1")))
        self.assertEqual(self._sent_aborts(tm), [])

        del tm.rid_to_state["X_10"]  # the colliding request finished
        with patch.object(tokenizer_manager, "_ORPHAN_ABORT_RETRY_S", 0.0):
            asyncio.run(tm._handle_batch_output(_make_batch_str_output("X_1")))

        self.assertEqual([a.rid for a in self._sent_aborts(tm)], ["X_1"])

    def test_a_declined_abort_is_not_reported_as_a_repeat(self):
        """The blocked path stamps the same map, so "again" has to key off
        whether an abort actually went out, not off the stamp existing."""
        tm = _make_tokenizer_manager(self)
        self._capture_dispatch(tm)
        tm.rid_to_state["X_10"] = _make_req_state("X_10")
        asyncio.run(tm._handle_batch_output(_make_batch_str_output("X_1")))
        del tm.rid_to_state["X_10"]

        with patch.object(tokenizer_manager, "_ORPHAN_ABORT_RETRY_S", 0.0):
            with self.assertLogs(tokenizer_manager.logger, level="ERROR") as logs:
                asyncio.run(tm._handle_batch_output(_make_batch_str_output("X_1")))

        self.assertFalse(any("again" in line for line in logs.output))

    def test_empty_rid_is_logged_once_per_window(self):
        """A caller may send rid=""; the normalizers only replace None, so
        this path is client-reachable and must not log per message."""
        tm = _make_tokenizer_manager(self)
        self._capture_dispatch(tm)
        with self.assertLogs(tokenizer_manager.logger, level="ERROR") as logs:
            for _ in range(5):
                asyncio.run(tm._handle_batch_output(_make_batch_str_output("")))

        self.assertEqual(len(logs.output), 1)

    def test_a_repeated_attempt_moves_the_rid_out_of_the_eviction_line(self):
        """_remember_orphan_abort pops before re-inserting; without that a
        retried rid keeps its original slot and is evicted mid-window."""
        tm = _make_tokenizer_manager(self)
        now = time.monotonic()
        tm._aborted_orphan_rids = {
            f"old_{i}": (now, True) for i in range(_MAX_TRACKED_ORPHAN_RIDS)
        }
        tm._remember_orphan_abort("old_0", now, sent=True)  # now the newest
        tm._remember_orphan_abort("newcomer", now, sent=True)

        self.assertIn("old_0", tm._aborted_orphan_rids)
        self.assertNotIn("old_1", tm._aborted_orphan_rids)

    def test_a_successful_abort_is_counted_and_a_failed_one_is_not(self):
        tm = _make_tokenizer_manager(self)
        tm.enable_metrics = True
        tm.metrics_collector = MagicMock(spec=TokenizerMetricsCollector)
        tm.metrics_collector.labels = {}  # an instance attribute, not on the class
        tm._dispatch_to_scheduler = Mock()
        tm._abort_orphaned_rid("orphan_counted")
        observe = tm.metrics_collector.observe_one_aborted_request
        self.assertEqual(observe.call_count, 1)

        tm._dispatch_to_scheduler = Mock(side_effect=RuntimeError("zmq down"))
        tm._abort_orphaned_rid("orphan_uncounted")
        self.assertEqual(observe.call_count, 1)

    def test_a_retried_abort_is_not_counted_twice(self):
        """already_counted reads the previous attempt's flag, so it has to be
        taken before the map is overwritten -- after it, it would equal `sent`
        and the metric would never fire at all."""
        tm = _make_tokenizer_manager(self)
        tm.enable_metrics = True
        tm.metrics_collector = MagicMock(spec=TokenizerMetricsCollector)
        tm.metrics_collector.labels = {}
        tm._dispatch_to_scheduler = Mock()
        rid = "orphan_counted_once"
        tm._abort_orphaned_rid(rid)
        with patch.object(tokenizer_manager, "_ORPHAN_ABORT_RETRY_S", 0.0):
            tm._abort_orphaned_rid(rid)

        self.assertEqual(tm._dispatch_to_scheduler.call_count, 2)
        observe = tm.metrics_collector.observe_one_aborted_request
        observe.assert_called_once_with(tm.metrics_collector.labels)

    def test_evicting_an_in_window_entry_is_reported_once_per_window(self):
        """Eviction inside the window un-throttles that rid; the degradation
        says so, and says it once."""
        tm = _make_tokenizer_manager(self)
        now = time.monotonic()
        tm._aborted_orphan_rids = {
            f"old_{i}": (now, True) for i in range(_MAX_TRACKED_ORPHAN_RIDS)
        }
        with self.assertLogs(tokenizer_manager.logger, level="ERROR") as logs:
            tm._remember_orphan_abort("newcomer", now, sent=True)
            tm._remember_orphan_abort("newcomer2", now, sent=True)

        self.assertEqual(len(logs.output), 1)
        self.assertIn("throttle full", logs.output[0])
        # Reports how many were evicted in-window, not just one sample victim.
        self.assertIn("1 entries evicted", logs.output[0])

    def test_orphan_abort_is_stamped_for_the_owning_worker(self):
        """Multi-tokenizer mode routes by the stamp, so it must survive."""
        tm = _make_tokenizer_manager(self)
        tm.tokenizer_ipc_name = "ipc:///tmp/worker-7"
        with patch.object(tokenizer_manager, "sock_send") as sock_send:
            tm._abort_orphaned_rid("orphan_mw")

        sent = sock_send.call_args.args[1]
        self.assertIsInstance(sent, AbortReq)
        self.assertEqual(sent.rid, "orphan_mw")
        self.assertEqual(sent.http_worker_ipc, "ipc:///tmp/worker-7")


class TestInitReqStateDuplicateDetection(CustomTestCase):
    """Test that _init_req_state raises ValueError for duplicate rids."""

    def test_duplicate_rid_raises_error(self):
        """_init_req_state should raise ValueError if rid already exists."""
        tm = _make_tokenizer_manager(self)
        rid = "duplicate_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None

        with self.assertRaises(ValueError) as ctx:
            tm._init_req_state(obj)
        self.assertIn("Duplicate request ID", str(ctx.exception))

    def test_unique_rid_succeeds(self):
        """_init_req_state should succeed with a unique rid."""
        tm = _make_tokenizer_manager(self)
        rid = "unique_rid"

        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None

        tm._init_req_state(obj)
        self.assertIn(rid, tm.rid_to_state)


class TestResubmitAfterCompletion(CustomTestCase):
    """End-to-end test: complete a request, then resubmit with the same rid."""

    def test_complete_then_resubmit_same_rid(self):
        """A request that completes normally should allow resubmission with the same rid."""
        tm = _make_tokenizer_manager(self)
        rid = "complete_resubmit_rid"

        # Phase 1: simulate a request in rid_to_state, then complete it
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        batch_output = _make_batch_str_output(rid, finished_reason={"type": "length"})
        asyncio.run(tm._handle_batch_output(batch_output))

        # rid should be cleaned up
        self.assertNotIn(rid, tm.rid_to_state)

        # Phase 2: resubmit with the same rid — should succeed
        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None
        tm._init_req_state(obj)

        self.assertIn(rid, tm.rid_to_state)

    def test_abort_then_resubmit_same_rid(self):
        """An aborted request should allow resubmission with the same rid."""
        tm = _make_tokenizer_manager(self)
        rid = "abort_resubmit_rid"

        # Phase 1: simulate a request, then abort it
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        abort_req = _make_abort_req(rid)
        tm._handle_abort_req(abort_req)

        self.assertNotIn(rid, tm.rid_to_state)

        # Phase 2: resubmit with the same rid — should succeed
        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None
        tm._init_req_state(obj)

        self.assertIn(rid, tm.rid_to_state)


class _DummyAsyncCM:
    """Reusable no-op async context manager (stands in for an RW lock)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _make_tm_for_generate(case) -> TokenizerManager:
    """Augment the mocked TokenizerManager with what generate_request needs."""
    tm = _make_tokenizer_manager(case)
    tm.server_args.language_only = False
    tm.server_args.tokenizer_worker_num = 1
    tm.server_args.enable_strict_thinking = False
    tm.auto_create_handle_loop = Mock()
    tm._set_default_priority = Mock()
    tm.request_logger = Mock()
    tm.tokenizer = None
    tm.is_pause = False
    tm.is_pause_cond = asyncio.Condition()
    tm.model_update_lock = Mock()
    tm.model_update_lock.reader_lock = _DummyAsyncCM()
    tm._validate_and_resolve_lora = AsyncMock(return_value=None)
    return tm


def _make_generate_obj(rid, is_single):
    obj = MagicMock(spec=GenerateReqInput)
    obj.routed_dp_rank = None
    obj.is_single = is_single
    obj.rid = rid
    obj.received_time = 0.0
    obj.external_trace_header = None
    obj.bootstrap_room = None
    obj.max_thinking_tokens = None
    obj.normalize_batch_and_arguments = Mock()
    if not is_single:
        obj.__getitem__.side_effect = lambda i: Mock()
    return obj


class TestReleaseReqStatesOnFailure(CustomTestCase):
    """Direct tests for _release_req_states_on_failure."""

    def test_undelivered_single_is_dropped(self):
        tm = _make_tokenizer_manager(self)
        rid = "d_single"
        tm.rid_to_state[rid] = _make_req_state(rid)
        tm._release_req_states_on_failure([rid])
        self.assertNotIn(rid, tm.rid_to_state)

    def test_undelivered_batch_removes_all(self):
        tm = _make_tokenizer_manager(self)
        rids = ["d0", "d1", "d2"]
        for r in rids:
            tm.rid_to_state[r] = _make_req_state(r)
        tm._release_req_states_on_failure(rids)
        for r in rids:
            self.assertNotIn(r, tm.rid_to_state)

    def test_ignores_already_removed(self):
        """A rid that is no longer present must not raise."""
        tm = _make_tokenizer_manager(self)
        tm.rid_to_state["p1"] = _make_req_state("p1")
        tm._release_req_states_on_failure(["p1", "already_gone"])
        self.assertNotIn("p1", tm.rid_to_state)

    def test_dispatched_single_is_aborted_and_state_kept(self):
        tm = _make_tokenizer_manager(self)
        tm.server_args.tokenizer_worker_num = 1
        tm._dispatch_to_scheduler = Mock()
        tm.enable_metrics = True
        tm.metrics_collector = MagicMock()
        rid = "d_live"
        state = _make_req_state(rid)
        state.dispatched = True
        tm.rid_to_state[rid] = state
        tm._release_req_states_on_failure([rid])
        tm._release_req_states_on_failure([rid])

        sent = [c.args[0] for c in tm._dispatch_to_scheduler.call_args_list]
        self.assertEqual(
            [type(m) for m in sent], [AbortReq], "expected exactly one AbortReq"
        )
        self.assertEqual(sent[0].rid, rid)
        self.assertIn(rid, tm.rid_to_state)
        self.assertTrue(state.abort_sent)
        tm.metrics_collector.observe_one_aborted_request.assert_called_once()

    def test_dispatched_batch_aborts_delivered_and_drops_rest(self):
        tm = _make_tokenizer_manager(self)
        tm.server_args.tokenizer_worker_num = 1
        tm._dispatch_to_scheduler = Mock()
        delivered, undelivered = "d_delivered", "d_undelivered"
        live = _make_req_state(delivered)
        live.dispatched = True
        tm.rid_to_state[delivered] = live
        tm.rid_to_state[undelivered] = _make_req_state(undelivered)
        tm._release_req_states_on_failure([delivered, undelivered])

        sent = [c.args[0] for c in tm._dispatch_to_scheduler.call_args_list]
        self.assertEqual([type(m) for m in sent], [AbortReq])
        self.assertEqual(sent[0].rid, delivered)
        self.assertIn(delivered, tm.rid_to_state)
        self.assertNotIn(undelivered, tm.rid_to_state)

    def test_abort_failure_does_not_stop_cleanup(self):
        tm = _make_tokenizer_manager(self)
        tm.server_args.tokenizer_worker_num = 1
        tm._dispatch_to_scheduler = Mock(side_effect=RuntimeError("send failed"))
        delivered, undelivered = "live", "pending"
        live = _make_req_state(delivered)
        live.dispatched = True
        tm.rid_to_state[delivered] = live
        tm.rid_to_state[undelivered] = _make_req_state(undelivered)

        with self.assertLogs(level="ERROR"):
            tm._release_req_states_on_failure([delivered, undelivered])

        self.assertIn(delivered, tm.rid_to_state)
        self.assertFalse(live.abort_sent)
        self.assertNotIn(undelivered, tm.rid_to_state)


class TestParallelStreamTaskCleanup(CustomTestCase):
    def test_failing_choice_cancels_and_closes_sibling_waiters(self):
        tm = _make_tokenizer_manager(self)

        async def drive():
            sibling_closed = asyncio.Event()

            async def failing_choice():
                await asyncio.sleep(0)
                raise RuntimeError("choice failed")
                yield  # pragma: no cover

            async def blocked_choice():
                try:
                    await asyncio.Event().wait()
                    yield  # pragma: no cover
                finally:
                    sibling_closed.set()

            stream = tm._stream_batch_responses(
                [failing_choice(), blocked_choice()],
                ["choice-0", "choice-1"],
            )
            with self.assertRaisesRegex(RuntimeError, "choice failed"):
                await stream.__anext__()
            self.assertTrue(sibling_closed.is_set())

        asyncio.run(drive())

    def test_failing_non_stream_choice_cancels_and_closes_sibling_waiters(self):
        tm = _make_tokenizer_manager(self)

        async def drive():
            sibling_closed = asyncio.Event()

            async def failing_choice():
                await asyncio.sleep(0)
                raise RuntimeError("choice failed")
                yield  # pragma: no cover

            async def blocked_choice():
                try:
                    await asyncio.Event().wait()
                    yield  # pragma: no cover
                finally:
                    sibling_closed.set()

            with self.assertRaisesRegex(RuntimeError, "choice failed"):
                await tm._collect_batch_responses([failing_choice(), blocked_choice()])
            self.assertTrue(sibling_closed.is_set())

        asyncio.run(drive())


class TestGenerateRequestCleanupOnDispatchFailure(CustomTestCase):
    """generate_request must not leak rid_to_state when dispatch fails.

    Regression guard: _init_req_state creates rid_to_state entries up front,
    and the only remover is the scheduler-response path. A failure before the
    request reaches the scheduler (e.g. input-length validation rejecting an
    over-context request) used to leak those entries permanently.
    """

    def test_single_failure_before_dispatch_cleans_up(self):
        tm = _make_tm_for_generate(self)
        rid = "single_overlen"
        obj = _make_generate_obj(rid, is_single=True)
        # Simulate over-length rejection during tokenization/validation.
        tm._tokenize_one_request = AsyncMock(side_effect=ValueError("input too long"))
        tm._send_one_request = Mock()

        async def drive():
            await tm.generate_request(obj).__anext__()

        with self.assertRaises(ValueError):
            asyncio.run(drive())

        # Got past _init_req_state (which created the entry) ...
        tm._tokenize_one_request.assert_awaited_once()
        tm._send_one_request.assert_not_called()
        # ... and the entry was cleaned up rather than leaked.
        self.assertNotIn(rid, tm.rid_to_state)

    def test_batch_failure_before_dispatch_cleans_up_all(self):
        tm = _make_tm_for_generate(self)
        rids = ["b0", "b1", "b2"]
        obj = _make_generate_obj(list(rids), is_single=False)

        # One over-length sub-request makes the whole batch dispatch raise.
        async def _boom(*args, **kwargs):
            raise ValueError("input too long")
            yield  # pragma: no cover  (marks this an async generator)

        tm._handle_batch_request = _boom

        async def drive():
            await tm.generate_request(obj).__anext__()

        with self.assertRaises(ValueError):
            asyncio.run(drive())

        # All sub-request entries created by _init_req_state are cleaned up.
        for r in rids:
            self.assertNotIn(r, tm.rid_to_state)

    def test_parallel_sampling_failure_cleans_generated_rid(self):
        tm = _make_tm_for_generate(self)
        obj = GenerateReqInput(
            text=["hello"],
            rid=["base"],
            sampling_params={"n": 2},
        )
        tokenized = MagicMock()
        tokenized.mm_inputs = None
        tokenized.sampling_params = MagicMock()
        tm._tokenize_one_request = AsyncMock(return_value=tokenized)
        tm._send_one_request = Mock(side_effect=RuntimeError("dispatch failed"))

        async def drive():
            await tm.generate_request(obj).__anext__()

        with self.assertRaisesRegex(RuntimeError, "dispatch failed"):
            asyncio.run(drive())

        self.assertFalse(tm.rid_to_state)

    def test_thinking_budget_rejects_runtime_without_strict_thinking(self):
        tm = _make_tm_for_generate(self)
        obj = GenerateReqInput(
            text="hello",
            rid="thinking-budget",
            sampling_params={},
            max_thinking_tokens=32,
        )

        async def drive():
            await tm.generate_request(obj).__anext__()

        with self.assertRaisesRegex(ValueError, "--enable-strict-thinking"):
            asyncio.run(drive())

        self.assertFalse(tm.rid_to_state)


class TestWaitOneResponseAfterStateFreed(CustomTestCase):
    """A waiter built before its request finishes must still deliver the output.

    Batch dispatch builds every waiter before advancing any, and the
    scheduler-response path drops rid_to_state as soon as a request finishes.
    """

    def test_generator_built_before_finish_still_delivers_output(self):
        tm = _make_tokenizer_manager(self)
        tm.request_logger = Mock()
        tm.request_metrics_exporter_manager = MagicMock()
        tm.request_metrics_exporter_manager.exporter_enabled.return_value = False
        rid = "freed_state_rid"
        state = _make_req_state(rid)
        state.obj.background = True  # skip the fastapi disconnect probe
        tm.rid_to_state[rid] = state

        async def drive():
            waiter = tm._wait_one_response(state.obj, None)
            await tm._handle_batch_output(_make_batch_str_output(rid))
            self.assertNotIn(rid, tm.rid_to_state)
            return await waiter.__anext__()

        out = asyncio.run(drive())
        self.assertEqual(out["meta_info"]["id"], rid)
        self.assertEqual(out["text"], "hello")


class TestDisconnectAfterDispatchAbortsRequest(CustomTestCase):
    """Cancellation after dispatch must stop the scheduler request."""

    @patch(
        "sglang.srt.managers.tokenizer_manager.wrap_shm_features",
        side_effect=lambda obj: obj,
    )
    def test_cancel_after_dispatch_sends_abort_and_keeps_state(self, _wrap_shm):
        tm = _make_tm_for_generate(self)
        tm.cuda_vmm_feature_transport = Mock()
        tm.cuda_vmm_feature_transport.prepare_for_dispatch_async = AsyncMock(
            return_value=[]
        )
        tm._dispatch_to_scheduler = Mock()
        rid = "disconnect_zombie"
        obj = _make_generate_obj(rid, is_single=True)
        obj.return_prompt_token_ids = False
        tokenized = MagicMock()
        tokenized.rid = rid
        tokenized.mm_inputs = None
        tm._tokenize_one_request = AsyncMock(return_value=tokenized)

        async def drive():
            task = asyncio.create_task(tm.generate_request(obj).__anext__())
            for _ in range(100):
                await asyncio.sleep(0)
                if tm._dispatch_to_scheduler.called:
                    break
            self.assertTrue(
                tm._dispatch_to_scheduler.called, "request never dispatched"
            )
            state = tm.rid_to_state.get(rid)
            self.assertIsNotNone(state)
            self.assertTrue(state.dispatched)

            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(drive())

        sent = [c.args[0] for c in tm._dispatch_to_scheduler.call_args_list]
        aborts = [m for m in sent if isinstance(m, AbortReq) and m.rid == rid]
        self.assertTrue(aborts, "disconnect must send an AbortReq to the scheduler")
        self.assertIn(rid, tm.rid_to_state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
