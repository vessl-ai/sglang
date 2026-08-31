"""A stop string must be matched against the answer, never the thinking.

solar-pro4's thinking almost always opens ``1.  **Identify ...``, so a client
stop of ``**`` fires on the first line of the thought and the turn ends before
an answer exists. `Req` therefore suppresses stop matching while the think
block is open, and clamps the decode window to `_content_token_offset` once it
closes.

Each test guards one direction, because a suppression this broad is only
correct if it also stops suppressing:

- while the block is open, a stop in the thinking must not finish the req --
  and the same tokens without `require_reasoning` must finish it, so the
  fixture is proven capable of matching at all;
- once it closes, a stop in the answer must finish the req and trim to it;
- the token-sized tail window must not reach back over the boundary into the
  thinking it just left;
- a speculative run that carries `<|think:end|>` *and* answer tokens after it
  must not lose the stop those answer tokens contain -- the ordinary stop check
  runs before the block is known to have closed, so `_maybe_update_reasoning_
  tokens` re-runs it.

Drives the real `Req.update_finish_state` and the real result processor with a
fake tokenizer; pure CPU.
"""

import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# A single-token <|think:end|> that decodes to nothing, the way a sentinel the
# template hides does: the harshest case for a window sized in tokens, since it
# costs the window a slot without contributing text to it.
THINK_END = 9
THINK_END_IDS = [THINK_END]
STAR = 20  # "*" -- three of these spell the stop
FILL = 40  # "z" -- ordinary answer text
ID_TO_TEXT = {THINK_END: "", STAR: "*", FILL: "z"}

STOP = "***"


class _FakeTokenizer:
    eos_token_id = -1
    additional_stop_token_ids = None

    def decode(self, ids):
        return "".join(ID_TO_TEXT[int(i)] for i in ids)


class _CharTokenizer:
    """One 'token' per character, so stop_str_max_len is the string's length."""

    def encode(self, s, add_special_tokens=False):
        return list(range(len(s)))


def _make_req(output_ids, *, require_reasoning=True, stop=(STOP,)):
    sp = SamplingParams(max_new_tokens=1000, stop=list(stop))
    sp.normalize(tokenizer=_CharTokenizer())
    req = Req(
        rid="t",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=sp,
        require_reasoning=require_reasoning,
        vocab_size=10_000,
    )
    req.tokenizer = _FakeTokenizer()
    req.output_ids = array("q", output_ids)
    return req


def _make_processor():
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(enable_metrics=False),
        model_config=SimpleNamespace(think_end_ids=THINK_END_IDS),
        token_to_kv_pool_allocator=None,
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=SimpleNamespace(),
        draft_worker=None,
        model_worker=SimpleNamespace(on_verify_complete_cpu=lambda *a, **k: None),
        logprob_result_processor=None,
        output_streamer=SimpleNamespace(),
        abort_request=lambda *a, **k: None,
    )


def _accept(req, proc, run):
    """One decode step: the scheduler's own order, extend then finish then count."""
    req.output_ids.extend(run)
    req.update_finish_state(len(run))
    proc._maybe_update_reasoning_tokens(req, run)


class TestStopStrAgainstReasoning(unittest.TestCase):
    def test_stop_inside_thinking_does_not_finish(self):
        req = _make_req([])
        _accept(req, _make_processor(), [STAR, STAR, STAR])
        self.assertFalse(req.finished())
        self.assertIsNone(req._content_token_offset)

    def test_same_tokens_finish_without_require_reasoning(self):
        """The other direction: without the carve-out this fixture does match,
        so the test above is measuring the carve-out and not a dead fixture."""
        req = _make_req([], require_reasoning=False)
        _accept(req, _make_processor(), [STAR, STAR, STAR])
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_reason.matched, STOP)

    def test_stop_inside_the_answer_finishes_and_trims(self):
        req, proc = _make_req([]), _make_processor()
        _accept(req, proc, [STAR, STAR, STAR])  # thinking, suppressed
        _accept(req, proc, [THINK_END])
        self.assertEqual(req._content_token_offset, 4)
        self.assertFalse(req.finished())

        _accept(req, proc, [STAR, STAR, STAR])  # the answer's own stop
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_reason.matched, STOP)
        # Trimmed at the stop, stop included: 4 pre-answer tokens plus "***".
        self.assertEqual(req.finished_len, 7)

    def test_window_does_not_reach_back_over_the_boundary(self):
        """The tail window is sized in tokens and widens with the accepted run,
        so an unclamped one still spans the thinking for several steps after
        the block closes."""
        req, proc = _make_req([STAR, STAR, STAR]), _make_processor()
        req.reasoning_tokens = 3
        _accept(req, proc, [THINK_END, FILL, FILL])
        self.assertEqual(req._content_token_offset, 4)
        self.assertFalse(req.finished(), "matched the thinking it had just left")

    def test_spec_run_carrying_think_end_keeps_the_answer_stop(self):
        """`update_finish_state` runs before the block is known to have closed,
        so without the re-run this stop is never looked at again."""
        req, proc = _make_req([STAR, STAR, STAR]), _make_processor()
        req.reasoning_tokens = 3
        _accept(req, proc, [THINK_END, STAR, STAR, STAR])
        self.assertTrue(req.finished())
        self.assertEqual(req.finished_reason.matched, STOP)
        self.assertEqual(req.finished_len, 7)
        # The sentinel is the last reasoning token; the answer is not counted.
        self.assertEqual(req.reasoning_tokens, 4)


if __name__ == "__main__":
    unittest.main()
