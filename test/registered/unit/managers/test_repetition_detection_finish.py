"""Regression for Req._check_repetition_finish: the vLLM-parity N-gram loop
detector (SamplingParams.repetition_detection). Drives the real
`Req.update_finish_state` with a fake tokenizer; pure CPU. Each test guards a
distinct branch of `_check_repetition_finish` / `_has_repeating_pattern`."""

import unittest
from array import array

from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    FINISH_MATCHED_TOKEN,
    FINISH_REPETITION,
    Req,
    apply_repetition_detection_gate,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTokenizer:
    eos_token_id = -1
    additional_stop_token_ids = None

    def decode(self, ids):
        return ""


class _MockTokenizerForNormalize:
    """Mock tokenizer for normalize() - returns char-count as token list."""

    def encode(self, s, add_special_tokens=False):
        return list(range(len(s)))  # One "token" per character


def _make_req(
    output_ids,
    *,
    repetition_detection=None,
    max_new_tokens=1000,
    min_new_tokens=0,
    stop_token_ids=None,
    eos_token_ids=frozenset(),
):
    sp = SamplingParams(
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        repetition_detection=repetition_detection,
        stop_token_ids=stop_token_ids,
    )
    sp.normalize(tokenizer=_MockTokenizerForNormalize())
    req = Req(
        rid="t",
        origin_input_text="",
        origin_input_ids=array("q", [0]),
        sampling_params=sp,
        eos_token_ids=eos_token_ids,
        vocab_size=10_000,
    )
    req.tokenizer = _FakeTokenizer()
    req.output_ids = array("q", output_ids)
    return req


class TestRepetitionDetectionFinish(unittest.TestCase):
    def test_period_2_block_repeated_min_count_times_finishes_untrimmed(self):
        # Tail = "56" repeated 3 times (period 2, min_count 3): must finish
        # with reason type "repetition", and output must NOT be trimmed.
        req = _make_req(
            [5, 6, 5, 6, 5, 6],
            repetition_detection={
                "max_pattern_size": 2,
                "min_pattern_size": 2,
                "min_count": 3,
            },
        )
        req.update_finish_state(new_accepted_len=6)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_REPETITION)
        self.assertEqual(req.finished_reason.pattern_len, 2)
        self.assertEqual(
            req.finished_reason.to_json(), {"type": "repetition", "matched": 2}
        )
        # No trimming: finished_len stays unset and the full output survives.
        self.assertIsNone(req.finished_len)
        self.assertEqual(list(req.output_ids_through_stop), list(req.output_ids))

    def test_repeated_only_min_count_minus_one_times_does_not_finish(self):
        # Only 2 genuine repeats of "56" at the tail (the token 3 blocks back
        # doesn't match) -> must not finish.
        req = _make_req(
            [9, 9, 9, 5, 6, 5, 6],
            repetition_detection={
                "max_pattern_size": 2,
                "min_pattern_size": 2,
                "min_count": 3,
            },
        )
        req.update_finish_state(new_accepted_len=4)
        self.assertFalse(req.finished())

    def test_block_longer_than_max_pattern_size_does_not_finish(self):
        # True period is 3 ("567" x3), but max_pattern_size caps the scan at
        # 2 -> the loop detector must not see it.
        req = _make_req(
            [5, 6, 7, 5, 6, 7, 5, 6, 7],
            repetition_detection={
                "max_pattern_size": 2,
                "min_pattern_size": 1,
                "min_count": 3,
            },
        )
        req.update_finish_state(new_accepted_len=9)
        self.assertFalse(req.finished())

    def test_field_absent_skips_the_check_entirely(self):
        # Same tail as the finishing case above, but repetition_detection is
        # never set -> the check must be a no-op (single None-check).
        req = _make_req([5, 6, 5, 6, 5, 6], repetition_detection=None)
        req.update_finish_state(new_accepted_len=6)
        self.assertFalse(req.finished())

    def test_min_new_tokens_gate_blocks_an_otherwise_matching_loop(self):
        # The tail satisfies the loop, but output length is still below
        # min_new_tokens -> must not finish yet.
        req = _make_req(
            [5, 6, 5, 6, 5, 6],
            repetition_detection={
                "max_pattern_size": 2,
                "min_pattern_size": 2,
                "min_count": 3,
            },
            min_new_tokens=100,
            max_new_tokens=1000,
        )
        req.update_finish_state(new_accepted_len=6)
        self.assertFalse(req.finished())

    def test_max_new_tokens_wins_over_repetition_in_the_same_step(self):
        # Tail also satisfies the loop, but max_new_tokens is hit first:
        # length must win (checked before repetition in the driver).
        req = _make_req(
            [5, 6, 5, 6, 5, 6],
            repetition_detection={
                "max_pattern_size": 2,
                "min_pattern_size": 2,
                "min_count": 3,
            },
            max_new_tokens=6,
        )
        req.update_finish_state(new_accepted_len=6)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_LENGTH)

    def test_stop_token_wins_over_repetition_in_the_same_step(self):
        # The tail [EOS, EOS, EOS] also satisfies a period-1 loop, but the
        # token-based stop check runs first and must win.
        EOS_ID = 2
        req = _make_req(
            [10, 11, EOS_ID, EOS_ID, EOS_ID],
            repetition_detection={
                "max_pattern_size": 1,
                "min_pattern_size": 1,
                "min_count": 3,
            },
            eos_token_ids={EOS_ID},
        )
        req.update_finish_state(new_accepted_len=3)
        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_MATCHED_TOKEN)
        self.assertEqual(req.finished_len, 3)

    def test_multi_token_accept_only_checks_the_final_position(self):
        # Accepting 7 tokens at once: the interior sub-sequence
        # [5,6,5,6,5,6] would have matched the loop, but the actual final
        # token (9) breaks the tail pattern -> must not finish (pins the
        # final-position-only semantics under multi-token acceptance).
        req = _make_req(
            [1, 2, 5, 6, 5, 6, 5, 6, 9],
            repetition_detection={
                "max_pattern_size": 2,
                "min_pattern_size": 2,
                "min_count": 3,
            },
        )
        req.update_finish_state(new_accepted_len=7)
        self.assertFalse(req.finished())


class TestRepetitionDetectionValidation(unittest.TestCase):
    def test_min_pattern_size_greater_than_max_pattern_size_raises(self):
        with self.assertRaises(ValueError):
            SamplingParams(
                repetition_detection={
                    "max_pattern_size": 2,
                    "min_pattern_size": 5,
                    "min_count": 3,
                }
            )

    def test_min_count_below_2_with_max_pattern_size_set_raises(self):
        with self.assertRaises(ValueError):
            SamplingParams(
                repetition_detection={
                    "max_pattern_size": 2,
                    "min_pattern_size": 1,
                    "min_count": 1,
                }
            )

    def test_max_pattern_size_zero_disables_without_raising(self):
        sp = SamplingParams(repetition_detection={"max_pattern_size": 0})
        self.assertIsNone(sp.repetition_detection)


class _FakeServerArgs:
    def __init__(self, enable_repetition_detection: bool):
        self.enable_repetition_detection = enable_repetition_detection


class TestApplyRepetitionDetectionGate(unittest.TestCase):
    """Regression for the request-intake gate (ServerArgs.enable_repetition_detection):
    flag off must null the field before it ever reaches a Req; flag on must
    preserve it untouched."""

    def _normalized_sp(self):
        sp = SamplingParams(
            repetition_detection={
                "max_pattern_size": 2,
                "min_pattern_size": 2,
                "min_count": 3,
            }
        )
        sp.normalize(tokenizer=_MockTokenizerForNormalize())
        return sp

    def test_flag_off_nulls_repetition_detection(self):
        sp = self._normalized_sp()
        self.assertIsNotNone(sp.repetition_detection)
        apply_repetition_detection_gate(_FakeServerArgs(False), sp)
        self.assertIsNone(sp.repetition_detection)

    def test_flag_on_preserves_repetition_detection(self):
        sp = self._normalized_sp()
        expected = dict(sp.repetition_detection)
        apply_repetition_detection_gate(_FakeServerArgs(True), sp)
        self.assertEqual(sp.repetition_detection, expected)

    def test_flag_off_is_noop_when_field_already_none(self):
        sp = self._normalized_sp()
        sp.repetition_detection = None
        apply_repetition_detection_gate(_FakeServerArgs(False), sp)
        self.assertIsNone(sp.repetition_detection)


if __name__ == "__main__":
    unittest.main()
