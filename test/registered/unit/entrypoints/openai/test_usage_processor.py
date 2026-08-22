"""Unit tests for UsageProcessor -- no server, no model loading."""

import unittest

from sglang.srt.entrypoints.openai.protocol import PromptTokensDetails
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _resp(cached_tokens: int, completion_tokens: int = 1) -> dict:
    return {
        "meta_info": {
            "prompt_tokens": 10,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
        }
    }


class TestCalculateResponseUsage(CustomTestCase):
    def test_cache_report_on_zero_cached(self):
        """A cache miss must still report {"cached_tokens": 0}, not omit the field."""
        responses = [_resp(0, completion_tokens=2)]
        usage = UsageProcessor.calculate_response_usage(
            responses, n_choices=1, enable_cache_report=True
        )
        self.assertEqual(
            usage.model_dump()["prompt_tokens_details"], {"cached_tokens": 0}
        )

    def test_cache_report_on_n_choices_sums_first_choice_only(self):
        """With n_choices>1, cached tokens are only summed over the first choice of
        each prompt (choices of the same prompt share the same prompt-side cache
        state, so counting every choice would double-count)."""
        responses = [
            _resp(5),
            _resp(999),  # second choice of prompt 0, must be ignored
            _resp(7),
            _resp(999),  # second choice of prompt 1, must be ignored
        ]
        usage = UsageProcessor.calculate_response_usage(
            responses, n_choices=2, enable_cache_report=True
        )
        self.assertEqual(
            usage.prompt_tokens_details.model_dump(), {"cached_tokens": 12}
        )

    def test_cache_report_off(self):
        """With cache reporting disabled, prompt_tokens_details stays None even
        though cached_tokens is nonzero in the raw meta_info."""
        responses = [_resp(5, completion_tokens=2)]
        usage = UsageProcessor.calculate_response_usage(
            responses, n_choices=1, enable_cache_report=False
        )
        self.assertIsNone(usage.prompt_tokens_details)


class TestCalculateStreamingUsage(CustomTestCase):
    def test_cache_report_on_zero_cached(self):
        """A cache miss must still report {"cached_tokens": 0}, not omit the field."""
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10, 1: 10},
            reasoning_tokens={0: 0, 1: 0},
            completion_tokens={0: 2, 1: 2},
            cached_tokens={0: 0, 1: 0},
            n_choices=2,
            enable_cache_report=True,
        )
        self.assertEqual(
            usage.model_dump()["prompt_tokens_details"], {"cached_tokens": 0}
        )

    def test_cache_report_on_n_choices_sums_first_choice_only(self):
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10, 1: 10, 2: 10, 3: 10},
            reasoning_tokens={0: 0, 1: 0, 2: 0, 3: 0},
            completion_tokens={0: 1, 1: 1, 2: 1, 3: 1},
            cached_tokens={0: 5, 1: 999, 2: 7, 3: 999},
            n_choices=2,
            enable_cache_report=True,
        )
        self.assertEqual(
            usage.prompt_tokens_details.model_dump(), {"cached_tokens": 12}
        )

    def test_cache_report_off(self):
        usage = UsageProcessor.calculate_streaming_usage(
            prompt_tokens={0: 10, 1: 10},
            reasoning_tokens={0: 0, 1: 0},
            completion_tokens={0: 2, 1: 2},
            cached_tokens={0: 5, 1: 5},
            n_choices=2,
            enable_cache_report=False,
        )
        self.assertIsNone(usage.prompt_tokens_details)


class TestCalculateTokenUsage(CustomTestCase):
    def test_cached_zero_details_plus_multimodal_still_attach(self):
        """A cache-miss PromptTokensDetails (cached_tokens=0) must not be dropped
        when multimodal counts are also attached to the same object."""
        cached_details = PromptTokensDetails(cached_tokens=0)
        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=10,
            completion_tokens=2,
            cached_tokens=cached_details,
            image_tokens=3,
        )
        self.assertEqual(
            usage.prompt_tokens_details.model_dump(),
            {"cached_tokens": 0, "image_tokens": 3},
        )

    def test_no_cache_no_multimodal_is_none(self):
        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=10,
            completion_tokens=2,
            cached_tokens=None,
        )
        self.assertIsNone(usage.prompt_tokens_details)


if __name__ == "__main__":
    unittest.main()
