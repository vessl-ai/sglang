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


class TestUsageProcessor(CustomTestCase):
    def test_cache_details_follow_reporting_flag(self):
        for enable_cache_report, cached_tokens in (
            (False, 7),
            (True, 0),
            (True, 7),
        ):
            with self.subTest(
                enable_cache_report=enable_cache_report,
                cached_tokens=cached_tokens,
            ):
                usages = (
                    UsageProcessor.calculate_response_usage(
                        [{"meta_info": {"cached_tokens": cached_tokens}}],
                        enable_cache_report=enable_cache_report,
                    ),
                    UsageProcessor.calculate_streaming_usage(
                        prompt_tokens={0: 0},
                        reasoning_tokens={0: 0},
                        completion_tokens={0: 0},
                        cached_tokens={0: cached_tokens},
                        n_choices=1,
                        enable_cache_report=enable_cache_report,
                    ),
                )
                for usage in usages:
                    details = usage.prompt_tokens_details
                    if enable_cache_report:
                        self.assertIsNotNone(details)
                        self.assertEqual(details.cached_tokens, cached_tokens)
                    else:
                        self.assertIsNone(details)

    def test_response_usage_sums_cached_over_first_choice_only(self):
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

    def test_streaming_usage_sums_cached_over_first_choice_only(self):
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
