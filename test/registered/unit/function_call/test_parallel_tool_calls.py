import json

"""
Test case for parallel tool call parsing.

This test verifies that the parser correctly handles parallel tool calls
with array parameters in JSON array format.

Scenario:
- Model outputs two parallel tool calls in JSON array format
- Both tools have array parameters (e.g., "title": ["7.8.9 H-9 ..."])
- First tool completes with closing braces
- Second tool starts with opening brace
- The parser must correctly handle the '[' characters in array parameters
  without confusing them with the JSON array start

Expected behavior: Both tools should be parsed correctly.
"""

import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.function_call.solar_open2_detector import TOOL_CALL_END
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-c-test-cpu")


class TestParallelToolCalls(unittest.TestCase):
    """Test case for parallel tool call parsing with array parameters."""

    def setUp(self):
        """Set up test tools and detector."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="search_docs",
                    description="Search documents",
                    parameters={
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Document title",
                            }
                        },
                        "required": ["title"],
                    },
                ),
            ),
        ]
        self.detector = JsonArrayParser()

    def _accumulate_tool_calls(self, tool_calls, result):
        """Helper method to accumulate tool call results from parsing output."""
        if not result.calls:
            return
        for call in result.calls:
            if call.tool_index is None:
                continue
            while len(tool_calls) <= call.tool_index:
                tool_calls.append({"name": "", "parameters": ""})
            if call.name:
                tool_calls[call.tool_index]["name"] = call.name
            if call.parameters:
                tool_calls[call.tool_index]["parameters"] += call.parameters

    def test_parallel_tool_calls_with_array_parameters(self):
        """
        Test parsing two parallel tool calls where both have array parameters.

        This test reproduces the specific scenario:
        - Two tool calls separated by comma
        - Both tools have array parameters containing '[' character
        - First tool completes with '}},'
        - Second tool starts with '{"name": ..., "parameters": {"title": ["'

        Expected: Both tools should be parsed correctly without errors.
        """
        # Simulate more realistic streaming chunks where
        # the key issue is the comma separator followed by second tool with array param
        chunks = [
            "[\n",
            '  {"name": "search_docs", "parameters": {"title": ["7.8.9"',
            '], "filename": "doc1"}},\n',
            '  {"name": "search_docs", "parameters": {"title": ',
            '["4.8"], "filename": "doc2"}}',
            "]",
        ]

        tool_calls = []
        errors = []

        for i, chunk in enumerate(chunks):
            try:
                result = self.detector.parse_streaming_increment(chunk, self.tools)
                # Collect tool calls
                self._accumulate_tool_calls(tool_calls, result)

            except Exception as e:
                errors.append(f"Chunk {i} ({repr(chunk)}): {type(e).__name__}: {e}")

        # Verify no errors occurred
        if errors:
            self.fail("Errors occurred during parsing:\n" + "\n".join(errors))

        # Verify both tool calls were parsed
        self.assertEqual(len(tool_calls), 2, "Should have parsed exactly 2 tool calls")

        # Verify first tool call
        self.assertEqual(
            tool_calls[0]["name"],
            "search_docs",
            "First tool name should be search_docs",
        )
        params1 = json.loads(tool_calls[0]["parameters"])
        self.assertEqual(params1["title"], ["7.8.9"], "First tool title should match")
        self.assertEqual(
            params1["filename"], "doc1", "First tool filename should be doc1"
        )

        # Verify second tool call
        self.assertEqual(
            tool_calls[1]["name"],
            "search_docs",
            "Second tool name should be search_docs",
        )
        params2 = json.loads(tool_calls[1]["parameters"])
        self.assertEqual(params2["title"], ["4.8"], "Second tool title should match")
        self.assertEqual(
            params2["filename"], "doc2", "Second tool filename should be doc2"
        )


class TestSolarOpen2ParallelToolCallsFalse(unittest.TestCase):
    """The legacy structural tag solar_open2 uses for required/named
    tool_choice (see SolarOpen2Detector.structure_info) only exposes
    ``at_least_one``; it has no knob to cap the number of calls. So
    ``parallel_tool_calls=False`` is not enforced by the grammar here —
    OpenAIServingChat._solar_single_call_stop_matched enforces it instead,
    by injecting ``<|tool_call:end|>`` as a stop string and gluing it back
    onto the trimmed text before handing it to the detector.
    """

    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                ),
            ),
        ]
        self.parser = FunctionCallParser(self.tools, "solar_open2")

    def test_structural_tag_ignores_parallel_tool_calls_false(self):
        result = self.parser.get_structure_constraint(
            "required", parallel_tool_calls=False
        )
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "structural_tag")
        self.assertTrue(result[1].at_least_one)

    def test_stop_trimmed_call_glued_back_yields_one_call_non_stream(self):
        trimmed = '<|tool_call:start|>get_weather\n{"location": "Seoul"}'
        text = trimmed + TOOL_CALL_END
        normal_text, calls = self.parser.parse_non_stream(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(json.loads(calls[0].parameters), {"location": "Seoul"})

    def test_stop_trimmed_call_glued_back_yields_one_call_stream(self):
        trimmed = '<|tool_call:start|>get_weather\n{"location": "Seoul"}'
        text = trimmed + TOOL_CALL_END
        collected = []
        for chunk in (text[: len(text) // 2], text[len(text) // 2 :]):
            _, calls = self.parser.parse_stream_chunk(chunk)
            collected.extend(calls)
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0].name, "get_weather")
        self.assertEqual(json.loads(collected[0].parameters), {"location": "Seoul"})


if __name__ == "__main__":
    unittest.main()
