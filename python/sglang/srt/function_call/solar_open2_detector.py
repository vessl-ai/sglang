# SPDX-License-Identifier: Apache-2.0
"""Tool-call detector for Solar Open2 (Upstage).

Wire format (all markers are single special tokens in the tokenizer):

    <|tool_call:start|>{function_name}
    <|tool_arg:start|>{arg_name}<|tool_arg:value|>{arg_value}<|tool_arg:end|>
    <|tool_call:end|>

Argument values arrive as raw strings; they are coerced to the type declared on
``request.tools`` when a matching JSON-schema entry exists, falling back to the
original string on a lookup miss or a failed conversion. Ported from the Upstage
vLLM fork (``vllm/tool_parsers/solar_open2_tool_parser.py``).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)

TOOL_CALL_START = "<|tool_call:start|>"
TOOL_CALL_END = "<|tool_call:end|>"
TOOL_ARG_START = "<|tool_arg:start|>"
TOOL_ARG_VALUE = "<|tool_arg:value|>"
TOOL_ARG_END = "<|tool_arg:end|>"


def _coerce(value: str, arg_type: Optional[str]) -> Any:
    """Coerce a raw wire string to the JSON-schema type, or keep it as-is."""
    if value == "null":
        return None
    if arg_type in (None, "string"):
        return value
    try:
        if arg_type == "integer":
            return int(value)
        if arg_type in ("number", "float"):
            return float(value)
        if arg_type == "boolean":
            return value.strip().lower() in ("true", "1", "yes")
        if arg_type in ("array", "object"):
            return json.loads(value)
    except (ValueError, TypeError, json.JSONDecodeError):
        return value
    return value


def _param_type(
    func_name: str, param_name: str, tools: Optional[List[Tool]]
) -> Optional[str]:
    if not tools:
        return None
    for tool in tools:
        fn = getattr(tool, "function", None)
        if fn is None or getattr(fn, "name", None) != func_name:
            continue
        params = getattr(fn, "parameters", None)
        if not isinstance(params, dict):
            return None
        prop = (params.get("properties") or {}).get(param_name)
        if not isinstance(prop, dict):
            return None
        t = prop.get("type")
        if isinstance(t, list):
            t = next((x for x in t if x != "null"), None)
        return t
    return None


class SolarOpen2Detector(BaseFormatDetector):
    """Non-streaming + buffered-streaming detector for the Solar Open2 format."""

    def __init__(self):
        super().__init__()
        self.bot_token = TOOL_CALL_START
        self.eot_token = TOOL_CALL_END
        self.tool_call_pattern = re.compile(
            rf"{re.escape(TOOL_CALL_START)}(.+?)\n"
            rf"((?:{re.escape(TOOL_ARG_START)}.*?{re.escape(TOOL_ARG_END)}\n?)*)"
            rf"{re.escape(TOOL_CALL_END)}",
            re.DOTALL,
        )
        self.tool_arg_pattern = re.compile(
            rf"{re.escape(TOOL_ARG_START)}(.*?){re.escape(TOOL_ARG_VALUE)}"
            rf"(.*?){re.escape(TOOL_ARG_END)}",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        return TOOL_CALL_START in text

    def _parse_calls(self, text: str, tools: List[Tool]) -> List[ToolCallItem]:
        indices = self._get_tool_indices(tools)
        calls: List[ToolCallItem] = []
        for match in self.tool_call_pattern.finditer(text):
            name = match.group(1).strip()
            if name not in indices:
                logger.warning("Solar Open2: unknown tool name %r, skipping", name)
                continue
            args = {}
            for arg_match in self.tool_arg_pattern.finditer(match.group(2) or ""):
                key = arg_match.group(1).strip()
                raw = arg_match.group(2)
                args[key] = _coerce(raw, _param_type(name, key, tools))
            calls.append(
                ToolCallItem(
                    tool_index=indices[name],
                    name=name,
                    parameters=json.dumps(args, ensure_ascii=False),
                )
            )
        return calls

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if TOOL_CALL_START not in text:
            return StreamingParseResult(normal_text=text, calls=[])
        normal_text = text[: text.index(TOOL_CALL_START)]
        return StreamingParseResult(
            normal_text=normal_text, calls=self._parse_calls(text, tools)
        )

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Buffer until whole calls are closed, then emit them at once.

        The wire format is not incrementally valid JSON, so per-token argument
        streaming would emit unparsable fragments. Callers get complete
        ToolCallItems one call at a time instead.
        """
        self._buffer += new_text

        if TOOL_CALL_START not in self._buffer:
            # Hold back a suffix that could be the start of the marker.
            hold = self._ends_with_partial_token(self._buffer, self.bot_token)
            if hold:
                emit, self._buffer = self._buffer[:-hold], self._buffer[-hold:]
            else:
                emit, self._buffer = self._buffer, ""
            return StreamingParseResult(normal_text=emit, calls=[])

        head = self._buffer[: self._buffer.index(TOOL_CALL_START)]
        rest = self._buffer[self._buffer.index(TOOL_CALL_START) :]
        if TOOL_CALL_END not in rest:
            self._buffer = rest
            return StreamingParseResult(normal_text=head, calls=[])

        cut = rest.rindex(TOOL_CALL_END) + len(TOOL_CALL_END)
        complete, self._buffer = rest[:cut], rest[cut:]
        return StreamingParseResult(
            normal_text=head, calls=self._parse_calls(complete, tools)
        )

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f"{TOOL_CALL_START}{name}\n",
            end=TOOL_CALL_END,
            trigger=TOOL_CALL_START,
        )
