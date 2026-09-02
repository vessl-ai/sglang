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

Tolerated degenerate shape: the model sometimes opens a call with a markdown
code fence carrying the function name instead of the start marker::

    ```{function_name}
    <|tool_arg:start|>{arg_name}<|tool_arg:value|>{arg_value}<|tool_arg:end|>
    <|tool_call:end|>

A fence line is treated as a call opener only when the next line begins with
``<|tool_arg:start|>``, so ordinary fenced code blocks in model output are
unaffected (a fence line not followed by an argument marker is never a call opener, so
a zero-argument call must use the real start marker).

A call whose function name is not in ``request.tools`` is emitted to the client
as-is (with a warning) rather than discarded: the client owns name validation
and can surface the mismatch back to the model, whereas a dropped call yields
an empty response with no diagnostic.

Forced tool calls: ``tool_choice="required"`` or a named choice are constrained
by the JSON-schema array (``supports_structural_tag()`` is False -- see that
method), so the output is ``[{"name": ..., "parameters": {...}}]`` parsed on the
serving layer's JSON path, not by this detector. ``parallel_tool_calls=False``
is then ``maxItems=1`` in that schema. For ``tool_choice="auto"`` there is no
grammar, so serving_chat injects ``<|tool_call:end|>`` as a stop string and
glues it back before parsing, capping generation at the first call (see
``OpenAIServingChat._solar_single_call_stop_matched``).

A call body with no argument markers is still accepted as a JSON object when it
parses as one (the shape the legacy structural tag used to force; values arrive
already typed, so schema coercion is skipped); a non-empty body that is neither
marker-formed nor a JSON object is kept as ``{"__raw": body}`` with a warning
rather than dropped.
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

# A fence-opened call: the fence's info string is the function name and the
# next line must start an argument marker (see module docstring).
FENCE_CALL_OPEN = re.compile(
    rf"(?:^|\n)```([\w.-]+)[ \t]*\n(?={re.escape(TOOL_ARG_START)})"
)


def partial_fence_open_len(text: str) -> int:
    """Length of a trailing segment of ``text`` that may still grow into a
    ``FENCE_CALL_OPEN`` match, or 0. Used to hold back streamed output so a
    fence-opened call is not emitted as plain text before its first argument
    marker arrives."""
    at = text.rfind("\n```")
    if at != -1:
        start = at + 1
    elif text.startswith("```"):
        start = 0
    else:
        # A run of one or two backticks at a line start may still grow into
        # the fence (the backticks can arrive split across increments).
        m = re.search(r"(?:^|\n)(`{1,2})$", text)
        return len(m.group(1)) if m else 0
    tail = text[start:]
    m = re.fullmatch(r"```[\w.-]*[ \t]*(?:\n(.*))?", tail, re.DOTALL)
    if m is None:
        return 0
    after_newline = m.group(1)
    if not after_newline:
        return len(tail)
    return len(tail) if TOOL_ARG_START.startswith(after_newline) else 0


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
            v = value.strip().lower()
            if v in ("true", "1", "yes"):
                return True
            if v in ("false", "0", "no"):
                return False
            logger.warning(
                "solar_open2: failed to coerce %r to bool; returning as string.",
                value,
            )
            return value
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


def has_call_opener(text: str) -> bool:
    """Whether ``text`` contains a call opener (marker or fence form) -- the
    serving layer's test for gluing a trimmed terminator back on."""
    return TOOL_CALL_START in text or FENCE_CALL_OPEN.search(text) is not None


class SolarOpen2Detector(BaseFormatDetector):
    """Non-streaming + buffered-streaming detector for the Solar Open2 format."""

    def __init__(self):
        super().__init__()
        # Streaming only: output from the first opener on that did not parse
        # as a call, held until a later call parses (then it is not content)
        # or the stream ends (then it is, as on the non-streaming path).
        self._unparsed: str = ""
        # Streaming only, after a call: a whitespace run is held until real
        # text (then it is content) or the next opener / stream end (then it
        # is not) -- the vendor's ``_stream_pending_ws``.
        self._pending_ws: str = ""
        # Streaming only: real (non-whitespace) content has been emitted; the
        # vendor's ``_stream_content_emitted`` -- whitespace that trails it
        # before the first call is content, a whitespace-only prefix is not.
        self._content_emitted: bool = False
        self.bot_token = TOOL_CALL_START
        self.eot_token = TOOL_CALL_END
        # The name group excludes a sentinel prefix: with the vendor's ``(.+?)``
        # a call missing its newline followed by a well-formed one parses as
        # one call named "<name><|tool_call:end|>..." (a fake call, no log).
        self.tool_call_pattern = re.compile(
            rf"{re.escape(TOOL_CALL_START)}((?:(?!<\|)[^\n])+?)\n"
            rf"(.*?)"
            rf"{re.escape(TOOL_CALL_END)}",
            re.DOTALL,
        )
        self.fence_call_pattern = re.compile(
            rf"(?:^|\n)```([\w.-]+)[ \t]*\n"
            rf"((?:{re.escape(TOOL_ARG_START)}.*?{re.escape(TOOL_ARG_END)}\s*)+)"
            rf"{re.escape(TOOL_CALL_END)}",
            re.DOTALL,
        )
        self.tool_arg_pattern = re.compile(
            rf"{re.escape(TOOL_ARG_START)}(.*?){re.escape(TOOL_ARG_VALUE)}"
            rf"(.*?){re.escape(TOOL_ARG_END)}",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        return TOOL_CALL_START in text or bool(self.fence_call_pattern.search(text))

    def _call_starts_all(self, text: str) -> List[int]:
        """Every call opener in ``text`` (marker or fence form), in order."""
        starts = [m.start() for m in re.finditer(re.escape(TOOL_CALL_START), text)]
        starts += [
            m.start() + (1 if text[m.start()] == "\n" else 0)
            for m in FENCE_CALL_OPEN.finditer(text)
        ]
        return sorted(starts)

    def _call_starts(self, text: str) -> List[int]:
        """Start offsets of call openers in ``text`` (marker or fence form)."""
        starts = []
        at = text.find(TOOL_CALL_START)
        if at != -1:
            starts.append(at)
        m = FENCE_CALL_OPEN.search(text)
        if m is not None:
            starts.append(m.start() + (1 if text[m.start()] == "\n" else 0))
        return starts

    def _parse_calls(
        self,
        text: str,
        tools: List[Tool],
        streaming: bool = False,
        text_out: Optional[List[str]] = None,
    ) -> List[ToolCallItem]:
        """``streaming``: number the calls sequentially across the stream
        (``current_tool_id``), the index a client accumulates deltas by --
        two calls of the same tool must not share it. Non-streaming keeps
        the tools-list index (SGLang's convention for every detector; the
        serving layer passes it through as ``tool_calls[].index``, which
        non-streaming clients do not accumulate by).

        ``text_out`` (streaming): receives the content found between the
        parsed calls and after the last one, routed by :meth:`_route_text`
        so the result does not depend on how many calls one chunk carried.
        Text before the first parsed call inside ``text`` (an opener that
        did not parse) is not content, as in non-streaming."""
        indices = self._get_tool_indices(tools)
        calls: List[ToolCallItem] = []
        matches = sorted(
            list(self.tool_call_pattern.finditer(text))
            + list(self.fence_call_pattern.finditer(text)),
            key=lambda m: m.start(),
        )
        consumed_until = 0
        for match in matches:
            if match.start() < consumed_until:
                continue
            if streaming and text_out is not None:
                between = text[consumed_until : match.start()]
                if calls:
                    text_out.append(self._route_text(between, before_opener=True))
                elif between.strip():
                    logger.debug(
                        "Solar Open2: dropping %d chars before the first parsed "
                        "call",
                        len(between),
                    )
            consumed_until = match.end()
            name = match.group(1).strip()
            if name not in indices:
                logger.warning(
                    "Solar Open2: tool name %r not in request.tools; "
                    "emitting the call for the client to handle",
                    name,
                )
            body = match.group(2) or ""
            args = {}
            if TOOL_ARG_START in body:
                for arg_match in self.tool_arg_pattern.finditer(body):
                    key = arg_match.group(1).strip()
                    raw = arg_match.group(2)
                    args[key] = _coerce(raw, _param_type(name, key, tools))
            elif body.strip():
                # Tolerated JSON-object body (the shape the legacy structural
                # tag used to force; required/named now take the JSON-array
                # path and never reach this detector), values already typed.
                try:
                    parsed = json.loads(body)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    args = parsed
                else:
                    logger.warning(
                        "Solar Open2: call body for %r is neither argument "
                        "markers nor a JSON object; passing it through raw",
                        name,
                    )
                    args = {"__raw": body}
            if streaming:
                self.current_tool_id += 1
                tool_index = self.current_tool_id
            else:
                tool_index = indices.get(name, len(calls))
            calls.append(
                ToolCallItem(
                    tool_index=tool_index,
                    name=name,
                    parameters=json.dumps(args, ensure_ascii=False),
                )
            )
        if not streaming and calls:
            openers = len(self._call_starts_all(text))
            if openers > len(calls):
                logger.warning(
                    "Solar Open2: %d opener(s) in the output did not parse as a "
                    "call and are not content",
                    openers - len(calls),
                )
        if streaming and text_out is not None and calls:
            trailing = text[consumed_until:]
            if self._call_starts(trailing):
                # A closed call after the last parsed one that did not parse.
                logger.warning(
                    "Solar Open2: closed tool call did not parse after %d "
                    "completed call(s) (%d chars); dropping it",
                    self.current_tool_id + 1,
                    len(trailing),
                )
            else:
                text_out.append(self._route_text(trailing))
        return calls

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """The vendor's non-streaming rule: with at least one parsed call,
        content is what precedes the first opener -- nothing if that is only
        whitespace -- and the rest belongs to the calls; with none (a start
        marker alone, a malformed or truncated call) the whole output stays
        content rather than losing the suffix after the marker
        (``SolarOpen2ToolParser.extract_tool_calls``)."""
        starts = self._call_starts(text)
        if not starts:
            return StreamingParseResult(normal_text=text, calls=[])
        calls = self._parse_calls(text, tools)
        if not calls:
            logger.warning(
                "Solar Open2: tool call marker present but no call parsed "
                "(%d chars); returning the output as content",
                len(text),
            )
            return StreamingParseResult(normal_text=text, calls=[])
        prefix = text[: min(starts)]
        return StreamingParseResult(
            normal_text=prefix if prefix.strip() else "", calls=calls
        )

    def holding_open_call(self) -> bool:
        """Streaming: an opener has arrived whose call is not closed yet (the
        text the serving layer must glue a trimmed terminator onto)."""
        return bool(self._call_starts(self._buffer))

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Buffer until whole calls are closed, then emit them at once.

        A kept delta from the vendor's streaming parser, which streams
        string-typed argument values per token: this detector emits complete
        ToolCallItems only (every call closed in the buffer at once), so a
        client never assembles a partial call.

        Content is decided as the vendor's streaming parser decides it, and
        independently of where the chunks are cut: real text is content
        wherever it is; a whitespace run is content only when it trails real
        text before the first opener -- not as the whole prefix, not between
        or after calls when it abuts the next opener or the end (the
        vendor's ``_stream_pending_ws`` / ``_stream_content_emitted``; its
        non-streaming parser keeps what precedes the first call unless that
        is only whitespace); output from the first opener on that does not
        parse is held (``_unparsed``) -- dropped if a later call parses, else
        released as content by :meth:`finish`.
        """
        self._buffer += new_text

        starts = self._call_starts(self._buffer)
        if not starts:
            # Hold back a suffix that could be the start of either opener.
            hold = max(
                self._ends_with_partial_token(self._buffer, self.bot_token) or 0,
                partial_fence_open_len(self._buffer),
            )
            if hold:
                emit, self._buffer = self._buffer[:-hold], self._buffer[-hold:]
            else:
                emit, self._buffer = self._buffer, ""
            return StreamingParseResult(normal_text=self._route_text(emit), calls=[])

        head = self._route_text(self._buffer[: min(starts)], before_opener=True)
        rest = self._buffer[min(starts) :]
        if TOOL_CALL_END not in rest:
            self._buffer = rest
            return StreamingParseResult(normal_text=head, calls=[])

        cut = rest.rindex(TOOL_CALL_END) + len(TOOL_CALL_END)
        complete, self._buffer = rest[:cut], rest[cut:]
        gap_text: List[str] = []
        calls = self._parse_calls(complete, tools, streaming=True, text_out=gap_text)
        if not calls:
            # Closed but not parseable (e.g. no newline after the name).
            if self.current_tool_id >= 0:
                logger.warning(
                    "Solar Open2: closed tool call did not parse after %d "
                    "completed call(s) (%d chars); dropping it",
                    self.current_tool_id + 1,
                    len(complete),
                )
            else:
                logger.warning(
                    "Solar Open2: closed tool call did not parse (%d chars); "
                    "holding it -- content unless a later call parses",
                    len(complete),
                )
                self._unparsed += complete
            return StreamingParseResult(normal_text=head, calls=[])
        if self._unparsed:
            logger.debug(
                "Solar Open2: a call parsed; dropping %d chars of earlier "
                "unparsed output",
                len(self._unparsed),
            )
            self._unparsed = ""
        return StreamingParseResult(normal_text=head + "".join(gap_text), calls=calls)

    def _route_text(self, text: str, before_opener: bool = False) -> str:
        """Text outside a call, as content or not (see the class rule).
        ``before_opener``: the text ends where a call opener starts."""
        if self.current_tool_id < 0 and self._unparsed:
            self._unparsed += text
            return ""
        text = self._pending_ws + text
        self._pending_ws = ""
        if not text:
            return ""
        if not text.strip():
            if before_opener:
                # Whitespace abutting an opener is content only when it trails
                # real text before the first call ("text\n" + call keeps the
                # newline, as in non-streaming); a whitespace-only prefix and
                # whitespace between calls are not.
                if self.current_tool_id < 0 and self._content_emitted:
                    return text
                return ""
            # Hold it: real text may follow in a later chunk.
            self._pending_ws = text
            return ""
        self._content_emitted = True
        if before_opener:
            # Before the first call the whole prefix is content (vendor).
            # Between calls the vendor emits whatever share of the run its
            # chunking put before the opener; rstrip makes ours cut-invariant.
            return text if self.current_tool_id < 0 else text.rstrip()
        body = text.rstrip()
        self._pending_ws = text[len(body) :]
        return body

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """The stream is over: release what was held back waiting for a marker
        that can no longer arrive. Before any call was emitted, a partial
        opener / fence candidate, held unparsed output and an unfinished call
        (opened, never closed -- e.g. cut by max_tokens) are returned as
        content, so nothing is dropped silently -- the vendor's non-streaming
        parser keeps such output as content (its streaming parser has already
        emitted as deltas whatever it had parsed; a call cut inside its name
        is emitted by neither). After a call was emitted, only real trailing
        text is content (see parse_streaming_increment); a trailing partial
        opener and trailing whitespace are not."""
        held, self._buffer = self._buffer, ""
        unparsed, self._unparsed = self._unparsed, ""
        pending_ws, self._pending_ws = self._pending_ws, ""
        if self.current_tool_id >= 0:
            # A call was already emitted. An unfinished later call is not
            # content (the non-streaming rule; the vendor's streaming parser
            # has no end-of-stream hook either).
            if TOOL_CALL_START in held or FENCE_CALL_OPEN.search(held):
                logger.warning(
                    "Solar Open2: stream ended inside an unfinished tool call "
                    "after %d completed call(s) (%d chars); dropping it",
                    self.current_tool_id + 1,
                    len(held),
                )
                return StreamingParseResult()
            hold = max(
                self._ends_with_partial_token(held, self.bot_token) or 0,
                partial_fence_open_len(held),
            )
            if hold:
                held = held[:-hold]
            body = (pending_ws + held).rstrip()
            return StreamingParseResult(normal_text=body, calls=[])
        if TOOL_CALL_START in held or FENCE_CALL_OPEN.search(held):
            logger.warning(
                "Solar Open2: stream ended inside an unfinished tool call "
                "(%d chars); returning it as content",
                len(held),
            )
        return StreamingParseResult(normal_text=pending_ws + unparsed + held, calls=[])

    def supports_structural_tag(self) -> bool:
        """``required`` / named tool_choice use the JSON-schema constraint
        (a JSON array of calls, parsed by the JSON path), as the vendor's vLLM
        serving does (``ToolParser.adjust_request`` -> ``StructuredOutputsParams
        (json=...)``; ``SolarOpen2ToolParser`` keeps the default
        ``supports_required_and_named``). Under the legacy structural tag
        built from :meth:`structure_info` the model was observed to spell
        ``<|tool_call:end|>`` out as text (a structural tag's begin/end/
        trigger are all text; see ``_has_grammar`` in the FSM), and the Solar
        Open2 FSM (``srt/sampling/solar_open2_fsm.py``), which tracks the
        tool-call envelope by sentinel id, then never saw the call close. With JSON output no sentinel is emitted at all and the FSM
        stays in CONTENT, where the grammar owns the phase -- the vendor's
        exact rule."""
        return False

    def structure_info(self) -> _GetInfoFunc:
        """Envelope of the legacy structural tag. Required by the base class;
        not reached for ``solar_open2`` since :meth:`supports_structural_tag`
        is False (required/named take the JSON-array path). If it were used,
        xgrammar would fill a JSON object between ``begin`` and ``end``, the
        body shape ``_parse_calls`` still accepts."""
        return lambda name: StructureInfo(
            begin=f"{TOOL_CALL_START}{name}\n",
            end=TOOL_CALL_END,
            trigger=TOOL_CALL_START,
        )
