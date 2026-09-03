# SPDX-License-Identifier: Apache-2.0
"""Tool-call detector for the Solar Open2 chat format (Solar Pro 4).

Wire format (every marker is one special token)::

    <|tool_call:start|>{function_name}
    <|tool_arg:start|>{arg_name}<|tool_arg:value|>{arg_value}<|tool_arg:end|>
    <|tool_call:end|>

Argument values arrive as raw strings and are coerced to the type the request's
tool schema declares (``_coerce``); the literal ``null`` is always None. A call
whose name is not among the request's tools is still emitted, with a warning:
the client owns name validation. A body without argument markers that parses
as a JSON object is accepted as the arguments; any other non-empty body is
kept as ``{"__raw": body}``.
An explicit ``"type": null`` in a parameter schema is no type at all (the
value stays a string); a non-dict ``parameters``/``properties`` likewise.

``tool_choice="required"`` and a named choice are not parsed here: they are
constrained to a JSON array of calls by the JSON-schema grammar
(``supports_structural_tag`` is False) and parsed by the serving layer's JSON
path. ``tool_choice="auto"`` with ``parallel_tool_calls=False`` is capped at
one call by a stop string (``entrypoints/openai/solar_open2_serving``).

Content rules (what is *not* a call):

* non-streaming: with at least one parsed call, content is the text before
  the first opener -- nothing when that is only whitespace; with none (a
  marker alone, a malformed or truncated call, a blank name) the whole output
  is content;
* streaming: real text is content wherever it is; a whitespace run is content
  only when it trails real text before the first call -- not as the whole
  prefix, and not between or after calls when it abuts the next opener or
  the end of the stream.

Streaming emits complete calls only, and its result does not depend on where
the chunks are cut: whitespace before the next opener is rstripped. Before any
call is emitted, everything from an opener that did not parse (a blank name
included) onward, prose included, is held to the end of the stream -- content
if no call ever parses, dropped once one does. After a call, a closed unparsed
call is dropped from its opener to its terminator and an unfinished one from
its opener on; the prose around them is content. ``_StreamContent`` owns those
rules; ``_parse_calls`` is pure.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import msgspec

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)

TOOL_CALL_START = "<|tool_call:start|>"
TOOL_CALL_END = "<|tool_call:end|>"
TOOL_ARG_START = "<|tool_arg:start|>"
TOOL_ARG_VALUE = "<|tool_arg:value|>"
TOOL_ARG_END = "<|tool_arg:end|>"

# Name: newlines before it are allowed, and it stops at a sentinel -- so a call
# missing its newline followed by a well-formed one does not parse as one call
# named "<name><|tool_call:end|>...". Body: cannot run across another opener,
# so an unfinished call followed by a well-formed one yields the latter.
_CALL = re.compile(
    rf"{re.escape(TOOL_CALL_START)}\n*((?:(?!<\|)[^\n])+?)\n"
    rf"((?:(?!{re.escape(TOOL_CALL_START)}).)*?)"
    rf"{re.escape(TOOL_CALL_END)}",
    re.DOTALL,
)
_ARG = re.compile(
    rf"{re.escape(TOOL_ARG_START)}(.*?){re.escape(TOOL_ARG_VALUE)}"
    rf"(.*?){re.escape(TOOL_ARG_END)}",
    re.DOTALL,
)


class _ParsedCall(msgspec.Struct):
    start: int  # offset of the opener in the parsed text
    end: int  # offset just past the terminator
    name: str
    arguments: Dict[str, Any]


class _StreamContent:
    """Which streamed text outside a call is content (module docstring,
    "Content rules"), independent of chunking.

    ``pending_ws``: a whitespace run waiting for real text (content) or an
    opener (not); at the end of the stream it is content unless a call was
    emitted. ``emitted``: real text has been sent, so whitespace trailing it
    before the first call is content. ``unparsed``: everything from an opener
    that did not parse onward, prose included, held until a call parses
    (dropped) or the stream ends (content). ``calls``: calls emitted so far --
    after the first, prose is content but whitespace and unparsed calls are
    not."""

    def __init__(self) -> None:
        self.pending_ws = ""
        self.emitted = False
        self.unparsed = ""
        self.calls = 0

    @property
    def after_call(self) -> bool:
        return self.calls > 0

    def text(self, text: str, *, before_opener: bool = False) -> str:
        """Content in ``text``, which holds no opener. ``before_opener``: the
        text ends where a call opener starts."""
        if not self.after_call and self.unparsed:
            self.unparsed += text
            return ""
        text, self.pending_ws = self.pending_ws + text, ""
        if not text:
            return ""
        if not text.strip():
            if before_opener:
                return text if (not self.after_call and self.emitted) else ""
            self.pending_ws = text  # real text may still follow
            return ""
        self.emitted = True
        if before_opener:
            return text if not self.after_call else text.rstrip()
        body = text.rstrip()
        self.pending_ws = text[len(body) :]
        return body

    def segment(self, segment: str, *, before_opener: bool = False) -> str:
        """Content in a segment after a parsed call that may hold unparsed
        calls: prose around them is content; a closed call that did not parse
        is not, from its opener to its terminator, nor is an unfinished one
        from its opener on."""
        out: List[str] = []
        while True:
            at = segment.find(TOOL_CALL_START)
            if at == -1:
                out.append(self.text(segment, before_opener=before_opener))
                return "".join(out)
            out.append(self.text(segment[:at], before_opener=True))
            end = segment.find(TOOL_CALL_END, at)
            if end == -1:
                self._dropped("unfinished tool call", size=len(segment) - at)
                return "".join(out)
            end += len(TOOL_CALL_END)
            self._dropped("closed tool call that did not parse", size=end - at)
            segment = segment[end:]

    def unparsed_call(self, complete: str) -> str:
        """Closed output from an opener on with no parsed call in it: after a
        call the segment rule applies; before one it is held."""
        if self.after_call:
            return self.segment(complete)
        logger.warning(
            "Solar Open2: closed tool call did not parse (%d chars); holding it "
            "-- content unless a later call parses",
            len(complete),
        )
        self.unparsed += complete
        return ""

    def call_parsed(self) -> None:
        if self.unparsed:
            logger.warning(
                "Solar Open2: a call parsed; dropping %d chars held since an "
                "opener that did not parse (%d of them not whitespace)",
                len(self.unparsed),
                len("".join(self.unparsed.split())),
            )
            self.unparsed = ""
        self.calls += 1

    def end(self, held: str, *, partial_opener: int) -> str:
        """Content in what the detector still holds at the end of the stream.
        ``partial_opener``: length of a trailing partial ``<|tool_call:start|>``,
        cut only after a call (before one it is content, as in non-streaming)."""
        if self.after_call:  # after a call ``unparsed`` is always empty
            at = held.find(TOOL_CALL_START)
            if at != -1:
                self._dropped("unfinished tool call at stream end", size=len(held) - at)
                return self.text(held[:at], before_opener=True)
            pending, self.pending_ws = self.pending_ws, ""
            if partial_opener:
                logger.warning(
                    "Solar Open2: stream ended on a partial opener after %d "
                    "completed call(s); dropping %d chars",
                    self.calls,
                    partial_opener,
                )
                held = held[:-partial_opener]
            return (pending + held).rstrip()
        pending, self.pending_ws = self.pending_ws, ""
        unparsed, self.unparsed = self.unparsed, ""
        if TOOL_CALL_START in held:
            logger.warning(
                "Solar Open2: stream ended inside an unfinished tool call "
                "(%d chars); returning it as content",
                len(held),
            )
        return pending + unparsed + held

    def _dropped(self, what: str, *, size: int) -> None:
        logger.warning(
            "Solar Open2: %s after %d completed call(s) (%d chars); dropping it",
            what,
            self.calls,
            size,
        )


class SolarOpen2Detector(BaseFormatDetector):
    """Non-streaming + buffered-streaming detector for the Solar Open2 format."""

    def __init__(self):
        super().__init__()
        self.bot_token = TOOL_CALL_START
        self.eot_token = TOOL_CALL_END
        self._content = _StreamContent()

    def has_tool_call(self, text: str) -> bool:
        return TOOL_CALL_START in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Non-streaming rule (module docstring). Calls carry the tools-list
        index, SGLang's convention for every detector."""
        first = text.find(TOOL_CALL_START)
        if first == -1:
            return StreamingParseResult(normal_text=text, calls=[])
        calls, blank_name = _parse_calls(text, tools=tools)
        if not calls or blank_name:
            logger.warning(
                "Solar Open2: tool call marker present but %s (%d chars); "
                "returning the output as content",
                "a call has a blank name" if blank_name else "no call parsed",
                len(text),
            )
            return StreamingParseResult(normal_text=text, calls=[])
        unparsed = text.count(TOOL_CALL_START) - len(calls)
        if unparsed:
            logger.warning(
                "Solar Open2: %d opener(s) in the output did not parse as a call "
                "and are not content",
                unparsed,
            )
        indices = self._get_tool_indices(tools)
        prefix = text[:first]
        return StreamingParseResult(
            normal_text=prefix if prefix.strip() else "",
            calls=[
                _item(call, index=indices.get(call.name, i))
                for i, call in enumerate(calls)
            ],
        )

    def holding_open_call(self) -> bool:
        """Streaming: an opener has arrived whose call is not closed yet (the
        text the serving layer must glue a trimmed terminator onto)."""
        return TOOL_CALL_START in self._buffer

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Buffer until whole calls are closed, then emit them at once, numbered
        sequentially across the stream (``current_tool_id``, the index a
        client accumulates deltas by). Content follows ``_StreamContent``."""
        self._buffer += new_text
        first = self._buffer.find(TOOL_CALL_START)
        if first == -1:
            # Hold back a suffix that could be the start of an opener.
            hold = self._ends_with_partial_token(self._buffer, self.bot_token)
            emit = self._buffer[: len(self._buffer) - hold]
            self._buffer = self._buffer[len(self._buffer) - hold :]
            return StreamingParseResult(normal_text=self._content.text(emit), calls=[])

        head = self._content.text(self._buffer[:first], before_opener=True)
        rest = self._buffer[first:]
        if TOOL_CALL_END not in rest:
            self._buffer = rest
            return StreamingParseResult(normal_text=head, calls=[])

        cut = rest.rindex(TOOL_CALL_END) + len(TOOL_CALL_END)
        complete, self._buffer = rest[:cut], rest[cut:]
        # A blank-name call is an unparsed segment here (dropped after a call,
        # held before one): the calls around it are emitted either way.
        calls, _ = _parse_calls(complete, tools=tools)
        if not calls:
            text = head + self._content.unparsed_call(complete)
            return StreamingParseResult(normal_text=text, calls=[])
        # ``complete`` starts at an opener; text before the first parsed call
        # is an opener that did not parse: after an earlier call the segment
        # rule applies to it, before any call it is not content. Judged before
        # this delta's calls are counted; the text between and after them is
        # judged after.
        leading = complete[: calls[0].start]
        if self._content.after_call:
            head += self._content.segment(leading, before_opener=True)
        elif leading.strip():
            logger.warning(
                "Solar Open2: dropping %d chars of output before the first "
                "parsed call (an opener that did not parse)",
                len(leading),
            )
        items = []
        for call in calls:
            self._content.call_parsed()
            self.current_tool_id += 1
            items.append(_item(call, index=self.current_tool_id))
        for prev, call in zip(calls, calls[1:]):
            between = complete[prev.end : call.start]
            head += self._content.segment(between, before_opener=True)
        head += self._content.segment(complete[calls[-1].end :])
        return StreamingParseResult(normal_text=head, calls=items)

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """The stream is over: release what was held back waiting for a marker
        that can no longer arrive (``_StreamContent.end``)."""
        held, self._buffer = self._buffer, ""
        partial = self._ends_with_partial_token(held, self.bot_token)
        text = self._content.end(held, partial_opener=partial)
        return StreamingParseResult(normal_text=text, calls=[])

    def supports_structural_tag(self) -> bool:
        """``required`` / named tool_choice use the JSON-schema array constraint
        instead. A structural tag constrains text, so the model can spell
        ``<|tool_call:end|>`` out as text while the FSM tracks sentinel ids
        and never sees the call close."""
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError("structure_info not used: no structural tag")


# ---------------------------------------------------------------------------
# Utilities


def _parse_calls(text: str, *, tools: List[Tool]) -> Tuple[List[_ParsedCall], bool]:
    """Every well-formed call in ``text``, in order, with its span, and whether
    a call with a blank name was seen (skipped here). Pure."""
    known = {t.function.name for t in tools}
    calls: List[_ParsedCall] = []
    blank_name = False
    for match in _CALL.finditer(text):
        name = match.group(1).strip()
        if not name:
            logger.warning("Solar Open2: tool call with an empty name; not a call")
            blank_name = True
            continue
        if name not in known:
            logger.warning(
                "Solar Open2: tool name %r not in request.tools; emitting the "
                "call for the client to handle",
                name,
            )
        arguments = _parse_arguments(name, body=match.group(2), tools=tools)
        calls.append(
            _ParsedCall(
                start=match.start(), end=match.end(), name=name, arguments=arguments
            )
        )
    return calls, blank_name


def _parse_arguments(name: str, *, body: str, tools: List[Tool]) -> Dict[str, Any]:
    if TOOL_ARG_START in body:
        return {
            m.group(1).strip(): _coerce(
                m.group(2), arg_type=_param_type(name, m.group(1).strip(), tools=tools)
            )
            for m in _ARG.finditer(body)
        }
    if not body.strip():
        return {}
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    logger.warning(
        "Solar Open2: call body for %r is neither argument markers nor a JSON "
        "object; passing it through raw",
        name,
    )
    return {"__raw": body}


def _param_type(func_name: str, param_name: str, *, tools: List[Tool]) -> Optional[str]:
    """JSON-schema ``type`` of a parameter from the request's tools, or None:
    the first ``function`` tool of that name whose schema declares the
    parameter (any other ``Tool.type`` is skipped, as the reference does)."""
    for tool in tools:
        if tool.type != "function" or tool.function.name != func_name:
            continue
        params = tool.function.parameters
        props = params.get("properties") if isinstance(params, dict) else None
        prop = props.get(param_name) if isinstance(props, dict) else None
        if not isinstance(prop, dict):
            continue
        t = prop.get("type")
        if isinstance(t, list):
            t = next((x for x in t if x != "null"), None)
        return None if t is None else str(t)
    return None


def _coerce(value: str, *, arg_type: Optional[str]) -> Any:
    """Convert a raw wire string to its JSON-schema type. Never raises: a value
    that does not convert is returned as the string, with a warning, so broken
    output still reaches the client as something inspectable; an unknown type
    is the string as well, without one. ``null`` (any case) is None whatever
    the type, and a ``null``/``none``-typed parameter is None whatever the
    value; a ``number`` with no fractional part is an int."""
    if value.strip().lower() == "null":
        return None
    pt = (arg_type or "string").strip().lower()
    if pt in ("string", "str", "text", "varchar", "char", "enum"):
        return value
    try:
        if pt in ("integer", "int"):
            return int(value)
        if pt in ("number", "float", "double"):
            f = float(value)
            return int(f) if f - int(f) == 0 else f
        if pt in ("boolean", "bool"):
            v = value.strip().lower()
            if v in ("true", "1", "yes"):
                return True
            if v in ("false", "0", "no"):
                return False
            raise ValueError(v)
        if pt in ("array", "list", "object", "dict"):
            return json.loads(value)
        if pt in ("null", "none"):
            return None
    except (ValueError, TypeError, OverflowError, json.JSONDecodeError):
        logger.warning(
            "solar_open2: failed to coerce %r to %s; returning as string.",
            value,
            arg_type,
        )
    return value


def _item(call: _ParsedCall, *, index: int) -> ToolCallItem:
    return ToolCallItem(
        tool_index=index,
        name=call.name,
        parameters=json.dumps(call.arguments, ensure_ascii=False),
    )
