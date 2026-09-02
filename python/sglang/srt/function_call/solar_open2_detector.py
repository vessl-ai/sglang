# SPDX-License-Identifier: Apache-2.0
"""Tool-call detector for Solar Open2 (Upstage).

Wire format (all markers are single special tokens in the tokenizer):

    <|tool_call:start|>{function_name}
    <|tool_arg:start|>{arg_name}<|tool_arg:value|>{arg_value}<|tool_arg:end|>
    <|tool_call:end|>

Argument values arrive as raw strings; they are coerced to the type declared on
``request.tools`` when a matching JSON-schema entry exists, falling back to the
original string on a lookup miss or a failed conversion. Ported from the Upstage
vLLM fork (``vllm/tool_parsers/solar_open2_tool_parser.py``); the verbatim
vendor parser lives in the test tree (``solar_open2_vendor_reference.py``) and
``TestSolarOpen2VendorDifferential`` keeps this detector aligned with it.

Tolerated degenerate shape: the model sometimes opens a call with a markdown
code fence carrying the function name instead of the start marker::

    ```{function_name}
    <|tool_arg:start|>{arg_name}<|tool_arg:value|>{arg_value}<|tool_arg:end|>
    <|tool_call:end|>

A fence line is treated as a call opener only when the next line begins with
``<|tool_arg:start|>``, so ordinary fenced code blocks in model output are
unaffected (a fence line not followed by an argument marker is never a call
opener, so a zero-argument call must use the real start marker).

A call whose function name is not in ``request.tools`` is emitted to the client
as-is (with a warning) rather than discarded: the client owns name validation
and can surface the mismatch back to the model, whereas a dropped call yields
an empty response with no diagnostic.

Forced tool calls: ``tool_choice="required"`` or a named choice are constrained
by the JSON-schema array (``supports_structural_tag()`` is False -- see that
method), so the output is ``[{"name": ..., "parameters": {...}}]`` parsed on the
serving layer's JSON path, not by this detector. ``parallel_tool_calls=False``
is then ``maxItems=1`` in that schema. For ``tool_choice="auto"`` there is no
grammar, so the serving layer injects ``<|tool_call:end|>`` as a stop string
and glues it back before parsing, capping generation at the first call (see
``entrypoints/openai/solar_open2_serving.py``).

A call body with no argument markers is still accepted as a JSON object when it
parses as one (the shape the legacy structural tag used to force; values arrive
already typed, so schema coercion is skipped); a non-empty body that is neither
marker-formed nor a JSON object is kept as ``{"__raw": body}`` with a warning
rather than dropped.

Content rules (what is *not* a call) follow the vendor:

* non-streaming (``extract_tool_calls``): with at least one parsed call, content
  is the text before the first opener -- nothing if that is only whitespace;
  with none (a marker alone, a malformed or truncated call, a blank name) the
  whole output is content;
* streaming (``extract_tool_calls_streaming``): real text is content wherever
  it is; a whitespace run is content only when it trails real text before the
  first call -- not as the whole prefix, not between or after calls when it
  abuts the next opener or the end (the vendor's pending-whitespace rule).

This detector emits complete calls only (the vendor streams argument values per
token) and makes the streaming result independent of where the chunks are cut:
the whitespace between calls is rstripped where the vendor keeps whatever share
arrived in the same delta, and output from an opener that did not parse is held
to the end of the stream -- content if no call ever parses (the vendor's
non-streaming rule), dropped otherwise. ``_StreamContent`` below owns those
rules; the parser itself (``_parse_calls``) is pure.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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

# The name group stops at a sentinel: with the vendor's ``(.+?)`` a call
# missing its newline followed by a well-formed one parses as one call named
# "<name><|tool_call:end|>..." (a fake call, no log). Newlines before the name
# are accepted as the vendor's lazy group plus ``strip()`` accepts them
# (FSM-legal: TOOL_CALL_BEGIN advances on any ordinary token).
# The body cannot run across another opener: an unfinished call followed by a
# well-formed one would otherwise parse as one call with the first name and
# the second body (the vendor's ``(.*?)`` does; ours yields the second call).
_MARKER_CALL = re.compile(
    rf"{re.escape(TOOL_CALL_START)}\n*((?:(?!<\|)[^\n])+?)\n"
    rf"((?:(?!{re.escape(TOOL_CALL_START)}).)*?)"
    rf"{re.escape(TOOL_CALL_END)}",
    re.DOTALL,
)
_FENCE_CALL = re.compile(
    rf"(?:^|\n)```([\w.-]+)[ \t]*\n"
    rf"((?:{re.escape(TOOL_ARG_START)}.*?{re.escape(TOOL_ARG_END)}\s*)+)"
    rf"{re.escape(TOOL_CALL_END)}",
    re.DOTALL,
)
_ARG = re.compile(
    rf"{re.escape(TOOL_ARG_START)}(.*?){re.escape(TOOL_ARG_VALUE)}"
    rf"(.*?){re.escape(TOOL_ARG_END)}",
    re.DOTALL,
)


def call_openers(text: str) -> List[int]:
    """Start offsets of every call opener in ``text`` (marker or fence form),
    ascending. A fence opener's offset is the fence line itself, not the
    newline before it."""
    starts = [m.start() for m in re.finditer(re.escape(TOOL_CALL_START), text)]
    starts += [
        m.start() + (1 if text[m.start()] == "\n" else 0)
        for m in FENCE_CALL_OPEN.finditer(text)
    ]
    return sorted(starts)


def has_call_opener(text: str) -> bool:
    """Whether ``text`` contains a call opener (marker or fence form)."""
    return TOOL_CALL_START in text or FENCE_CALL_OPEN.search(text) is not None


def partial_fence_open_len(text: str) -> int:
    """Length of a trailing segment of ``text`` that may still grow into a
    ``FENCE_CALL_OPEN`` match, or 0. Used to hold back streamed output so a
    fence-opened call is not emitted as plain text before its first argument
    marker arrives."""
    at = text.rfind("\n```")
    if at != -1:
        tail = text[at + 1 :]
    elif text.startswith("```"):
        tail = text
    else:
        tail = ""
    m = re.fullmatch(r"```([\w.-]*)[ \t]*(\n(.*))?", tail, re.DOTALL) if tail else None
    if m is not None:
        name, newline, after_newline = m.group(1), m.group(2), m.group(3)
        # A fence line whose name is still being written is a candidate; one
        # already closed by a newline is only a candidate with a name and an
        # argument marker (or its prefix) following.
        if not newline or (name and TOOL_ARG_START.startswith(after_newline)):
            return len(tail)
    # No live fence candidate: a run of one or two backticks at a line start
    # may still grow into one (the backticks can arrive split across
    # increments).
    m = re.search(r"(?:^|\n)(`{1,2})$", text)
    return len(m.group(1)) if m else 0


def _coerce(value: str, arg_type: Optional[str]) -> Any:
    """Coerce a raw wire string to the JSON-schema type, or keep it as-is.
    The literal ``null`` (any case, surrounding whitespace ignored) is None
    whatever the declared type, as in the vendor."""
    if value.strip().lower() == "null":
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


@dataclass
class _ParsedCall:
    start: int  # offset of the opener in the parsed text
    end: int  # offset just past the terminator
    name: str
    arguments: Dict[str, Any]


def _parse_arguments(name: str, body: str, tools: List[Tool]) -> Dict[str, Any]:
    if TOOL_ARG_START in body:
        return {
            m.group(1).strip(): _coerce(
                m.group(2), _param_type(name, m.group(1).strip(), tools)
            )
            for m in _ARG.finditer(body)
        }
    if not body.strip():
        return {}
    # Tolerated JSON-object body (the shape the legacy structural tag used to
    # force; required/named now take the JSON-array path and never reach this
    # detector), values already typed.
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


def _parse_calls(text: str, tools: List[Tool]) -> Tuple[List[_ParsedCall], bool]:
    """Every well-formed call in ``text``, in order, with its span, and whether
    a call with a blank name was seen (skipped here; the vendor's
    non-streaming parser then treats the whole output as content, while its
    streaming parser and ours keep the calls around it -- see the callers).
    Pure: no streaming state, no indices."""
    known = {t.function.name for t in tools}
    matches = sorted(
        list(_MARKER_CALL.finditer(text)) + list(_FENCE_CALL.finditer(text)),
        key=lambda m: m.start(),
    )
    calls: List[_ParsedCall] = []
    blank_name = False
    consumed_until = 0
    for match in matches:
        if match.start() < consumed_until:
            continue
        consumed_until = match.end()
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
        start = match.start() + (1 if text[match.start()] == "\n" else 0)
        calls.append(
            _ParsedCall(
                start,
                match.end(),
                name,
                _parse_arguments(name, match.group(2) or "", tools),
            )
        )
    return calls, blank_name


def _item(call: _ParsedCall, index: int) -> ToolCallItem:
    return ToolCallItem(
        tool_index=index,
        name=call.name,
        parameters=json.dumps(call.arguments, ensure_ascii=False),
    )


class _StreamContent:
    """Decides, for the streaming detector, which text outside a call is
    content -- the vendor's streaming rules, made independent of chunking
    (module docstring, "Content rules").

    State: ``pending_ws`` -- a whitespace run waiting for real text (content)
    or an opener / the end (not); ``emitted`` -- real text has been sent
    (whitespace trailing it before the first call is content, the vendor's
    ``_stream_content_emitted``); ``unparsed`` -- output from an opener that
    did not parse, held until a call parses (dropped) or the stream ends
    (content); ``calls`` -- calls emitted so far (after the first, prose is
    content but whitespace and unparsed calls are not)."""

    def __init__(self) -> None:
        self.pending_ws = ""
        self.emitted = False
        self.unparsed = ""
        self.calls = 0

    @property
    def after_call(self) -> bool:
        return self.calls > 0

    def text(self, text: str, before_opener: bool = False) -> str:
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
                # Whitespace abutting an opener is content only when it
                # trails real text before the first call ("text\n" + call
                # keeps the newline, as in non-streaming).
                return text if (not self.after_call and self.emitted) else ""
            self.pending_ws = text  # real text may still follow
            return ""
        self.emitted = True
        if before_opener:
            # Before the first call the whole prefix is content (vendor).
            # Between calls the vendor emits whatever share of the trailing
            # run its chunking put before the opener; rstrip is cut-invariant.
            return text if not self.after_call else text.rstrip()
        body = text.rstrip()
        self.pending_ws = text[len(body) :]
        return body

    def segment(self, segment: str, before_opener: bool = False) -> str:
        """Content in a segment after a parsed call that may hold unparsed
        calls: prose around them is content; a closed call that did not parse
        is not, from its opener to its terminator, nor is an unfinished one
        from its opener on (as in non-streaming)."""
        out: List[str] = []
        while True:
            openers = call_openers(segment)
            if not openers:
                out.append(self.text(segment, before_opener=before_opener))
                return "".join(out)
            at = openers[0]
            out.append(self.text(segment[:at], before_opener=True))
            end = segment.find(TOOL_CALL_END, at)
            if end == -1:
                self._dropped("unfinished tool call", len(segment) - at)
                return "".join(out)
            end += len(TOOL_CALL_END)
            self._dropped("closed tool call that did not parse", end - at)
            segment = segment[end:]

    def unparsed_call(self, complete: str) -> str:
        """Closed output from an opener on, with no parsed call in it (e.g. no
        newline after a name). After a call the segment rule applies (the
        prose around the unparsed calls is content); before one it is held."""
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
            logger.debug(
                "Solar Open2: a call parsed; dropping %d chars of earlier "
                "unparsed output",
                len(self.unparsed),
            )
            self.unparsed = ""
        self.calls += 1

    def end(self, held: str, partial_opener: int) -> str:
        """Content in what the detector still holds when the stream ends
        (``partial_opener``: length of a trailing partial opener / fence)."""
        pending, self.pending_ws = self.pending_ws, ""
        unparsed, self.unparsed = self.unparsed, ""
        if self.after_call:
            openers = call_openers(held)
            if openers:
                self._dropped(
                    "unfinished tool call at stream end", len(held) - openers[0]
                )
                return self.text(held[: openers[0]], before_opener=True)
            if partial_opener:
                held = held[:-partial_opener]
            return (pending + held).rstrip()  # a partial marker is not text
        if has_call_opener(held):
            logger.warning(
                "Solar Open2: stream ended inside an unfinished tool call "
                "(%d chars); returning it as content",
                len(held),
            )
        return pending + unparsed + held

    def _dropped(self, what: str, size: int) -> None:
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
        return has_call_opener(text)

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """The vendor's non-streaming rule (module docstring). Calls carry the
        tools-list index, SGLang's convention for every detector."""
        openers = call_openers(text)
        if not openers:
            return StreamingParseResult(normal_text=text, calls=[])
        calls, blank_name = _parse_calls(text, tools)
        if not calls or blank_name:
            # The vendor: no parsed call, or any call with a blank name, makes
            # the whole output content.
            logger.warning(
                "Solar Open2: tool call marker present but %s (%d chars); "
                "returning the output as content",
                "a call has a blank name" if blank_name else "no call parsed",
                len(text),
            )
            return StreamingParseResult(normal_text=text, calls=[])
        if len(openers) > len(calls):
            logger.warning(
                "Solar Open2: %d opener(s) in the output did not parse as a call "
                "and are not content",
                len(openers) - len(calls),
            )
        indices = self._get_tool_indices(tools)
        prefix = text[: openers[0]]
        return StreamingParseResult(
            normal_text=prefix if prefix.strip() else "",
            calls=[_item(c, indices.get(c.name, i)) for i, c in enumerate(calls)],
        )

    def holding_open_call(self) -> bool:
        """Streaming: an opener has arrived whose call is not closed yet (the
        text the serving layer must glue a trimmed terminator onto)."""
        return has_call_opener(self._buffer)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Buffer until whole calls are closed, then emit them at once, numbered
        sequentially across the stream (``current_tool_id``, the index a
        client accumulates deltas by -- two calls of the same tool must not
        share it). Content follows ``_StreamContent``."""
        self._buffer += new_text
        openers = call_openers(self._buffer)
        if not openers:
            hold = self._partial_opener_len(self._buffer)
            emit, self._buffer = (
                self._buffer[: len(self._buffer) - hold],
                self._buffer[len(self._buffer) - hold :],
            )
            return StreamingParseResult(normal_text=self._content.text(emit), calls=[])

        head = self._content.text(self._buffer[: openers[0]], before_opener=True)
        rest = self._buffer[openers[0] :]
        if TOOL_CALL_END not in rest:
            self._buffer = rest
            return StreamingParseResult(normal_text=head, calls=[])

        cut = rest.rindex(TOOL_CALL_END) + len(TOOL_CALL_END)
        complete, self._buffer = rest[:cut], rest[cut:]
        # A blank-name call is an unparsed segment here (dropped after a call,
        # held before one): the calls around it were or will be emitted, so
        # the non-streaming whole-output rule cannot apply.
        calls, _ = _parse_calls(complete, tools)
        if not calls:
            text = head + self._content.unparsed_call(complete)
            return StreamingParseResult(normal_text=text, calls=[])

        # Text inside ``complete`` before the first parsed call starts at an
        # opener that did not parse: after an earlier call the segment rule
        # applies (prose around it is content); before any call it is not
        # content, as the calls that follow rule out the whole-output rule.
        leading = complete[: calls[0].start]
        text_out = [head]
        if self._content.after_call:
            text_out.append(self._content.segment(leading, before_opener=True))
        elif leading.strip():
            logger.warning(
                "Solar Open2: dropping %d chars of output before the first "
                "parsed call (an opener that did not parse)",
                len(leading),
            )
        items: List[ToolCallItem] = []
        for i, call in enumerate(calls):
            if i:
                text_out.append(
                    self._content.segment(
                        complete[calls[i - 1].end : call.start], before_opener=True
                    )
                )
            self._content.call_parsed()
            self.current_tool_id += 1
            items.append(_item(call, self.current_tool_id))
        text_out.append(self._content.segment(complete[calls[-1].end :]))
        return StreamingParseResult(normal_text="".join(text_out), calls=items)

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """The stream is over: release what was held back waiting for a marker
        that can no longer arrive (``_StreamContent.end``)."""
        held, self._buffer = self._buffer, ""
        # A fence candidate at the end is ordinary text (the vendor has no
        # fence form); only a partial marker is not.
        partial_marker = self._ends_with_partial_token(held, self.bot_token) or 0
        text = self._content.end(held, partial_marker)
        return StreamingParseResult(normal_text=text, calls=[])

    def _partial_opener_len(self, text: str) -> int:
        """Length of a trailing suffix that could grow into either opener."""
        return max(
            self._ends_with_partial_token(text, self.bot_token) or 0,
            partial_fence_open_len(text),
        )

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
        tool-call envelope by sentinel id, then never saw the call close. With
        JSON output no sentinel is emitted at all and the FSM stays in
        CONTENT, where the grammar owns the phase -- the vendor's exact
        rule."""
        return False

    def structure_info(self) -> _GetInfoFunc:
        """Envelope of the legacy structural tag. Required by the base class;
        not reached for ``solar_open2`` since :meth:`supports_structural_tag`
        is False (required/named take the JSON-array path). If it were used,
        xgrammar would fill a JSON object between ``begin`` and ``end``, the
        body shape ``_parse_arguments`` still accepts."""
        return lambda name: StructureInfo(
            begin=f"{TOOL_CALL_START}{name}\n",
            end=TOOL_CALL_END,
            trigger=TOOL_CALL_START,
        )
