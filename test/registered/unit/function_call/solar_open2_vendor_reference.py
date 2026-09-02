# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference copy of the vendor's vLLM tool parser for Solar Open2
(UpstageAI patch set for vLLM 0.25.0, 2026-09-01: ``01-tool-parser.patch``,
``vllm/tool_parsers/solar_open2_tool_parser.py``), kept verbatim apart from
the vLLM imports (replaced by the minimal stand-ins below so the parser runs
inside this test tree without vLLM), the two ``timeout=`` keyword arguments of
its regex ``finditer`` calls, and the line wrapping of one regex literal. It is the ground truth the
SGLang detector is compared against in ``test_function_call_parser.py``
(``TestSolarOpen2VendorDifferential``). Do not "fix" it: a difference from
the detector is either a documented delta or a detector bug.
"""

import json
import re as _re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional


def init_logger(name):
    import logging

    return logging.getLogger(name)


TokenizerLike = object


class _Record:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __repr__(self):
        return f"{type(self).__name__}({self.__dict__})"


class ChatCompletionRequest(_Record):
    pass


class FunctionCall(_Record):
    pass


class ToolCall(_Record):
    pass


class DeltaFunctionCall(_Record):
    pass


class DeltaToolCall(_Record):
    pass


class DeltaMessage(_Record):
    pass


class ExtractedToolCallInformation(_Record):
    pass


class ToolParser:
    def __init__(self, tokenizer, *args, **kwargs):
        self.model_tokenizer = tokenizer
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool = []


class _Compiled:
    """``regex``'s ``timeout=`` keyword, accepted and ignored (stdlib ``re``
    underneath)."""

    def __init__(self, pattern):
        self._p = pattern

    def finditer(self, text, timeout=None):
        return self._p.finditer(text)

    def search(self, text, timeout=None):
        return self._p.search(text)

    def match(self, text, timeout=None):
        return self._p.match(text)


class _Regex:
    DOTALL = _re.DOTALL
    MULTILINE = _re.MULTILINE

    @staticmethod
    def compile(pattern, flags=0):
        return _Compiled(_re.compile(pattern, flags))

    @staticmethod
    def escape(s):
        return _re.escape(s)


re = _Regex()  # noqa: F811 -- the vendor's ``regex`` module
logger = init_logger(__name__)


class SolarOpen2ToolParser(ToolParser):
    """
    Tool call parser for Solar Open2 model.

    Parses the format:
        <|tool_call:start|>{function_name}
        <|tool_arg:start|>{arg_name}<|tool_arg:value|>{arg_value}<|tool_arg:end|>
        <|tool_call:end|>

    Argument values are surfaced as raw strings in the wire format. When a
    matching JSON-schema entry is found on ``request.tools``, values are
    coerced to the declared type (int/float/bool/array/object/null). The
    literal string ``"null"`` always coerces to ``None``. Falls back to the
    original string on lookup miss or conversion failure so malformed output
    still round-trips.
    """

    TOOL_CALL_START = "<|tool_call:start|>"
    TOOL_CALL_END = "<|tool_call:end|>"
    TOOL_ARG_START = "<|tool_arg:start|>"
    TOOL_ARG_VALUE = "<|tool_arg:value|>"
    TOOL_ARG_END = "<|tool_arg:end|>"

    # Streaming state machine states.
    _STATE_WAITING_FOR_TOOL = "waiting_for_tool"
    _STATE_READING_FUNCTION_NAME = "reading_function_name"
    _STATE_WAITING_IN_CALL = "waiting_in_call"
    _STATE_READING_ARG_NAME = "reading_arg_name"
    _STATE_READING_ARG_VALUE = "reading_arg_value"

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        # v0.22.0 instantiates tool parsers as `cls(tokenizer, tools)`; accept
        # and forward extra positional/keyword args so the signature stays
        # forward-compatible (mirrors the reasoning parser).
        super().__init__(tokenizer, *args, **kwargs)
        self._reset_stream_state()

        tc_start = re.escape(self.TOOL_CALL_START)
        tc_end = re.escape(self.TOOL_CALL_END)
        ta_start = re.escape(self.TOOL_ARG_START)
        ta_value = re.escape(self.TOOL_ARG_VALUE)
        ta_end = re.escape(self.TOOL_ARG_END)

        self.tool_call_pattern = re.compile(
            rf"{tc_start}(.+?)\n" rf"((?:{ta_start}.*?{ta_end}\n?)*)" rf"{tc_end}",
            re.DOTALL,
        )
        self.tool_arg_pattern = re.compile(
            rf"{ta_start}(.*?){ta_value}(.*?){ta_end}",
            re.DOTALL,
        )

    def _get_param_type(
        self,
        func_name: str,
        param_name: str,
        tools: Sequence | None,
    ) -> str:
        """Return JSON-schema ``type`` for the param, or ``"string"`` if unknown."""
        if not tools:
            return "string"
        for tool in tools:
            if getattr(tool, "type", None) != "function":
                continue
            fn = getattr(tool, "function", None)
            if fn is None or getattr(fn, "name", None) != func_name:
                continue
            params = getattr(fn, "parameters", None)
            if not isinstance(params, dict):
                continue
            props = params.get("properties")
            if not isinstance(props, dict):
                continue
            spec = props.get(param_name)
            if not isinstance(spec, dict):
                continue
            t = spec.get("type", "string")
            # JSON Schema allows ``type`` to be a list (union). Prefer the
            # first non-null entry so e.g. ``["integer", "null"]`` still
            # coerces to int. This is more permissive than qwen3xml /
            # qwen3coder / step3p5 / seed_oss, which ``str()`` the list
            # and so fall through to string for any union.
            #
            # For a genuine multi-type union like ``["string", "integer"]``
            # this heuristic is order-dependent — values that should be
            # ``int`` but appear later in the list will stay as strings.
            # A proper fix is the priority-based approach used by
            # ``minimax_m2_tool_parser`` (``_extract_types_from_schema`` +
            # ``_convert_param_value_with_types``), which also handles
            # ``anyOf`` / ``oneOf`` / ``allOf``. Left as follow-up work.
            if isinstance(t, list):
                for cand in t:
                    if cand != "null":
                        return str(cand)
                return "string"
            return str(t)
        return "string"

    def _coerce(self, value: str, param_type: str) -> Any:
        """Convert raw captured string to the target JSON-schema type.

        Never raises — falls back to the original string on conversion
        failure so broken model output still round-trips to the client as
        something inspectable.
        """
        if value.strip().lower() == "null":
            return None

        pt = param_type.strip().lower()
        # Match the string-alias set used by qwen3xml / qwen3coder /
        # step3p5 / seed_oss. ``enum`` and ``varchar`` are not JSON Schema
        # standard ``type`` values, but real-world schemas occasionally
        # place them in the ``type`` field, so they are accepted
        # defensively and treated as strings.
        if pt in ("string", "str", "text", "varchar", "char", "enum"):
            return value
        if pt in ("integer", "int"):
            try:
                return int(value)
            except (ValueError, TypeError):
                logger.warning(
                    "solar_open2: failed to coerce %r to int "
                    "(param_type=%s); returning as string.",
                    value,
                    param_type,
                )
                return value
        if pt in ("number", "float", "double"):
            try:
                f = float(value)
                # Match the downcast heuristic used by qwen3coder / step3p5 /
                # seed_oss / minimax_m2: if the fractional part is zero,
                # return int. JSON Schema ``number`` permits floats, but
                # type-strict evaluators (e.g. BFCL) often expect int when
                # the ground truth is an int, and the model's textual
                # rendering of an int is an int.
                return int(f) if f - int(f) == 0 else f
            except (ValueError, TypeError):
                logger.warning(
                    "solar_open2: failed to coerce %r to float "
                    "(param_type=%s); returning as string.",
                    value,
                    param_type,
                )
                return value
        if pt in ("boolean", "bool"):
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
        if pt in ("array", "list", "object", "dict"):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "solar_open2: failed to json.loads %r "
                    "(param_type=%s); returning as string.",
                    value,
                    param_type,
                )
                return value
        if pt in ("null", "none"):
            return None
        return value

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if self.TOOL_CALL_START not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        # Content is everything before the first tool call
        first_tc_idx = model_output.index(self.TOOL_CALL_START)
        content_prefix = model_output[:first_tc_idx]
        content = (
            content_prefix if content_prefix and not content_prefix.isspace() else None
        )

        tools = getattr(request, "tools", None)

        tool_calls: list[ToolCall] = []
        try:
            for match in self.tool_call_pattern.finditer(
                model_output,
            ):
                func_name = match.group(1).strip()
                if not func_name:
                    return ExtractedToolCallInformation(
                        tools_called=False,
                        tool_calls=[],
                        content=model_output,
                    )
                args_block = match.group(2)

                args_dict: dict[str, Any] = {}
                for arg_match in self.tool_arg_pattern.finditer(
                    args_block,
                ):
                    arg_name = arg_match.group(1)
                    arg_value_raw = arg_match.group(2)
                    param_type = self._get_param_type(func_name, arg_name, tools)
                    args_dict[arg_name] = self._coerce(arg_value_raw, param_type)

                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=func_name,
                            arguments=json.dumps(args_dict, ensure_ascii=False),
                        ),
                        id=f"call_{uuid.uuid4().hex[:24]}",
                    )
                )
        except TimeoutError:
            logger.warning(
                "Regex timeout occurred when parsing a Solar Open2 tool call."
            )
            logger.debug(
                "Solar Open2 tool-call regex timed out on model output: %s",
                model_output,
            )
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        if not tool_calls:
            # A start marker alone is not a valid call. Preserve malformed or
            # truncated output as ordinary content instead of dropping the
            # unmatched suffix after the marker.
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content,
        )

    def _reset_stream_state(self) -> None:
        """Reset streaming-only fields. Safe to call from ``__init__`` and from
        the start of every new stream (detected via empty ``previous_text``).
        """
        self._stream_buffer: str = ""
        # Whitespace run held back while WAITING_FOR_TOOL. The non-streaming
        # parser normalizes whitespace-only content around tool calls to
        # ``None``; to keep streamed conversations byte-equivalent we hold
        # pure whitespace until we know whether real content follows (flush
        # together) or a tool call starts (drop, matching non-streaming).
        self._stream_pending_ws: str = ""
        # True once any non-whitespace content has been streamed. Used to
        # mirror the non-streaming rule exactly: content is everything before
        # the FIRST tool call (so a whitespace run that trails real content
        # is kept), while whitespace-only content is normalized away.
        self._stream_content_emitted: bool = False
        self._stream_state: str = self._STATE_WAITING_FOR_TOOL
        self._stream_func_name: str = ""
        self._stream_arg_name: str = ""
        # Raw bytes consumed while deciding whether a tool-call start is
        # valid. If its function name is empty, these bytes are restored as
        # ordinary content instead of being lost behind an invalid tool call.
        self._stream_candidate_text: str = ""
        # First-arg flag per current tool call: on the first completed arg
        # in a call we emit the opening ``{``; after that we emit ``,`` separators.
        self._stream_first_arg_in_call: bool = True
        # Per-argument state for incrementally streaming pure string values.
        # Other schema types stay buffered until TOOL_ARG_END so ``_coerce``
        # can preserve the non-streaming conversion/fallback contract.
        self._stream_arg_param_type: str = "string"
        self._stream_partial_string: bool = False
        self._stream_value_key_opened: bool = False

    @staticmethod
    def _is_string_param_type(param_type: str) -> bool:
        """Return whether ``param_type`` is safe to stream as a JSON string."""
        return param_type.strip().lower() in (
            "string",
            "str",
            "text",
            "varchar",
            "char",
            "enum",
        )

    @staticmethod
    def _json_string_body(value: str) -> str:
        """JSON-escape ``value`` without its surrounding quotes."""
        return json.dumps(value, ensure_ascii=False)[1:-1]

    @staticmethod
    def _may_still_be_null_literal(raw: str) -> bool:
        """Return whether appending bytes could still make ``raw`` JSON null.

        ``_coerce`` maps a stripped, case-insensitive ``null`` literal to
        ``None`` even for string schema types. Do not irreversibly open a JSON
        string while the partial value can still become that literal.
        """
        stripped = raw.strip().lower()
        if not stripped or stripped == "null":
            return True
        return "null".startswith(stripped)

    def _append_arg_fragment(
        self,
        fragment: str,
        tool_calls_out: list[DeltaToolCall],
    ) -> None:
        """Emit one argument delta and keep both serving trackers identical.

        Fragments for the same tool call within one DeltaMessage are coalesced
        into a single DeltaToolCall entry: OpenAI and upstream vLLM parsers
        emit at most one ``tool_calls`` entry per index per SSE chunk, and
        clients that read only ``delta.tool_calls[0]`` (or merge same-index
        entries by overwrite instead of concat) silently drop every extra
        entry — e.g. the first fragment of a string value emitted alongside
        its ``{"key": "`` opener.
        """
        if not fragment:
            return
        tool_idx = self.current_tool_id
        combined = self.streamed_args_for_tool[tool_idx] + fragment
        self.streamed_args_for_tool[tool_idx] = combined
        self.prev_tool_call_arr[tool_idx]["arguments"] = combined
        if tool_calls_out and tool_calls_out[-1].index == tool_idx:
            last = tool_calls_out[-1]
            if last.function is None:
                last.function = DeltaFunctionCall(arguments=fragment)
            else:
                last.function.arguments = (last.function.arguments or "") + fragment
            return
        tool_calls_out.append(
            DeltaToolCall(
                index=tool_idx,
                function=DeltaFunctionCall(arguments=fragment),
            )
        )

    def _stream_safe_string_prefix(
        self,
        tool_calls_out: list[DeltaToolCall],
    ) -> None:
        """Emit and consume the safe prefix of the current string argument.

        Only a possible suffix of ``TOOL_ARG_END`` remains buffered. Consuming
        emitted raw text avoids re-escaping the full value on every token.
        """
        holdback = self._holdback_len(self._stream_buffer, (self.TOOL_ARG_END,))
        flush_end = len(self._stream_buffer) - holdback
        if flush_end <= 0:
            return

        raw_prefix = self._stream_buffer[:flush_end]
        if not self._stream_value_key_opened and self._may_still_be_null_literal(
            raw_prefix
        ):
            return

        if not self._stream_value_key_opened:
            object_prefix = "{" if self._stream_first_arg_in_call else ", "
            self._stream_first_arg_in_call = False
            key_json = json.dumps(self._stream_arg_name, ensure_ascii=False)
            self._append_arg_fragment(f'{object_prefix}{key_json}: "', tool_calls_out)
            self._stream_value_key_opened = True

        self._append_arg_fragment(self._json_string_body(raw_prefix), tool_calls_out)
        self._stream_buffer = self._stream_buffer[flush_end:]

    def _holdback_len(self, buf: str, sentinels: tuple[str, ...]) -> int:
        """Number of trailing bytes of ``buf`` that could be the start of any
        sentinel in ``sentinels`` and therefore must stay in the buffer until
        the next delta arrives.

        Example: if ``buf`` ends with ``"<|tool_ca"`` and one sentinel is
        ``"<|tool_call:start|>"``, return 9 so the caller flushes everything
        before those 9 bytes and carries those 9 bytes forward.
        """
        if not buf or not sentinels:
            return 0
        # Anchor on the last ``<`` — all solar_open2 sentinels start with ``<|``.
        last_lt = buf.rfind("<")
        if last_lt == -1:
            return 0
        tail = buf[last_lt:]
        for s in sentinels:
            # Only holdback if ``tail`` is a proper, non-empty prefix of ``s``.
            # If ``tail == s`` we don't hold back (the full sentinel is already
            # present and will be consumed by the state machine).
            if len(tail) < len(s) and s.startswith(tail):
                return len(tail)
        return 0

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # Fresh stream: reset both shared (base-class) and streaming-only state.
        if not previous_text:
            self.prev_tool_call_arr = []
            self.current_tool_id = -1
            self.current_tool_name_sent = False
            self.streamed_args_for_tool = []
            self._reset_stream_state()

        if delta_text:
            self._stream_buffer += delta_text

        tools = getattr(request, "tools", None)
        content_out = ""
        tool_calls_out: list[DeltaToolCall] = []

        while True:
            if self._stream_state == self._STATE_WAITING_FOR_TOOL:
                idx = self._stream_buffer.find(self.TOOL_CALL_START)
                if idx == -1:
                    # No sentinel yet — flush everything as content except
                    # any trailing bytes that could be the start of the
                    # sentinel we're waiting on.
                    hb = self._holdback_len(
                        self._stream_buffer, (self.TOOL_CALL_START,)
                    )
                    flush_end = len(self._stream_buffer) - hb
                    if flush_end > 0:
                        chunk = (
                            self._stream_pending_ws + self._stream_buffer[:flush_end]
                        )
                        self._stream_buffer = self._stream_buffer[flush_end:]
                        if chunk.strip() == "":
                            # Pure whitespace: hold — it may precede a tool
                            # call, in which case non-streaming drops it.
                            self._stream_pending_ws = chunk
                        else:
                            # Real content: flush, but keep any trailing
                            # whitespace run pending (it may abut the next
                            # tool call).
                            body = chunk.rstrip()
                            self._stream_pending_ws = chunk[len(body) :]
                            content_out += body
                            self._stream_content_emitted = True
                    break
                # Sentinel found. Content before it flushes only if it
                # contains non-whitespace (matching the non-streaming
                # whitespace-only -> None normalization).
                prefix = self._stream_pending_ws + self._stream_buffer[:idx]
                self._stream_pending_ws = ""
                suppressed_prefix = ""
                if prefix.strip() != "":
                    content_out += prefix
                    self._stream_content_emitted = True
                elif (
                    prefix
                    and self._stream_content_emitted
                    and self.current_tool_id == -1
                ):
                    # Whitespace trailing real content before the FIRST tool
                    # call is part of non-streaming content ("text\n" + call
                    # -> content "text\n"). Between/after calls it is not.
                    content_out += prefix
                else:
                    suppressed_prefix = prefix
                self._stream_buffer = self._stream_buffer[
                    idx + len(self.TOOL_CALL_START) :
                ]
                self._stream_candidate_text = suppressed_prefix + self.TOOL_CALL_START
                self._stream_state = self._STATE_READING_FUNCTION_NAME
                continue

            if self._stream_state == self._STATE_READING_FUNCTION_NAME:
                nl = self._stream_buffer.find("\n")
                if nl == -1:
                    # Wait for the newline that terminates the function name.
                    break
                raw_func_name = self._stream_buffer[:nl]
                self._stream_func_name = raw_func_name.strip()
                if not self._stream_func_name:
                    content_out += (
                        self._stream_candidate_text
                        + raw_func_name
                        + self._stream_buffer[nl]
                    )
                    self._stream_candidate_text = ""
                    self._stream_buffer = self._stream_buffer[nl + 1 :]
                    self._stream_content_emitted = True
                    self._stream_state = self._STATE_WAITING_FOR_TOOL
                    continue
                self._stream_candidate_text = ""
                self._stream_buffer = self._stream_buffer[nl + 1 :]

                # Register a new tool call and emit the name delta.
                self.current_tool_id += 1
                tool_idx = self.current_tool_id
                while len(self.prev_tool_call_arr) <= tool_idx:
                    self.prev_tool_call_arr.append({"name": "", "arguments": ""})
                while len(self.streamed_args_for_tool) <= tool_idx:
                    self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr[tool_idx]["name"] = self._stream_func_name

                self._stream_first_arg_in_call = True
                tool_calls_out.append(
                    DeltaToolCall(
                        index=tool_idx,
                        id=f"call_{uuid.uuid4().hex[:24]}",
                        type="function",
                        function=DeltaFunctionCall(
                            name=self._stream_func_name, arguments=""
                        ),
                    )
                )
                self.current_tool_name_sent = True
                self._stream_state = self._STATE_WAITING_IN_CALL
                continue

            if self._stream_state == self._STATE_WAITING_IN_CALL:
                ta_idx = self._stream_buffer.find(self.TOOL_ARG_START)
                tc_idx = self._stream_buffer.find(self.TOOL_CALL_END)
                if ta_idx == -1 and tc_idx == -1:
                    # Could be partway through either sentinel — hold back
                    # any trailing prefix match and wait for more.
                    break
                pick_arg = ta_idx != -1 and (tc_idx == -1 or ta_idx < tc_idx)
                if pick_arg:
                    self._stream_buffer = self._stream_buffer[
                        ta_idx + len(self.TOOL_ARG_START) :
                    ]
                    self._stream_state = self._STATE_READING_ARG_NAME
                    continue
                # TOOL_CALL_END → close current JSON object.
                self._stream_buffer = self._stream_buffer[
                    tc_idx + len(self.TOOL_CALL_END) :
                ]
                close_str = "{}" if self._stream_first_arg_in_call else "}"
                self._append_arg_fragment(close_str, tool_calls_out)
                self._stream_state = self._STATE_WAITING_FOR_TOOL
                continue

            if self._stream_state == self._STATE_READING_ARG_NAME:
                v_idx = self._stream_buffer.find(self.TOOL_ARG_VALUE)
                if v_idx == -1:
                    break
                self._stream_arg_name = self._stream_buffer[:v_idx]
                self._stream_buffer = self._stream_buffer[
                    v_idx + len(self.TOOL_ARG_VALUE) :
                ]
                self._stream_arg_param_type = self._get_param_type(
                    self._stream_func_name, self._stream_arg_name, tools
                )
                self._stream_partial_string = self._is_string_param_type(
                    self._stream_arg_param_type
                )
                self._stream_value_key_opened = False
                self._stream_state = self._STATE_READING_ARG_VALUE
                continue

            if self._stream_state == self._STATE_READING_ARG_VALUE:
                e_idx = self._stream_buffer.find(self.TOOL_ARG_END)
                if e_idx == -1:
                    if self._stream_partial_string:
                        self._stream_safe_string_prefix(tool_calls_out)
                    # Non-string types stay fully buffered so ``_coerce`` sees
                    # the complete value atomically.
                    break
                raw_value = self._stream_buffer[:e_idx]
                self._stream_buffer = self._stream_buffer[
                    e_idx + len(self.TOOL_ARG_END) :
                ]

                if self._stream_value_key_opened:
                    # Earlier raw prefixes have already been escaped, emitted,
                    # and consumed. Finish only the remaining suffix.
                    self._append_arg_fragment(
                        self._json_string_body(raw_value) + '"',
                        tool_calls_out,
                    )
                    self._stream_value_key_opened = False
                    self._stream_partial_string = False
                    self._stream_state = self._STATE_WAITING_IN_CALL
                    continue

                # Atomic path: non-string types, null, empty strings, short
                # null-prefix strings, or a complete value in one input delta.
                coerced = self._coerce(raw_value, self._stream_arg_param_type)
                prefix = "{" if self._stream_first_arg_in_call else ", "
                self._stream_first_arg_in_call = False
                key_json = json.dumps(self._stream_arg_name, ensure_ascii=False)
                value_json = json.dumps(coerced, ensure_ascii=False)
                self._append_arg_fragment(
                    f"{prefix}{key_json}: {value_json}", tool_calls_out
                )
                self._stream_partial_string = False
                self._stream_state = self._STATE_WAITING_IN_CALL
                continue

            # Unreachable — every state above either `continue`s or `break`s.
            break

        if not content_out and not tool_calls_out:
            return None
        return DeltaMessage(
            content=content_out or None,
            tool_calls=tool_calls_out,
        )
