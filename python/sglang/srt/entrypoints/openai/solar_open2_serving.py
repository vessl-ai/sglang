"""Serving-side rules for the Solar Open2 chat format (Solar Pro 4), in one
place; ``OpenAIServingChat`` calls each at one point.

1. ``validate_request`` -- ``reasoning_effort`` must be one of the named tiers
   (a number would silently pre-close the think block and skip the FSM budget).
2. ``normalize_reasoning_effort`` -- hands the requested effort and the tools
   flag to the scheduler-side FSM through ``custom_params`` (budget table in
   ``solar_open2_fsm``), then lower-cases the effort and folds xhigh/max to
   "high" for the chat template, which opens thinking only for medium/high
   with an exact, case-sensitive test.
3. ``injects_single_call_stop`` -- ``tool_choice="auto"`` with
   ``parallel_tool_calls=False`` has no grammar to cap the call count, so
   generation is halted at the first call's terminator (a stop string);
   required/named use the JSON-schema array with ``maxItems=1`` instead.
4. ``single_call_stop_matched`` / ``glue_for_text`` / ``glue_for_stream`` --
   the detokenizer trims a matched stop string, so the terminator the detector
   requires is glued back before parsing, but only onto an open call: without
   an opener the internal stop string must not reach the client as content.
   The internal stop is never reported as ``matched_stop``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, get_args

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ReasoningEffortTier,
    Tool,
    ToolChoice,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.solar_open2_detector import (
    TOOL_CALL_END,
    TOOL_CALL_START,
)
from sglang.srt.sampling.solar_open2_fsm import EFFORT_PARAM, TOOLS_PARAM

logger = logging.getLogger(__name__)

PARSER_NAME = "solar_open2"
# The template opens thinking only for medium/high; an effort above high would
# otherwise silently disable reasoning (the FSM still budgets the requested one).
EFFORT_FOLD = ("xhigh", "max")
EFFORT_TIERS = get_args(ReasoningEffortTier)


def is_solar_cell(
    reasoning_parser: Optional[str], tool_call_parser: Optional[str]
) -> bool:
    """Either Solar parser marks a Solar cell (the FSM reads custom_params)."""
    return PARSER_NAME in (reasoning_parser, tool_call_parser)


def validate_request(request: ChatCompletionRequest) -> Optional[str]:
    """Rule 1. ``chat_template_kwargs`` is checked too: the message pipeline
    moves its copy into the request field after validation."""
    efforts = (
        request.reasoning_effort,
        (request.chat_template_kwargs or {}).get("reasoning_effort"),
    )
    if any(isinstance(e, (int, float)) and not isinstance(e, bool) for e in efforts):
        return (
            "reasoning_effort must be one of "
            + ", ".join(EFFORT_TIERS)
            + " for this model."
        )
    return None


def _for_template(effort: str) -> str:
    effort = effort.strip().lower()
    return "high" if effort in EFFORT_FOLD else effort


def normalize_reasoning_effort(
    request: ChatCompletionRequest, tools_available: bool
) -> None:
    """Rule 2. ``tools_available``: whether a tool call can be answered at all
    (tools offered, tool_choice not "none", a parser configured) -- otherwise
    the FSM forbids ``<|tool_call:start|>`` in every state (it matters in
    CONTENT, where a model shut out of EOS takes it as the exit)."""
    ctk = request.chat_template_kwargs
    requested = (
        request.reasoning_effort
        if request.reasoning_effort is not None
        else (ctk or {}).get("reasoning_effort")
    )
    # custom_params rides SamplingParams to the Req. The entrypoint owns the
    # effort key: a value a client put there itself is not a budget override.
    custom = dict(request.custom_params or {})
    custom.pop(EFFORT_PARAM, None)
    if isinstance(requested, str) and requested.strip():
        custom[EFFORT_PARAM] = requested.strip().lower()
    custom[TOOLS_PARAM] = tools_available
    request.custom_params = custom

    if isinstance(request.reasoning_effort, str):
        request.reasoning_effort = _for_template(request.reasoning_effort)
    elif request.reasoning_effort is not None:
        # Only a server default (--default-chat-template-kwargs) can put a
        # non-string here past validate_request: not silently no reasoning,
        # but the template default, with a warning.
        logger.warning(
            "solar_open2: ignoring non-string reasoning_effort %r from the "
            "server defaults; using the template default",
            request.reasoning_effort,
        )
        request.reasoning_effort = None
    if ctk:
        effort = ctk.get("reasoning_effort")
        if isinstance(effort, str) and effort.strip():
            ctk["reasoning_effort"] = _for_template(effort)
        elif effort is not None and not isinstance(effort, str):
            ctk.pop("reasoning_effort")
        if isinstance(request.reasoning_effort, str) and "reasoning_effort" in ctk:
            # One source for the template: the request's (folded) effort. A
            # server default (--default-chat-template-kwargs) lands in ctk
            # after the client's value was moved to the request field, and the
            # template reads ctk last.
            ctk["reasoning_effort"] = request.reasoning_effort


def _auto_single_call(
    request: ChatCompletionRequest, effective_tools: List[Tool]
) -> bool:
    return bool(
        effective_tools
        and request.tool_choice != "none"
        and request.tool_choice != "required"
        and not isinstance(request.tool_choice, ToolChoice)
        and request.parallel_tool_calls is False
    )


def injects_single_call_stop(
    tool_call_parser: Optional[str],
    request: ChatCompletionRequest,
    effective_tools: List[Tool],
) -> bool:
    """Rule 3: whether ``<|tool_call:end|>`` is added to the request's stops."""
    return tool_call_parser == PARSER_NAME and _auto_single_call(
        request, effective_tools
    )


def single_call_stop_matched(
    tool_call_parser: Optional[str],
    request: ChatCompletionRequest,
    effective_tools: List[Tool],
    finish_reason: Optional[Dict[str, Any]],
) -> bool:
    """Rule 4: this request injected the terminator as a stop string and
    generation halted on exactly that stop -- and the detokenizer trimmed it
    (``no_stop_trim`` keeps it in the text, where the parser consumes it, so
    gluing it back would leak a second copy)."""
    return bool(
        injects_single_call_stop(tool_call_parser, request, effective_tools)
        and not request.no_stop_trim
        and finish_reason
        and finish_reason.get("type") == "stop"
        and finish_reason.get("matched") == TOOL_CALL_END
    )


def glue_for_text(text: str) -> str:
    """Non-streaming: the terminator to append before parsing, or nothing when
    the stop matched without an opener."""
    return TOOL_CALL_END if TOOL_CALL_START in text else ""


def glue_for_stream(parser: FunctionCallParser) -> str:
    """Streaming: the terminator to feed the detector before the stream-end
    flush, or nothing when it holds no open call."""
    return TOOL_CALL_END if parser.detector.holding_open_call() else ""
