"""Serving-side rules for Solar Open2 (Solar Pro 4), in one place.

``OpenAIServingChat`` calls these at five points; each is a one-liner there.
The rules mirror the vendor's vLLM serving of the same checkpoint (Upstage
patch set for vLLM 0.25.0, 2026-09-01) unless a comment says otherwise.

1. ``validate_request`` -- a float ``reasoning_effort`` (an SGLang extension)
   is rejected: the vendor's request model accepts the named tiers only (422),
   and here a float would silently pre-close the think block.
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

from typing import Any, Dict, List, Optional

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    Tool,
    ToolChoice,
)
from sglang.srt.function_call.solar_open2_detector import (
    TOOL_CALL_END,
    has_call_opener,
)
from sglang.srt.sampling.solar_open2_fsm import EFFORT_PARAM, TOOLS_PARAM

PARSER_NAME = "solar_open2"
# The template opens thinking only for medium/high; an effort above high would
# otherwise silently disable reasoning (the FSM still budgets the requested one).
EFFORT_FOLD = ("xhigh", "max")
EFFORT_TIERS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def is_solar_cell(
    reasoning_parser: Optional[str], tool_call_parser: Optional[str]
) -> bool:
    """Either Solar parser marks a Solar cell (the FSM reads custom_params)."""
    return PARSER_NAME in (reasoning_parser, tool_call_parser)


def validate_request(request: ChatCompletionRequest) -> Optional[str]:
    if isinstance(request.reasoning_effort, float):
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
    if ctk:
        effort = ctk.get("reasoning_effort")
        if isinstance(effort, str) and effort.strip():
            ctk["reasoning_effort"] = _for_template(effort)
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
    the stop matched without an opener (marker or fence form)."""
    return TOOL_CALL_END if has_call_opener(text) else ""


def glue_for_stream(parser: Any) -> str:
    """Streaming: the terminator to feed the detector before the stream-end
    flush, or nothing when it holds no open call."""
    detector = getattr(parser, "detector", None)
    holding = getattr(detector, "holding_open_call", None)
    return TOOL_CALL_END if holding is not None and holding() else ""
