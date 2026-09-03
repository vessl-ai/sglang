"""Serving-side rules for the Solar Open2 chat format (Solar Pro 4), in one
place; ``OpenAIServingChat`` calls them. Only the ``finish_reason`` rewrite
and the glue-before-flush order of rule 4 sit at the call sites.

1. ``validate_request`` -- ``reasoning_effort`` must name a tier, in the
   request field (typed upstream) and in ``chat_template_kwargs`` (untyped):
   anything else would silently pre-close the think block and skip the FSM
   budget.
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
    for effort in efforts:
        if effort is None or (
            isinstance(effort, str) and effort.strip().lower() in EFFORT_TIERS
        ):
            continue
        return (
            "reasoning_effort must be one of "
            + ", ".join(EFFORT_TIERS)
            + " for this model."
        )
    return None


_WARNED_EFFORTS: set = set()


def _template_effort(value: Any) -> Optional[str]:
    """The effort handed to the chat template: lower-cased, xhigh/max folded to
    high. Anything that is not a tier -- only a server default can get here
    past ``validate_request`` -- falls back to the template default, with a
    warning."""
    tier = value.strip().lower() if isinstance(value, str) else None
    if tier in EFFORT_FOLD:
        return "high"
    if tier in EFFORT_TIERS:
        return tier
    if repr(value) not in _WARNED_EFFORTS:
        # Once per distinct value: a bad server default would otherwise log
        # at request rate.
        _WARNED_EFFORTS.add(repr(value))
        logger.warning(
            "solar_open2: ignoring reasoning_effort %r (not a tier); using the "
            "template default",
            value,
        )
    return None


def normalize_reasoning_effort(
    request: ChatCompletionRequest, tools_available: bool
) -> None:
    """Rule 2. ``tools_available``: whether a tool call can be answered at all
    (tools offered, tool_choice not "none", a parser configured); the FSM
    forbids ``<|tool_call:start|>`` everywhere when it cannot (see
    ``solar_open2_fsm.TOOLS_PARAM``)."""
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

    raw = request.reasoning_effort
    if raw is not None:
        request.reasoning_effort = _template_effort(raw)
    if ctk and "reasoning_effort" in ctk:
        # Only a server default (--default-chat-template-kwargs) can still be
        # here: the client's copy was moved into the request field before
        # this, and so was the default when that field was empty -- then the
        # fold above already judged it. The template reads ctk last, so it
        # gets one source: the request's folded effort when there is one
        # (a present ``None`` would fail the template's test and close the
        # think block while the parser starts in reasoning).
        default = ctk["reasoning_effort"]
        folded = (
            request.reasoning_effort
            if default in (raw, None)
            else _template_effort(default)
        )
        effort = request.reasoning_effort or folded
        if effort is None:
            ctk.pop("reasoning_effort")
        else:
            ctk["reasoning_effort"] = effort


def injects_single_call_stop(
    tool_call_parser: Optional[str],
    *,
    request: ChatCompletionRequest,
    effective_tools: List[Tool],
) -> bool:
    """Rule 3: whether ``<|tool_call:end|>`` is added to the request's stops."""
    return bool(
        tool_call_parser == PARSER_NAME
        and effective_tools
        and request.tool_choice == "auto"
        and request.parallel_tool_calls is False
    )


def single_call_stop_matched(
    tool_call_parser: Optional[str],
    *,
    request: ChatCompletionRequest,
    effective_tools: List[Tool],
    finish_reason: Optional[Dict[str, Any]],
) -> bool:
    """Rule 4: this request injected the terminator as a stop string and
    generation halted on exactly that stop -- and the detokenizer trimmed it
    (``no_stop_trim`` keeps it in the text, where the parser consumes it, so
    gluing it back would leak a second copy)."""
    return bool(
        injects_single_call_stop(
            tool_call_parser, request=request, effective_tools=effective_tools
        )
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
