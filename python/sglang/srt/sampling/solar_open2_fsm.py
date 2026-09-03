# SPDX-License-Identifier: Apache-2.0
"""Solar Open2 chat-envelope FSM: a per-request logits mask for the Solar Pro 4
chat template.

The template wraps a reply in ``<|think:start|>`` ... ``<|think:end|>`` followed
by content that may hold tool calls (``<|tool_call:start|>`` name, argument
runs ``<|tool_arg:start|>`` ``<|tool_arg:value|>`` ``<|tool_arg:end|>``,
``<|tool_call:end|>``). This module keeps a generation inside that envelope:

* REASONING: EOS, ``<|im:end|>`` and every control sentinel but ``<|think:end|>``
  are masked, so the block can only end through ``<|think:end|>``; once the
  reasoning budget is spent the next token is forced to ``<|think:end|>``; the
  token right after ``<|think:start|>`` may not be a newline run.
* CONTENT: a turn that has produced no content token yet may not end (EOS and
  ``<|im:end|>`` masked); after the first content token only stray sentinels
  are masked. A request that offers no tools also loses ``<|tool_call:start|>``
  (``TOOLS_PARAM``).
* Tool-call states (``TOOL_CALL_BEGIN`` .. ``TOOL_CALL_END``): only the
  envelope sentinel(s) the sub-state expects are open, so a call cannot be
  cut short by a turn end or a stray sentinel (after ``<|tool_call:end|>``
  the turn may end). When the request carries a grammar (json_schema / regex
  / ebnf / structural_tag) the grammar owns the CONTENT phase and this mask
  stays out of it.

The states, their allowed-sentinel tables, the transitions and the budget table
(``_MASK_SPEC_BY_STATE``, ``_MASK_SPEC_CONTENT``, ``_TRANSITIONS``,
``_EFFORT_BUDGETS``) are the ones the model was trained against. Ids are
resolved from the served tokenizer at first use, never hardcoded.

The reasoning budget is a fixed number of tokens per ``reasoning_effort``
(``none`` / ``minimal`` close the block at once; nothing above the hard limit;
default effort ``high``), independent of ``max_tokens``. The chat entrypoint
hands the request's effort and whether it offers tools to the scheduler through
``custom_params`` (``EFFORT_PARAM``, ``TOOLS_PARAM``); ``/generate`` and
``/v1/completions`` carry neither and get the defaults. The chat template opens
a think block only for ``medium`` / ``high`` (other efforts pre-close it), so
the ``low`` budget only applies to a block the model opens itself, e.g. after a
tool result.

Enabled by ``SOLAR_FSM=1`` with ``SOLAR_FSM_TOKENIZER_DIR`` pointing at the
served tokenizer; a resolution failure fails loud at first use rather than
masking with wrong ids. Optional tunables: ``SOLAR_REASONING_BUDGET_LOW`` ..
``_MAX`` (per-effort budgets), ``SOLAR_REASONING_BUDGET_HARD_LIMIT`` (0 disables
the ceiling), ``SOLAR_REASONING_BUDGET_DEFAULT_EFFORT``,
``SOLAR_REASONING_BUDGET_NO_REASONING_EFFORTS`` (comma list),
``SOLAR_OPEN2_EOS_TOKEN_IDS`` and ``SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS``
(JSON lists of ints overriding the tokenizer-derived sets; the sentinel ids
themselves always come from the tokenizer), and
``SOLAR_FSM_SPEC_ALWAYS_EAGER=1`` (speculative decoding: mask every verify step
instead of only the steps the folded path cannot cover).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import msgspec
import torch

logger = logging.getLogger(__name__)

NEG_INF = float("-inf")

# Control sentinels by field name, resolved from the tokenizer's added tokens.
# Every one present in the served tokenizer is a member of ``all_controls``.
_TOKEN_TEXT_BY_FIELD = {
    "im_start": "<|im:start|>",
    "im_end": "<|im:end|>",
    "think_start": "<|think:start|>",
    "think_end": "<|think:end|>",
    "im_content": "<|im:content|>",
    "tool_start": "<|tool:start|>",
    "tool_end": "<|tool:end|>",
    "tool_call_start": "<|tool_call:start|>",
    "tool_call_end": "<|tool_call:end|>",
    "tool_arg_start": "<|tool_arg:start|>",
    "tool_arg_value": "<|tool_arg:value|>",
    "tool_arg_end": "<|tool_arg:end|>",
    "tool_response_start": "<|tool_response:start|>",
    "tool_response_end": "<|tool_response:end|>",
}
_SENTINEL_TOKENS = tuple(_TOKEN_TEXT_BY_FIELD.values())

# FSM states, as small ints (they live in __slots__ and are compared on every
# decode step).
REASONING = 0
CONTENT = 1
TOOL_CALL_BEGIN = 2
TOOL_CALL_NAME = 3
TOOL_ARG_BEGIN = 4
TOOL_ARG_NAME = 5
TOOL_ARG_VALUE_BEGIN = 6
TOOL_ARG_VALUE = 7
TOOL_ARG_END = 8
TOOL_CALL_END = 9
_STATE_NAMES = (
    "REASONING",
    "CONTENT",
    "TOOL_CALL_BEGIN",
    "TOOL_CALL_NAME",
    "TOOL_ARG_BEGIN",
    "TOOL_ARG_NAME",
    "TOOL_ARG_VALUE_BEGIN",
    "TOOL_ARG_VALUE",
    "TOOL_ARG_END",
    "TOOL_CALL_END",
)

# Per-state mask specification: (allowed sentinel fields, eos_masked). Bare
# EOS is forbidden wherever ending the turn is template-illegal -- everywhere
# except CONTENT-with-progress and TOOL_CALL_END. CONTENT is keyed by the
# content-progress flag instead.
_MASK_SPEC_BY_STATE = {
    REASONING: (("think_end",), True),
    TOOL_CALL_BEGIN: ((), True),
    TOOL_CALL_NAME: (("tool_arg_start", "tool_call_end"), True),
    TOOL_ARG_BEGIN: ((), True),
    TOOL_ARG_NAME: (("tool_arg_value",), True),
    TOOL_ARG_VALUE_BEGIN: (("tool_arg_end",), True),
    TOOL_ARG_VALUE: (("tool_arg_end",), True),
    TOOL_ARG_END: (("tool_arg_start", "tool_call_end"), True),
    TOOL_CALL_END: (("tool_call_start", "im_end"), False),
}
_MASK_SPEC_CONTENT = {
    # content_progress=True: the turn may legally end -> EOS stays available.
    True: (("tool_call_start", "im_end"), False),
    # Fresh CONTENT (no content yet): turn end (im_end + bare EOS) forbidden.
    False: (("tool_call_start",), True),
}
# Sentinel -> next state, and the states a non-transition token advances by
# itself.
_TRANSITIONS = (
    ("think_start", REASONING),
    ("think_end", CONTENT),
    ("tool_call_start", TOOL_CALL_BEGIN),
    ("tool_call_end", TOOL_CALL_END),
    ("tool_arg_start", TOOL_ARG_BEGIN),
    ("tool_arg_value", TOOL_ARG_VALUE_BEGIN),
    ("tool_arg_end", TOOL_ARG_END),
)
_AUTO_ADVANCE = {
    TOOL_CALL_BEGIN: TOOL_CALL_NAME,
    TOOL_ARG_BEGIN: TOOL_ARG_NAME,
    TOOL_ARG_VALUE_BEGIN: TOOL_ARG_VALUE,
    TOOL_CALL_END: CONTENT,
}

# Reasoning budget per effort (tokens).
_EFFORT_BUDGETS: Dict[str, int] = {
    "low": 4 * 1024,
    "medium": 16 * 1024,
    "high": 32 * 1024,
    "xhigh": 64 * 1024,
    "max": 128 * 1024,
}
_NO_REASONING_EFFORTS = frozenset({"none", "minimal"})
_DEFAULT_EFFORT = "high"
_HARD_LIMIT = 128 * 1024
_NO_HARD_LIMIT = 1 << 62  # SOLAR_REASONING_BUDGET_HARD_LIMIT=0: no ceiling
# Leading-newline rule after <|think:start|>: ids resolved by
# _leading_newline_ids.
_NEWLINE_BYTELEVEL = "Ċ"  # byte-level "\n"
# custom_params key the chat entrypoint uses to hand the request's reasoning
# effort to the scheduler-side FSM (solar_open2_serving.normalize_reasoning_effort).
EFFORT_PARAM = "solar_reasoning_effort"
# custom_params key: whether the request offers tools (chat entrypoint). A
# request without tools can never have a <|tool_call:start|> answered, so the
# no-tools table forbids the token in every state. It matters in fresh
# CONTENT: with EOS shut there, the model otherwise takes the token as the
# exit ("<|tool_call:start|>think", then stop).
TOOLS_PARAM = "solar_tools_available"


class _Config:
    """Resolved once per process."""

    enabled: bool = False
    im_start: Optional[int] = None
    im_end: Optional[int] = None
    think_start: Optional[int] = None
    think_end: Optional[int] = None
    im_content: Optional[int] = None
    tool_start: Optional[int] = None
    tool_end: Optional[int] = None
    tool_call_start: Optional[int] = None
    tool_call_end: Optional[int] = None
    tool_arg_start: Optional[int] = None
    tool_arg_value: Optional[int] = None
    tool_arg_end: Optional[int] = None
    tool_response_start: Optional[int] = None
    tool_response_end: Optional[int] = None
    all_controls: frozenset = frozenset()
    # sentinel id -> next state (_TRANSITIONS resolved against the tokenizer).
    transitions: Dict[int, int] = {}
    # (state, content_progress) -> forbidden ids: the control sentinels the
    # state does not allow, plus EOS where ending the turn is illegal. The
    # no-tools variant forbids <|tool_call:start|> in every state as well.
    forbidden: Dict[Tuple[int, bool], Tuple[int, ...]] = {}
    forbidden_notools: Dict[Tuple[int, bool], Tuple[int, ...]] = {}
    # Named views into the tables. reasoning_forbidden / reasoning_open_forbidden
    # are the live REASONING lookups; the content_* four serve logging and the
    # tests:
    #   REASONING            : allow think_end only, EOS masked
    #   CONTENT (no content) : allow tool_call_start only, EOS masked
    #                          <- this is what stops the model from emitting EOS
    #                             the instant it leaves reasoning (empty answer)
    #   CONTENT (has content): allow tool_call_start + im_end, EOS free
    reasoning_forbidden: Tuple[int, ...] = ()
    # reasoning_forbidden plus the leading-newline ids: the set for the one
    # token that directly follows <|think:start|>.
    reasoning_open_forbidden: Tuple[int, ...] = ()
    leading_newline_forbidden: Tuple[int, ...] = ()
    content_fresh_forbidden: Tuple[int, ...] = ()
    content_done_forbidden: Tuple[int, ...] = ()
    # The same two sets for a request that offers no tools: tool_call_start
    # joins them.
    content_fresh_forbidden_notools: Tuple[int, ...] = ()
    content_done_forbidden_notools: Tuple[int, ...] = ()
    # Reasoning budget: a fixed number of tokens per effort, clamped by hard_limit.
    effort_budgets: Dict[str, int] = dict(_EFFORT_BUDGETS)
    default_effort: str = _DEFAULT_EFFORT
    hard_limit: int = _HARD_LIMIT
    no_reasoning_efforts: frozenset = _NO_REASONING_EFFORTS
    # Speculative decoding: masking the verify logits costs the folded in-graph
    # accept, so by default only the steps the folded path cannot cover leave
    # it (plan_gate). True masks every verify step (fuller enforcement, slower).
    spec_always_eager: bool = False
    _mask_cache: Dict[Tuple[Tuple[int, ...], str], torch.Tensor] = {}


CFG = _Config()


def _added_token_ids(tokenizer_dir: str) -> Dict[str, int]:
    """text -> id for added/special tokens, straight from tokenizer files."""
    out: Dict[str, int] = {}
    tc_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
    if os.path.isfile(tc_path):
        tc = json.load(open(tc_path, encoding="utf-8"))
        for tid, spec in (tc.get("added_tokens_decoder") or {}).items():
            content = spec.get("content") if isinstance(spec, dict) else None
            if content:
                out[content] = int(tid)
    if not out:
        tk_path = os.path.join(tokenizer_dir, "tokenizer.json")
        if os.path.isfile(tk_path):
            tk = json.load(open(tk_path, encoding="utf-8"))
            for entry in tk.get("added_tokens") or []:
                out[entry["content"]] = int(entry["id"])
    return out


def _eos_ids(tokenizer_dir: str) -> List[int]:
    """Turn-terminating ids, masked where ending the turn is illegal: the union
    of ``config.json`` / ``generation_config.json`` ``eos_token_id`` and the
    tokenizer's ``eos_token`` text; ``SOLAR_OPEN2_EOS_TOKEN_IDS`` (JSON list of
    ints) overrides everything. Sentinels are removed by ``configure_ids`` -- they
    stay state-managed."""
    override = _env_id_list("SOLAR_OPEN2_EOS_TOKEN_IDS")
    if override is not None:
        return sorted(override)
    ids = set()
    for name in ("config.json", "generation_config.json"):
        path = os.path.join(tokenizer_dir, name)
        if os.path.isfile(path):
            raw = json.load(open(path, encoding="utf-8")).get("eos_token_id")
            if isinstance(raw, int):
                ids.add(raw)
            elif isinstance(raw, (list, tuple)):
                ids.update(int(x) for x in raw)
    tc_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
    if os.path.isfile(tc_path):
        eos_token = json.load(open(tc_path, encoding="utf-8")).get("eos_token")
        if isinstance(eos_token, dict):
            eos_token = eos_token.get("content")
        if isinstance(eos_token, str):
            tid = _added_token_ids(tokenizer_dir).get(eos_token)
            if tid is not None:
                ids.add(tid)
    if not ids:
        logger.warning(
            "[SOLAR-FSM] no EOS id found in %s (config.json / generation_config.json "
            "eos_token_id, tokenizer eos_token); only the sentinels will be masked "
            "-- set SOLAR_OPEN2_EOS_TOKEN_IDS if the model has a bare EOS",
            tokenizer_dir,
        )
    return sorted(ids)


_RETIRED_ENV = {
    "SOLAR_FSM_EOS_IDS": "SOLAR_OPEN2_EOS_TOKEN_IDS",
    "SOLAR_FSM_LEADING_NEWLINE_IDS": "SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS",
    "SOLAR_FSM_HARD_LIMIT": "SOLAR_REASONING_BUDGET_HARD_LIMIT",
    "SOLAR_FSM_DEFAULT_EFFORT": "SOLAR_REASONING_BUDGET_DEFAULT_EFFORT",
    "SOLAR_FSM_BUDGET_LOW": "SOLAR_REASONING_BUDGET_LOW",
    "SOLAR_FSM_BUDGET_MEDIUM": "SOLAR_REASONING_BUDGET_MEDIUM",
    "SOLAR_FSM_BUDGET_HIGH": "SOLAR_REASONING_BUDGET_HIGH",
    "SOLAR_FSM_BUDGET_XHIGH": "SOLAR_REASONING_BUDGET_XHIGH",
    "SOLAR_FSM_BUDGET_MAX": "SOLAR_REASONING_BUDGET_MAX",
    "SOLAR_FSM_CONTENT_MASK": None,
    "SOLAR_FSM_BUDGET_POLICY": None,
    "SOLAR_FSM_BUDGET_RATIO": None,
    "SOLAR_FSM_BUDGET_ABS": None,
}


def _warn_retired_env() -> None:
    """Names an earlier version read and this one does not: renamed knobs point
    at their replacement; removed ones (the content-mask switch, the
    max_tokens-relative budget) are ignored -- the envelope is always enforced
    and the budget always follows the effort table."""
    for name, replacement in _RETIRED_ENV.items():
        if name in os.environ:
            logger.warning(
                "[SOLAR-FSM] %s is not read any more%s",
                name,
                f"; use {replacement}" if replacement else " (removed)",
            )


def _env_id_list(name: str) -> Optional[set]:
    """A JSON list of non-negative token ids from ``name``, or None when the
    variable is unset or blank. Malformed values fail loud."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} must be a JSON list of ints, got {raw!r}") from exc
    if not isinstance(data, list) or not all(
        isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in data
    ):
        raise RuntimeError(
            f"{name} must be a JSON list of non-negative ints, got {raw!r}"
        )
    return set(data)


def _env_int(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def _leading_newline_ids(tokenizer_dir: str) -> Tuple[int, ...]:
    """Ids forbidden as the first token after ``<|think:start|>``.

    Every token of ``tokenizer.json``'s vocab and of the added tokens
    (``tokenizer_config.json`` / ``tokenizer.json``) whose text is a pure
    newline run, plus a verbatim newline token;
    ``SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS`` (JSON list of ints) overrides,
    ``[]`` switches the rule off, and a blank value is the same as unset. Like
    the think ids, a vocab that cannot supply them fails loud rather than
    silently dropping the rule.
    """
    override = _env_id_list("SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS")
    if override is not None:
        return tuple(sorted(override))
    tk_path = os.path.join(tokenizer_dir, "tokenizer.json")
    vocab = {}
    if os.path.isfile(tk_path):
        with open(tk_path, encoding="utf-8") as f:
            model = json.load(f).get("model") or {}
        vocab = model.get("vocab") if isinstance(model.get("vocab"), dict) else {}
    table = dict(vocab)
    table.update(_added_token_ids(tokenizer_dir))
    ids = {
        int(tid)
        for text, tid in table.items()
        if text and set(text) == {_NEWLINE_BYTELEVEL}
    }
    ids |= {int(table[t]) for t in ("\n", "\n\n") if t in table}
    if not ids:
        raise RuntimeError(
            f"SOLAR_FSM: leading-newline tokens ('Ċ', 'ĊĊ') not found "
            f"in {tokenizer_dir}'s vocab or added tokens; set "
            "SOLAR_OPEN2_THINK_LEADING_FORBIDDEN_IDS to a JSON list of the ids, or "
            "to [] to switch the rule off"
        )
    return tuple(sorted(ids))


_LOG_INTERVAL = 60.0
_EFFORT_LOG = {"last": -_LOG_INTERVAL, "num_suppressed": 0}


def _rate_limited(state: Dict[str, Any], *, count: int = 1) -> Optional[int]:
    """One report per ``_LOG_INTERVAL``: the number suppressed since the last
    report, or None while suppressing (``count`` more suppressed)."""
    now = time.monotonic()
    if now - state["last"] < _LOG_INTERVAL:
        state["num_suppressed"] += count
        return None
    suppressed = state["num_suppressed"]
    state["last"], state["num_suppressed"] = now, 0
    return suppressed


def _warn_unknown_effort(effort) -> None:
    """Rate-limited (one line per minute, with the suppressed count): a fleet
    of clients sending a bad value should stay visible for the life of the
    server, unlike the once-only wiring-defect warnings."""
    suppressed = _rate_limited(_EFFORT_LOG)
    if suppressed is None:
        return
    logger.warning(
        "[SOLAR-FSM] unknown reasoning effort %r; using the default effort %r "
        "budget. %d earlier occurrence(s) suppressed.",
        effort,
        CFG.default_effort,
        suppressed,
    )


def _budget_for(effort) -> int:
    """Reasoning budget (tokens) for one request. ``effort`` is the value the
    entrypoint attached (a string), or None for the default."""
    if effort is None:
        key = CFG.default_effort
    elif isinstance(effort, str):
        key = effort.strip().lower()
    else:
        key = None
    if key in CFG.no_reasoning_efforts:
        return 0
    budget = CFG.effort_budgets.get(key) if key is not None else None
    if budget is None:
        _warn_unknown_effort(effort)
        budget = (
            0
            if CFG.default_effort in CFG.no_reasoning_efforts
            else (CFG.effort_budgets[CFG.default_effort])
        )
    return min(budget, CFG.hard_limit)


def _req_effort(req):
    """The reasoning effort the chat entrypoint attached to the request
    (``EFFORT_PARAM`` in ``sampling_params.custom_params``), or None."""
    params = getattr(getattr(req, "sampling_params", None), "custom_params", None)
    if isinstance(params, dict):
        return params.get(EFFORT_PARAM)
    return None


def _req_tools(req) -> bool:
    """Whether the request offers tools (``TOOLS_PARAM``); True when the
    entrypoint did not say (the tables that allow a tool call)."""
    params = getattr(getattr(req, "sampling_params", None), "custom_params", None)
    if isinstance(params, dict) and TOOLS_PARAM in params:
        value = params[TOOLS_PARAM]
        if isinstance(value, bool):
            return value
        if not _WARNED["tools"]:
            logger.warning(
                "[SOLAR-FSM] custom_params[%r] is %r, not a bool; treating the "
                "request as offering tools. This warning is logged once.",
                TOOLS_PARAM,
                value,
            )
            _WARNED["tools"] = True
    return True


def _forbidden_for(
    state: int, *, content_progress: bool, tools: bool
) -> Tuple[int, ...]:
    """The forbidden set for a non-REASONING step: the
    (state, content_progress) entry of the forbidden table, or of the
    no-tools variant when the request offers no tools."""
    table = CFG.forbidden if tools else CFG.forbidden_notools
    return table.get((state, bool(content_progress)), ())


def _reasoning_forbidden(state) -> Tuple[int, ...]:
    """REASONING forbid set for ``state`` (a ``SolarReqFSM`` or ``_SimState``):
    the leading-newline ids join it on the token right after ``<|think:start|>``."""
    return (
        CFG.reasoning_open_forbidden if state.at_think_open else CFG.reasoning_forbidden
    )


def configure_ids(
    ids: Dict[str, Optional[int]],
    eos: Sequence[int],
    leading_newline: Sequence[int] = (),
) -> None:
    """Resolve the sentinel ids and build every derived table from them.

    ``ids`` maps the field names of ``_TOKEN_TEXT_BY_FIELD`` to token ids,
    ``None`` for a sentinel the tokenizer does not carry. Per (state,
    content_progress) the forbidden set holds the control sentinels the state
    does not allow, plus the bare EOS ids wherever ending the turn is
    template-illegal. The unit tests configure the module through this as
    well, so a test and the server can never disagree on how a set is
    derived from the spec.
    """
    for field in _TOKEN_TEXT_BY_FIELD:
        setattr(CFG, field, ids.get(field))
    if CFG.think_start is None or CFG.think_end is None:
        raise RuntimeError("SOLAR_FSM: think_start / think_end ids are required")
    sentinels = {i for i in (ids.get(f) for f in _TOKEN_TEXT_BY_FIELD) if i is not None}
    CFG.all_controls = frozenset(sentinels)
    transitions: Dict[int, int] = {}
    for field, state in _TRANSITIONS:
        tok = ids.get(field)
        if tok is not None:
            transitions.setdefault(tok, state)
    CFG.transitions = transitions
    eos_set = set(eos) - sentinels

    def build(allowed_fields, *, mask_eos, block_tool_call_start):
        allowed = {ids.get(f) for f in allowed_fields} - {None}
        f = sentinels - allowed
        if block_tool_call_start and CFG.tool_call_start is not None:
            f.add(CFG.tool_call_start)
        if mask_eos:
            f |= eos_set
        return tuple(sorted(f))

    for block, attr in ((False, "forbidden"), (True, "forbidden_notools")):
        table: Dict[Tuple[int, bool], Tuple[int, ...]] = {}
        for state, (allowed, mask_eos) in _MASK_SPEC_BY_STATE.items():
            table[(state, False)] = table[(state, True)] = build(
                allowed, mask_eos=mask_eos, block_tool_call_start=block
            )
        for progress, (allowed, mask_eos) in _MASK_SPEC_CONTENT.items():
            table[(CONTENT, progress)] = build(
                allowed, mask_eos=mask_eos, block_tool_call_start=block
            )
        setattr(CFG, attr, table)
    CFG.reasoning_forbidden = CFG.forbidden[(REASONING, False)]
    CFG.content_fresh_forbidden = CFG.forbidden[(CONTENT, False)]
    CFG.content_done_forbidden = CFG.forbidden[(CONTENT, True)]
    CFG.content_fresh_forbidden_notools = CFG.forbidden_notools[(CONTENT, False)]
    CFG.content_done_forbidden_notools = CFG.forbidden_notools[(CONTENT, True)]
    CFG.leading_newline_forbidden = tuple(sorted(set(leading_newline)))
    CFG.reasoning_open_forbidden = tuple(
        sorted(set(CFG.reasoning_forbidden) | set(CFG.leading_newline_forbidden))
    )
    CFG._mask_cache.clear()


def init_from_env() -> None:
    """Resolve ids + budget. Called by ``is_active`` once SOLAR_FSM=1 is seen."""
    tok_dir = os.environ.get("SOLAR_FSM_TOKENIZER_DIR", "")
    if not tok_dir or not os.path.isdir(tok_dir):
        raise RuntimeError(
            "SOLAR_FSM=1 but SOLAR_FSM_TOKENIZER_DIR is unset or not a directory: "
            f"{tok_dir!r}"
        )
    _configure_from_tokenizer(tok_dir)
    _configure_budget_from_env()
    CFG.spec_always_eager = os.environ.get("SOLAR_FSM_SPEC_ALWAYS_EAGER", "0") == "1"
    _warn_retired_env()
    CFG.enabled = True
    logger.info(
        "[SOLAR-FSM] enabled: think_start=%s think_end=%s im_end=%s | "
        "forbidden reasoning=%s reasoning_open=%s content_fresh=%s "
        "content_done=%s (no tools: %s / %s) tool states=%s | "
        "budget per effort %s default=%s hard_limit=%s",
        CFG.think_start,
        CFG.think_end,
        CFG.im_end,
        CFG.reasoning_forbidden,
        CFG.reasoning_open_forbidden,
        CFG.content_fresh_forbidden,
        CFG.content_done_forbidden,
        CFG.content_fresh_forbidden_notools,
        CFG.content_done_forbidden_notools,
        {
            _STATE_NAMES[st]: CFG.forbidden[(st, False)]
            for st in range(TOOL_CALL_BEGIN, TOOL_CALL_END + 1)
        },
        CFG.effort_budgets,
        CFG.default_effort,
        "off" if CFG.hard_limit >= _NO_HARD_LIMIT else CFG.hard_limit,
    )


def _configure_from_tokenizer(tok_dir: str) -> None:
    """Sentinel, EOS and leading-newline ids from the served tokenizer."""
    table = _added_token_ids(tok_dir)
    missing = [t for t in ("<|think:start|>", "<|think:end|>") if t not in table]
    if missing:
        raise RuntimeError(
            f"SOLAR_FSM: required special tokens not found in {tok_dir}: {missing}"
        )
    configure_ids(
        {field: table.get(text) for field, text in _TOKEN_TEXT_BY_FIELD.items()},
        eos=_eos_ids(tok_dir),
        leading_newline=_leading_newline_ids(tok_dir),
    )


def _configure_budget_from_env() -> None:
    """Hard limit, per-effort budgets, no-reasoning efforts and the default
    effort from ``SOLAR_REASONING_BUDGET_*``."""
    # A hard limit of 0 disables the server-wide ceiling; a negative value is a
    # configuration error.
    CFG.hard_limit = _env_int("SOLAR_REASONING_BUDGET_HARD_LIMIT", default=_HARD_LIMIT)
    if CFG.hard_limit < 0:
        raise RuntimeError(
            "SOLAR_REASONING_BUDGET_HARD_LIMIT must be >= 0 (0 disables), got "
            f"{CFG.hard_limit}"
        )
    if CFG.hard_limit == 0:
        CFG.hard_limit = _NO_HARD_LIMIT
    budgets = {}
    for effort, default in _EFFORT_BUDGETS.items():
        name = f"SOLAR_REASONING_BUDGET_{effort.upper()}"
        value = _env_int(name, default=default)
        if value < 0:
            # A negative per-effort budget falls back to the built-in default
            # with a warning rather than failing every request; a non-integer
            # fails loud in _env_int; a blank value takes the default.
            logger.warning(
                "[SOLAR-FSM] ignoring %s=%d (must be >= 0); using the built-in "
                "%s budget %d.",
                name,
                value,
                effort,
                default,
            )
            value = default
        if value > CFG.hard_limit:
            if name in os.environ:
                logger.warning(
                    "[SOLAR-FSM] %s=%d exceeds SOLAR_REASONING_BUDGET_HARD_LIMIT=%d; "
                    "clamped.",
                    name,
                    value,
                    CFG.hard_limit,
                )
            value = CFG.hard_limit
        budgets[effort] = value
    CFG.effort_budgets = budgets
    raw = os.environ.get("SOLAR_REASONING_BUDGET_NO_REASONING_EFFORTS")
    if raw is not None:
        CFG.no_reasoning_efforts = frozenset(
            e.strip().lower() for e in raw.split(",") if e.strip()
        )
    CFG.default_effort = (
        os.environ.get("SOLAR_REASONING_BUDGET_DEFAULT_EFFORT", _DEFAULT_EFFORT)
        .strip()
        .lower()
    )
    if (
        CFG.default_effort not in CFG.effort_budgets
        and CFG.default_effort not in CFG.no_reasoning_efforts
    ):
        raise RuntimeError(
            f"SOLAR_REASONING_BUDGET_DEFAULT_EFFORT={CFG.default_effort!r} is not "
            f"a known reasoning effort: {sorted(CFG.effort_budgets)} / "
            f"{sorted(CFG.no_reasoning_efforts)}"
        )


def is_active() -> bool:
    """Whether the FSM is on. Resolves config lazily on first use."""
    if CFG.enabled:
        return True
    if os.environ.get("SOLAR_FSM", "0") != "1":
        return False
    init_from_env()
    return CFG.enabled


class SolarReqFSM:
    """Per-request walk of the ten-state envelope machine. Lives on the Req,
    so a row permutation can never hand one request another request's
    state."""

    __slots__ = (
        "state",
        "at_think_open",
        "count",
        "consumed",
        "budget",
        "forced",
        "content_progress",
        "tools",
    )

    def __init__(
        self,
        prompt_ids: Sequence[int],
        *,
        effort: Optional[str] = None,
        tools: bool = True,
    ):
        # Initial state: inside reasoning iff the last think_start in the
        # prompt is not followed by a think_end.
        self.state = REASONING if _starts_in_reasoning(prompt_ids) else CONTENT
        # True while the next token is the first one after <|think:start|>
        # (the chat template ends the prompt with it when reasoning is on).
        self.at_think_open = bool(
            self.state == REASONING
            and len(prompt_ids) > 0
            and prompt_ids[-1] == CFG.think_start
        )
        # Completed-call seed: a tool call the prompt's *current* assistant
        # turn already completed counts as content.
        self.content_progress = _prompt_completed_tool_call(prompt_ids)
        self.count = 0
        self.consumed = 0
        self.budget = _budget_for(effort)
        self.tools = bool(tools)
        self.forced = False

    def _step(self, tok: int) -> None:
        """Per-token transition. Reused verbatim as ``_SimState.step``, which
        walks a throwaway copy of this state over a speculative chain, so the
        two can never drift. Touches only ``state`` / ``at_think_open`` /
        ``count`` / ``content_progress`` -- the slots ``_SimState`` also
        carries."""
        state = self.state
        if tok not in CFG.all_controls:
            # Ordinary token: state-implied bookkeeping only.
            if state == REASONING:
                self.count += 1
            elif state == CONTENT:
                self.content_progress = True
            else:
                nxt = _AUTO_ADVANCE.get(state)
                if nxt is not None:
                    self.state = nxt
            self.at_think_open = False
            return
        nxt = CFG.transitions.get(tok)
        if nxt is not None:
            self.state = nxt
        else:
            nxt = _AUTO_ADVANCE.get(state)
            if nxt is not None:
                self.state = nxt
        # Reasoning-budget accounting: <|think:start|> (re)opens the block and
        # resets the counter; a sentinel that keeps the FSM in REASONING counts.
        if tok == CFG.think_start:
            self.count = 0
        elif state == REASONING and self.state == REASONING:
            self.count += 1
        if tok == CFG.tool_call_end:
            # A completed tool call counts as turn content.
            self.content_progress = True
        self.at_think_open = tok == CFG.think_start

    @property
    def in_reasoning(self) -> bool:
        return self.state == REASONING

    def advance(self, output_ids: Sequence[int]) -> None:
        """Catch up to ``req.output_ids``. A no-op when ``commit()`` has already
        fed this run: ``consumed`` counts tokens and ``req.output_ids`` only ever
        grows (a retraction re-prefills the tokens it already holds), so
        ``len(output_ids) <= self.consumed`` means the FSM is level with or ahead
        of it, never that it has to rewind.
        """
        n = len(output_ids)
        if n <= self.consumed:
            return
        for i in range(self.consumed, n):
            self._step(output_ids[i])
        self.consumed = n

    def commit(self, tokens: Sequence[int]) -> None:
        """Step over a run of tokens this request has just committed.

        The scheduler's commit path feeds the run here as soon as the verify
        result lands, which is one scheduler step before ``req.output_ids``
        grows by the same tokens. ``consumed`` counts tokens, not
        ``output_ids`` entries, so the later ``advance(req.output_ids)`` sees
        nothing new and the run is stepped over exactly once.
        """
        for tok in tokens:
            self._step(tok)
        self.consumed += len(tokens)

    def budget_exhausted(self) -> bool:
        return self.in_reasoning and self.count >= self.budget


def _has_grammar(req) -> bool:
    """Whether structured outputs (a grammar) constrain this request.

    A grammar owns the CONTENT phase: this FSM masks nothing there and keeps
    masking the tool states. That holds because a grammar never produces a
    sentinel -- a required / named tool call is a JSON-schema grammar (a JSON
    array of calls, no ``<|tool_call:start|>``), so the FSM never enters a
    tool state under one. ``SolarOpen2Detector.supports_structural_tag`` is
    False for the same reason: a structural tag constrains *text*, while this
    FSM tracks sentinel *ids*, so the two can disagree (opener sampled as the
    sentinel id, closer spelled out as text), leaving the FSM inside
    TOOL_CALL_NAME with ``<|im:end|>`` forbidden.

    Judged from the request's own constraint (json_schema / regex / ebnf /
    structural_tag), not from ``req.grammar``: ``--enable-strict-thinking``
    hangs a server-side reasoning grammar with no inner grammar on every
    reasoning request, and that must not switch the CONTENT rules off.
    """
    sp = getattr(req, "sampling_params", None)
    return sp is not None and any(
        getattr(sp, name, None) is not None
        for name in ("json_schema", "regex", "ebnf", "structural_tag")
    )


def _fsm_stale(req) -> bool:
    """True when the FSM hung off ``req`` predates the request's last retraction.

    A retraction throws away whatever the in-flight commit already fed the FSM:
    the scheduler drops the pending run instead of appending it to
    ``req.output_ids`` (see ``process_batch_result_decode``'s
    ``req.finished() or req.is_retracted`` skip under overlap). That would leave
    ``consumed`` permanently ahead of ``len(req.output_ids)``, and ``advance()``
    only ever moves forward -- so it would silently skip that many real tokens,
    including a ``<|think:end|>``. Rebuild instead and let ``advance()`` replay
    ``req.output_ids``, which a retraction keeps intact.
    """
    return getattr(req, "_solar_fsm_retractions", 0) != req.retraction_count


def _req_fsm(req) -> SolarReqFSM:
    """Get (or build) the persistent FSM hung off a request."""
    fsm = getattr(req, "_solar_fsm", None)
    if fsm is None or _fsm_stale(req):
        fsm = SolarReqFSM(
            getattr(req, "origin_input_ids", ()) or (),
            effort=_req_effort(req),
            tools=_req_tools(req),
        )
        req._solar_fsm = fsm
        req._solar_fsm_retractions = req.retraction_count
    return fsm


def _rindex(values: Sequence[int], *, needle: int) -> Optional[int]:
    for i in range(len(values) - 1, -1, -1):
        if values[i] == needle:
            return i
    return None


def _starts_in_reasoning(prompt_ids: Sequence[int]) -> bool:
    if CFG.think_start is None or CFG.think_end is None:
        return False
    last_start = _rindex(prompt_ids, needle=CFG.think_start)
    if last_start is None:
        return False
    last_end = _rindex(prompt_ids, needle=CFG.think_end)
    return last_end is None or last_start > last_end


def _prompt_completed_tool_call(prompt_ids: Sequence[int]) -> bool:
    """Whether the prompt's *current* assistant turn has a completed tool call
    (a ``<|tool_call:end|>`` after the last ``<|im:start|>``). Earlier turns'
    calls are always followed by a newer message start, so they never seed
    the flag; with no ``<|im:start|>`` id the boundary is unknown and the flag
    conservatively stays False."""
    if CFG.tool_call_end is None or CFG.im_start is None:
        return False
    last_call_end = _rindex(prompt_ids, needle=CFG.tool_call_end)
    if last_call_end is None:
        return False
    last_im_start = _rindex(prompt_ids, needle=CFG.im_start)
    return last_im_start is not None and last_im_start < last_call_end


def _mask_tensor(ids: Tuple[int, ...], *, device: torch.device) -> torch.Tensor:
    key = (ids, str(device))
    t = CFG._mask_cache.get(key)
    if t is None:
        t = torch.tensor(ids, dtype=torch.long, device=device)
        CFG._mask_cache[key] = t
    return t


def _mask_and_force(
    logits: torch.Tensor,
    *,
    mask_rows: Dict[Tuple[int, ...], List[int]],
    force_rows: List[int],
) -> Tuple[List[int], List[int]]:
    """Write ``NEG_INF`` over each forbidden set in its rows, then force
    ``<|think:end|>`` on ``force_rows``. A row the grammar has already closed
    to ``<|think:end|>`` would become fully -inf if forced -- the sampler then
    returns an arbitrary id that the grammar rejects on accept -- so it is left
    to the grammar. Returns ``(forced rows, rows left to the grammar)``."""
    for ids, rows_i in mask_rows.items():
        if not ids or not rows_i:
            continue
        idx = _mask_tensor(ids, device=logits.device)
        rsel = torch.tensor(rows_i, dtype=torch.long, device=logits.device)
        logits[rsel.unsqueeze(1), idx.unsqueeze(0)] = NEG_INF
    if not force_rows:
        return [], []
    rsel = torch.tensor(force_rows, dtype=torch.long, device=logits.device)
    allowed = torch.isfinite(logits[rsel, CFG.think_end]).tolist()
    forced = [r for r, ok in zip(force_rows, allowed) if ok]
    blocked = [r for r, ok in zip(force_rows, allowed) if not ok]
    if forced:
        rsel = torch.tensor(forced, dtype=torch.long, device=logits.device)
        keep = logits[rsel, CFG.think_end].clone()
        logits[rsel, :] = NEG_INF
        logits[rsel, CFG.think_end] = keep
    return forced, blocked


_CONFLICT_LOG = {"last": -_LOG_INTERVAL, "num_suppressed": 0}


def _log_force_conflict(rows, *, stride: int, rids) -> None:
    """Report rows where the FSM wanted to force <|think:end|> but something
    had already forbidden it, so forcing would leave the row fully masked.

    Two sources, and the message names both because they are not
    distinguishable here: a grammar reading different committed state, and the
    in-graph CONTENT mask, which forbids ``<|think:end|>`` on a row committed
    in CONTENT-with-progress. The second reaches this only for a zero-budget
    request (effort ``none``/``minimal``) whose chain drafts a
    ``<|think:start|>``: ``plan_verify`` then wants the budget force on a row
    the content mask has already shut. Rate-limited like
    ``_warn_unknown_effort``."""
    num_suppressed = _rate_limited(_CONFLICT_LOG, count=len(rows))
    if num_suppressed is None:
        return
    if rids is not None and stride > 0:
        named = [rids[r // stride] for r in rows if r // stride < len(rids)]
    else:
        named = []
    logger.warning(
        "[SOLAR-FSM] <|think:end|> (id=%s) is already forbidden by the grammar "
        "or by the in-graph CONTENT mask on %d row(s) %s (reqs %s); skipping "
        "the budget force there so the row keeps the set it already has. "
        "%d earlier occurrence(s) suppressed.",
        CFG.think_end,
        len(rows),
        rows,
        named,
        num_suppressed,
    )


# --------------------------------------------------------------------------
# Hooks called from the sampler, the batch-info build/filter/merge, the batch
# result processor, the scheduler, and the DSpark verify worker and graph.
# --------------------------------------------------------------------------
def attach_rows(sampling_info, batch) -> None:
    if not os.environ.get("SOLAR_FSM", "0") == "1":
        return
    sampling_info.solar_fsm_rows = list(batch.reqs)


def filter_rows(sampling_info, keep_indices: List[int]) -> None:
    rows = getattr(sampling_info, "solar_fsm_rows", None)
    if rows is not None:
        sampling_info.solar_fsm_rows = [rows[i] for i in keep_indices]


def merge_rows(sampling_info, other) -> None:
    rows = getattr(sampling_info, "solar_fsm_rows", None)
    other_rows = getattr(other, "solar_fsm_rows", None)
    if rows is None and other_rows is None:
        return
    if (rows is None) != (other_rows is None) and is_active():
        # from_schedule_batch attaches rows to every batch while the FSM is on,
        # so one side arriving without them means a construction site skipped
        # attach_rows: the merged list would be shorter than the batch and
        # apply() would then skip the whole step. Merge what is there, loudly.
        if not _WARNED["merge"]:
            logger.warning(
                "[SOLAR-FSM] merge_batch: one side has no solar_fsm_rows "
                "(self=%s other=%s); the merged rows will be short of the batch. "
                "This warning is logged once.",
                "missing" if rows is None else len(rows),
                "missing" if other_rows is None else len(other_rows),
            )
            _WARNED["merge"] = True
    sampling_info.solar_fsm_rows = list(rows or []) + list(other_rows or [])


def advance_committed(result, batch) -> None:
    """Advance each request's FSM over the tokens THIS batch committed.

    Called from the same commit path that advances the grammar FSM -- eagerly
    from the scheduler's grammar barrier inside verify(), so the FSM state a
    following verify step plans from is on the same time base as the grammar's,
    and lazily from the spec result resolver otherwise. Idempotent via
    ``result.solar_fsm_advanced``.

    The run fed here is the one that lands in ``req.output_ids``: the
    grammar-truncated run when the request has a grammar, the raw accepted run
    otherwise, and the single prefilled token for an extend result.
    """
    if result.solar_fsm_advanced:
        return
    if not is_active():
        return
    is_decode = batch.forward_mode.is_decode()
    if not (is_decode or batch.forward_mode.is_extend()):
        return
    if result.copy_done is not None:
        result.copy_done.synchronize()
    next_token_ids = result.next_token_ids.tolist()

    if not is_decode:
        # Extend: advance over the single token each completed-prefill req
        # emitted (mirrors advance_grammar_fsm's extend branch).
        for i, req in enumerate(batch.reqs):
            if req.is_retracted or req.finished() or req.inflight_middle_chunks > 0:
                continue
            _req_fsm(req).commit([next_token_ids[i]])
        result.solar_fsm_advanced = True
        return

    # Decode: non-spec decode has no accepted run here and is driven by
    # advance(req.output_ids) instead.
    if result.accept_lens is None:
        return
    accept_lens = result.accept_lens.tolist()
    stride = result.speculative_num_draft_tokens
    assert stride is not None, "spec-v2 result missing speculative_num_draft_tokens"
    for i, req in enumerate(batch.reqs):
        if req.is_retracted or req.finished():
            continue
        if req.grammar is not None and result.grammar_retained_tokens is not None:
            run = result.grammar_retained_tokens[i]
            if run is None:
                continue
        else:
            run = next_token_ids[i * stride : i * stride + accept_lens[i]]
        _req_fsm(req).commit(run)
    result.solar_fsm_advanced = True


_WARNED = {"shape": False, "rows": False, "merge": False, "tools": False}


def apply(logits: torch.Tensor, sampling_info) -> None:
    """Mask ``logits`` in place according to each row's FSM state."""
    if not is_active():
        return

    rows = getattr(sampling_info, "solar_fsm_rows", None)
    if rows is None:
        # Every SamplingBatchInfo on the sampler path comes from
        # from_schedule_batch, which attaches the rows whenever the FSM is on
        # (the one other construction site, the EAGLE draft cuda-graph runner,
        # picks draft tokens itself and never reaches the Sampler). Reaching
        # here means a copy or a new construction site lost them. Say so, once.
        if not _WARNED["rows"]:
            logger.warning(
                "[SOLAR-FSM] sampling_info carries no solar_fsm_rows while the "
                "FSM is active (logits rows=%d): think-block masks are NOT "
                "applied on this path. This warning is logged once.",
                logits.shape[0],
            )
            _WARNED["rows"] = True
        return
    if logits.shape[0] != len(rows):
        # The sampler sees one logits row per request, and the verify step of
        # speculative decoding does not come through here (it masks its own
        # strided logits via plan_verify). A mismatch -- including an empty
        # rows list under a non-empty batch -- means the rows fell out of step
        # with the batch somewhere (a filter or merge site).
        if not _WARNED["shape"]:
            logger.warning(
                "[SOLAR-FSM] skipping mask: logits rows=%d != batch rows=%d; "
                "the rows list is out of step with the batch. This warning is "
                "logged once.",
                logits.shape[0],
                len(rows),
            )
            _WARNED["shape"] = True
        return
    if not rows:
        return

    force_rows: List[int] = []
    mask_rows: Dict[Tuple[int, ...], List[int]] = {}
    for i, req in enumerate(rows):
        fsm = _req_fsm(req)
        fsm.advance(req.output_ids)
        if fsm.in_reasoning:
            if fsm.budget_exhausted():
                force_rows.append(i)
            else:
                mask_rows.setdefault(_reasoning_forbidden(fsm), []).append(i)
        elif fsm.state == CONTENT and _has_grammar(req):
            continue  # structured outputs own the CONTENT phase
        else:
            # Fresh CONTENT forbids EOS/im_end until real content exists
            # (otherwise the model leaves reasoning and stops at once with an
            # empty answer); inside a tool call only the envelope sentinel(s)
            # the sub-state expects are open; no tools -> see TOOLS_PARAM.
            mask_rows.setdefault(
                _forbidden_for(
                    fsm.state, content_progress=fsm.content_progress, tools=fsm.tools
                ),
                [],
            ).append(i)

    forced, blocked = _mask_and_force(
        logits, mask_rows=mask_rows, force_rows=force_rows
    )
    if blocked:
        _log_force_conflict(blocked, stride=1, rids=[r.rid for r in rows])
    for i in forced:
        fsm = rows[i]._solar_fsm
        if not fsm.forced:
            fsm.forced = True
            logger.info(
                "[SOLAR-FSM] reasoning budget %d exhausted -> forcing "
                "<|think:end|> (req %s)",
                fsm.budget,
                rows[i].rid,
            )


# --------------------------------------------------------------------------
# Speculative verify path (DSpark).
#
# `apply()` above is called from layers/sampler.py, which the DSpark verify
# path never reaches, so these helpers mask the *verify* logits instead.
#
# Row layout is NOT the 1-row-1-token convention of ordinary decode. The verify
# scatter kernel (dspark_verify_window.py:537-549) computes
#     i = row // stride ; w = row % stride ; valid iff w < verify_lens[i]
# so rows are (request-major, chain-minor) and rows past a request's verify_len
# are padding. `verify_ids_2d[i]` is [anchor, draft_1 .. draft_gamma]: the anchor
# is already committed, so the FSM state for row (i, w) is the committed state
# advanced over draft tokens 1..w.
#
# The persistent FSM is never advanced from draft tokens -- rejected drafts
# therefore need no rollback. Only a throwaway copy walks the chain.
#
# Two decisions split across the target-verify launch:
#   * `plan_gate` runs before the launch, off whatever committed state is
#     available then -- which can lag by one accepted run, since the grammar
#     barrier that feeds `advance_committed` hasn't run yet for this step. It
#     decides only whether the folded in-graph accept path must be left (a
#     conservative, sync-free check); it builds no mask.
#   * `plan_verify` runs after the launch, once the caller has run the grammar
#     barrier (`advance_committed` / `advance_grammar_fsm`), so the FSM state it
#     reads and the grammar bitmask built alongside it describe the same
#     committed prefix. It builds the actual per-row masks.
# --------------------------------------------------------------------------
class _SimState:
    """Throwaway FSM state used to walk a speculative chain."""

    __slots__ = ("state", "at_think_open", "count", "content_progress")

    def __init__(self, fsm: SolarReqFSM):
        self.state = fsm.state
        self.at_think_open = fsm.at_think_open
        self.count = fsm.count
        self.content_progress = fsm.content_progress

    @property
    def in_reasoning(self) -> bool:
        return self.state == REASONING

    # The persistent FSM's own transition, reused rather than mirrored: a
    # drift between the chain walk and the committed state is exactly the
    # class of bug this module exists to avoid.
    step = SolarReqFSM._step

    def exhausted(self, budget: int) -> bool:
        return self.in_reasoning and self.count >= budget


class VerifyPlan(msgspec.Struct, kw_only=True):
    """Per-(request, chain position) masks for one target-verify step."""

    force_rows: List[int]
    mask_rows: Dict[Tuple[int, ...], List[int]]
    stride: int
    bs: int
    rids: Optional[List[str]]

    def apply(self, logits: torch.Tensor, verify_lens=None) -> None:
        expect_rows = self.bs * self.stride
        if logits.shape[0] != expect_rows:
            # Fail loud: a silent shape mismatch would mask the wrong request.
            raise RuntimeError(
                f"[SOLAR-FSM] verify logits rows={logits.shape[0]} != "
                f"bs*stride={expect_rows}; row mapping would be wrong"
            )
        valid = None
        if verify_lens is not None:
            vl = verify_lens.tolist() if hasattr(verify_lens, "tolist") else verify_lens
            valid = set()
            for i, n in enumerate(vl[: self.bs]):
                for w in range(min(int(n), self.stride)):
                    valid.add(i * self.stride + w)

        keep = lambda rs: [r for r in rs if valid is None or r in valid]
        _, blocked = _mask_and_force(
            logits,
            mask_rows={ids: keep(rows_i) for ids, rows_i in self.mask_rows.items()},
            force_rows=keep(self.force_rows),
        )
        if blocked:
            _log_force_conflict(blocked, stride=self.stride, rids=self.rids)


def apply_folded_mask(logits, row_flags, forbid_ids) -> None:
    """Write ``NEG_INF`` over ``forbid_ids`` in every row ``row_flags`` marks.

    The row-wise half of the reasoning mask, split out from the epilogue that
    calls it so it can be exercised without a GPU: it is plain tensor work with
    no FSM state, and it is the piece the folded accept path depends on.

    ``logits`` is ``(bs * stride, vocab)`` in the verify step's own row order,
    request-major and chain-minor; ``row_flags`` is a bool tensor over those
    rows. Written in place. Every operand's shape is fixed once ``bs`` is, which
    is what lets these kernels sit inside a captured cuda graph and be replayed:
    only the *contents* of ``row_flags`` change between steps, so an unarmed
    step runs the same kernels and writes each selected logit back unchanged.

    ``masked_fill`` rather than ``torch.where``: it keeps ``logits``' dtype,
    where a float32 -inf operand would promote and then fail the index_put_ on a
    half-precision logits tensor.
    """
    rows = row_flags.unsqueeze(1)
    logits[:, forbid_ids] = logits[:, forbid_ids].masked_fill(rows, NEG_INF)


def _reasoning_needs_eager(fsm, *, window: int) -> bool:
    """Inside the think block: the leading-newline set on the step after
    ``<|think:start|>`` and the forced ``<|think:end|>`` at a spent budget
    (also every step of a zero-budget none/minimal block) are plan_verify's
    alone. Judged only while inside the block -- a spent or zero budget must
    not keep a request that has left reasoning eager for the rest of its
    life."""
    return fsm.at_think_open or fsm.count + window >= fsm.budget


def _content_needs_eager(fsm, *, req) -> bool:
    """Outside the think block: fresh CONTENT (no content token yet) and every
    step inside a tool call take their EOS/``<|im:end|>``-shutting sets from
    plan_verify; the folded mask does not carry them. A CONTENT row under a
    grammar is exempt: the grammar owns CONTENT."""
    if fsm.state != CONTENT:
        return True
    return not fsm.content_progress and not _has_grammar(req)


def folded_content_mask_flags(reqs, stride: int) -> Optional[List[bool]]:
    """Per-(request, chain position) flags for the in-graph CONTENT mask.

    True where the request's **committed** state is CONTENT with content
    already produced and no grammar -- exactly the rows :func:`plan_gate`
    declines to send eager, and therefore the rows whose forbidden set nothing
    else applies. Without this the second bullet of
    :func:`folded_mask_flags` is open: a drafted ``<|think:start|>`` (or any
    other sentinel CONTENT does not allow) is accepted unmasked on a
    content-with-progress row, which is how a control token reaches the client
    as text.

    Committed state only, for the same reason as ``folded_mask_flags``: the
    draft chain lives on device and reading it before the target launch is the
    sync this path exists to avoid. The two therefore disagree wherever a draft
    moves the state, and in the same bounded way -- at most ``stride - 1`` rows,
    cleared once the next step's committed state catches up.

    Fresh CONTENT is not flagged: ``plan_gate`` sends those rows eager, where
    ``plan_verify`` applies the stricter fresh set (EOS and ``<|im:end|>``
    shut). Rows under a grammar are not flagged either -- the grammar owns
    CONTENT.

    Returns None when the FSM is inactive, so the caller keeps stock behaviour.
    """
    if not is_active() or not reqs or stride <= 0:
        return None
    flags: List[bool] = []
    for req in reqs:
        fsm = getattr(req, "_solar_fsm", None)
        if fsm is None or _fsm_stale(req):
            # plan_gate sent this step eager; plan_verify will judge it with
            # fresh state after the barrier.
            flags.extend([False] * stride)
            continue
        fsm.advance(req.output_ids)
        on = bool(
            fsm.state == CONTENT and fsm.content_progress and not _has_grammar(req)
        )
        flags.extend([on] * stride)
    return flags


def plan_gate(reqs, stride: int) -> bool:
    """Whether this verify step must leave the folded in-graph accept path.

    The folded epilogue accepts inside the cuda graph off its own buffers and
    ``plan_verify`` runs after the grammar barrier, so what only plan_verify
    can do -- named by the two row predicates above -- must be decided here,
    per row, from committed state. Outside a tool call a generation spends
    only its boundary steps eager. The masks themselves are applied inside the
    graph (``folded_mask_flags`` for REASONING, ``folded_content_mask_flags``
    for content-with-progress), and the budget window is ``2 * stride``
    because the state read here can lag by one accepted run. Known folded-path
    gaps, each <= stride-1 chain rows and closed by the next step's committed
    state: a drafted sentinel under a grammar, and on a content-with-progress
    row the no-tools variant of the CONTENT set -- the in-graph buffer carries
    the tools-available one. Host-only and sync-free: it never touches the
    draft tokens.
    """
    if not is_active():
        return False
    if not reqs or stride <= 0:
        return False
    if CFG.spec_always_eager:
        return True
    window = 2 * stride
    for req in reqs:
        fsm = getattr(req, "_solar_fsm", None)
        if fsm is None or _fsm_stale(req):
            # No committed state to judge from yet (or it predates a retraction
            # and _req_fsm will rebuild it after the barrier).
            return True
        fsm.advance(req.output_ids)
        if fsm.in_reasoning:
            if _reasoning_needs_eager(fsm, window=window):
                return True
        elif _content_needs_eager(fsm, req=req):
            return True
    return False


def folded_mask_flags(reqs, stride: int) -> Optional[List[bool]]:
    """Per-(request, chain position) flags for the in-graph reasoning mask.

    True where the request's **committed** state is in REASONING with budget
    left. A row whose budget is spent needs a forced ``<|think:end|>`` instead,
    which only ``plan_verify`` can write, and :func:`plan_gate` has already sent
    that step to the eager path -- so it is left False here. The CONTENT rows
    this leaves False are armed by :func:`folded_content_mask_flags` instead.

    Committed state only, which is what keeps this sync-free: the draft chain
    lives on device and reading it before the target launch is the sync this
    whole path exists to avoid. ``plan_verify`` *does* walk the chain, so the
    two disagree wherever a draft moves the state, in both directions:

    * A drafted ``<|think:end|>`` takes ``plan_verify`` out of REASONING at that
      position; these flags stay True for the rest of the chain. Overmasking,
      and bounded -- at most ``stride - 1`` rows, cleared once the next step's
      committed state catches up. It is not free: EOS is forbidden where the
      model may legitimately want it, so a non-EOS token is accepted and
      committed after the answer. It is the cheaper error.
    * A drafted ``<|think:start|>`` puts ``plan_verify`` *into* REASONING from
      that position; these flags stay False. The row is not unmasked, though:
      it is a content-with-progress row, so :func:`folded_content_mask_flags`
      arms it with the CONTENT set, which forbids ``<|think:start|>`` in the
      first place. What is left is undermasking of the following rows against
      the *reasoning* set, bounded the same way as above.

    Returns None when the FSM is inactive, so the caller keeps stock behaviour.
    """
    if not is_active() or not reqs or stride <= 0:
        return None
    flags: List[bool] = []
    for req in reqs:
        fsm = getattr(req, "_solar_fsm", None)
        if fsm is None or _fsm_stale(req):
            # plan_gate sent this step eager; plan_verify will judge it with
            # fresh state after the barrier.
            flags.extend([False] * stride)
            continue
        fsm.advance(req.output_ids)
        # _SimState.exhausted's own predicate, read off the committed FSM:
        # in REASONING with the budget not yet spent.
        on = bool(fsm.in_reasoning and fsm.count < fsm.budget)
        flags.extend([on] * stride)
    return flags


def plan_verify(reqs, chain_ids, stride: int) -> Optional[VerifyPlan]:
    """Build the verify-step masks. Host work; call after the target forward,
    once the grammar barrier has advanced the committed state.

    ``chain_ids`` is a host-resident copy of the verify chain ``(bs, stride)``,
    staged before the target launch (the caller already owns one for the
    grammar path -- ``GrammarTree.resolve()``), so ``.tolist()`` here is not a
    device sync.

    Returns None when the FSM is inactive, so the caller keeps stock behaviour.
    """
    if not is_active():
        return None
    if not reqs or stride <= 0:
        return None

    chain = chain_ids.tolist()
    bs = min(len(reqs), len(chain))
    force_rows: List[int] = []
    mask_rows: Dict[Tuple[int, ...], List[int]] = {}
    rids = [reqs[i].rid for i in range(bs)]

    for i in range(bs):
        req = reqs[i]
        fsm = _req_fsm(req)
        # Committed tokens only -- drafts never touch the persistent state.
        # Usually a no-op here: the grammar barrier's advance_committed() has
        # already fed this run through commit().
        fsm.advance(req.output_ids)
        sim = _SimState(fsm)
        row_ids = chain[i]
        for w in range(stride):
            # row (i, w) predicts the token after drafts 1..w; w == 0 is the
            # committed state (row_ids[0] is the already-committed anchor).
            if w > 0 and w < len(row_ids):
                sim.step(int(row_ids[w]))
            row = i * stride + w
            if sim.in_reasoning:
                if sim.exhausted(fsm.budget):
                    force_rows.append(row)
                    if w == 0 and not fsm.forced:
                        fsm.forced = True
                        logger.info(
                            "[SOLAR-FSM] reasoning budget %d exhausted -> "
                            "forcing <|think:end|> (req %s, verify)",
                            fsm.budget,
                            rids[i],
                        )
                else:
                    mask_rows.setdefault(_reasoning_forbidden(sim), []).append(row)
            elif sim.state == CONTENT and _has_grammar(reqs[i]):
                continue  # structured outputs own the CONTENT phase
            else:
                mask_rows.setdefault(
                    _forbidden_for(
                        sim.state,
                        content_progress=sim.content_progress,
                        tools=fsm.tools,
                    ),
                    [],
                ).append(row)

    return VerifyPlan(
        force_rows=force_rows, mask_rows=mask_rows, stride=stride, bs=bs, rids=rids
    )
