# SPDX-License-Identifier: Apache-2.0
"""Solar Open2 chat-envelope FSM enforcer (SGLang port).

Ported from the Upstage vLLM fork's
``vllm/v1/sample/logits_processor/solar_open2.py``
(``SolarOpen2TokenFSMEnforcer``). The fork ships this as part of *serving* the
model, not as an optional extra: it structurally closes the illegal exits from
the REASONING block so the turn can only end through ``<|think:end|>``, and it
force-emits ``<|think:end|>`` once the reasoning budget is spent.

Scope of this port (agreed with the campaign owner):
  * REASONING state: sentinel/EOS mask + ``_FORCE_THINK_END`` on budget.
  * CONTENT / TOOL_CALL_* states: **not** masked here - the tool-call parser
    (``srt/function_call/solar_open2_detector.py``) covers the tool envelope.

Two deliberate deviations from the fork, both campaign decisions:
  1. **Reasoning budget is relative to the request's ``max_new_tokens``**
     (default 75%), not the fork's absolute 128K. The absolute default can
     never fire for our request sizes, which would make the port a no-op.
     ``SOLAR_FSM_BUDGET_RATIO`` / ``SOLAR_FSM_BUDGET_ABS`` override it.
  2. Token ids are resolved from the served tokenizer rather than hardcoded
     (the fork's defaults are its own tokenizer's layout).

Enabled by ``SOLAR_FSM=1``; a resolution failure disables it loudly rather
than masking with wrong ids.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

NEG_INF = float("-inf")

_SENTINEL_TOKENS = (
    "<|im:start|>",
    "<|im:end|>",
    "<|im:content|>",
    "<|think:start|>",
    "<|think:end|>",
    "<|tool:start|>",
    "<|tool:end|>",
    "<|tool_call:start|>",
    "<|tool_call:end|>",
    "<|tool_arg:start|>",
    "<|tool_arg:value|>",
    "<|tool_arg:end|>",
)

_FORK_ABS_BUDGET = 128 * 1024


class _Config:
    """Resolved once per process."""

    enabled: bool = False
    think_start: Optional[int] = None
    think_end: Optional[int] = None
    im_end: Optional[int] = None
    tool_call_start: Optional[int] = None
    tool_call_end: Optional[int] = None
    all_controls: frozenset = frozenset()
    # Per-state forbidden ids, mirroring the fork's _MASK_SPEC_BY_STATE /
    # _MASK_SPEC_CONTENT (solar_open2.py:329-350):
    #   REASONING            : allow think_end only, EOS masked
    #   CONTENT (no content) : allow tool_call_start only, EOS masked
    #                          <- this is what stops the model from emitting EOS
    #                             the instant it leaves reasoning (empty answer)
    #   CONTENT (has content): allow tool_call_start + im_end, EOS free
    reasoning_forbidden: Tuple[int, ...] = ()
    content_fresh_forbidden: Tuple[int, ...] = ()
    content_done_forbidden: Tuple[int, ...] = ()
    budget_ratio: float = 0.75
    # R4 measured this as a REGRESSION (GPQA 74.0% -> 68.0%): forbidding EOS in
    # fresh CONTENT forces a model that wanted to stop to emit something, which
    # raised truncation (1->7) and unparsable answers (1->6). The fork's rule
    # only makes sense together with the rest of its FSM; ported in isolation it
    # hurts. Kept as an opt-in so the asset survives, default OFF.
    content_mask: bool = False
    # R5-8: masking the verify logits costs the folded in-graph accept, so by
    # default we only leave the folded path on steps where the reasoning budget
    # actually falls inside the draft chain. Set 1 to mask every verify step
    # (fuller enforcement, measurable throughput cost -- BUILDLOG §[R5-8] ⑤).
    spec_always_eager: bool = False
    budget_abs: int = _FORK_ABS_BUDGET
    _mask_cache: Dict[Tuple[Tuple[int, ...], int, str], torch.Tensor] = {}


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
    ids: List[int] = []
    gc_path = os.path.join(tokenizer_dir, "generation_config.json")
    if os.path.isfile(gc_path):
        raw = json.load(open(gc_path, encoding="utf-8")).get("eos_token_id")
        if isinstance(raw, int):
            ids = [raw]
        elif isinstance(raw, (list, tuple)):
            ids = [int(x) for x in raw]
    return ids


def init_from_env() -> None:
    """Resolve ids + budget. Called lazily on first sampler pass."""
    if not os.environ.get("SOLAR_FSM", "0") == "1":
        CFG.enabled = False
        return
    tok_dir = os.environ.get("SOLAR_FSM_TOKENIZER_DIR", "")
    if not tok_dir or not os.path.isdir(tok_dir):
        raise RuntimeError(
            "SOLAR_FSM=1 but SOLAR_FSM_TOKENIZER_DIR is unset or not a directory: "
            f"{tok_dir!r}"
        )
    table = _added_token_ids(tok_dir)
    missing = [t for t in ("<|think:start|>", "<|think:end|>") if t not in table]
    if missing:
        raise RuntimeError(
            f"SOLAR_FSM: required special tokens not found in {tok_dir}: {missing}"
        )
    CFG.think_start = table["<|think:start|>"]
    CFG.think_end = table["<|think:end|>"]

    CFG.im_end = table.get("<|im:end|>")
    CFG.tool_call_start = table.get("<|tool_call:start|>")
    CFG.tool_call_end = table.get("<|tool_call:end|>")
    sentinels = {table[t] for t in _SENTINEL_TOKENS if t in table}
    CFG.all_controls = frozenset(sentinels)
    eos = set(_eos_ids(tok_dir))

    def build(allowed, mask_eos):
        f = sentinels - {i for i in allowed if i is not None}
        if mask_eos:
            f |= eos
        return tuple(sorted(f))

    CFG.reasoning_forbidden = build({CFG.think_end}, True)
    CFG.content_fresh_forbidden = build({CFG.tool_call_start}, True)
    CFG.content_done_forbidden = build({CFG.tool_call_start, CFG.im_end}, False)

    CFG.budget_ratio = float(os.environ.get("SOLAR_FSM_BUDGET_RATIO", "0.75"))
    CFG.content_mask = os.environ.get("SOLAR_FSM_CONTENT_MASK", "0") == "1"
    CFG.spec_always_eager = os.environ.get("SOLAR_FSM_SPEC_ALWAYS_EAGER", "0") == "1"
    CFG.budget_abs = int(os.environ.get("SOLAR_FSM_BUDGET_ABS", str(_FORK_ABS_BUDGET)))
    CFG.enabled = True
    logger.info(
        "[SOLAR-FSM] enabled: think_start=%s think_end=%s im_end=%s | "
        "forbidden reasoning=%s content_fresh=%s content_done=%s | "
        "content_mask=%s budget=min(%d, max_new_tokens*%.2f)",
        CFG.think_start,
        CFG.think_end,
        CFG.im_end,
        CFG.reasoning_forbidden,
        CFG.content_fresh_forbidden,
        CFG.content_done_forbidden,
        CFG.content_mask,
        CFG.budget_abs,
        CFG.budget_ratio,
    )


class SolarReqFSM:
    """Per-request REASONING tracker. Lives on the Req, so a row permutation
    can never hand one request another request's state."""

    __slots__ = (
        "in_reasoning",
        "count",
        "last_len",
        "budget",
        "forced",
        "content_progress",
    )

    def __init__(self, prompt_ids: Sequence[int], max_new_tokens: Optional[int]):
        # Mirrors the fork's _initial_state: inside reasoning iff the last
        # think_start in the prompt is not followed by a think_end.
        self.in_reasoning = _starts_in_reasoning(prompt_ids)
        self.content_progress = False
        self.count = 0
        self.last_len = 0
        if max_new_tokens and max_new_tokens > 0:
            self.budget = min(CFG.budget_abs, int(max_new_tokens * CFG.budget_ratio))
        else:
            self.budget = CFG.budget_abs
        self.forced = False

    def advance(self, output_ids: Sequence[int]) -> None:
        n = len(output_ids)
        if n < self.last_len:  # retraction / restart
            self.last_len = 0
            self.count = 0
        for i in range(self.last_len, n):
            tok = output_ids[i]
            if tok == CFG.think_start:
                self.in_reasoning = True
                self.content_progress = False
                self.count = 0
            elif tok == CFG.think_end:
                self.in_reasoning = False
            elif self.in_reasoning:
                self.count += 1
            elif tok == CFG.tool_call_end or tok not in CFG.all_controls:
                # a completed tool call, or any ordinary token, counts as the
                # turn having produced content -> the turn may now legally end
                self.content_progress = True
        self.last_len = n

    def budget_exhausted(self) -> bool:
        return self.in_reasoning and self.count >= self.budget


def _rindex(values: Sequence[int], needle: int) -> Optional[int]:
    for i in range(len(values) - 1, -1, -1):
        if values[i] == needle:
            return i
    return None


def _starts_in_reasoning(prompt_ids: Sequence[int]) -> bool:
    if CFG.think_start is None or CFG.think_end is None:
        return False
    last_start = _rindex(prompt_ids, CFG.think_start)
    if last_start is None:
        return False
    last_end = _rindex(prompt_ids, CFG.think_end)
    return last_end is None or last_start > last_end


def _mask_tensor(ids: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    key = (ids, 0, str(device))
    t = CFG._mask_cache.get(key)
    if t is None:
        t = torch.tensor(ids, dtype=torch.long, device=device)
        CFG._mask_cache[key] = t
    return t


# --------------------------------------------------------------------------
# Hooks called from patched sglang core (see solar_patch.py)
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
    if rows is not None and other_rows is not None:
        sampling_info.solar_fsm_rows = rows + other_rows


_WARNED = {"shape": False}


def apply(logits: torch.Tensor, sampling_info) -> None:
    """Mask ``logits`` in place according to each row's FSM state."""
    if not CFG.enabled:
        if os.environ.get("SOLAR_FSM", "0") != "1":
            return
        init_from_env()
        if not CFG.enabled:
            return

    rows = getattr(sampling_info, "solar_fsm_rows", None)
    if not rows:
        return
    if logits.shape[0] != len(rows):
        # Speculative decoding and other multi-token-per-row paths land here.
        if not _WARNED["shape"]:
            logger.warning(
                "[SOLAR-FSM] skipping mask: logits rows=%d != batch rows=%d "
                "(spec decode?). This warning is logged once.",
                logits.shape[0],
                len(rows),
            )
            _WARNED["shape"] = True
        return

    force_rows: List[int] = []
    mask_rows: Dict[Tuple[int, ...], List[int]] = {}
    for i, req in enumerate(rows):
        fsm = getattr(req, "_solar_fsm", None)
        if fsm is None:
            fsm = SolarReqFSM(
                getattr(req, "origin_input_ids", ()) or (),
                getattr(getattr(req, "sampling_params", None), "max_new_tokens", None),
            )
            req._solar_fsm = fsm
        fsm.advance(req.output_ids)
        if fsm.in_reasoning:
            if fsm.budget_exhausted():
                force_rows.append(i)
            else:
                mask_rows.setdefault(CFG.reasoning_forbidden, []).append(i)
        elif not CFG.content_mask:
            continue  # R2-3rd behaviour: CONTENT unmasked (parser owns it)
        elif fsm.content_progress:
            mask_rows.setdefault(CFG.content_done_forbidden, []).append(i)
        else:
            # Fresh CONTENT: EOS/im_end forbidden until real content exists.
            # Without this the model leaves reasoning and immediately stops,
            # yielding finish=stop with an empty answer (observed: 6/13 GPQA
            # errors in R2-3rd had content_len == 0).
            mask_rows.setdefault(CFG.content_fresh_forbidden, []).append(i)

    for ids, rows_i in mask_rows.items():
        if not ids:
            continue
        idx = _mask_tensor(ids, logits.device)
        rsel = torch.tensor(rows_i, dtype=torch.long, device=logits.device)
        logits[rsel.unsqueeze(1), idx.unsqueeze(0)] = NEG_INF

    if force_rows:
        rsel = torch.tensor(force_rows, dtype=torch.long, device=logits.device)
        keep = logits[rsel, CFG.think_end].clone()
        logits[rsel, :] = NEG_INF
        logits[rsel, CFG.think_end] = keep
        for i in force_rows:
            fsm = rows[i]._solar_fsm
            if not fsm.forced:
                fsm.forced = True
                logger.info(
                    "[SOLAR-FSM] reasoning budget %d exhausted -> forcing "
                    "<|think:end|>",
                    fsm.budget,
                )


# --------------------------------------------------------------------------
# Speculative verify path (R5-8). See BUILDLOG §[R5-8].
#
# `apply()` above is injected into layers/sampler.py, which the DSpark verify
# path never reaches -- so with spec ON the reasoning budget was never enforced
# (measured: `<|think:end|>` fired at the budget position in 12/12 spec-OFF
# generations and 0/12 spec-ON). These helpers mask the *verify* logits instead.
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
# --------------------------------------------------------------------------
class _SimState:
    """Throwaway FSM state used to walk a speculative chain."""

    __slots__ = ("in_reasoning", "count", "content_progress")

    def __init__(self, fsm: SolarReqFSM):
        self.in_reasoning = fsm.in_reasoning
        self.count = fsm.count
        self.content_progress = fsm.content_progress

    def step(self, tok: int) -> None:
        """Mirror of SolarReqFSM.advance()'s per-token transition."""
        if tok == CFG.think_start:
            self.in_reasoning = True
            self.content_progress = False
            self.count = 0
        elif tok == CFG.think_end:
            self.in_reasoning = False
        elif self.in_reasoning:
            self.count += 1
        elif tok == CFG.tool_call_end or tok not in CFG.all_controls:
            self.content_progress = True

    def exhausted(self, budget: int) -> bool:
        return self.in_reasoning and self.count >= budget


class VerifyPlan:
    """Per-(request, chain position) masks for one target-verify step."""

    __slots__ = ("force_rows", "mask_rows", "stride", "bs")

    def __init__(self, force_rows, mask_rows, stride, bs):
        self.force_rows = force_rows
        self.mask_rows = mask_rows
        self.stride = stride
        self.bs = bs

    @property
    def needs_eager(self) -> bool:
        """Whether this step must leave the folded in-graph accept path.

        The folded epilogue accepts inside the cuda graph off its own buffers,
        where a mask applied to `next_token_logits` never lands. Forcing eager
        costs the in-graph accept, so we only pay it on steps where the FSM
        actually has to intervene -- i.e. a budget boundary falls inside the
        chain. (Design B, BUILDLOG §[R5-8] ⑤; SOLAR_FSM_SPEC_ALWAYS_EAGER=1
        selects the always-on variant.)
        """
        if CFG.spec_always_eager:
            return bool(self.force_rows or self.mask_rows)
        return bool(self.force_rows)

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

        for ids, rows_i in self.mask_rows.items():
            rows_i = keep(rows_i)
            if not ids or not rows_i:
                continue
            idx = _mask_tensor(ids, logits.device)
            rsel = torch.tensor(rows_i, dtype=torch.long, device=logits.device)
            logits[rsel.unsqueeze(1), idx.unsqueeze(0)] = NEG_INF

        force = keep(self.force_rows)
        if force:
            rsel = torch.tensor(force, dtype=torch.long, device=logits.device)
            kept = logits[rsel, CFG.think_end].clone()
            logits[rsel, :] = NEG_INF
            logits[rsel, CFG.think_end] = kept


def plan_verify(reqs, verify_ids_2d, stride: int) -> Optional[VerifyPlan]:
    """Build the verify-step masks. Host work; call before the target forward.

    Returns None when the FSM is inactive, so the caller keeps stock behaviour.
    """
    if not CFG.enabled:
        if os.environ.get("SOLAR_FSM", "0") != "1":
            return None
        init_from_env()
        if not CFG.enabled:
            return None
    if not reqs or stride <= 0:
        return None

    # Fast path: `verify_ids_2d.tolist()` is a device->host copy, i.e. a sync on
    # every decode step. The chain walk can only change the outcome when a
    # request is within `stride` tokens of its budget, and that is decidable
    # from committed state alone. Nothing near the boundary -> no plan, no sync.
    # (With CONTENT masking off -- the default -- the budget force is the only
    # thing this plan produces.)
    near = False
    for req in reqs:
        fsm = getattr(req, "_solar_fsm", None)
        if fsm is None:
            near = True  # unseen request: fall through and build its state
            break
        fsm.advance(req.output_ids)
        if fsm.in_reasoning and fsm.count + stride >= fsm.budget:
            near = True
            break
    if not near and not CFG.content_mask and not CFG.spec_always_eager:
        return None

    chain = verify_ids_2d.tolist()
    bs = min(len(reqs), len(chain))
    force_rows: List[int] = []
    mask_rows: Dict[Tuple[int, ...], List[int]] = {}

    for i in range(bs):
        req = reqs[i]
        fsm = getattr(req, "_solar_fsm", None)
        if fsm is None:
            fsm = SolarReqFSM(
                getattr(req, "origin_input_ids", ()) or (),
                getattr(getattr(req, "sampling_params", None), "max_new_tokens", None),
            )
            req._solar_fsm = fsm
        # committed tokens only -- drafts never touch the persistent state
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
                else:
                    mask_rows.setdefault(CFG.reasoning_forbidden, []).append(row)
            elif not CFG.content_mask:
                continue  # R2-3rd behaviour: CONTENT unmasked (parser owns it)
            elif sim.content_progress:
                mask_rows.setdefault(CFG.content_done_forbidden, []).append(row)
            else:
                mask_rows.setdefault(CFG.content_fresh_forbidden, []).append(row)

    return VerifyPlan(force_rows, mask_rows, stride, bs)
