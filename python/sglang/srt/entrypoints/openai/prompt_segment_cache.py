r"""Reuse the token ids of prompt segments that earlier requests already encoded.

A HF fast tokenizer splits the text at added-token matches first and then runs
the normalizer, pre-tokenizer and model on each piece independently (see
``parallel_prompt_encode``). A rendered chat prompt therefore splits, right
after every added-token match (role and turn markers), into segments whose ids
do not depend on the text around them: ``encode(text)`` equals the
concatenation of ``encode(segment)`` over the segments. Long agent and RAG
prompts repeat almost all of their segments from request to request (the
shared system prompt, earlier turns, tool outputs), which is what makes their
prefix-cache hit high; the server nevertheless tokenized the full prompt again
for every request, and in PD both the prefill and the decode server do it.

This module keeps an LRU of segment text -> ids as an ``array('q')`` and
encodes only the missing segments (one ``encode_batch`` call, which releases
the GIL). A hit costs a dict lookup and a buffer copy instead of tokenizing the
segment and building a Python list of its ids under the GIL. The result is an
``array('q')``, the representation the tokenized request already carries, so
the tokenizer manager's ``array("q", input_ids)`` becomes a buffer copy.

It uses ``parallel_prompt_encode``'s checks of the tokenizer pipeline (stock HF
fast tokenizer, position-independent pre-tokenizer, id-preserving
post-processor that adds no special tokens, no lstrip/rstrip/single_word added
tokens, no truncation, padding or split_special_tokens) and cuts only at
added-token ends. Normalized added tokens (the role markers of DeepSeek-V3/V4
tokenizers) count as cut points only when the normalizer is an identity (none
or an empty Sequence). Anything outside that falls back to ``tokenizer.encode``.

Off unless SGLANG_OPT_PROMPT_SEGMENT_CACHE_MB > 0 (the cache's byte budget).
Segments shorter than SGLANG_OPT_PROMPT_SEGMENT_CACHE_MIN_CHARS are encoded
every time (with the misses) and not stored.
"""

import json
import logging
import threading
import weakref
from array import array
from collections import OrderedDict
from typing import List, Optional, Union

from sglang.srt.entrypoints.openai import parallel_prompt_encode as ppe
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_ITEM_BYTES = array("q").itemsize
# Log the hit rate once per this many prompts (confirms the switch in a log).
_REPORT_EVERY = 1000


class PromptSegmentCache:
    """Thread-safe LRU of segment text -> token ids (``array('q')``)."""

    def __init__(self, budget_bytes: int, min_chars: int):
        self.budget_bytes = budget_bytes
        self.min_chars = min_chars
        self._entries: OrderedDict[str, array] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.hit_tokens = 0
        self.miss_tokens = 0
        self.calls = 0

    @staticmethod
    def _cost(text: str, ids: array) -> int:
        # The key string is held too; count both so the budget means memory.
        return len(ids) * _ITEM_BYTES + len(text)

    def get(self, text: str) -> Optional[array]:
        with self._lock:
            ids = self._entries.get(text)
            if ids is not None:
                self._entries.move_to_end(text)
            return ids

    def put(self, text: str, ids: array) -> None:
        cost = self._cost(text, ids)
        if cost > self.budget_bytes:
            return
        with self._lock:
            old = self._entries.pop(text, None)
            if old is not None:
                self._bytes -= self._cost(text, old)
            self._entries[text] = ids
            self._bytes += cost
            while self._bytes > self.budget_bytes:
                old_text, old_ids = self._entries.popitem(last=False)
                self._bytes -= self._cost(old_text, old_ids)

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        # An empty cache is still a cache (``cache or default`` must not skip it).
        return True

    @property
    def nbytes(self) -> int:
        return self._bytes


_cache: Optional[PromptSegmentCache] = None
_cache_lock = threading.Lock()
_plans = weakref.WeakKeyDictionary()


def _normalizer_is_identity(cfg) -> bool:
    if cfg is None:
        return True
    if cfg.get("type") == "Sequence":
        return all(_normalizer_is_identity(c) for c in cfg.get("normalizers", []))
    return False


def _build_plan(tokenizer):
    """(backend, specials_added, pattern, npattern) or None.

    parallel_prompt_encode's checks, plus: when the normalizer is an identity
    (none, or an empty Sequence as in DeepSeek-V3/V4 tokenizers), normalized
    added tokens match at the same offsets as in the raw text, so their ends
    are cut points too. That is where the role markers of those tokenizers are.
    """
    plan = ppe._plan_for(tokenizer)
    if plan is None:
        return None
    backend, specials_added, pattern, npattern, _intra = plan
    if npattern is None:
        try:
            identity = _normalizer_is_identity(
                json.loads(backend.to_str()).get("normalizer")
            )
        except Exception:
            identity = False
        if identity:
            normed = [
                t.content
                for t in backend.get_added_tokens_decoder().values()
                if t.normalized and t.content
            ]
            npattern = ppe._alternation(normed)
    if pattern is None and npattern is None:
        return None
    return backend, specials_added, pattern, npattern


def _plan_for(tokenizer):
    try:
        return _plans[tokenizer]
    except KeyError:
        pass
    except TypeError:
        return _build_plan(tokenizer)
    plan = _build_plan(tokenizer)
    _plans[tokenizer] = plan
    return plan


def get_prompt_segment_cache() -> Optional[PromptSegmentCache]:
    """The process-wide cache, or None when the switch is off."""
    global _cache
    budget_mb = envs.SGLANG_OPT_PROMPT_SEGMENT_CACHE_MB.get()
    if budget_mb <= 0:
        return None
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = PromptSegmentCache(
                    budget_bytes=budget_mb * 1024 * 1024,
                    min_chars=envs.SGLANG_OPT_PROMPT_SEGMENT_CACHE_MIN_CHARS.get(),
                )
                logger.info(
                    "Prompt segment cache on: budget %d MiB, segments >= %d chars",
                    budget_mb,
                    _cache.min_chars,
                )
    return _cache


def split_segments(text: str, pattern, npattern) -> List[str]:
    """Cut ``text`` right after every added-token match (and nowhere else)."""
    segments, start = [], 0
    for _, end in ppe._added_spans(text, pattern, npattern):
        if end > start:
            segments.append(text[start:end])
            start = end
    if start < len(text):
        segments.append(text[start:])
    return segments


def encode_with_segment_cache(
    tokenizer, text: str, cache: Optional[PromptSegmentCache] = None
) -> Union[array, List[int]]:
    """Drop-in for ``tokenizer.encode(text)`` (default kwargs).

    Returns ``array('q')`` when the cache path is taken, otherwise whatever
    ``tokenizer.encode`` returns.
    """
    cache = cache if cache is not None else get_prompt_segment_cache()
    if cache is None or not isinstance(text, str) or not text:
        return tokenizer.encode(text)
    plan = _plan_for(tokenizer)
    if plan is None:
        return tokenizer.encode(text)
    backend, specials_added, pattern, npattern = plan
    # encode() adds the post-processor's special tokens by default; the
    # segment path can only reproduce a post-processor that adds none.
    if specials_added:
        return tokenizer.encode(text)
    if (
        backend.truncation is not None
        or backend.padding is not None
        or backend.encode_special_tokens
    ):
        return tokenizer.encode(text)

    segments = split_segments(text, pattern, npattern)
    parts: List[Optional[array]] = [None] * len(segments)
    miss_idx = []
    for i, seg in enumerate(segments):
        if len(seg) >= cache.min_chars:
            ids = cache.get(seg)
            if ids is not None:
                parts[i] = ids
                continue
        miss_idx.append(i)

    if miss_idx:
        encodings = backend.encode_batch(
            [segments[i] for i in miss_idx], add_special_tokens=False
        )
        for i, enc in zip(miss_idx, encodings):
            ids = array("q", enc.ids)
            parts[i] = ids
            if len(segments[i]) >= cache.min_chars:
                cache.put(segments[i], ids)

    out = array("q")
    for ids in parts:
        out.extend(ids)
    miss_tokens = sum(len(parts[i]) for i in miss_idx)
    hit_tokens = len(out) - miss_tokens
    with cache._lock:
        cache.hits += len(segments) - len(miss_idx)
        cache.misses += len(miss_idx)
        cache.hit_tokens += hit_tokens
        cache.miss_tokens += miss_tokens
        cache.calls += 1
        report = cache.calls % _REPORT_EVERY == 0
    if report:
        logger.info(
            "Prompt segment cache: %d prompts, token hit %.1f%% (%d / %d), "
            "%d entries, %.0f MiB",
            cache.calls,
            100.0 * cache.hit_tokens / max(1, cache.hit_tokens + cache.miss_tokens),
            cache.hit_tokens,
            cache.hit_tokens + cache.miss_tokens,
            len(cache),
            cache.nbytes / 2**20,
        )
    return out
