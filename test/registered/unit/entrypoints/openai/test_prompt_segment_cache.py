"""The prompt segment cache must return exactly ``tokenizer.encode(text)``.

`encode_with_segment_cache` cuts a rendered prompt right after every added
token, reuses the ids of segments seen before and encodes only the rest, so
its whole contract is id identity with the single encode -- a wrong cut or a
stale entry is not a crash, it is a silently different prompt. The backend
here mirrors the DeepSeek-V4.x pipeline (digit / CJK / word Split regexes,
ByteLevel without regex, empty normalizer Sequence, ByteLevel post-processor,
BPE, special role tokens), built in memory with a small vocabulary.
"""

import random
import threading
import unittest
from array import array
from functools import lru_cache
from types import SimpleNamespace
from unittest.mock import Mock

from tokenizers import (
    AddedToken,
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizerFast

from sglang.srt.entrypoints.openai import prompt_segment_cache as psc
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_DS_WORD_REGEX = (
    r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?"
    r"[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_SPECIALS = ["<bos>", "<User>", "<Assistant>", "<think>", "</think>", "<eos>"]
_ROLES = ["<|System|>", "<|Tool|>"]
_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "numbers 12345 67890 11111 98765 and 3.14159",
    "def main(argv):\n    return 0\n",
    "中文文本 日本語 한국어 émigré naïve Ωmega",
    "it's they're I'm we'll you'd she's",
]


@lru_cache(maxsize=1)
def _ds_backend_json():
    backend = Tokenizer(models.BPE())
    backend.normalizer = normalizers.Sequence([])
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"\p{N}{1,3}"), behavior="isolated"),
            pre_tokenizers.Split(
                Regex("[一-龥぀-ゟ゠-ヿ]+"),
                behavior="isolated",
            ),
            pre_tokenizers.Split(Regex(_DS_WORD_REGEX), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    backend.post_processor = processors.ByteLevel(trim_offsets=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        _CORPUS * 20,
        trainers.BpeTrainer(
            vocab_size=700,
            show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[],
        ),
    )
    backend.add_tokens(
        [AddedToken(t, normalized=False, special=True) for t in _SPECIALS]
    )
    return backend.to_str()


def _tokenizer(mutate=None, normalized_roles=False):
    backend = Tokenizer.from_str(_ds_backend_json())
    if normalized_roles:
        # DeepSeek-V3/V4 mark the role tokens normalized=True; with the empty
        # normalizer Sequence they still match at the raw offsets.
        backend.add_tokens(
            [AddedToken(t, normalized=True, special=True) for t in _ROLES]
        )
    if mutate is not None:
        mutate(backend)
    return PreTrainedTokenizerFast(tokenizer_object=backend)


def _words(rng, n):
    return " ".join(str(rng.randrange(10000, 100000)) for _ in range(n))


def _prompt(system, user, tail="<Assistant></think>"):
    return "<bos>" + system + "<User>" + user + tail


class TestPromptSegmentCache(CustomTestCase):
    def setUp(self):
        self.tok = _tokenizer()
        self.cache = psc.PromptSegmentCache(budget_bytes=64 << 20, min_chars=16)

    def _check(self, text, cache=None):
        got = psc.encode_with_segment_cache(
            self.tok, text, cache if cache is not None else self.cache
        )
        self.assertIsInstance(got, array)
        self.assertEqual(got.typecode, "q")
        self.assertEqual(list(got), self.tok.encode(text))
        return got

    def test_segments_end_right_after_added_tokens(self):
        backend, _, pattern, npattern, _ = psc.ppe._plan_for(self.tok)
        text = _prompt("sys text", "user text")
        self.assertEqual(
            psc.split_segments(text, pattern, npattern),
            ["<bos>", "sys text<User>", "user text<Assistant>", "</think>"],
        )
        self.assertEqual(psc.split_segments("plain", pattern, npattern), ["plain"])
        self.assertEqual(psc.split_segments("", pattern, npattern), [])

    def test_identity_on_edge_texts(self):
        rng = random.Random(0)
        texts = [
            "a",
            " ",
            "\n",
            "<User>",
            "<User><Assistant>",
            "x<User>",
            "<User>x",
            "no specials at all 12345\nline two\r\n",
            _prompt("", ""),
            _prompt(" leading space", "trailing space "),
            _prompt("中文文本 日本語 한국어", "émigré naïve Ωmega 123456789"),
            _prompt(_words(rng, 300), _words(rng, 30), tail=""),
            "<bos><User>\n\n<Assistant><think>\n</think> 42<eos>",
            "<User " + _words(rng, 5) + " User>",  # not an added token
        ]
        for text in texts:
            with self.subTest(text=text[:40]):
                self._check(text)

    def test_repeated_segments_hit_and_stay_identical(self):
        rng = random.Random(1)
        systems = [_words(rng, 400) for _ in range(3)]
        for i in range(30):
            text = _prompt(systems[i % 3], "[%08x] " % i + _words(rng, 40))
            self._check(text)
        # 3 systems x 1 miss each; every later system segment is a hit.
        self.assertGreaterEqual(self.cache.hits, 27)
        self.assertGreater(self.cache.hit_tokens, self.cache.miss_tokens)

    def test_lru_budget_evicts_oldest(self):
        rng = random.Random(2)
        seg_a, seg_b = _words(rng, 200), _words(rng, 200)
        ids_a = array("q", self.tok.encode(seg_a))
        cost_a = psc.PromptSegmentCache._cost(seg_a, ids_a)
        cache = psc.PromptSegmentCache(budget_bytes=int(cost_a * 1.5), min_chars=16)
        self._check(_prompt(seg_a, "u"), cache)
        self.assertIsNotNone(cache.get(seg_a + "<User>"))
        self._check(_prompt(seg_b, "u"), cache)
        self.assertIsNone(cache.get(seg_a + "<User>"))
        self.assertLessEqual(cache.nbytes, cache.budget_bytes)
        # An entry larger than the whole budget is never stored.
        tiny = psc.PromptSegmentCache(budget_bytes=64, min_chars=16)
        self._check(_prompt(seg_a, "u"), tiny)
        self.assertEqual(len(tiny), 0)

    def test_short_segments_are_not_stored(self):
        cache = psc.PromptSegmentCache(budget_bytes=1 << 20, min_chars=1000)
        self._check(_prompt("short system", "short user"), cache)
        self.assertEqual(len(cache), 0)

    def test_concurrent_callers_get_identical_ids(self):
        rng = random.Random(3)
        systems = [_words(rng, 300) for _ in range(4)]
        texts = [
            _prompt(systems[i % 4], "[%03d] " % i + _words(rng, 20)) for i in range(64)
        ]
        want = [self.tok.encode(t) for t in texts]
        errors = []

        def run(k):
            for j in range(k, len(texts), 8):
                got = psc.encode_with_segment_cache(self.tok, texts[j], self.cache)
                if list(got) != want[j]:
                    errors.append(j)

        threads = [threading.Thread(target=run, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    def test_falls_back_when_post_processor_adds_specials(self):
        def add_bos(backend):
            backend.post_processor = processors.TemplateProcessing(
                single="<bos> $A", special_tokens=[("<bos>", 0)]
            )

        tok = _tokenizer(add_bos)
        text = _prompt("system", "user")
        got = psc.encode_with_segment_cache(tok, text, self.cache)
        self.assertNotIsInstance(got, array)
        self.assertEqual(got, tok.encode(text))
        self.assertEqual(len(self.cache), 0)

    def test_falls_back_with_truncation_enabled(self):
        tok = _tokenizer(lambda b: b.enable_truncation(max_length=8))
        text = _prompt("system words here", "user")
        got = psc.encode_with_segment_cache(tok, text, self.cache)
        self.assertNotIsInstance(got, array)
        self.assertEqual(got, tok.encode(text))

    def test_switch_off_means_no_cache(self):
        with envs.SGLANG_OPT_PROMPT_SEGMENT_CACHE_MB.override(0):
            self.assertIsNone(psc.get_prompt_segment_cache())
            text = _prompt("system", "user")
            got = psc.encode_with_segment_cache(self.tok, text)
            self.assertNotIsInstance(got, array)
            self.assertEqual(got, self.tok.encode(text))


class TestNormalizedAddedTokens(CustomTestCase):
    """Normalized added tokens are cut points only under an identity normalizer."""

    def _segments(self, tok, text):
        plan = psc._plan_for(tok)
        self.assertIsNotNone(plan)
        return psc.split_segments(text, plan[2], plan[3])

    def test_identity_normalizer_cuts_at_normalized_tokens(self):
        tok = _tokenizer(normalized_roles=True)
        text = "<bos><|System|>sys 123<|Tool|>tool 456<User>u<Assistant>"
        self.assertEqual(
            self._segments(tok, text),
            [
                "<bos>",
                "<|System|>",
                "sys 123<|Tool|>",
                "tool 456<User>",
                "u<Assistant>",
            ],
        )
        rng = random.Random(4)
        cache = psc.PromptSegmentCache(budget_bytes=8 << 20, min_chars=16)
        system = _words(rng, 300)
        for i in range(6):
            text = "<bos><|System|>" + system + "<|Tool|>" + _words(rng, 30) + "<User>q"
            got = psc.encode_with_segment_cache(tok, text, cache)
            self.assertEqual(list(got), tok.encode(text))
        self.assertGreaterEqual(cache.hits, 5)

    def test_non_identity_normalizer_ignores_normalized_tokens(self):
        def lowercase(backend):
            backend.normalizer = normalizers.Lowercase()

        tok = _tokenizer(lowercase, normalized_roles=True)
        text = "<bos><|System|>Sys ABC<User>U<Assistant>"
        segs = self._segments(tok, text)
        self.assertNotIn("<|System|>", segs)  # not a cut point under Lowercase
        cache = psc.PromptSegmentCache(budget_bytes=8 << 20, min_chars=1)
        got = psc.encode_with_segment_cache(tok, text, cache)
        self.assertEqual(list(got), tok.encode(text))


class TestServingChatHook(CustomTestCase):
    """`_encode_rendered_prompt` takes the cache only when the switch is on
    and the request is not parallel-sampled."""

    def _chat(self, tok):
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

        chat = OpenAIServingChat.__new__(OpenAIServingChat)
        chat.tokenizer_manager = SimpleNamespace(tokenizer=tok)
        return chat

    def test_hook(self):
        tok = _tokenizer()
        chat = self._chat(tok)
        text = _prompt("system words " * 20, "user words")
        with envs.SGLANG_OPT_PROMPT_SEGMENT_CACHE_MB.override(0):
            got = chat._encode_rendered_prompt(text, SimpleNamespace(n=1))
            self.assertIsInstance(got, list)
            self.assertEqual(got, tok.encode(text))
        with envs.SGLANG_OPT_PROMPT_SEGMENT_CACHE_MB.override(8):
            try:
                got = chat._encode_rendered_prompt(text, SimpleNamespace(n=1))
                self.assertIsInstance(got, array)
                self.assertEqual(list(got), tok.encode(text))
                got = chat._encode_rendered_prompt(text, SimpleNamespace(n=2))
                self.assertIsInstance(got, list)
                self.assertEqual(got, tok.encode(text))
            finally:
                psc._cache = None

    def test_engine_prompt_treats_array_like_list(self):
        from sglang.srt.entrypoints.openai.protocol import MessageProcessingResult

        chat = self._chat(Mock())
        chat._prompt_text_round_trip_is_lossy = True
        for spec in ("dsv41", "dsv4"):
            chat.chat_encoding_spec = spec
            for ids in ([4, 5, 6], array("q", [4, 5, 6])):
                result = MessageProcessingResult(
                    prompt="rendered",
                    prompt_ids=ids,
                    image_data=None,
                    audio_data=None,
                    video_data=None,
                    modalities=[],
                    stop=[],
                )
                with self.subTest(spec=spec, kind=type(ids).__name__):
                    key, value = chat._engine_prompt(result, is_multimodal=True)
                    self.assertEqual(key, "input_ids")
                    self.assertIs(value, ids)

    def test_assistant_prefix_accepts_array(self):
        tok = Mock()
        tok.encode.return_value = [7, 8]
        tok.bos_token_id = 1
        chat = self._chat(tok)
        out = chat._append_assistant_prefix_to_prompt_ids(array("q", [4, 5]), "x")
        self.assertEqual(out, [4, 5, 7, 8])


if __name__ == "__main__":
    unittest.main(verbosity=2)
