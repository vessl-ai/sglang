import os
import unittest
from contextlib import contextmanager
from unittest import mock

from sglang.kernels.ops.attention.dsa import sm80
from sglang.srt.runtime_context import cp_min_extend_tokens
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_V1 = "SGLANG_SM80_SPARSE_MLA_V1_CFG"
_SPLIT = "SGLANG_SM80_SPARSE_MLA_DECODE_SPLIT"
_CP = "SGLANG_CP_MIN_EXTEND_TOKENS"


def _clear_caches():
    cp_min_extend_tokens.cache_clear()
    sm80.sm80_v1_cfg.cache_clear()
    sm80.decode_split_enabled.cache_clear()
    sm80.validate_sm80_fp8kv.cache_clear()


@contextmanager
def _env(**values):
    """Run with exactly these settings variables; unset the others."""
    with mock.patch.dict(os.environ, {}):
        for name in (_V1, _SPLIT, _CP):
            os.environ.pop(name, None)
        os.environ.update(values)
        _clear_caches()
        try:
            yield
        finally:
            _clear_caches()


class TestCpMinExtendTokens(CustomTestCase):
    def test_unset_is_off(self):
        with _env():
            self.assertEqual(cp_min_extend_tokens(), 0)
        with _env(**{_CP: ""}):
            self.assertEqual(cp_min_extend_tokens(), 0)

    def test_valid(self):
        with _env(**{_CP: "4096"}):
            self.assertEqual(cp_min_extend_tokens(), 4096)
        with _env(**{_CP: " 0 "}):
            self.assertEqual(cp_min_extend_tokens(), 0)

    def test_malformed_raises_with_the_name(self):
        for bad in ("abc", "1.5", "4k"):
            with _env(**{_CP: bad}), self.assertRaisesRegex(ValueError, _CP):
                cp_min_extend_tokens()

    def test_negative_raises(self):
        with _env(**{_CP: "-1"}), self.assertRaisesRegex(ValueError, _CP):
            cp_min_extend_tokens()


class TestSm80Settings(CustomTestCase):
    def test_v1_cfg_unset_and_valid(self):
        with _env():
            self.assertIsNone(sm80.sm80_v1_cfg())
            sm80.validate_sm80_settings()
        with _env(**{_V1: "32,1,128"}):
            self.assertEqual(sm80.sm80_v1_cfg(), (32, 1, 128))
            sm80.validate_sm80_settings()

    def test_v1_cfg_malformed_raises_with_the_name(self):
        for bad in ("32,1", "32,1,128,4", "a,b,c", "32,0,128", "-1,1,128"):
            with _env(**{_V1: bad}), self.assertRaisesRegex(ValueError, _V1):
                sm80.validate_sm80_settings()

    def test_decode_split(self):
        with _env():
            self.assertFalse(sm80.decode_split_enabled())
        for on in ("1", "true"):
            with _env(**{_SPLIT: on}):
                self.assertTrue(sm80.decode_split_enabled())
        with _env(**{_SPLIT: "0"}):
            self.assertFalse(sm80.decode_split_enabled())
        with _env(**{_SPLIT: "maybe"}), self.assertRaisesRegex(ValueError, _SPLIT):
            sm80.validate_sm80_settings()

    def test_fp8_kv_boot_check(self):
        with _env():  # unset: the fp8 KV default tile is not pipelined
            sm80.validate_sm80_fp8kv()
        with _env(**{_V1: "32,1,128"}):
            sm80.validate_sm80_fp8kv()
        with _env(**{_V1: "32,2,128"}), self.assertRaisesRegex(ValueError, _V1):
            sm80.validate_sm80_fp8kv()


if __name__ == "__main__":
    unittest.main()
