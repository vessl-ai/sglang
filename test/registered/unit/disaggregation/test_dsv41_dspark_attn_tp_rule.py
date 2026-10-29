"""Unit tests for disaggregation/utils.dsv41_dspark_attn_tp_compatible.

The rule decides at the first bootstrap whether a DeepSeek-V4.1 DSpark PD pair may
transfer at all. Expected values are written from the rule itself (equal attention TP;
MLA CP prefill with TP x CP == decode TP; and, only behind the switch, a CP-1 prefill
with attention TP > 1 feeding an attention-TP-1 decode), never by calling it twice.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


import unittest

from sglang.srt.disaggregation.utils import dsv41_dspark_attn_tp_compatible
from sglang.test.test_utils import CustomTestCase


def _rule(p_tp, p_cp, d_tp, d_cp=1, mla=True, allow=False):
    return dsv41_dspark_attn_tp_compatible(
        prefill_attn_tp=p_tp,
        prefill_attn_cp=p_cp,
        decode_attn_tp=d_tp,
        decode_attn_cp=d_cp,
        mla_backend=mla,
        allow_tp_prefill_dp_decode=allow,
    )


class TestDsparkAttnTpRuleUnchanged(CustomTestCase):
    """With the switch off the rule is the one upstream #40323 ships."""

    def test_equal_attention_tp(self):
        self.assertTrue(_rule(4, 1, 4))  # e3: prefill TP4 -> decode TP4
        self.assertTrue(_rule(1, 4, 1))  # prefill CP4 -> decode DP4 (upstream B200 pair)
        self.assertTrue(_rule(1, 1, 1))  # prefill DP4 -> decode DP4

    def test_cp_prefill_world_matches_decode_tp(self):
        self.assertTrue(_rule(1, 4, 4))  # prefill CP4 -> decode TP4
        self.assertFalse(_rule(1, 4, 2))
        self.assertFalse(_rule(1, 4, 4, mla=False))

    def test_tp_prefill_to_dp_decode_is_refused_without_switch(self):
        self.assertFalse(_rule(4, 1, 1))
        self.assertFalse(_rule(2, 1, 1))


class TestDsparkAttnTpRuleSwitch(CustomTestCase):
    def test_tp_prefill_to_dp_decode_with_switch(self):
        self.assertTrue(_rule(4, 1, 1, allow=True))
        self.assertTrue(_rule(8, 1, 1, allow=True))

    def test_switch_does_not_widen_other_pairs(self):
        self.assertFalse(_rule(4, 1, 2, allow=True))  # decode attention TP must be 1
        self.assertFalse(_rule(4, 1, 1, mla=False, allow=True))  # MLA layout only
        self.assertFalse(_rule(4, 1, 1, d_cp=2, allow=True))  # decode CP must be 1
        self.assertFalse(_rule(1, 4, 2, allow=True))  # CP rule unchanged


if __name__ == "__main__":
    unittest.main()
