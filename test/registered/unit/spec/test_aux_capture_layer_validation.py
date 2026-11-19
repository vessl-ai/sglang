"""Unit tests for aux hidden-state capture-layer validation.

``_resolve_dflash_aux_hidden_state`` picks the target layers whose hidden states
feed a DFLASH/DSPARK draft. ``DFlashDraftConfig.resolve_target_layer_ids``
range-checks the ids it builds, but the DSPARK branch replaces them wholesale
with ``dspark_target_layer_ids`` and nothing re-checked the replacement.

Both failure modes this covers are silent -- the engine boots, serves and
answers either way. An out-of-range id makes the draft read a layer that is not
there; a tap on a linear-attention layer of a hybrid target hands the draft a
tensor it was not trained on and only costs accept rate.

Scope note: these exercise the validator itself. That the DSPARK branch actually
calls it is a wiring fact, covered by booting a cell with a deliberately wrong
draft config rather than here -- ``ModelConfig.from_server_args`` needs real
checkpoint files, which a CPU unit test has none of.
"""

import types
import unittest

from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    _validate_aux_capture_layers,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _plain_target():
    """A target that declares no layer_types -- every layer is full attention."""
    return types.SimpleNamespace()


def _hybrid_target(num_layers, full_attention_ids):
    full = set(full_attention_ids)
    return types.SimpleNamespace(
        layer_types=[
            "full_attention" if i in full else "linear_attention"
            for i in range(num_layers)
        ]
    )


def _validate(
    ids,
    num_layers,
    hf_text_config,
    source="dspark_target_layer_ids",
    declared=True,
):
    _validate_aux_capture_layers(
        target_layer_ids=list(ids),
        target_num_layers=num_layers,
        hf_text_config=hf_text_config,
        source=source,
        taps_are_declared=declared,
    )


class TestCaptureLayerRange(CustomTestCase):
    def test_in_range_ids_pass(self):
        _validate([0, 11, 47], 48, _plain_target())

    def test_empty_ids_are_not_this_validator_s_rule(self):
        # resolve_target_layer_ids already rejects an empty explicit list; this
        # validator must not invent a second, differently-worded rule for it.
        _validate([], 48, _plain_target())

    def test_id_equal_to_the_layer_count_is_out_of_range(self):
        with self.assertRaises(ValueError) as cm:
            _validate([0, 48], 48, _plain_target())
        msg = str(cm.exception)
        self.assertIn("out-of-range", msg)
        self.assertIn("target_layer_ids[1]=48", msg)
        self.assertIn("target_num_layers=48", msg)

    def test_negative_id_is_out_of_range(self):
        with self.assertRaises(ValueError):
            _validate([-1], 48, _plain_target())

    def test_message_names_where_the_ids_came_from(self):
        with self.assertRaises(ValueError) as cm:
            _validate([99], 48, _plain_target(), source="dspark_target_layer_ids")
        self.assertIn("dspark_target_layer_ids", str(cm.exception))

    def test_tap_count_semantics_in_the_draft_config(self):
        # A draft published with vLLM semantics carries the tap count where
        # SGLang reads the target's layer count. Solar-Pro-4 has 48 layers, so a
        # tap at 53 is what that mismatch looks like by the time it reaches here.
        with self.assertRaises(ValueError) as cm:
            _validate([5, 17, 29, 41, 53], 48, _plain_target())
        self.assertIn("target_layer_ids[4]=53", str(cm.exception))


class TestCaptureLayerAttentionKind(CustomTestCase):
    # 12 full-attention layers out of 48 -- the Solar-Open2 hybrid split.
    FULL = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)

    def test_taps_on_full_attention_layers_pass(self):
        _validate([11, 23, 35, 47], 48, _hybrid_target(48, self.FULL))

    def test_a_tap_on_a_linear_attention_layer_raises(self):
        with self.assertRaises(ValueError) as cm:
            _validate([11, 23, 35, 44], 48, _hybrid_target(48, self.FULL))
        msg = str(cm.exception)
        self.assertIn("not full attention", msg)
        self.assertIn("[44]", msg)
        self.assertIn("linear_attention", msg)
        # The message has to say where a tap may go, not only that it is wrong.
        self.assertIn("full-attention layers:", msg)

    def test_every_bad_tap_is_reported_not_just_the_first(self):
        with self.assertRaises(ValueError) as cm:
            _validate([2, 11, 44], 48, _hybrid_target(48, self.FULL))
        self.assertIn("[2, 44]", str(cm.exception))

    def test_rule_is_skipped_on_a_target_without_layer_types(self):
        _validate([11, 23, 35, 44], 48, _plain_target())

    def test_rule_is_skipped_when_layer_types_cannot_be_lined_up(self):
        # A layer_types whose length disagrees with the layer count cannot be
        # indexed by layer id, so the rule is unreadable rather than violated.
        mismatched = types.SimpleNamespace(layer_types=["full_attention"] * 4)
        _validate([11, 23, 35, 44], 48, mismatched)

    def test_empty_layer_types_is_skipped(self):
        _validate([11, 44], 48, types.SimpleNamespace(layer_types=[]))

    def test_range_rule_still_applies_on_a_hybrid_target(self):
        with self.assertRaises(ValueError) as cm:
            _validate([48], 48, _hybrid_target(48, self.FULL))
        self.assertIn("out-of-range", str(cm.exception))


class TestAutoBuiltTaps(CustomTestCase):
    """Ids from ``build_target_layer_ids`` are not a config author's choice.

    That builder spaces its picks evenly and knows nothing about hybrid layouts,
    so on a hybrid target it lands on linear-attention layers as a matter of
    course. Failing those would reject a decision nobody made, so the
    attention-kind rule is gated on the taps being declared.
    """

    FULL = TestCaptureLayerAttentionKind.FULL

    def test_attention_kind_rule_does_not_apply(self):
        # Evenly spaced picks, three of four on linear-attention layers.
        _validate(
            [9, 21, 33, 45],
            48,
            _hybrid_target(48, self.FULL),
            source="dflash_config.target_layer_ids",
            declared=False,
        )

    def test_range_rule_still_applies(self):
        with self.assertRaises(ValueError) as cm:
            _validate([48], 48, _hybrid_target(48, self.FULL), declared=False)
        self.assertIn("out-of-range", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
