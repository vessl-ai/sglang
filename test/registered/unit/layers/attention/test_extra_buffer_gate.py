from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.overrides import supports_mamba_cache_extra_buffer


def _view(**kwargs):
    base = {
        "linear_attn_backend": "triton",
        "linear_attn_prefill_backend": None,
        "linear_attn_decode_backend": None,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


class TestSupportsMambaCacheExtraBuffer(unittest.TestCase):
    """Who gets the extra_buffer strategy, and on what backend.

    Saying no here is not an error: the strategy falls back to ``no_buffer``,
    which force-disables the overlap scheduler and makes any ``page_size`` above
    1 fail a later assertion. So a wrong answer shows up as a capability the
    operator quietly does not have, or as a startup failure far from its cause.
    """

    def test_in_tree_arch_is_supported(self):
        self.assertTrue(
            supports_mamba_cache_extra_buffer(_view(), "SolarOpen2ForCausalLM")
        )

    def test_out_of_tree_arch_may_declare_support_on_its_spec(self):
        """``LinearAttnModelSpec`` advertises this as a registration point.

        A model that sets the field has to actually get the strategy, or the
        field is a promise the code does not keep. The registry is a plain list
        filled at import time and this lookup imports nothing, so the
        registration is done here explicitly -- which is also what makes it a
        model being served rather than merely existing.
        """
        from sglang.srt.configs.linear_attn_model_registry import (
            LinearAttnModelSpec,
            _LINEAR_ATTN_MODEL_REGISTRY,
            register_linear_attn_model,
        )

        arch = "OutOfTreeLinearAttnForCausalLM"
        register_linear_attn_model(
            LinearAttnModelSpec(
                config_class=type("OutOfTreeConfig", (), {}),
                backend_class_name=(
                    "sglang.srt.layers.attention.linear.kda_backend.KDAAttnBackend"
                ),
                arch_names=[arch],
                support_mamba_cache_extra_buffer=True,
            )
        )
        try:
            self.assertTrue(supports_mamba_cache_extra_buffer(_view(), arch))
        finally:
            _LINEAR_ATTN_MODEL_REGISTRY[:] = [
                spec
                for spec in _LINEAR_ATTN_MODEL_REGISTRY
                if arch not in spec.arch_names
            ]

    def test_unknown_arch_is_not_supported(self):
        self.assertFalse(
            supports_mamba_cache_extra_buffer(_view(), "NoSuchModelForCausalLM")
        )

    def test_non_triton_prefill_backend_is_not_supported(self):
        """The h[] snapshots come from the prefill kernel, so that is what counts.

        Checking only ``linear_attn_backend`` let
        ``--linear-attn-prefill-backend flashkda`` through, enabling extra_buffer
        against a kernel whose chunk layout the index math does not assume.
        """
        self.assertFalse(
            supports_mamba_cache_extra_buffer(
                _view(linear_attn_prefill_backend="flashkda"),
                "SolarOpen2ForCausalLM",
            )
        )

    def test_decode_backend_does_not_gate(self):
        """Decode is deliberately out of scope.

        h[] is written at extend time, and this runs before the per-phase
        backends are resolved -- so gating on decode would reject explicit
        configurations that work today while still missing the ones it aims at.
        """
        self.assertTrue(
            supports_mamba_cache_extra_buffer(
                _view(linear_attn_decode_backend="flashinfer"),
                "SolarOpen2ForCausalLM",
            )
        )

    def test_partial_view_without_per_phase_fields(self):
        """Callers build partial views; a missing override means "not set".

        Reading the per-phase fields directly turned those callers into
        AttributeError instead of a verdict.
        """
        self.assertFalse(
            supports_mamba_cache_extra_buffer(
                SimpleNamespace(linear_attn_backend="fla"), "Qwen3NextForCausalLM"
            )
        )
        self.assertTrue(
            supports_mamba_cache_extra_buffer(
                SimpleNamespace(linear_attn_backend="triton"),
                "Qwen3NextForCausalLM",
            )
        )


if __name__ == "__main__":
    unittest.main()
