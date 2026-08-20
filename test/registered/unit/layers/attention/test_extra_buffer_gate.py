from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.arg_groups.overrides import supports_mamba_cache_extra_buffer
from sglang.srt.server_args import ServerArgs


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

        Reading the per-phase fields directly makes those callers an
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


class TestValidateMambaExtraBuffer(unittest.TestCase):
    """The page_size guard in ``ServerArgs._validate_mamba_extra_buffer``.

    It runs during ``__post_init__``, before ``_handle_page_size`` has defaulted
    ``page_size`` -- so it must not reach for anything that resolves page_size,
    and it must tolerate ``None``.
    """

    def _validate(self, page_size, model_chunk=None):
        args = ServerArgs.__new__(ServerArgs)
        hf_config = SimpleNamespace()
        if model_chunk is not None:
            hf_config.mamba_chunk_size = model_chunk
        view = SimpleNamespace(
            page_size=page_size,
            mamba_radix_cache_strategy="extra_buffer",
            speculative_num_draft_tokens=None,
            # unrelated pre-existing check asserts track_interval % page_size == 0
            mamba_track_interval=(page_size or 64) * 2,
            chunked_prefill_size=None,
            disaggregation_mode="null",
            speculative_algorithm=None,
        )
        # This validator asserts the platform (CUDA/MUSA/NPU/ROCm) before it
        # reaches the page_size ceiling, and this test runs on a CPU runner --
        # without the patch every case here would be catching that assert
        # instead, and the "refused" case would pass for the wrong reason.
        #
        # The checks that already lived here reach for the
        # mamba_cache_chunk_size property, which resolves a full ServerArgs, so
        # it is stubbed too -- with the real max(chunk, page_size) semantics
        # rather than a constant. A constant would let a tautological ceiling
        # (`page_size <= self.mamba_cache_chunk_size`, i.e. page <= max(chunk,
        # page), always true) pass this suite. The ceiling's own code
        # deliberately does not go through that property.
        with mock.patch(
            "sglang.srt.server_args.is_cuda", return_value=True
        ), mock.patch.object(
            ServerArgs,
            "get_model_config",
            lambda self: SimpleNamespace(hf_config=hf_config),
        ), mock.patch.object(
            ServerArgs,
            "mamba_cache_chunk_size",
            new_callable=mock.PropertyMock,
            side_effect=lambda: max(model_chunk or FLA_CHUNK_SIZE, page_size or 1),
        ), mock.patch(
            "sglang.srt.arg_groups.overrides.supports_mamba_cache_extra_buffer",
            return_value=True,
        ):
            ServerArgs._validate_mamba_extra_buffer(args, view, "SolarOpen2ForCausalLM")

    def test_unset_page_size_does_not_raise(self):
        """The default launch path: page_size is still None here."""
        self._validate(page_size=None)

    def test_page_size_within_the_kernel_chunk_is_accepted(self):
        self._validate(page_size=FLA_CHUNK_SIZE)

    def test_page_size_above_the_kernel_chunk_is_refused(self):
        with self.assertRaises(AssertionError):
            self._validate(page_size=FLA_CHUNK_SIZE * 2)

    def test_model_declared_chunk_raises_the_ceiling(self):
        """A Mamba2-family model chunks at its own config value, not the FLA one.

        Asserting FLA_CHUNK_SIZE for them would refuse configurations that serve
        today -- NemotronH, FalconH1 and GraniteMoeHybrid all ship a
        mamba_chunk_size well above 64.
        """
        self._validate(page_size=256, model_chunk=256)
        with self.assertRaises(AssertionError):
            self._validate(page_size=512, model_chunk=256)


if __name__ == "__main__":
    unittest.main()
