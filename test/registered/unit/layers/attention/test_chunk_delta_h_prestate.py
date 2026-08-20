from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

try:
    from sglang.kernels.ops.attention.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
    )

    _IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    chunk_gated_delta_rule_fwd_h = None
    _IMPORT_ERROR = e

CHUNK = 64


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestChunkDeltaHIsPreState(unittest.TestCase):
    """Pin down which chunk ``h[j]`` corresponds to.

    ``h`` is the intermediate-state buffer that the hybrid-SSM prefix-cache
    restore path reads from: ``_init_track_ssm_indices`` picks an entry out of it
    for rounds that do not land on a chunk boundary, and whatever it picks is
    written into the radix cache and later resumed from. Getting the index wrong
    by one chunk does not raise -- it resumes a request from 64 tokens earlier
    and answers slightly differently, with no error and no metric.

    The kernel stores ``h[i_t]`` at the top of its chunk loop, before chunk
    ``i_t`` is accumulated, so ``h[j]`` is the state *entering* chunk ``j``:
    the state after exactly ``j`` completed chunks, never including chunk ``j``
    itself. That convention is what makes ``offset + len // C`` the last
    completed boundary, and it is asserted here because it lives in Triton and
    is invisible to any reader of the consuming code.
    """

    def setUp(self):
        if chunk_gated_delta_rule_fwd_h is None:
            self.skipTest(f"kernel import failed: {_IMPORT_ERROR}")
        self.dev = "cuda"
        self.B, self.T = 1, 3 * CHUNK + 8  # 4 chunks, last one ragged
        self.Hg = self.H = 4
        self.K = self.V = 128

    def _inputs(self):
        g = torch.Generator(device=self.dev).manual_seed(1234)
        # Small magnitudes on purpose: the delta rule diverges for random inputs
        # at unit scale (h reaches ~1e7) and every comparison below becomes
        # meaningless without the recurrence staying contractive.
        k = (
            torch.randn(
                self.B, self.T, self.Hg, self.K, generator=g, device=self.dev,
                dtype=torch.bfloat16,
            )
            * 0.02
        )
        w = (
            torch.randn(
                self.B, self.T, self.H, self.K, generator=g, device=self.dev,
                dtype=torch.bfloat16,
            )
            * 0.02
        )
        u = (
            torch.randn(
                self.B, self.T, self.H, self.V, generator=g, device=self.dev,
                dtype=torch.bfloat16,
            )
            * 0.02
        )
        gk = torch.zeros(
            self.B, self.T, self.Hg, self.K, device=self.dev, dtype=torch.float32
        )
        return k, w, u, gk

    def _run(self, k, w, u, gk, initial_state):
        # The wrapper passes INPLACE_UPDATE=True, so the kernel writes the final
        # state back over initial_state. Two runs must not share one buffer, or
        # the second silently starts from the first one's output.
        state = initial_state.clone()
        indices = torch.zeros(self.B, device=self.dev, dtype=torch.int32)
        h, _ = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            gk=gk,
            use_exp2=True,
            initial_state=state,
            initial_state_indices=indices,
        )
        return h

    def test_h_entry_excludes_its_own_chunk(self):
        """Perturbing chunk ``j`` must leave ``h[0..j]`` untouched."""
        initial_state = (
            torch.randn(
                1, self.H, self.V, self.K, device=self.dev, dtype=torch.float32
            )
            * 0.02
        )

        base = self._inputs()
        pert = self._inputs()
        target = 2
        lo, hi = target * CHUNK, (target + 1) * CHUNK
        pert[0][:, lo:hi] += 0.5
        pert[2][:, lo:hi] += 0.5

        h_base = self._run(*base, initial_state)
        h_pert = self._run(*pert, initial_state)

        self.assertTrue(torch.isfinite(h_base).all(), "recurrence diverged")
        self.assertEqual(h_base.shape[1], (self.T + CHUNK - 1) // CHUNK)

        for j in range(target + 1):
            self.assertTrue(
                torch.equal(h_base[:, j], h_pert[:, j]),
                f"h[{j}] moved when only chunk {target} changed, so h[{j}] "
                f"includes chunk {j}. The restore path assumes it does not: "
                f"offset + len // C is meant to be the last COMPLETED chunk "
                f"boundary. If this fires, _init_track_ssm_indices needs the "
                f"index re-derived, not this assertion relaxed.",
            )

        self.assertFalse(
            torch.equal(h_base[:, target + 1], h_pert[:, target + 1]),
            f"h[{target + 1}] did not move when chunk {target} changed -- the "
            f"perturbation never reached the kernel, so this test proves nothing.",
        )

    def test_first_entry_is_the_incoming_state(self):
        """``h[0]`` is the state entering chunk 0, i.e. the initial state."""
        initial_state = (
            torch.randn(
                1, self.H, self.V, self.K, device=self.dev, dtype=torch.float32
            )
            * 0.02
        )
        h = self._run(*self._inputs(), initial_state)
        # h is bf16 while the state pool is fp32, so this is a downcast, not an
        # exact copy: compare at bf16 resolution rather than with torch.equal.
        torch.testing.assert_close(
            h[:, 0].float(),
            initial_state.float(),
            rtol=0,
            atol=torch.finfo(torch.bfloat16).eps * initial_state.abs().max().item(),
            msg="h[0] is not the incoming state, so h[j] is not a pre-state",
        )


if __name__ == "__main__":
    unittest.main()
