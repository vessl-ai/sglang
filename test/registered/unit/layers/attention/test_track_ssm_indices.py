from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import types
import unittest
from unittest import mock

import torch

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)


def _make_backend():
    """A backend instance carrying only what _init_track_ssm_indices reads.

    The method is pure CPU tensor arithmetic -- it touches ``self.device`` and
    its own class (to tell Mamba2 from the FLA-convention backends) and nothing
    else -- so it can be exercised without a GPU, a model, or a kernel. Building
    the object this way keeps the test on the arithmetic rather than on a
    backend's constructor.
    """

    class _FlaConventionBackend(MambaAttnBackendBase):
        pass

    backend = object.__new__(_FlaConventionBackend)
    backend.device = torch.device("cpu")
    return backend


def _forward_batch(extend_seq_lens, prefix_lens, track_seqlens, track_mask):
    return types.SimpleNamespace(
        extend_seq_lens=torch.tensor(extend_seq_lens, dtype=torch.int64),
        extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int64),
        mamba_track_seqlens=torch.tensor(track_seqlens, dtype=torch.int64),
        mamba_track_mask=torch.tensor(track_mask, dtype=torch.bool),
        mamba_track_indices=torch.arange(len(extend_seq_lens), dtype=torch.int64),
    )


def _call(backend, forward_batch, chunk_size=CHUNK_SIZE):
    cache_indices = torch.arange(len(forward_batch.extend_seq_lens), dtype=torch.int64)
    server_args = types.SimpleNamespace(mamba_cache_chunk_size=chunk_size)
    with mock.patch(
        "sglang.srt.layers.attention.hybrid_linear_attn_backend.get_server_args",
        return_value=server_args,
    ):
        return backend._init_track_ssm_indices(cache_indices, forward_batch)


class TestTrackSsmIndices(unittest.TestCase):
    """Pin the h[] index arithmetic in ``_init_track_ssm_indices``.

    This is where a prefix-cache restore picks which intermediate state a
    sequence resumes from. Picking the wrong one does not raise: the state goes
    into the radix cache, a later request resumes from it, and the model answers
    slightly differently with no error and no metric.

    Two facts fix the correct index for the FLA-convention backends (KDA, GDN,
    and anything else that is not Mamba2). ``h[j]`` is the state *entering*
    chunk ``j`` -- the kernel stores it at the top of its chunk loop, before that
    chunk is accumulated, which
    ``test_chunk_delta_h_prestate.py`` asserts directly. And ``num_h_states`` is
    ``ceil(len / C)`` while the h branch is taken only when ``len % C != 0``, so
    ``len // C == num_h_states - 1`` exactly: the last entry of the sequence's
    own block. Both halves are needed, and neither is visible from the
    expression itself.
    """

    def test_source_is_last_entry_of_own_block(self):
        # Three sequences, all unaligned so all take the h branch.
        lens = [200, 100, 328]  # 3*64+8, 1*64+36, 5*64+8
        fb = _forward_batch(
            extend_seq_lens=lens,
            prefix_lens=[0, 0, 0],
            track_seqlens=lens,
            track_mask=[True, True, True],
        )
        h_src, _, _, _ = _call(_make_backend(), fb)

        num_h = torch.tensor([(n - 1) // CHUNK_SIZE + 1 for n in lens])
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), num_h.cumsum(0)[:-1]])
        expected = offsets + num_h - 1

        self.assertTrue(
            torch.equal(h_src, expected),
            f"track_ssm_h_src={h_src.tolist()} but the last completed chunk "
            f"boundary of each sequence is {expected.tolist()}. h[j] is the "
            f"state entering chunk j, so the source is offset + num_h_states - 1 "
            f"(== offset + len // C for an unaligned length). Re-derive the "
            f"index against the kernel rather than adjusting this expectation.",
        )

    def test_source_stays_inside_the_sequences_own_block(self):
        """The failure this guards is reading a *neighbour's* state.

        ``h`` is one flat batch-wide buffer and the gather is a plain advanced
        index, so an index outside a sequence's own window silently returns
        another sequence's recurrent state -- including negative indices, which
        Python wraps to the end of the buffer rather than raising.
        """
        # Deliberately includes a sequence shorter than one chunk, which is the
        # case that turns an over-eager "- 1" into a negative index.
        lens = [70, 40, 200]
        fb = _forward_batch(
            extend_seq_lens=lens,
            prefix_lens=[0, 0, 0],
            track_seqlens=lens,
            track_mask=[True, True, True],
        )
        h_src, _, _, _ = _call(_make_backend(), fb)

        num_h = torch.tensor([(n - 1) // CHUNK_SIZE + 1 for n in lens])
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), num_h.cumsum(0)[:-1]])
        # Select windows with the same predicate production uses, not by row
        # order. Slicing the first len(h_src) rows happens to line up only while
        # every sequence is both unaligned and tracked; add an aligned or
        # masked-out one and the windows would silently pair with the wrong
        # sequences while the containment check kept passing.
        rows = (torch.tensor(lens) % CHUNK_SIZE) != 0
        lo = offsets[rows]
        hi = lo + num_h[rows]

        self.assertTrue(
            bool(((h_src >= lo) & (h_src < hi)).all()),
            f"track_ssm_h_src={h_src.tolist()} escapes its own h[] window "
            f"(lo={lo.tolist()}, hi={hi.tolist()}). Out-of-window is not an "
            f"error at the gather -- it reads a neighbouring sequence's state.",
        )

    def test_force_track_h_lands_on_the_requested_boundary(self):
        """The aligned+1 trick only works under the pre-state convention.

        When the tracked position is already chunk-aligned, the scheduler asks
        for ``aligned + 1`` so the round takes the h branch at all
        (``_force_track_h``). Floor division then yields the number of completed
        chunks, which under the pre-state convention is exactly the entry
        holding the state at that boundary. Any shift breaks the pairing between
        what is cached and what it is labelled as.
        """
        k = 3
        aligned = k * CHUNK_SIZE
        fb = _forward_batch(
            extend_seq_lens=[aligned + 40],
            prefix_lens=[0],
            track_seqlens=[aligned + 1],  # what _force_track_h passes
            track_mask=[True],
        )
        h_src, _, _, _ = _call(_make_backend(), fb)

        self.assertTrue(
            torch.equal(h_src, torch.tensor([k])),
            f"track_ssm_h_src={h_src.tolist()}, expected [{k}] -- the state "
            f"after {k} completed chunks, which is the boundary the scheduler "
            f"asked to cache.",
        )

    def test_aligned_rounds_do_not_use_h(self):
        """Aligned lengths take the ssm_states branch, not the h branch."""
        fb = _forward_batch(
            extend_seq_lens=[2 * CHUNK_SIZE, 130],
            prefix_lens=[0, 0],
            track_seqlens=[2 * CHUNK_SIZE, 130],
            track_mask=[True, True],
        )
        h_src, _, final_src, _ = _call(_make_backend(), fb)

        self.assertEqual(
            h_src.numel(), 1, "only the unaligned sequence belongs in the h branch"
        )
        self.assertEqual(
            final_src.numel(),
            1,
            "the aligned sequence restores its live state, not an h entry",
        )

    def test_untracked_sequences_still_consume_their_offset(self):
        """A masked-out sequence must still occupy its slice of ``h``.

        The offsets are a cumulative sum over every sequence in the batch, taken
        before the track mask is applied, because the kernel writes an h block
        for each sequence whether or not the scheduler wants a snapshot of it.
        If the cumsum were ever taken after masking, every sequence downstream
        of an untracked one would read a neighbour's block -- the same failure
        as an off-by-one index, reached a different way.
        """
        lens = [200, 100, 328]
        fb = _forward_batch(
            extend_seq_lens=lens,
            prefix_lens=[0, 0, 0],
            track_seqlens=lens,
            track_mask=[True, False, True],
        )
        h_src, _, _, _ = _call(_make_backend(), fb)

        num_h = torch.tensor([(n - 1) // CHUNK_SIZE + 1 for n in lens])
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64), num_h.cumsum(0)[:-1]])
        expected = (offsets + num_h - 1)[torch.tensor([True, False, True])]

        self.assertTrue(
            torch.equal(h_src, expected),
            f"track_ssm_h_src={h_src.tolist()}, expected {expected.tolist()}. "
            f"Sequence 1 is untracked but still owns h entries; skipping its "
            f"block would shift sequence 2 onto sequence 1's state.",
        )

    def test_prefix_length_does_not_shift_the_index(self):
        """``num_h_states`` comes from the extend length, the index from the tracked one.

        Those are two different quantities and a nonzero prefix is what separates
        them: the h block is sized by how many tokens this round computes, while
        the entry within it is chosen by how far into the sequence the snapshot
        should sit. Every other case here runs with an empty prefix, which hides
        a swap between the two.
        """
        prefix, extend = 512, 200
        fb = _forward_batch(
            extend_seq_lens=[extend],
            prefix_lens=[prefix],
            track_seqlens=[prefix + extend],
            track_mask=[True],
        )
        h_src, _, _, _ = _call(_make_backend(), fb)

        expected = extend // CHUNK_SIZE  # offset is 0 for the first sequence
        self.assertTrue(
            torch.equal(h_src, torch.tensor([expected])),
            f"track_ssm_h_src={h_src.tolist()}, expected [{expected}] -- the "
            f"prefix is already in the cache, so only the {extend} tokens this "
            f"round computed have h entries.",
        )


if __name__ == "__main__":
    unittest.main()
