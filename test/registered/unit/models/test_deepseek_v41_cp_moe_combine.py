"""The V4.1 prefill-CP MoE exit, rank by rank on CPU.

Each CP rank gathers the batch rank-major, runs its TP shard of the experts and
holds a partial sum of every row. The exit must hand rank r the complete sum of
its own shard (chunk r). The all-reduce + split and the reduce-scatter exits are
run against simulated collectives over the same per-rank partials.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.layer_boundary import ops as boundary_ops
from sglang.srt.models import deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

CP_SIZE = 4
ROWS_PER_RANK = 3
HIDDEN = 5


class _Collectives:
    """The partials of every rank; each collective answers for ``rank``."""

    def __init__(self, partials):
        self.partials = partials
        self.total = torch.stack(partials).sum(dim=0)
        self.calls = []

    def all_reduce(self, tensor):
        self.calls.append("all_reduce")
        return self.total.clone()

    def reduce_scatter_tensor(self, rank, output, tensor):
        self.calls.append("reduce_scatter")
        output.copy_(self.total.tensor_split(CP_SIZE)[rank])


@contextlib.contextmanager
def _as_rank(rank, collectives, *, tp_size, attn_cp_size):
    parallel = SimpleNamespace(
        tp_size=tp_size,
        attn_cp_size=attn_cp_size,
        attn_cp_rank=rank,
        tp_group=SimpleNamespace(all_reduce=collectives.all_reduce),
    )
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(deepseek_v4, "get_parallel", return_value=parallel)
        )
        stack.enter_context(
            mock.patch.object(boundary_ops, "get_parallel", return_value=parallel)
        )
        stack.enter_context(
            mock.patch.object(
                boundary_ops,
                "attn_cp_reduce_scatter_tensor",
                lambda output, tensor: collectives.reduce_scatter_tensor(
                    rank, output, tensor
                ),
            )
        )
        yield


class TestDeepseekV41CpMoeCombine(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        # Small integers: every summation order gives the same value exactly.
        self.partials = [
            torch.randint(
                -8, 8, (CP_SIZE * ROWS_PER_RANK, HIDDEN), generator=generator
            ).float()
            for _ in range(CP_SIZE)
        ]

    def _exit(self, rank, *, reduce_scatter, tp_size=CP_SIZE, attn_cp_size=CP_SIZE):
        collectives = _Collectives(self.partials)
        with (
            envs.SGLANG_DSV41_CP_MOE_REDUCE_SCATTER.override(reduce_scatter),
            _as_rank(rank, collectives, tp_size=tp_size, attn_cp_size=attn_cp_size),
        ):
            shard = deepseek_v4._cp_moe_return_shard(
                self.partials[rank].clone(), dsv41=True
            )
        return shard, collectives

    def test_both_exits_return_the_complete_sum_of_the_own_shard(self):
        total = torch.stack(self.partials).sum(dim=0)
        for rank in range(CP_SIZE):
            expected = total.tensor_split(CP_SIZE)[rank]
            with self.subTest(rank=rank):
                all_reduced, ar = self._exit(rank, reduce_scatter=False)
                scattered, rs = self._exit(rank, reduce_scatter=True)
                self.assertEqual(ar.calls, ["all_reduce"])
                self.assertEqual(rs.calls, ["reduce_scatter"])
                self.assertEqual(scattered.shape, (ROWS_PER_RANK, HIDDEN))
                torch.testing.assert_close(all_reduced, expected, rtol=0, atol=0)
                torch.testing.assert_close(scattered, expected, rtol=0, atol=0)

    def test_attention_tp_above_one_keeps_the_all_reduce(self):
        # TP 8 with CP 4: the CP group holds half of each row's expert partials.
        _, collectives = self._exit(0, reduce_scatter=True, tp_size=8)
        self.assertEqual(collectives.calls, ["all_reduce"])

    def test_decision(self):
        cases = {
            # (env, tp, attn cp) -> reduce-scatter
            (False, 4, 4): False,
            (True, 4, 4): True,
            (True, 8, 4): False,
            (True, 8, 8): True,
        }
        for (env, tp_size, attn_cp_size), expected in cases.items():
            parallel = SimpleNamespace(tp_size=tp_size, attn_cp_size=attn_cp_size)
            with (
                self.subTest(env=env, tp=tp_size, cp=attn_cp_size),
                envs.SGLANG_DSV41_CP_MOE_REDUCE_SCATTER.override(env),
                mock.patch.object(deepseek_v4, "get_parallel", return_value=parallel),
            ):
                self.assertEqual(
                    deepseek_v4._cp_moe_combine_reduce_scatters(), expected
                )


if __name__ == "__main__":
    unittest.main()
