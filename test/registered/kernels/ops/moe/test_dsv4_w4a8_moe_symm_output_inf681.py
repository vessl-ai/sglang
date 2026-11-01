"""INF-681 SGLANG_OPT_DSV4_DP_SYMM_MOE_OUTPUT on the SM90 W4A8 decode MoE (#104, and its v2 sub-switch).

With the switch on, Mxfp4MarlinMoEMethod.apply allocates the routed-expert output inside
use_symmetric_memory(tp_group, disabled=not is_allocation_symmetric()), the kernel writes into that
buffer, and the result is bit-identical to the switch off. In the engine that context is the TP
group's NCCL symmetric pool (ncclMemAlloc + window registration), which needs a real TP group; here a
private torch MemPool stands in for it, so the test can check that the returned buffer comes from the
context's pool. Whether the pool grows the same way on every rank is a 4-rank engine question (the
DP4 decoder pre-check), not one this test can answer."""

import contextlib
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.utils.common import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm90_supported(),
    reason="SM90 wgmma W4A8 decode MoE",
)

HIDDEN, INTERMEDIATE = 5120, 576
SWIGLU_LIMIT = 10.0
NUM_EXPERTS, TOP_K, ROWS_PER_REQ = 384, 6, 4  # target layer, DSpark gamma 3 verify rows
TP_GROUP = "tp-group-sentinel"


def _weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    h, i, e = HIDDEN, INTERMEDIATE, NUM_EXPERTS

    def nibbles(*shape):
        return torch.randint(-128, 128, shape, generator=g, dtype=torch.int8)

    def scales(*shape):
        return torch.randint(119, 123, shape, generator=g, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )

    return {
        "w13_weight": nibbles(e, 2 * i, h // 2),
        "w2_weight": nibbles(e, h, i // 2),
        "w13_weight_scale_inv": scales(e, 2 * i, h // 32),
        "w2_weight_scale_inv": scales(e, h, i // 32),
    }


def _routing(bs, seed, masked_rows=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = bs * ROWS_PER_REQ
    req = torch.randn(bs, NUM_EXPERTS, device="cuda", generator=g).repeat_interleave(
        ROWS_PER_REQ, 0
    )
    logits = req + 0.5 * torch.randn(rows, NUM_EXPERTS, device="cuda", generator=g)
    w, ids = torch.topk(torch.softmax(logits, -1), TOP_K, dim=-1)
    ids = ids.int()
    if masked_rows:
        ids[-masked_rows:] = -1  # dp-attention pad rows
    x = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)
    return x, ids, (w / w.sum(-1, keepdim=True)).float()


def _method():
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    method = Mxfp4MarlinMoEMethod(
        SimpleNamespace(process_weights_after_loading=lambda layer: None), "test"
    )
    method.hidden_pad = 0
    layer = torch.nn.Module()
    layer.hidden_size = HIDDEN
    layer.intermediate_size_per_partition = INTERMEDIATE
    layer.dispatcher = SimpleNamespace()
    for name, tensor in _weights().items():
        layer.register_parameter(
            name, torch.nn.Parameter(tensor.cuda().clone(), requires_grad=False)
        )
    config = MoeRunnerConfig(
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTERMEDIATE,
        top_k=TOP_K,
        params_dtype=torch.bfloat16,
        activation="silu",
        is_gated=True,
        swiglu_limit=SWIGLU_LIMIT,
        gate_up_interleaved=False,
        inplace=False,
        layer=layer,
    )
    layer.moe_runner_config = config
    method.create_moe_runner(layer, config)
    method.process_weights_after_loading(layer)
    assert layer._dsv4_mxfp4_backend == "sm90_decode_w4a8"
    return method, layer


class _FakeSymmetricMemory:
    """Stands in for use_symmetric_memory: records every call and, when not disabled, routes the
    allocations made inside the context to a private MemPool (as the real context routes them to the
    NCCL pool)."""

    def __init__(self):
        self.pool = torch.cuda.MemPool()
        self.calls = []

    def __call__(self, group, disabled=False):
        self.calls.append((group, disabled))
        if disabled:
            return contextlib.nullcontext()
        return torch.cuda.use_mem_pool(self.pool)

    def owns(self, t):
        ptr = t.data_ptr()
        return any(
            seg["address"] <= ptr < seg["address"] + seg["total_size"]
            for seg in self.pool.snapshot()
        )


@pytest.mark.parametrize("v2", [False, True], ids=["w4a8", "v2"])
@pytest.mark.parametrize("symmetric", [True, False], ids=["max_len", "not_symmetric"])
def test_output_from_symmetric_pool(v2, symmetric, monkeypatch):
    import sglang.srt.distributed.device_communicators.pynccl_allocator as pynccl_allocator
    import sglang.srt.layers.dp_attention as dp_attention
    import sglang.srt.runtime_context as runtime_context
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2, dsv4_mxfp4_decode_moe_ws
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    monkeypatch.setenv("SGLANG_OPT_DSV4_SM90_DECODE_MOE_W4A8", "1")
    monkeypatch.setenv("SGLANG_OPT_DSV4_SM90_DECODE_MOE_W4A8_V2", "1" if v2 else "0")
    method, layer = _method()
    assert method._sm90_decode_w4a8_v2 == v2

    fake = _FakeSymmetricMemory()
    monkeypatch.setattr(pynccl_allocator, "use_symmetric_memory", fake)
    monkeypatch.setattr(dp_attention, "is_allocation_symmetric", lambda: symmetric)
    monkeypatch.setattr(
        runtime_context, "get_parallel", lambda: SimpleNamespace(tp_group=TP_GROUP)
    )
    kernel_module = dsv4_mxfp4_decode_moe_v2 if v2 else dsv4_mxfp4_decode_moe_ws
    other_module = dsv4_mxfp4_decode_moe_ws if v2 else dsv4_mxfp4_decode_moe_v2
    seen = []
    real = kernel_module.decode_moe

    def spy(*args, **kwargs):
        seen.append(kwargs.get("output"))
        return real(*args, **kwargs)

    def must_not_run(*args, **kwargs):
        raise AssertionError("the other W4A8 decode MoE kernel was called")

    monkeypatch.setattr(kernel_module, "decode_moe", spy)
    monkeypatch.setattr(other_module, "decode_moe", must_not_run)

    x, ids, wts = _routing(64, seed=5, masked_rows=7)
    dispatch = StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(topk_weights=wts, topk_ids=ids, router_logits=None),
    )
    with envs.SGLANG_OPT_DSV4_DP_SYMM_MOE_OUTPUT.override(True):
        out = method.apply(layer, dispatch).hidden_states
    assert fake.calls == [(TP_GROUP, not symmetric)]
    assert len(seen) == 1 and seen[0] is not None
    assert seen[0].data_ptr() == out.data_ptr()
    assert out.shape == x.shape and out.dtype == torch.bfloat16
    assert fake.owns(out) == symmetric

    with envs.SGLANG_OPT_DSV4_DP_SYMM_MOE_OUTPUT.override(False):
        ref = method.apply(layer, dispatch).hidden_states
    assert len(fake.calls) == 1, "switch off must not enter the symmetric context"
    assert not fake.owns(ref)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
