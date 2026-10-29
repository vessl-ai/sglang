"""DeepSeek-V4 MXFP4 experts on triton_kernels for prefill
(SGLANG_OPT_DSV4_TRITON_KERNELS_MOE): the host-sync-free expert routing against
bincount and a stable argsort, accuracy against a BF16-dequantized reference (next
to Marlin's) with masked (-1) slots and empty batches, and the refusal under CUDA
graph capture."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.utils.common import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm90_supported(),
    reason="Hopper triton_kernels MXFP4 experts",
)

HIDDEN, INTERMEDIATE = 5120, 576  # TP4 shard of 2304
SWIGLU_LIMIT = 10.0
NUM_EXPERTS, TOP_K = 384, 6
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def _weights(num_experts, seed=0):
    g = torch.Generator().manual_seed(seed)
    h, i = HIDDEN, INTERMEDIATE

    def nibbles(*shape):
        return torch.randint(-128, 128, shape, generator=g, dtype=torch.int8)

    def scales(*shape):
        return torch.randint(119, 123, shape, generator=g, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )

    return {
        "w13_weight": nibbles(num_experts, 2 * i, h // 2),
        "w2_weight": nibbles(num_experts, h, i // 2),
        "w13_weight_scale_inv": scales(num_experts, 2 * i, h // 32),
        "w2_weight_scale_inv": scales(num_experts, h, i // 32),
    }


def _layer(cpu_weights, backend, monkeypatch):
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    monkeypatch.setenv(
        "SGLANG_OPT_DSV4_TRITON_KERNELS_MOE",
        "1" if backend == "triton_kernels" else "0",
    )
    # The prefill path only; an exported decode or W4A8 switch would change it.
    monkeypatch.setenv("SGLANG_OPT_DSV4_TRITON_KERNELS_MOE_DECODE", "0")
    monkeypatch.setenv("SGLANG_OPT_DSV4_W4A8_PREFILL_MOE", "0")
    method = Mxfp4MarlinMoEMethod(
        SimpleNamespace(process_weights_after_loading=lambda layer: None), "test"
    )
    assert method._use_triton_kernels == (backend == "triton_kernels")
    method.hidden_pad = 0
    layer = torch.nn.Module()
    layer.hidden_size = HIDDEN
    layer.intermediate_size_per_partition = INTERMEDIATE
    layer.dispatcher = SimpleNamespace()
    for name, tensor in cpu_weights.items():
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
    return method, layer


def _dispatch(rows, seed, masked_rows=0):
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)
    logits = torch.randn(rows, NUM_EXPERTS, device="cuda", generator=g)
    weights, ids = torch.topk(torch.softmax(logits, -1), TOP_K, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).float()
    ids = ids.int()
    if masked_rows:
        # EP / DP-pad masking: whole rows and single slots.
        ids[-masked_rows:] = -1
        ids[::3, -1] = -1
    topk = StandardTopKOutput(topk_weights=weights, topk_ids=ids, router_logits=logits)
    return StandardDispatchOutput(
        hidden_states=x, hidden_states_scale=None, topk_output=topk
    )


def _dequant(weight, scale):
    lut = torch.tensor(E2M1, device="cuda")
    w = weight.cuda().view(torch.uint8).long()
    values = torch.stack([lut[w & 0xF], lut[w >> 4]], -1).flatten(-2)
    s = torch.exp2(scale.cuda().view(torch.uint8).float() - 127.0)
    return values * s.repeat_interleave(32, -1)


def _reference(dispatch, w13, w2):
    """FP32 math on BF16-exact dequantized experts ([w1; w3] gate/up order, clamped
    SwiGLU), rounding to bf16 where the engine does; -1 slots contribute nothing."""
    x = dispatch.hidden_states.float()
    ids = dispatch.topk_output.topk_ids
    weights = dispatch.topk_output.topk_weights
    out = torch.zeros_like(x)
    for e in torch.unique(ids).tolist():
        if e < 0:
            continue
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = (x[tok] @ w13[e].T).bfloat16().float()
        gate = h[:, :INTERMEDIATE].clamp(max=SWIGLU_LIMIT)
        up = h[:, INTERMEDIATE:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        act = (torch.nn.functional.silu(gate).bfloat16().float() * up).bfloat16()
        y = (act.float() @ w2[e].T) * weights[tok, slot, None]
        out.index_add_(0, tok, y.bfloat16().float())
    return out


def _rel(y, ref):
    return ((y.float() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize(
    "tokens,num_experts,top_k",
    [(1, 384, 6), (777, 384, 6), (4096, 384, 6), (513, 128, 3)],
)
def test_route_slots_by_expert_matches_bincount_and_stable_argsort(
    tokens, num_experts, top_k
):
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import route_slots_by_expert

    g = torch.Generator(device="cuda").manual_seed(tokens)
    ids = torch.randint(
        0, num_experts, (tokens, top_k), device="cuda", generator=g, dtype=torch.int32
    )
    ids[::7, 0] = -1
    ids[-1] = -1
    weights = torch.rand(tokens, top_k, device="cuda", generator=g)
    routing = route_slots_by_expert(ids, weights, num_experts)
    flat = torch.where(ids >= 0, ids, 0).flatten().long()
    ref_slots = torch.argsort(flat, stable=True)
    assert routing.expert_counts.dtype == torch.int32
    assert torch.equal(
        routing.expert_counts.long(), torch.bincount(flat, minlength=num_experts)
    )
    assert torch.equal(routing.combine_indx.long(), ref_slots)
    assert torch.equal(routing.gather_indx.long(), ref_slots // top_k)
    ref_w = torch.where(ids >= 0, weights, 0).flatten()[ref_slots]
    assert torch.equal(routing.sorted_weights, ref_w)


def test_route_slots_by_expert_has_no_host_sync():
    """Captured in a CUDA graph and replayed with new routing: a host-syncing
    histogram (torch.bincount) cannot be captured, and a replay must see the new ids."""
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import route_slots_by_expert

    tokens, num_experts, top_k = 300, 384, 6
    g = torch.Generator(device="cuda").manual_seed(7)

    def draw():
        ids = torch.randint(
            0, num_experts, (tokens, top_k), device="cuda", generator=g
        ).int()
        ids[::5, -1] = -1
        return ids, torch.rand(tokens, top_k, device="cuda", generator=g)

    ids, weights = draw()
    route_slots_by_expert(ids, weights, num_experts)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        routing = route_slots_by_expert(ids, weights, num_experts)
    new_ids, new_weights = draw()
    ids.copy_(new_ids)
    weights.copy_(new_weights)
    graph.replay()
    torch.cuda.synchronize()
    flat = torch.where(new_ids >= 0, new_ids, 0).flatten().long()
    ref_slots = torch.argsort(flat, stable=True)
    assert torch.equal(
        routing.expert_counts.long(), torch.bincount(flat, minlength=num_experts)
    )
    assert torch.equal(routing.combine_indx.long(), ref_slots)
    ref_w = torch.where(new_ids >= 0, new_weights, 0).flatten()[ref_slots]
    assert torch.equal(routing.sorted_weights, ref_w)
    del graph


def test_prefill_accuracy_vs_marlin(monkeypatch):
    cpu_weights = _weights(NUM_EXPERTS)
    w13 = _dequant(cpu_weights["w13_weight"], cpu_weights["w13_weight_scale_inv"])
    w2 = _dequant(cpu_weights["w2_weight"], cpu_weights["w2_weight_scale_inv"])
    tk, tk_layer = _layer(cpu_weights, "triton_kernels", monkeypatch)
    marlin, marlin_layer = _layer(cpu_weights, "marlin", monkeypatch)
    for rows in (1, 7, 1000, 4096):
        # Raw Marlin does not take masked (-1) slots, so compare on full routing.
        d = _dispatch(rows, seed=rows)
        ref = _reference(d, w13, w2)
        rel_tk = _rel(tk.apply(tk_layer, d).hidden_states, ref)
        rel_marlin = _rel(marlin.apply(marlin_layer, d).hidden_states, ref)
        print(
            f"rows={rows} rel_err triton_kernels={rel_tk:.5f} marlin={rel_marlin:.5f}"
        )
        assert rel_tk <= rel_marlin + 1e-3, (rows, rel_tk, rel_marlin)
        if rows > 1:
            masked = _dispatch(rows, seed=rows, masked_rows=1)
            out = tk.apply(tk_layer, masked).hidden_states
            assert torch.equal(out[-1], torch.zeros_like(out[-1]))
            rel_masked = _rel(out, _reference(masked, w13, w2))
            assert rel_masked <= rel_marlin + 1e-3, (rows, rel_masked, rel_marlin)


def test_zero_tokens_and_cuda_graph_capture(monkeypatch):
    tk, layer = _layer(_weights(NUM_EXPERTS), "triton_kernels", monkeypatch)
    empty = _dispatch(0, seed=0)
    assert tk.apply(layer, empty).hidden_states.shape == (0, HIDDEN)
    static = _dispatch(48, seed=1)
    tk.apply(layer, static)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="does not support CUDA graphs"):
        with torch.cuda.graph(graph):
            tk.apply(layer, static)
