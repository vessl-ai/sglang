"""DeepSeek-V4 MXFP4 experts on triton_kernels with decode tiles
(SGLANG_OPT_DSV4_TRITON_KERNELS_MOE_DECODE): accuracy against a BF16-dequantized
reference (next to Marlin's), CUDA graph capture/replay at every speculative decode
capture size, and the one-launch expert sort against a stable argsort."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.utils.common import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm90_supported(),
    reason="Hopper triton_kernels MXFP4 experts",
)

HIDDEN, INTERMEDIATE = 5120, 576  # TP4 shard of 2304
SWIGLU_LIMIT = 10.0
# (experts, top_k, rows per request): target layers verify 6 rows per request,
# DSpark draft stages run 5 rows over 128 experts.
SHAPES = {"target": (384, 6, 6), "draft": (128, 3, 5)}
# Speculative decode capture sizes up to --cuda-graph-max-bs-decode=128.
CAPTURE_BS = (
    list(range(1, 9))
    + list(range(10, 33, 2))
    + list(range(40, 65, 4))
    + list(range(72, 129, 8))
)
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


def _layer(cpu_weights, num_experts, top_k, backend, monkeypatch):
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    monkeypatch.setenv(
        "SGLANG_OPT_DSV4_TRITON_KERNELS_MOE_DECODE",
        "1" if backend == "triton_kernels_decode" else "0",
    )
    monkeypatch.setenv("SGLANG_OPT_DSV4_TRITON_KERNELS_MOE", "0")
    method = Mxfp4MarlinMoEMethod(
        SimpleNamespace(process_weights_after_loading=lambda layer: None), "test"
    )
    assert method._use_triton_kernels == (backend == "triton_kernels_decode")
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
        num_experts=num_experts,
        num_local_experts=num_experts,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=INTERMEDIATE,
        top_k=top_k,
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


def _dispatch(rows, num_experts, top_k, seed, masked_rows=0):
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)
    logits = torch.randn(rows, num_experts, device="cuda", generator=g)
    weights, ids = torch.topk(torch.softmax(logits, -1), top_k, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).float()
    ids = ids.int()
    if masked_rows:
        ids[-masked_rows:, -1] = -1
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
    """FP32 math on BF16-exact dequantized experts, rounding to bf16 where the
    engine does (gate/up output, SiLU, activation, per-expert output)."""
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


@pytest.mark.parametrize("shape", list(SHAPES))
def test_decode_tiles_accuracy_vs_marlin(shape, monkeypatch):
    num_experts, top_k, rows_per_req = SHAPES[shape]
    cpu_weights = _weights(num_experts)
    w13 = _dequant(cpu_weights["w13_weight"], cpu_weights["w13_weight_scale_inv"])
    w2 = _dequant(cpu_weights["w2_weight"], cpu_weights["w2_weight_scale_inv"])
    tk, tk_layer = _layer(
        cpu_weights, num_experts, top_k, "triton_kernels_decode", monkeypatch
    )
    marlin, marlin_layer = _layer(
        cpu_weights, num_experts, top_k, "marlin", monkeypatch
    )
    for bs in (1, 48, 64, 96, 128):
        # Raw Marlin does not take masked (-1) slots, so compare on full routing.
        d = _dispatch(bs * rows_per_req, num_experts, top_k, seed=bs)
        ref = _reference(d, w13, w2)
        rel_tk = _rel(tk.apply(tk_layer, d).hidden_states, ref)
        rel_marlin = _rel(marlin.apply(marlin_layer, d).hidden_states, ref)
        masked = _dispatch(
            bs * rows_per_req, num_experts, top_k, seed=bs, masked_rows=1
        )
        rel_masked = _rel(
            tk.apply(tk_layer, masked).hidden_states, _reference(masked, w13, w2)
        )
        print(
            f"{shape} bs={bs} rel_err triton_kernels={rel_tk:.5f} marlin={rel_marlin:.5f} "
            f"triton_kernels_masked={rel_masked:.5f}"
        )
        assert rel_tk <= rel_marlin + 1e-3, (bs, rel_tk, rel_marlin)
        assert rel_masked <= rel_marlin + 1e-3, (bs, rel_masked, rel_marlin)


@pytest.mark.parametrize("shape", list(SHAPES))
def test_decode_tiles_cuda_graph_replay(shape, monkeypatch):
    num_experts, top_k, rows_per_req = SHAPES[shape]
    cpu_weights = _weights(num_experts)
    w13 = _dequant(cpu_weights["w13_weight"], cpu_weights["w13_weight_scale_inv"])
    w2 = _dequant(cpu_weights["w2_weight"], cpu_weights["w2_weight_scale_inv"])
    tk, layer = _layer(
        cpu_weights, num_experts, top_k, "triton_kernels_decode", monkeypatch
    )
    for bs in CAPTURE_BS:
        rows = bs * rows_per_req
        static = _dispatch(rows, num_experts, top_k, seed=1000 + bs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            tk.apply(layer, static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = tk.apply(layer, static).hidden_states
        # Replay with new activations and new routing (one masked slot).
        fresh = _dispatch(rows, num_experts, top_k, seed=2000 + bs, masked_rows=1)
        static.hidden_states.copy_(fresh.hidden_states)
        static.topk_output.topk_ids.copy_(fresh.topk_output.topk_ids)
        static.topk_output.topk_weights.copy_(fresh.topk_output.topk_weights)
        graph.replay()
        torch.cuda.synchronize()
        eager = tk.apply(layer, fresh).hidden_states
        assert torch.equal(out, eager), bs
        rel = _rel(out, _reference(fresh, w13, w2))
        assert rel < 1e-2, (bs, rel)
        del graph


@pytest.mark.parametrize(
    "shape,tokens,decode_tiles",
    [
        # 16 routed rows per expert on average is the last decode-tile batch.
        ("target", 16 * 384 // 6, True),
        ("target", 16 * 384 // 6 + 1, False),
        ("draft", 16 * 128 // 3, True),
        ("draft", 16 * 128 // 3 + 1, False),
    ],
)
def test_decode_switch_large_batch_fallback_cuda_graph(
    shape, tokens, decode_tiles, monkeypatch
):
    """Past 16 routed rows per expert the decode switch falls back to the argsort
    routing and triton_kernels' own tiles (4 warps, fixed by the swizzle). Both
    sides of the boundary capture and replay in a CUDA graph and match eager."""
    num_experts, top_k, _ = SHAPES[shape]
    cpu_weights = _weights(num_experts)
    w13 = _dequant(cpu_weights["w13_weight"], cpu_weights["w13_weight_scale_inv"])
    w2 = _dequant(cpu_weights["w2_weight"], cpu_weights["w2_weight_scale_inv"])
    tk, layer = _layer(
        cpu_weights, num_experts, top_k, "triton_kernels_decode", monkeypatch
    )
    assert tk._uses_decode_tiles(tokens * top_k) == decode_tiles
    static = _dispatch(tokens, num_experts, top_k, seed=tokens)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        tk.apply(layer, static)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = tk.apply(layer, static).hidden_states
    fresh = _dispatch(tokens, num_experts, top_k, seed=tokens + 1, masked_rows=3)
    static.hidden_states.copy_(fresh.hidden_states)
    static.topk_output.topk_ids.copy_(fresh.topk_output.topk_ids)
    static.topk_output.topk_weights.copy_(fresh.topk_output.topk_weights)
    graph.replay()
    torch.cuda.synchronize()
    eager = tk.apply(layer, fresh).hidden_states
    assert torch.equal(out, eager)
    rel = _rel(out, _reference(fresh, w13, w2))
    assert rel < 1e-2, rel
    del graph


@pytest.mark.parametrize(
    "num_slots,num_experts,top_k",
    [(6, 384, 6), (2304, 384, 6), (4608, 384, 6), (640, 128, 3), (3000, 128, 3)],
)
def test_expert_sort_small_matches_stable_argsort(num_slots, num_experts, top_k):
    from sglang.kernels.ops.moe.expert_sort_small import expert_sort_small

    tokens = num_slots // top_k
    g = torch.Generator(device="cuda").manual_seed(num_slots)
    ids = torch.randint(
        0, num_experts, (tokens, top_k), device="cuda", generator=g, dtype=torch.int32
    )
    ids[::7, 0] = -1
    weights = torch.rand(tokens, top_k, device="cuda", generator=g)
    counts, slots, toks, sorted_w = expert_sort_small(ids, weights, num_experts)
    flat = torch.where(ids >= 0, ids, 0).flatten().long()
    ref_slots = torch.argsort(flat, stable=True)
    assert torch.equal(counts.long(), torch.bincount(flat, minlength=num_experts))
    assert torch.equal(slots.long(), ref_slots)
    assert torch.equal(toks.long(), ref_slots // top_k)
    ref_w = torch.where(ids >= 0, weights, 0).flatten()[ref_slots]
    assert torch.equal(sorted_w, ref_w)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
