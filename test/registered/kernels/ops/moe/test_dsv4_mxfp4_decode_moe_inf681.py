"""INF-681 decode MoE levers for the DeepSeek-V4 TP4 MXFP4 experts:
- SGLANG_OPT_DSV4_DECODE_MOE_ROWS_AWARE_TILES: triton_kernels row tile chosen from the routed rows
  per expert. Must give the same output bit for bit as the 16-row decode tiles.
- SGLANG_OPT_DSV4_SM90_DECODE_MOE_KERNEL: dsv4_mxfp4_decode_moe CUDA kernels. Exact dequantization
  (one-hot down rows reproduce the weights), same output as the triton_kernels decode path, masked
  slots, rows beyond one 32-row pass, and CUDA graph capture/replay with fresh inputs."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.utils.common import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm90_supported(),
    reason="SM90 MXFP4 decode MoE",
)

HIDDEN, INTERMEDIATE = 5120, 576  # TP4 shard of 2304
SWIGLU_LIMIT = 10.0
SHAPES = {"target": (384, 6, 6), "draft": (128, 3, 5)}  # (experts, top_k, rows per request)
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


def _layer(cpu_weights, num_experts, top_k, mode, monkeypatch):
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    monkeypatch.setenv("SGLANG_OPT_DSV4_TRITON_KERNELS_MOE", "0")
    monkeypatch.setenv(
        "SGLANG_OPT_DSV4_TRITON_KERNELS_MOE_DECODE", "1" if mode in ("tk", "rows") else "0"
    )
    monkeypatch.setenv(
        "SGLANG_OPT_DSV4_DECODE_MOE_ROWS_AWARE_TILES", "1" if mode == "rows" else "0"
    )
    monkeypatch.setenv(
        "SGLANG_OPT_DSV4_SM90_DECODE_MOE_KERNEL", "1" if mode == "sm90" else "0"
    )
    method = Mxfp4MarlinMoEMethod(
        SimpleNamespace(process_weights_after_loading=lambda layer: None), "test"
    )
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
    expected = "sm90_decode" if mode == "sm90" else "triton_kernels"
    assert layer._dsv4_mxfp4_backend == expected
    return method, layer


def _dispatch(rows, num_experts, top_k, seed, rows_per_req=1, masked_rows=0):
    """Rows of one request share most experts (DSpark verify rows do), so some experts get more
    than 16 and 32 rows."""
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)
    req_logits = torch.randn(
        (rows + rows_per_req - 1) // rows_per_req, num_experts, device="cuda", generator=g
    )
    logits = req_logits.repeat_interleave(rows_per_req, 0)[:rows]
    logits = logits + 0.5 * torch.randn(rows, num_experts, device="cuda", generator=g)
    weights, ids = torch.topk(torch.softmax(logits, -1), top_k, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).float()
    ids = ids.int()
    if masked_rows:
        ids[-masked_rows:] = -1
    topk = StandardTopKOutput(topk_weights=weights, topk_ids=ids, router_logits=logits)
    return StandardDispatchOutput(hidden_states=x, hidden_states_scale=None, topk_output=topk)


def _dequant(weight, scale):
    lut = torch.tensor(E2M1, device="cuda")
    w = weight.cuda().view(torch.uint8).long()
    values = torch.stack([lut[w & 0xF], lut[w >> 4]], -1).flatten(-2)
    s = torch.exp2(scale.cuda().view(torch.uint8).float() - 127.0)
    return values * s.repeat_interleave(32, -1)


def _apply(method, layer, d):
    return method.apply(layer, d).hidden_states


@pytest.mark.parametrize("shape", list(SHAPES))
def test_rows_aware_tiles_bit_identical(shape, monkeypatch):
    num_experts, top_k, rows_per_req = SHAPES[shape]
    w = _weights(num_experts)
    tk = _layer(w, num_experts, top_k, "tk", monkeypatch)
    rows = _layer(w, num_experts, top_k, "rows", monkeypatch)
    for bs in (8, 64, 125, 160, 192):
        d = _dispatch(bs * rows_per_req, num_experts, top_k, bs, rows_per_req, masked_rows=3)
        assert torch.equal(_apply(*rows, d), _apply(*tk, d)), bs
    if shape == "target":  # prefill-sized batches take the 32- and 64-row tiles
        for n in (1536, 4096):
            d = _dispatch(n, num_experts, top_k, n)
            assert torch.equal(_apply(*rows, d), _apply(*tk, d)), n


def test_sm90_down_dequant_exact():
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe as dk

    w = _weights(8)
    w13, s13 = w["w13_weight"].cuda(), w["w13_weight_scale_inv"].cuda()
    w2, s2 = w["w2_weight"].cuda(), w["w2_weight_scale_inv"].cuda()
    packed = dk.pack_experts(w13, s13, w2, s2)
    n = 40
    expert = torch.arange(n, device="cuda") % 8
    order = torch.argsort(expert, stable=True)
    col = torch.arange(n, device="cuda") * 7 % INTERMEDIATE
    h = torch.zeros(n, INTERMEDIATE, device="cuda", dtype=torch.bfloat16)
    h[torch.arange(n, device="cuda"), col[order]] = 1.0
    y = torch.empty(n, HIDDEN, device="cuda", dtype=torch.bfloat16)
    counts = torch.bincount(expert, minlength=8).int()
    dk._module().down(h, packed.w2_q, packed.w2_s, counts, order.int(), torch.ones(n, device="cuda"), y)
    ref = torch.stack([_dequant(w2[expert[o]], s2[expert[o]])[:, col[o]] for o in range(n)])
    assert torch.equal(y.float(), ref)


@pytest.mark.parametrize("shape", list(SHAPES))
def test_sm90_kernel_matches_triton_decode(shape, monkeypatch):
    num_experts, top_k, rows_per_req = SHAPES[shape]
    w = _weights(num_experts)
    tk = _layer(w, num_experts, top_k, "tk", monkeypatch)
    sm90 = _layer(w, num_experts, top_k, "sm90", monkeypatch)
    w13 = _dequant(w["w13_weight"], w["w13_weight_scale_inv"])
    for bs in (1, 8, 48, 125, 192):
        d = _dispatch(bs * rows_per_req, num_experts, top_k, bs, rows_per_req, masked_rows=2)
        counts = torch.bincount(d.topk_output.topk_ids.clamp_min(0).flatten().long(), minlength=num_experts)
        out = _apply(*sm90, d)
        assert not torch.isnan(out).any()
        assert torch.equal(out, _apply(*tk, d)), (bs, int(counts.max()))
    del w13


@pytest.mark.parametrize("shape", list(SHAPES))
def test_sm90_kernel_cuda_graph_replay(shape, monkeypatch):
    num_experts, top_k, rows_per_req = SHAPES[shape]
    w = _weights(num_experts)
    method, layer = _layer(w, num_experts, top_k, "sm90", monkeypatch)
    for bs in (1, 16, 64, 128):
        rows = bs * rows_per_req
        static = _dispatch(rows, num_experts, top_k, 1000 + bs, rows_per_req)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _apply(method, layer, static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _apply(method, layer, static)
        fresh = _dispatch(rows, num_experts, top_k, 2000 + bs, rows_per_req, masked_rows=1)
        static.hidden_states.copy_(fresh.hidden_states)
        static.topk_output.topk_ids.copy_(fresh.topk_output.topk_ids)
        static.topk_output.topk_weights.copy_(fresh.topk_output.topk_weights)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, _apply(method, layer, fresh)), bs


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
