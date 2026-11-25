"""SGLANG_OPT_DSV4_SM90_MLA_SCHED_META: the parallel large-batch FlashMLA schedule
(flashmla_sched_meta_large) gives the same metadata as FlashMLA's own SM90 kernel,
including its low-shared-memory variant (>= ~11.6k rows), and the sparse decode output
is bit-identical when FlashMLA is handed it. Also the backend gate: off by default,
skipped under CUDA graph capture unless SGLANG_OPT_DSV4_SM90_MLA_SCHED_META_IN_GRAPH,
and with that switch a captured decode graph recomputes the same schedule on every
replay from the replayed lengths."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or not get_platform().is_sm90,
    reason="checks FlashMLA's SM90 split-KV schedule",
)

H_Q, D_QK, D_V = 64, 512, 512
BYTES_PER_TOKEN, PAGE = 584, 256
# FlashMLA's DecodingSchedMeta ends in a `_pad` word it never writes.
DEFINED = slice(0, 7)
_ENV = "SGLANG_OPT_DSV4_SM90_MLA_SCHED_META"
_ENV_IN_GRAPH = "SGLANG_OPT_DSV4_SM90_MLA_SCHED_META_IN_GRAPH"


def _cache(pages):
    return torch.randint(
        0, 200, (pages, PAGE, 1, BYTES_PER_TOKEN), device="cuda", dtype=torch.uint8
    )


def _inputs(b, topk, extra_topk, mode, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((b, 1, H_Q, D_QK), device="cuda", dtype=torch.bfloat16, generator=g)
    kv, extra_kv = _cache(64), _cache(256)
    indices = torch.randint(
        0, 64 * PAGE, (b, 1, topk), device="cuda", dtype=torch.int32, generator=g
    )
    sink = torch.randn((H_Q,), device="cuda", dtype=torch.float32, generator=g)
    if mode == "full":
        topk_length = torch.full((b,), topk, device="cuda", dtype=torch.int32)
    elif mode == "zeros":
        topk_length = torch.zeros((b,), device="cuda", dtype=torch.int32)
    else:
        topk_length = torch.randint(
            0, topk + 1, (b,), device="cuda", dtype=torch.int32, generator=g
        )
    extra = None
    if extra_topk:
        extra_indices = torch.randint(
            0,
            256 * PAGE,
            (b, 1, extra_topk),
            device="cuda",
            dtype=torch.int32,
            generator=g,
        )
        if mode == "prefill":
            # A chunk's rows: the compressed history grows with the position.
            extra_len = torch.clamp(
                torch.arange(b, device="cuda", dtype=torch.int32) // 2 + 1,
                max=extra_topk,
            )
        else:
            extra_len = torch.randint(
                1, extra_topk + 1, (b,), device="cuda", dtype=torch.int32, generator=g
            )
        extra = (extra_kv, extra_indices, extra_len)
    return q, kv, indices, sink, topk_length, extra


def _call(q, kv, indices, sink, topk_length, extra, sched):
    out, lse = _launch(q, kv, indices, sink, topk_length, extra, sched)
    torch.cuda.synchronize()
    return out, lse


def _launch(q, kv, indices, sink, topk_length, extra, sched):
    import sgl_kernel.flash_mla as flash_mla

    kwargs = {}
    if extra is not None:
        kwargs = dict(
            extra_k_cache=extra[0],
            extra_indices_in_kvcache=extra[1],
            extra_topk_length=extra[2],
        )
    out, lse = flash_mla.flash_mla_with_kvcache(
        q,
        kv,
        None,
        None,
        D_V,
        sched,
        indices=indices,
        is_fp8_kvcache=True,
        softmax_scale=0.1,
        causal=False,
        topk_length=topk_length,
        attn_sink=sink,
        **kwargs,
    )
    return out, lse


# _ENV_IN_GRAPH: the step-table (chain) kernel where it fits, else the same as _ENV.
@pytest.mark.parametrize("env", [_ENV, _ENV_IN_GRAPH])
@pytest.mark.parametrize("b", [1, 6, 133, 512, 1728, 11300, 16320, 30000])
@pytest.mark.parametrize("extra_topk", [0, 512, 1024])
@pytest.mark.parametrize("mode", ["random", "full", "zeros", "prefill"])
def test_backend_schedule_matches_flashmla(monkeypatch, env, b, extra_topk, mode):
    import sgl_kernel.flash_mla as flash_mla

    from sglang.srt.layers.attention import deepseek_v4_backend as backend

    if mode == "prefill" and not extra_topk:
        pytest.skip("prefill-shaped lengths are for the extra cache")
    q, kv, indices, sink, topk_length, extra = _inputs(
        b, 128, extra_topk, mode, b * 31 + extra_topk + len(mode)
    )
    ref_sched = flash_mla.FlashMLASchedMeta()
    ref_out, ref_lse = _call(q, kv, indices, sink, topk_length, extra, ref_sched)

    monkeypatch.delenv(_ENV, raising=False)
    monkeypatch.delenv(_ENV_IN_GRAPH, raising=False)
    monkeypatch.setenv(env, "1")
    sched = flash_mla.FlashMLASchedMeta()
    backend._maybe_precompute_flashmla_sched_meta(
        sched,
        q=q,
        indices=indices,
        topk_length=topk_length,
        extra_indices=None if extra is None else extra[1],
        extra_topk_length=None if extra is None else extra[2],
    )
    assert sched.tile_scheduler_metadata is not None
    torch.cuda.synchronize()
    assert torch.equal(
        sched.tile_scheduler_metadata[:, DEFINED],
        ref_sched.tile_scheduler_metadata[:, DEFINED],
    )
    assert torch.equal(sched.num_splits, ref_sched.num_splits)
    out, lse = _call(q, kv, indices, sink, topk_length, extra, sched)
    assert torch.equal(out.view(torch.int16), ref_out.view(torch.int16))
    assert torch.equal(lse.view(torch.int32), ref_lse.view(torch.int32))


@pytest.mark.parametrize("b", [1, 133, 512])
@pytest.mark.parametrize("capacity", ["exact", "short"])
def test_chain_kernel_matches_flashmla(b, capacity):
    """The step-table kernel directly: with the backend's capacity, and with a
    capacity below the real total (the kernel's serial-walk fallback)."""
    import sgl_kernel.flash_mla as flash_mla

    from sglang.kernels.ops.attention.dsv4.flashmla_sched_meta_large import (
        META_INTS,
        chain_capacity,
        flashmla_sched_meta_chain,
    )

    q, kv, indices, sink, topk_length, extra = _inputs(b, 128, 512, "random", b + 7)
    ref_sched = flash_mla.FlashMLASchedMeta()
    _call(q, kv, indices, sink, topk_length, extra, ref_sched)
    parts = torch.cuda.get_device_properties(0).multi_processor_count
    cap = chain_capacity(
        b, block_size_n=64, fixed_overhead_num_blocks=5, topk=128, extra_topk=512
    )
    if capacity == "short":
        cap = 1
    meta = torch.full((parts, META_INTS), -7, dtype=torch.int32, device="cuda")
    num_splits = torch.full((b + 1,), -7, dtype=torch.int32, device="cuda")
    flashmla_sched_meta_chain(
        meta,
        num_splits,
        topk_length=topk_length,
        extra_topk_length=extra[2],
        block_size_n=64,
        fixed_overhead_num_blocks=5,
        topk=128,
        extra_topk=512,
        capacity=cap,
    )
    torch.cuda.synchronize()
    assert torch.equal(meta[:, DEFINED], ref_sched.tile_scheduler_metadata[:, DEFINED])
    assert torch.equal(num_splits, ref_sched.num_splits)


def _precompute(sched, q, indices, topk_length, extra):
    from sglang.srt.layers.attention import deepseek_v4_backend as backend

    backend._maybe_precompute_flashmla_sched_meta(
        sched,
        q=q,
        indices=indices,
        topk_length=topk_length,
        extra_indices=None if extra is None else extra[1],
        extra_topk_length=None if extra is None else extra[2],
    )


# 512 / 640 rows: EP8 decode at 128 / 160 requests per rank with 4 verify rows each.
@pytest.mark.parametrize("b", [1, 133, 512, 640, 1728])
@pytest.mark.parametrize("extra_topk", [0, 512])
def test_in_graph_schedule_matches_flashmla(monkeypatch, b, extra_topk):
    import sgl_kernel.flash_mla as flash_mla

    monkeypatch.delenv(_ENV, raising=False)
    monkeypatch.setenv(_ENV_IN_GRAPH, "1")
    q, kv, indices, sink, topk_length, extra = _inputs(
        b, 128, extra_topk, "random", b * 17 + extra_topk
    )
    # Eager once first: loads the JIT module outside the capture.
    _precompute(flash_mla.FlashMLASchedMeta(), q, indices, topk_length, extra)

    sched = flash_mla.FlashMLASchedMeta()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _precompute(sched, q, indices, topk_length, extra)
        assert sched.tile_scheduler_metadata is not None
        out, lse = _launch(q, kv, indices, sink, topk_length, extra, sched)

    g = torch.Generator(device="cuda").manual_seed(b + extra_topk)
    for step in range(3):
        if step:
            # A later replay sees other lengths in the same buffers (as the
            # engine's replay copies them in); the schedule must follow.
            topk_length.copy_(
                torch.randint(0, 129, (b,), device="cuda", dtype=torch.int32, generator=g)
            )
            if extra is not None:
                extra[2].copy_(
                    torch.randint(
                        1,
                        extra_topk + 1,
                        (b,),
                        device="cuda",
                        dtype=torch.int32,
                        generator=g,
                    )
                )
        graph.replay()
        torch.cuda.synchronize()
        ref_sched = flash_mla.FlashMLASchedMeta()
        ref_out, ref_lse = _call(q, kv, indices, sink, topk_length, extra, ref_sched)
        assert torch.equal(
            sched.tile_scheduler_metadata[:, DEFINED],
            ref_sched.tile_scheduler_metadata[:, DEFINED],
        )
        assert torch.equal(sched.num_splits, ref_sched.num_splits)
        assert torch.equal(out.view(torch.int16), ref_out.view(torch.int16))
        assert torch.equal(lse.view(torch.int32), ref_lse.view(torch.int32))


def test_backend_gate(monkeypatch):
    from sglang.srt.layers.attention import deepseek_v4_backend as backend

    q = torch.zeros((4096, 1, H_Q, D_QK), device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros((4096, 1, 128), device="cuda", dtype=torch.int32)
    kwargs = dict(q=q, indices=indices, topk_length=None, extra_indices=None)
    kwargs["extra_topk_length"] = None

    # Off by default: FlashMLA computes its own schedule.
    monkeypatch.delenv(_ENV, raising=False)
    meta = SimpleNamespace(tile_scheduler_metadata=None, num_splits=None)
    backend._maybe_precompute_flashmla_sched_meta(meta, **kwargs)
    assert meta.tile_scheduler_metadata is None

    # Never inside a CUDA graph capture (the late-layer graph keeps FlashMLA's).
    monkeypatch.setenv(_ENV, "1")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    meta = SimpleNamespace(tile_scheduler_metadata=None, num_splits=None)
    backend._maybe_precompute_flashmla_sched_meta(meta, **kwargs)
    assert meta.tile_scheduler_metadata is None

    # ... unless the in-graph switch is on (it alone also turns the path on).
    monkeypatch.delenv(_ENV, raising=False)
    monkeypatch.setenv(_ENV_IN_GRAPH, "1")
    meta = SimpleNamespace(tile_scheduler_metadata=None, num_splits=None)
    backend._maybe_precompute_flashmla_sched_meta(meta, **kwargs)
    assert meta.tile_scheduler_metadata is not None
    monkeypatch.delenv(_ENV_IN_GRAPH, raising=False)
    monkeypatch.setenv(_ENV, "1")

    # A schedule that does not fit in shared memory is left to FlashMLA.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    big = SimpleNamespace(
        shape=(80000, 1, H_Q, D_QK), ndim=4, is_cuda=True, device=q.device
    )
    meta = SimpleNamespace(tile_scheduler_metadata=None, num_splits=None)
    backend._maybe_precompute_flashmla_sched_meta(
        meta,
        q=big,
        indices=indices,
        topk_length=None,
        extra_indices=None,
        extra_topk_length=None,
    )
    assert meta.tile_scheduler_metadata is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
