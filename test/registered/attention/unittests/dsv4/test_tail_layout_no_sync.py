"""SGLANG_OPT_DSV4_TAIL_LAYOUT_NO_SYNC: the bounded-replay late-layer tail layout built
from host extend lengths gives the same tensors (values and dtypes) as the device-side
build, for every CP rank, and does not wait for the stream."""

import time
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_ENV = "SGLANG_OPT_DSV4_TAIL_LAYOUT_NO_SYNC"
SWA_WINDOW = 128

BATCHES = [
    [6912],
    [6904, 6904],
    [1, 2, 3, 127, 128, 129, 6750],
    [6750] * 9 + [1744],
    [300, 5, 6904, 77, 128],
]


def _layout(monkeypatch, on, extend, seq):
    from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
        late_layer_tail_layout,
    )

    monkeypatch.setenv(_ENV, "1" if on else "0")
    return late_layer_tail_layout(
        extend_lens_cpu=extend, seq_lens_cpu=seq, tail_len=SWA_WINDOW, device="cuda"
    )


def _forward_batch(extend):
    total = sum(extend)
    g = torch.Generator(device="cuda").manual_seed(total)
    return SimpleNamespace(
        extend_seq_lens_cpu=list(extend),
        batch_size=len(extend),
        positions=torch.randint(
            0, 1 << 20, (total,), device="cuda", dtype=torch.int64, generator=g
        ),
    )


@pytest.mark.parametrize("extend", BATCHES)
def test_tail_layout_same(monkeypatch, extend):
    seq = [n + 68096 for n in extend]
    a = _layout(monkeypatch, False, extend, seq)
    b = _layout(monkeypatch, True, extend, seq)
    torch.cuda.synchronize()
    assert a[1] == b[1]
    for x, y in ((a[0], b[0]), (a[2], b[2])):
        assert x.dtype == y.dtype and torch.equal(x, y)


@pytest.mark.parametrize("extend", BATCHES)
@pytest.mark.parametrize("cp_rank", [0, 1, 2, 3])
def test_cp_layout_same(monkeypatch, extend, cp_rank):
    from sglang.srt.layers.attention import deepseek_v4_backend as backend

    monkeypatch.setattr(
        backend,
        "get_parallel",
        lambda: SimpleNamespace(attn_cp_rank=cp_rank, attn_cp_size=4),
    )
    seq = [n + 68096 for n in extend]
    token_indices, tail_lens_cpu, _ = _layout(monkeypatch, False, extend, seq)
    tail_lens = torch.tensor(tail_lens_cpu, dtype=torch.int32, device="cuda")
    contiguous_start = extend[0] - tail_lens_cpu[0] if len(extend) == 1 else None
    fb = _forward_batch(extend)
    ref = backend.DeepseekV4AttnBackend._late_layer_tail_cp_layout(
        None, fb, token_indices, tail_lens
    )
    got = backend.DeepseekV4AttnBackend._late_layer_tail_cp_layout_host(
        None, fb, token_indices, tail_lens_cpu, contiguous_start
    )
    torch.cuda.synchronize()
    assert got["pad_rows"] == ref["pad_rows"]
    assert got["local_lens_cpu"] == ref["local_lens_cpu"]
    for key in ("local_token_indices", "local_positions"):
        assert got[key].dtype == ref[key].dtype and torch.equal(got[key], ref[key])
    rm, gm = ref["cp_metadata"], got["cp_metadata"]
    for f in ("per_rank_actual_token", "max_rank_len", "total_seq_lens", "bs"):
        assert getattr(gm, f) == getattr(rm, f), f
    assert gm.per_rank_logical_token == rm.per_rank_logical_token
    for f in ("gather_index", "local_index"):
        x, y = getattr(rm, f), getattr(gm, f)
        assert x.dtype == y.dtype and torch.equal(x, y), f


def test_no_stream_wait(monkeypatch):
    from sglang.srt.layers.attention import deepseek_v4_backend as backend

    monkeypatch.setattr(
        backend, "get_parallel", lambda: SimpleNamespace(attn_cp_rank=1, attn_cp_size=4)
    )
    extend = [6904, 6904, 300]
    seq = [n + 68096 for n in extend]
    fb = _forward_batch(extend)
    waits = {}
    for on in (False, True):
        # warm allocators and kernels
        token_indices, tail_lens_cpu, _ = _layout(monkeypatch, on, extend, seq)
        torch.cuda.synchronize()
        torch.cuda._sleep(2_000_000_000)  # ~1 s of GPU time queued ahead
        t0 = time.perf_counter()
        token_indices, tail_lens_cpu, _ = _layout(monkeypatch, on, extend, seq)
        if on:
            backend.DeepseekV4AttnBackend._late_layer_tail_cp_layout_host(
                None, fb, token_indices, tail_lens_cpu, None
            )
        else:
            tail_lens = torch.tensor(tail_lens_cpu, dtype=torch.int32, device="cuda")
            backend.DeepseekV4AttnBackend._late_layer_tail_cp_layout(
                None, fb, token_indices, tail_lens
            )
        waits[on] = time.perf_counter() - t0
        torch.cuda.synchronize()
    assert waits[False] > 0.3, waits
    assert waits[True] < 0.05, waits
