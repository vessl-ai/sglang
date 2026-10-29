"""INF-681 k-pcomm switches for the DeepSeek-V4.1 prefill-CP MoE collectives (all default off).

- SGLANG_OPT_DSV4_CP_SHARED_EXPERT_LOCAL: on only with the replicated (TP1) shared expert;
  the local shared expert of a shard equals the rows of the gathered-batch shared expert
  it replaces (per-token MLP), so adding it after the reduce-scatter gives the same sum.
- SGLANG_OPT_DSV4_CP_SHARED_EXPERT_OVERLAP: the side stream is used only for eager shards
  of at least _MIN_ROWS rows; inline otherwise; one stream for all layers.
- SGLANG_OPT_DSV4_W4A8_PREFETCH_AFTER_COMBINE: a held-back prefetch is launched exactly
  once, by fire_pending_prefetch (after the reduce-scatter) or, if nobody fired it, by the
  next layer's _acquire before it reads the FP8 scratch.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA streams")


@pytest.fixture
def v4(monkeypatch):
    import sglang.srt.models.deepseek_v4 as m

    monkeypatch.setattr(m, "_CP_SHARED_EXPERT_STREAM", None)
    monkeypatch.setattr(m, "_CP_SHARED_EXPERT_LOGGED", False)
    return m


def test_local_shared_needs_switch_and_tp1(v4, monkeypatch):
    tp1 = SimpleNamespace(shared_experts=object(), _shared_expert_tp1=True)
    tp4 = SimpleNamespace(shared_experts=object(), _shared_expert_tp1=False)
    none = SimpleNamespace(shared_experts=None, _shared_expert_tp1=True)
    monkeypatch.delenv("SGLANG_OPT_DSV4_CP_SHARED_EXPERT_LOCAL", raising=False)
    assert not v4._cp_shared_expert_local_on(tp1)
    monkeypatch.setenv("SGLANG_OPT_DSV4_CP_SHARED_EXPERT_LOCAL", "1")
    assert v4._cp_shared_expert_local_on(tp1)
    assert not v4._cp_shared_expert_local_on(tp4)
    assert not v4._cp_shared_expert_local_on(none)


def test_overlap_stream_gate(v4, monkeypatch):
    monkeypatch.delenv("SGLANG_OPT_DSV4_CP_SHARED_EXPERT_OVERLAP", raising=False)
    assert v4._cp_shared_expert_stream(65536) is None
    monkeypatch.setenv("SGLANG_OPT_DSV4_CP_SHARED_EXPERT_OVERLAP", "1")
    monkeypatch.setenv("SGLANG_OPT_DSV4_CP_SHARED_EXPERT_OVERLAP_MIN_ROWS", "4096")
    assert v4._cp_shared_expert_stream(1728) is None  # the 1-request CP4 shard: inline
    s1 = v4._cp_shared_expert_stream(16302)  # the full-step CP4 shard
    s2 = v4._cp_shared_expert_stream(4096)
    assert isinstance(s1, torch.cuda.Stream) and s1 is s2
    monkeypatch.setattr(v4, "get_is_capture_mode", lambda: True)
    assert v4._cp_shared_expert_stream(16302) is None  # never forks inside a CUDA graph


def test_sm_cap_restores_deep_gemm(v4, monkeypatch):
    deep_gemm = pytest.importorskip("deep_gemm")
    total = deep_gemm.get_num_sms()
    monkeypatch.setenv("SGLANG_OPT_DSV4_CP_SHARED_EXPERT_RESERVE_SMS", "24")
    with v4._cp_shared_expert_sm_cap():
        assert deep_gemm.get_num_sms() == total - 24
    assert deep_gemm.get_num_sms() == total
    monkeypatch.setenv("SGLANG_OPT_DSV4_CP_SHARED_EXPERT_RESERVE_SMS", "0")
    with v4._cp_shared_expert_sm_cap():
        assert deep_gemm.get_num_sms() == total


def test_local_shared_sum_equals_gathered_rows():
    """A per-token MLP on the rank's shard rows equals those rows of the MLP on the
    rank-major gathered batch (the replicated shared expert sees the same weights)."""
    torch.manual_seed(0)
    cp, rows, hidden, inter = 4, 96, 256, 128
    w1 = torch.randn(2 * inter, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(hidden, inter, device="cuda") / inter**0.5

    def mlp(x):
        g, u = (x @ w1.t()).chunk(2, dim=-1)
        return (torch.nn.functional.silu(g) * u) @ w2.t()

    shards = [torch.randn(rows, hidden, device="cuda") for _ in range(cp)]
    gathered = torch.cat(shards)
    full = mlp(gathered)
    for r in range(cp):
        torch.testing.assert_close(mlp(shards[r]), full[r * rows:(r + 1) * rows], rtol=1e-5, atol=1e-5)


class _Runner:
    def __init__(self):
        self.calls = []

    def _prefetch_next(self, n):
        self.calls.append(n)


def test_held_back_prefetch_fires_once():
    from sglang.srt.layers.quantization import mxfp4_w4a8_prefill as w

    r = _Runner()
    w._PENDING_PREFETCH[0] = (r, 16)
    w.fire_pending_prefetch()
    w.fire_pending_prefetch()
    assert r.calls == [16] and w._PENDING_PREFETCH[0] is None


def test_acquire_fires_a_prefetch_nobody_fired(monkeypatch):
    from sglang.srt.layers.quantization import mxfp4_w4a8_prefill as w

    order = []
    r = _Runner()
    r._prefetch_next = lambda n: order.append(("prefetch", n))
    w._PENDING_PREFETCH[0] = (r, 16)
    sc = SimpleNamespace(device=torch.device("cuda"), ready=None, holder=None, last_called=None)
    layer = SimpleNamespace(scratch=sc, successor=None)
    layer._single_launch = lambda: 1
    layer._convert_into_scratch = lambda n: order.append(("convert", n))
    w.Mxfp4W4A8PrefillExperts._acquire(layer)
    # the held-back prefetch first, then this layer's own conversion (holder was None)
    assert order == [("prefetch", 16), ("convert", 1)]
    assert w._PENDING_PREFETCH[0] is None
