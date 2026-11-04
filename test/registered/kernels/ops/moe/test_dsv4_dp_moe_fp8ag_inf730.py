"""INF-730 SGLANG_OPT_DSV4_DP_FP8_MOE_GATHER: the TP decode MoE input all-gathered as e4m3 rows.

Today every rank all-gathers the W ranks' BF16 rows and the v2 decode MoE quantizes all of them in its prologue.
The switch quantizes each rank's rows first (fp8ag_quant, the prologue's own code), gathers packed
[e4m3 | scale | top-k ids | top-k weights] rows and runs decode_moe_prequant on them. One GPU stands in for the W
ranks here: the all-gather only concatenates the ranks' packed rows (rank-major), so it is emulated by torch.cat.

Checked, bit for bit: quantize-then-gather equals gather-then-quantize (e4m3 rows and scales vs the prologue's on
the gathered BF16 rows), the gathered top-k, the bf16 copy into the rank's slot, and the MoE output against
decode_moe on the gathered BF16 rows (target and draft shapes, pad rows, skewed routing, CUDA graph replay);
and that the router GEMM gives a rank's rows the same logits when the other ranks' slots hold stale rows (what the
switch relies on to keep the router on the gathered buffer's shape). With the checkpoint mounted
(SGLANG_TEST_DSV41_MODEL, default /opt/lv/data/models/DeepSeek-V4.1-Flash-1017) the MoE check also runs on real
experts (TP4 rank slices of a target layer and the DSpark draft layer) and the router check on a real gate."""

import json
import os

import pytest
import torch

from sglang.srt.utils.common import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm90_supported(),
    reason="SM90 wgmma W4A8 decode MoE",
)

HIDDEN, INTER = 5120, 576
SWIGLU_LIMIT = 10.0
W = 4  # TP4 decode: 4 DP ranks
# (experts, top_k, rows per request): verify rows at DSpark block 3 (gamma 3 -> 4), draft rows (3)
SHAPES = {"target": (384, 6, 4), "draft": (128, 3, 3)}
MODEL = os.environ.get("SGLANG_TEST_DSV41_MODEL", "/opt/lv/data/models/DeepSeek-V4.1-Flash-1017")


def _synthetic_packed(num_experts, seed=0):
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

    g = torch.Generator().manual_seed(seed)
    nib = lambda *s: torch.randint(-128, 128, s, generator=g, dtype=torch.int8)
    sc = lambda *s: torch.randint(119, 123, s, generator=g, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    return ws.pack_experts(
        nib(num_experts, 2 * INTER, HIDDEN // 2).cuda(),
        sc(num_experts, 2 * INTER, HIDDEN // 32).cuda(),
        nib(num_experts, HIDDEN, INTER // 2).cuda(),
        sc(num_experts, HIDDEN, INTER // 32).cuda(),
    )


def _rank_rows(n, rows_per_req, num_experts, top_k, seed, pad=0, skew=0.0):
    """One rank's MAX_LEN-padded rows: correlated verify rows, optional zipf-skewed experts, `pad` trailing pad
    rows (stale values, top-k -1), one all-zero row and one large-magnitude row."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    reqs = (n + rows_per_req - 1) // rows_per_req
    pop = torch.zeros(num_experts, device="cuda")
    if skew:
        pop = (-skew * torch.log(torch.arange(1, num_experts + 1, device="cuda").float()))[
            torch.randperm(num_experts, device="cuda", generator=g)
        ]
    logits = pop + torch.randn(reqs, num_experts, device="cuda", generator=g).repeat_interleave(rows_per_req, 0)[:n]
    logits = logits + 0.5 * torch.randn(n, num_experts, device="cuda", generator=g)
    w, ids = torch.topk(torch.softmax(logits, -1), top_k, dim=-1)
    ids = ids.int()
    w = (w / w.sum(-1, keepdim=True)).float()
    x = (torch.randn(n, HIDDEN, device="cuda", generator=g) * 0.7).to(torch.bfloat16)
    if n > 2:
        x[0] = 0
        x[1] *= 300.0
    if pad:
        ids[n - pad :] = -1
        x[n - pad :] = torch.randn(pad, HIDDEN, device="cuda", generator=g).to(torch.bfloat16) * 1e4
    return x, ids, w


def _prologue_quant(x, ids, w, num_experts):
    """The v2 prologue's quantization of x (what decode_moe feeds the experts)."""
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2 as v2

    T, K = ids.shape
    dev = x.device
    x8 = torch.empty(T, HIDDEN, dtype=torch.uint8, device=dev)
    xs = torch.empty(T, dtype=torch.float32, device=dev)
    work = torch.empty(v2.WORKSPACE_BASE + 2 * (num_experts + T * K // 32 + 1), dtype=torch.int32, device=dev)
    v2._module().prep(
        ids.int().contiguous(),
        w.float().contiguous(),
        x.contiguous(),
        torch.empty(num_experts, dtype=torch.int32, device=dev),
        torch.empty(T * K, dtype=torch.int32, device=dev),
        torch.empty(T * K, dtype=torch.int32, device=dev),
        torch.empty(T * K, dtype=torch.float32, device=dev),
        x8,
        xs,
        work,
    )
    return x8, xs


def _fp8ag_path(ranks, top_k, packed_experts, global_rows=None):
    """Every rank: quantize + pack its rows (copying them into its slot), add its top-k; 'all-gather' = cat;
    unpack; decode_moe_prequant. Returns (output, x8, xs, ids, w, global_rows)."""
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2 as v2

    n = ranks[0][0].shape[0]
    if global_rows is None:
        global_rows = torch.empty(len(ranks) * n, HIDDEN, dtype=torch.bfloat16, device="cuda")
    packs = []
    for s, (x, ids, w) in enumerate(ranks):
        p = torch.empty(n, v2.fp8ag_row_bytes(top_k), dtype=torch.uint8, device="cuda")
        v2.fp8ag_quant(x, p, global_rows[s * n : (s + 1) * n])
        v2.fp8ag_meta(ids, w, p)
        packs.append(p)
    x8, xs, gids, gw = v2.fp8ag_unpack(torch.cat(packs), top_k)
    out = v2.decode_moe_prequant(x8, xs, gids, gw, packed_experts, SWIGLU_LIMIT)
    return out, x8, xs, gids, gw, global_rows


def _check(ranks, top_k, num_experts, packed_experts, tag):
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2 as v2

    X = torch.cat([r[0] for r in ranks])
    ids = torch.cat([r[1] for r in ranks])
    w = torch.cat([r[2] for r in ranks])
    ref = v2.decode_moe(X, ids, w, packed_experts, SWIGLU_LIMIT)  # today's path on the gathered BF16 rows
    ref_x8, ref_xs = _prologue_quant(X, ids, w, num_experts)
    out, x8, xs, gids, gw, grows = _fp8ag_path(ranks, top_k, packed_experts)
    assert torch.equal(grows, X), tag  # each rank's slot holds its BF16 rows
    assert torch.equal(x8, ref_x8) and torch.equal(xs, ref_xs), tag  # quantize-then-gather == gather-then-quantize
    assert torch.equal(gids, ids) and torch.equal(gw, w), tag
    assert torch.equal(out, ref), tag
    return out


@pytest.mark.parametrize("shape", list(SHAPES))
def test_fp8_gather_bit_identical(shape):
    num_experts, top_k, rpr = SHAPES[shape]
    packed = _synthetic_packed(num_experts)
    # n = padded rows per rank: 1 request up to 160 requests (verify) / 160 (draft)
    for reqs in (1, 16, 64, 107, 128, 160):
        n = reqs * rpr
        for skew in (0.0, 1.2):
            ranks = [
                _rank_rows(n, rpr, num_experts, top_k, seed=1000 * s + reqs, pad=(s * 3) % (n // 2 + 1), skew=skew)
                for s in range(W)
            ]
            _check(ranks, top_k, num_experts, packed, (shape, reqs, skew))


def test_fp8_gather_cuda_graph_replay():
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_v2 as v2

    num_experts, top_k, rpr = SHAPES["target"]
    packed = _synthetic_packed(num_experts)
    n = 96 * rpr
    bufs = [_rank_rows(n, rpr, num_experts, top_k, seed=s) for s in range(W)]
    v2.prepare(torch.cuda.current_device())
    out_holder = {}

    def run():
        out_holder["out"] = _fp8ag_path(bufs, top_k, packed)[0]

    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        run()
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=st):
        run()
    for it in range(5):
        for s in range(W):
            x, ids, w = _rank_rows(n, rpr, num_experts, top_k, seed=100 + 10 * it + s, pad=it, skew=0.8)
            bufs[s][0].copy_(x)
            bufs[s][1].copy_(ids)
            bufs[s][2].copy_(w)
        g.replay()
        torch.cuda.synchronize()
        X = torch.cat([b[0] for b in bufs])
        ids = torch.cat([b[1] for b in bufs])
        w = torch.cat([b[2] for b in bufs])
        assert torch.equal(out_holder["out"], v2.decode_moe(X, ids, w, packed, SWIGLU_LIMIT)), it


def _gemm_paths():
    from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32

    return {
        "fp32_linear": lambda h, wt: torch.nn.functional.linear(h.float(), wt.float()),
        "bf16_fp32_cublas": lambda h, wt: linear_bf16_fp32(h, wt.to(torch.bfloat16)),
    }


def _router_rows_independent(gate_w, tag):
    """A rank's router logits on the gathered-buffer shape do not depend on what the other slots hold."""
    for name, gemm in _gemm_paths().items():
        for allow_tf32 in (False, True):
            prev = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            try:
                for reqs in (1, 4, 16, 64, 107, 128, 160):
                    n = reqs * 4
                    full = (torch.randn(W * n, HIDDEN, device="cuda") * 0.7).to(torch.bfloat16)
                    ref = gemm(full, gate_w)
                    for s in range(W):
                        stale = (torch.randn(W * n, HIDDEN, device="cuda") * 50).to(torch.bfloat16)
                        stale[s * n : (s + 1) * n] = full[s * n : (s + 1) * n]
                        got = gemm(stale, gate_w)
                        assert torch.equal(got[s * n : (s + 1) * n], ref[s * n : (s + 1) * n]), (
                            tag,
                            name,
                            allow_tf32,
                            reqs,
                            s,
                        )
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prev


def test_router_rows_independent_of_other_slots():
    g = torch.Generator(device="cuda").manual_seed(7)
    _router_rows_independent(torch.randn(384, HIDDEN, device="cuda", generator=g) * 0.02, "synthetic")


# ---- real checkpoint (skipped when it is not mounted) ----

_real = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL, "model.safetensors.index.json")), reason="checkpoint not mounted"
)


_IDX, _HANDLES = {}, {}


def _load(names):
    from safetensors import safe_open

    if not _IDX:
        _IDX.update(json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"])
    out = {}
    for name in names:
        f = _IDX[name]
        if f not in _HANDLES:
            _HANDLES[f] = safe_open(os.path.join(MODEL, f), "pt", device="cpu")
        out[name] = _HANDLES[f].get_tensor(name)
    return out


def _real_tp_slice(prefix, num_experts, tp_rank):
    """Rank tp_rank's TP4 slice of every expert of a layer, packed for the v2 kernel."""
    from sglang.kernels.ops.moe import dsv4_mxfp4_decode_moe_ws as ws

    w = {k: [] for k in ("w1", "s1", "w3", "s3", "w2", "s2")}
    for e in range(num_experts):
        t = _load([f"{prefix}.experts.{e}.{m}.{k}" for m in ("w1", "w3", "w2") for k in ("weight", "scale")])
        for m in ("w1", "w3", "w2"):
            wt = t[f"{prefix}.experts.{e}.{m}.weight"].view(torch.uint8)
            sc = t[f"{prefix}.experts.{e}.{m}.scale"].view(torch.uint8)
            lo, hi = tp_rank * INTER, (tp_rank + 1) * INTER
            if m == "w2":  # [hidden, inter / 2] nibbles, [hidden, inter / 32] scales
                wt, sc = wt[:, lo // 2 : hi // 2], sc[:, lo // 32 : hi // 32]
            else:  # [inter, hidden / 2], [inter, hidden / 32]
                wt, sc = wt[lo:hi], sc[lo:hi]
            w["w" + m[1]].append(wt.contiguous())
            w["s" + m[1]].append(sc.contiguous())
    st = {k: torch.stack(v).cuda() for k, v in w.items()}
    return ws.pack_experts(
        torch.cat([st["w1"], st["w3"]], 1).contiguous(),
        torch.cat([st["s1"], st["s3"]], 1).contiguous().view(torch.float8_e8m0fnu),
        st["w2"],
        st["s2"].view(torch.float8_e8m0fnu),
    )


@_real
@pytest.mark.parametrize("case", [("layers.3.ffn", "target", 0), ("layers.25.ffn", "target", 3), ("mtp.0.ffn", "draft", 1)])
def test_fp8_gather_bit_identical_real_weights(case):
    prefix, shape, tp_rank = case
    num_experts, top_k, rpr = SHAPES[shape]
    try:
        packed = _real_tp_slice(prefix, num_experts, tp_rank)
    except KeyError as e:
        pytest.skip(f"{prefix}: {e} not in the checkpoint")
    for reqs in (1, 64, 128, 160):
        n = reqs * rpr
        for skew in (0.0, 1.0):
            ranks = [
                _rank_rows(n, rpr, num_experts, top_k, seed=77 * s + reqs, pad=(2 * s) % (n // 2 + 1), skew=skew)
                for s in range(W)
            ]
            out = _check(ranks, top_k, num_experts, packed, (prefix, reqs, skew))
            assert out.abs().sum() > 0


@_real
def test_router_rows_independent_real_gate():
    for prefix in ("layers.3.ffn", "layers.25.ffn"):
        name = f"{prefix}.gate.weight"
        try:
            gw = _load([name])[name]
        except KeyError:
            pytest.skip(f"{name} not in the checkpoint")
        _router_rows_independent(gw.cuda(), prefix)
