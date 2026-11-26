"""EP over the prefill-CP group for the W4A8 early-layer experts (INF-874 R3P, design 3).

With prefill CP the MoE runs TP over the CP group (the MoE TP group): every rank computes
all routed slots of the step with 1/4 of each expert's intermediate size, so W2 writes and
the top-k sum reads one bf16 hidden row per slot on every rank (83.8 GB per step on a 65k-
row step at TP4), and the reduce-scatter adds the four K-partials. Here each rank keeps
whole experts instead: the experts a placement table gives it plus the table's few "hot"
experts on every rank, and computes only the slots of those experts. The slots of a hot
expert are split over the ranks per step by water-filling the ranks' loads, so the ranks
stay balanced although the static placement cannot follow the step's routing. The reduce-
scatter is unchanged (it adds partials whatever their split), so are the gather before the
MoE, the shared expert and the late layers; each slot is computed on exactly one rank.

Weights: the checkpoint loader gives every rank its TP slice of all experts; at weight
processing the ranks exchange slices (all_to_all over the MoE TP group, layer by layer) so
that each rank holds its experts at full intermediate size, and the TP slices are freed.

Not the same bits as TP: an expert's output is one full-K GEMM instead of four K-partials
summed in bf16 by the reduce-scatter (GSM8K gate).
"""

from __future__ import annotations

import functools
import json
import logging
from typing import List, Optional, Sequence

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization import mxfp4_w4a8_prefill as _wp

logger = logging.getLogger(__name__)

RANKS = 4
MAX_HOT = 16


@functools.cache
def load_table(path: str) -> dict:
    """{layer id: (owner list[E], hot list)} from the JSON table."""
    with open(path) as fh:
        raw = json.load(fh)
    out = {}
    for k, v in raw["layers"].items():
        out[int(k)] = ([int(r) for r in v["owner"]], [int(e) for e in v.get("hot", [])])
    return out


def local_experts(owner: Sequence[int], hot: Sequence[int], rank: int) -> List[int]:
    """This rank's experts: the non-hot ones it owns (ascending), then every hot one (table order)."""
    hs = set(hot)
    return [e for e, r in enumerate(owner) if r == rank and e not in hs] + list(hot)


# ----------------------------------------------------------------------------- weights
def full_shape(kind: str, tp_shape) -> tuple:
    """Shape of n full-intermediate experts from a TP slice tensor's shape [E, ...] (n left as the first dim)."""
    _, a, c = tp_shape
    return (a * RANKS, c) if kind == "w13" else (a, c * RANKS)


def assemble_into(out: torch.Tensor, parts: torch.Tensor, kind: str) -> None:
    """Write full-intermediate experts into out from the RANKS TP slices of the same experts (MoE TP rank order),
    without temporaries. kind "w13": parts [RANKS, n, 2i, C] (each slice = [w1 rows; w3 rows] of that rank) ->
    out [n, 2I, C] = [w1; w3]; kind "w2": parts [RANKS, n, H, c] -> out [n, H, RANKS * c]. I = RANKS * i."""
    R, n = parts.shape[:2]
    if kind == "w13":
        two_i, c = parts.shape[2:]
        i = two_i // 2
        out[:, : R * i].view(n, R, i, c).copy_(parts[:, :, :i].permute(1, 0, 2, 3))
        out[:, R * i:].view(n, R, i, c).copy_(parts[:, :, i:].permute(1, 0, 2, 3))
    else:
        h, c = parts.shape[2:]
        out.view(n, h, R, c).copy_(parts.permute(1, 2, 0, 3))


def assemble_full(parts13, parts13s, parts2, parts2s):
    """assemble_into on new tensors, for all four tensors of a layer (tests)."""
    outs = []
    for p, kind in ((parts13, "w13"), (parts13s, "w13"), (parts2, "w2"), (parts2s, "w2")):
        R, n, a, c = p.shape
        out = torch.empty((n,) + ((a * R, c) if kind == "w13" else (a, c * R)), dtype=p.dtype, device=p.device)
        assemble_into(out, p, kind)
        outs.append(out)
    return outs


def exchange_one(t: torch.Tensor, local_lists: List[List[int]], group) -> torch.Tensor:
    """all_to_all of one TP slice tensor: rank s sends rank r its slice of local_lists[r]; returns this rank's
    [RANKS, n_me, ...] received slices (uint8 view; e8m0 scales are bytes)."""
    import torch.distributed as dist

    me = group.rank_in_group
    n = [len(x) for x in local_lists]
    tb = t.view(torch.uint8)
    idx = torch.tensor([e for lst in local_lists for e in lst], device=t.device)
    send = tb.index_select(0, idx)
    recv = torch.empty((RANKS * n[me],) + tuple(tb.shape[1:]), dtype=torch.uint8, device=t.device)
    dist.all_to_all_single(recv, send, output_split_sizes=[n[me]] * RANKS, input_split_sizes=n, group=group.device_group)
    del send
    return recv.view((RANKS, n[me]) + tuple(tb.shape[1:]))


def exchange_to_full(tensors, local_lists: List[List[int]], group) -> List[torch.Tensor]:
    """exchange_one for each tensor (tests)."""
    return [exchange_one(t, local_lists, group) for t in tensors]


# ----------------------------------------------------------------------------- per-step split
@triton.jit
def _ep_waterfill_kernel(
    COUNTS, OWNER, HOT, RANGES, NVALID, E, n_hot,
    R: tl.constexpr, H: tl.constexpr, BLOCK_E: tl.constexpr,
):
    # One program. Loads of the ranks from their owned non-hot experts, then each hot expert's slots
    # water-filled over the R ranks in table order: RANGES[j, r] = (first, end) of the slot ordinals of hot
    # expert j that rank r computes; NVALID[r] = rows of rank r.
    ex = tl.arange(0, BLOCK_E)
    em = ex < E
    c = tl.load(COUNTS + ex, mask=em, other=0)
    own = tl.load(OWNER + ex, mask=em, other=-1)
    rr = tl.arange(0, R)
    hj = tl.arange(0, H)
    hot = tl.load(HOT + hj, mask=hj < n_hot, other=-1)
    is_hot = tl.max(tl.where(ex[:, None] == hot[None, :], 1, 0), 1) > 0
    load = tl.sum(tl.where((own[:, None] == rr[None, :]) & ~is_hot[:, None], c[:, None], 0), 0)  # [R] int32
    for j in range(0, n_hot):
        e = tl.load(HOT + j)
        x = tl.load(COUNTS + e)
        lf = load.to(tl.float32)
        lo = tl.min(lf, 0)
        hi = tl.max(lf, 0) + x.to(tl.float32)
        for _ in range(48):  # level t with sum(max(t - load, 0)) = x
            mid = (lo + hi) * 0.5
            f = tl.sum(tl.maximum(mid - lf, 0.0), 0)
            lo = tl.where(f < x.to(tl.float32), mid, lo)
            hi = tl.where(f < x.to(tl.float32), hi, mid)
        # lo is below the level (sum < x), so the floors never hand out more than x slots
        give = tl.maximum(lo - lf, 0.0).to(tl.int32)
        rem = x - tl.sum(give, 0)
        # the remainder (< R) goes to the least loaded ranks, one slot each, lowest rank first on ties
        after = load + give
        for _k in range(R):
            m = tl.min(after, 0)
            first = tl.min(tl.where(after == m, rr, R), 0)
            add = (rr == first) & (rem > 0)
            give = give + add.to(tl.int32)
            after = after + add.to(tl.int32)
            rem = rem - tl.sum(add.to(tl.int32), 0)
        end = tl.cumsum(give, 0)
        tl.store(RANGES + (j * R + rr) * 2, end - give)
        tl.store(RANGES + (j * R + rr) * 2 + 1, end)
        load = load + give
    tl.store(NVALID + rr, load)


@triton.jit
def _ep_hist_kernel(IDS, COUNTS, n_slots, E, BINS: tl.constexpr, BLOCK: tl.constexpr):
    # per-block histogram, then one atomic per (block, expert) (one atomic per slot on 384 counters took 108 us
    # at 391k slots)
    s = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(IDS + s, mask=s < n_slots, other=-1)
    g = tl.where((g >= 0) & (g < E), g, BINS - 1)
    h = tl.histogram(g, BINS)
    o = tl.arange(0, BINS)
    tl.atomic_add(COUNTS + o, h, mask=(o < E) & (h > 0), sem="relaxed")


@triton.jit
def _ep_block_counts_kernel(IDS, HOT, BC, n_slots, n_hot, nblk, H: tl.constexpr, BLOCK: tl.constexpr):
    # BC[j, b] = slots of hot expert j in block b of BLOCK slots
    b = tl.program_id(0)
    s = b * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(IDS + s, mask=s < n_slots, other=-1)
    for j in tl.static_range(H):
        if j < n_hot:
            e = tl.load(HOT + j)
            tl.store(BC + j * nblk + b, tl.sum((g == e).to(tl.int32), 0))


@triton.jit
def _ep_local_ids_kernel(
    IDS, LOCAL, HOT, HOTIDX, PRE, RANGES, OUT, n_slots, n_hot, nblk, me,
    R: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr,
):
    # A slot of a hot expert j is computed here when its ordinal among j's slots (slot order: PRE[b, j] =
    # j's slots in the blocks before b, plus the running count inside the block) is in RANGES[j, me];
    # any other slot when this rank holds its expert. PRE is [H, nblk].
    b = tl.program_id(0)
    s = b * BLOCK + tl.arange(0, BLOCK)
    m = s < n_slots
    g = tl.load(IDS + s, mask=m, other=-1)
    ok = m & (g >= 0)
    gs = tl.where(ok, g, 0)
    loc = tl.load(LOCAL + gs, mask=ok, other=-1)
    j = tl.load(HOTIDX + gs, mask=ok, other=-1)
    take = j < 0
    for jj in tl.static_range(H):
        if jj < n_hot:
            e = tl.load(HOT + jj)
            hit = ok & (g == e)
            ord_ = tl.load(PRE + jj * nblk + b) + tl.cumsum(hit.to(tl.int32), 0) - 1
            first = tl.load(RANGES + (jj * R + me) * 2)
            end = tl.load(RANGES + (jj * R + me) * 2 + 1)
            take = take | (hit & (ord_ >= first) & (ord_ < end))
    tl.store(OUT + s, tl.where(ok & take, loc, -1), mask=m)


class EpPlan:
    """Device tables of one layer's placement for one rank."""

    def __init__(self, owner: Sequence[int], hot: Sequence[int], rank: int, device):
        E = len(owner)
        assert len(hot) <= MAX_HOT
        self.E, self.rank, self.hot = E, rank, list(hot)
        self.experts = local_experts(owner, hot, rank)
        local = torch.full((E,), -1, dtype=torch.int32)
        local[torch.tensor(self.experts, dtype=torch.long)] = torch.arange(len(self.experts), dtype=torch.int32)
        hot_idx = torch.full((E,), -1, dtype=torch.int32)
        for j, e in enumerate(hot):
            hot_idx[e] = j
        self.local = local.to(device)
        self.hot_idx = hot_idx.to(device)
        self.owner = torch.tensor(owner, dtype=torch.int32, device=device)
        self.hot_t = torch.tensor(list(hot) + [-1] * (MAX_HOT - len(hot)), dtype=torch.int32, device=device)

    def assign(self, topk_ids: torch.Tensor):
        """(local ids [T, k] int32, -1 for slots of other ranks; NVALID [R] int32 rows per rank) on the device."""
        dev = topk_ids.device
        ids = topk_ids.reshape(-1).to(torch.int32).contiguous()
        n = ids.numel()
        counts = torch.zeros(self.E, dtype=torch.int32, device=dev)
        block = 1024
        bins = triton.next_power_of_2(self.E + 1)
        _ep_hist_kernel[(triton.cdiv(n, 4096),)](ids, counts, n, self.E, bins, 4096, num_warps=8)
        ranges = torch.zeros(MAX_HOT * RANKS * 2, dtype=torch.int32, device=dev)
        nvalid = torch.empty(RANKS, dtype=torch.int32, device=dev)
        _ep_waterfill_kernel[(1,)](
            counts, self.owner, self.hot_t, ranges, nvalid, self.E, len(self.hot),
            RANKS, MAX_HOT, triton.next_power_of_2(self.E), num_warps=4,
        )
        nblk = triton.cdiv(n, block)
        out = torch.empty_like(ids)
        if self.hot:
            bc = torch.zeros((MAX_HOT, nblk), dtype=torch.int32, device=dev)
            _ep_block_counts_kernel[(nblk,)](ids, self.hot_t, bc, n, len(self.hot), nblk, MAX_HOT, block, num_warps=4)
            pre = (torch.cumsum(bc, 1, dtype=torch.int32) - bc).contiguous()  # along the last dim: fast
        else:
            pre = torch.zeros((MAX_HOT, nblk), dtype=torch.int32, device=dev)
        _ep_local_ids_kernel[(nblk,)](
            ids, self.local, self.hot_t, self.hot_idx, pre, ranges, out, n, len(self.hot), nblk, self.rank,
            RANKS, MAX_HOT, block, num_warps=4,
        )
        return out.view(topk_ids.shape), nvalid


class Mxfp4W4A8EPExperts(_wp.Mxfp4W4A8PrefillExperts):
    """Mxfp4W4A8PrefillExperts over this rank's full-intermediate experts (EpPlan.experts order); takes the
    global top-k ids of all rows, computes only this rank's slots and returns this rank's partial routed output
    (zero for rows without a local slot), which the prefill-CP reduce-scatter sums like the TP partials."""

    def __init__(self, *args, plan: EpPlan, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.num_experts == len(plan.experts)
        self.plan = plan
        self._nv_host = None
        self._nv_ready = None

    def __call__(self, hidden_states, topk_ids, topk_weights, add_to=None, row_quant=None):
        import contextlib

        import triton_kernels.matmul_details.opt_flags as opt_flags
        from triton_kernels.matmul import PrecisionConfig, matmul
        from triton_kernels.tensor import make_ragged_tensor_metadata

        num_tokens, top_k = topk_ids.shape
        if num_tokens == 0:
            return torch.empty_like(hidden_states) if add_to is None else add_to
        local_ids, nvalid = self.plan.assign(topk_ids)
        if not (self.counting_route and self.fused_swiglu and self.sorted_sum) or (
            0 < num_tokens <= self.small_step_rows
        ):
            # Other configurations: the base class on the local ids (every path takes -1 slots).
            return super().__call__(hidden_states, local_ids, topk_weights, add_to=add_to, row_quant=row_quant)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("SGLANG_OPT_DSV4_W4A8_EP_TABLE does not support CUDA graphs")
        if _wp._qcp_take_shared_add(self, hidden_states) is not None:
            raise NotImplementedError("SGLANG_OPT_DSV4_W4A8_EP_TABLE with the k-qcp shared add in the top-k sum")
        # The row count of this rank sizes the GEMM outputs (sizing them by all slots, as the TP path does, would
        # allocate 4x the rows computed). It reaches the host through a pinned copy that the host waits for only
        # after it has queued the routing, the metadata and the input quant, which do not need it, so the GPU
        # keeps working while the host waits (one wait per layer).
        if self._nv_host is None:
            self._nv_host = torch.empty(RANKS, dtype=torch.int32, pin_memory=True)
            self._nv_ready = torch.cuda.Event()
        self._nv_host.copy_(nvalid, non_blocking=True)
        self._nv_ready.record()
        routing = _wp._count_route(local_ids, topk_weights, self.num_experts)
        md = make_ragged_tensor_metadata(routing.expert_counts, num_tokens * top_k)  # a bound; the rows are n
        if row_quant is not None:
            x_fp8, x_scale = row_quant
        else:
            row = _wp._qcp.take_row_quant(self, hidden_states)  # INF-681 k-qcp dual quant
            x_fp8, x_scale = row if row is not None else _wp._row_quant(hidden_states.contiguous())
        self._acquire()
        self._nv_ready.synchronize()
        n = int(self._nv_host[self.plan.rank])
        if n == 0:
            out0 = hidden_states.new_zeros(hidden_states.shape)
            return out0 if add_to is None else add_to
        # every row is written by the top-k sum (rows without a local slot get 0)
        output = hidden_states.new_empty(hidden_states.shape)
        token = opt_flags._opt_flags_constraints.set({})
        try:
            act = matmul(
                x_fp8, self._fp8_view(self.w13, self.scratch.w13), None, a_ragged_metadata=md,
                gather_indx=routing.gather_indx[:n],
                precision_config=PrecisionConfig(
                    a_mx_tensor_scale=x_scale, b_mx_tensor_scale=self.w13_channel_scale_il,
                    out_dtype=torch.bfloat16,
                ),
                fused_activation=_wp._swiglu_activation(self.swiglu_limit),
            )
            a_fp8, a_scale = _wp._rows_quant(act.view(-1, act.shape[-1]))
            del act
            w2_flags = None
            if 0 < self.w2_two_cta_min_rows <= num_tokens and n >= 128 * self.num_experts:
                w2_flags = _wp._w2_two_cta_flags()
            with (opt_flags.scoped_opt_flags(w2_flags) if w2_flags is not None else contextlib.nullcontext()):
                down = matmul(
                    a_fp8, self._fp8_view(self.w2, self.scratch.w2), None, a_ragged_metadata=md,
                    precision_config=PrecisionConfig(
                        a_mx_tensor_scale=a_scale, b_mx_tensor_scale=self.w2_channel_scale,
                        out_dtype=torch.bfloat16,
                    ),
                    gammas=routing.sorted_weights[:n],
                )
        finally:
            opt_flags._opt_flags_constraints.reset(token)
        overlapped = self.prefetch_chunks > 0 and num_tokens >= _wp._OVERLAP_MIN_ROWS
        if self.prefetch and not overlapped:
            self._prefetch_next(self._single_launch())
        fuse_add = add_to is not None and self.fused_shared_add
        if fuse_add:
            output = add_to
        _wp._topk_sum_by_pos(down, routing.pos, output, top_k, fuse_add)
        if self.prefetch and overlapped:
            self._prefetch_next(self.prefetch_chunks)
        if add_to is not None and not fuse_add:
            return add_to.add_(output)
        return output


def prepare_ep_experts(layer, layer_id: int, table_path: str, make_kwargs: dict, swiglu_limit: float):
    """Exchange this layer's TP slices into the rank's full experts (freeing the slices) and build the EP
    experts. Every rank of the MoE TP group calls this for the same layers in the same order."""
    from sglang.srt.distributed.parallel_state import get_moe_tp_group
    from sglang.srt.runtime_context import get_parallel

    group = get_moe_tp_group()
    assert group.world_size == RANKS, f"EP table needs a MoE TP group of {RANKS}, got {group.world_size}"
    me = group.rank_in_group
    assert me == get_parallel().moe_tp_rank
    owner, hot = load_table(table_path)[layer_id]
    E = layer.w13_weight.shape[0]
    assert len(owner) == E
    lists = [local_experts(owner, hot, r) for r in range(RANKS)]
    dev = layer.w13_weight.device
    n_me = len(lists[me])
    params = (
        ("w13_weight", "w13"), ("w13_weight_scale_inv", "w13"), ("w2_weight", "w2"), ("w2_weight_scale_inv", "w2"),
    )
    # The rank's full experts are allocated first, in fresh segments of their exact size; every temporary
    # (send / receive buffers) is freed and released before the next layer. Allocated after the temporaries, the
    # final tensors split their cached segments and the remainders stayed reserved (the first engine run kept
    # ~0.55 GiB per layer, 11.6 GiB, out of the KV pool).
    torch.cuda.empty_cache()
    finals = []
    for name, kind in params:
        t = getattr(layer, name).data
        finals.append(torch.empty((n_me,) + full_shape(kind, t.shape), dtype=torch.uint8, device=dev))
    for (name, kind), out in zip(params, finals):
        p = getattr(layer, name)
        recv = exchange_one(p.data, lists, group)
        assemble_into(out, recv, kind)
        del recv
        dtype = p.data.dtype
        p.data = out.view(dtype)  # frees the TP slice
    torch.cuda.empty_cache()
    f13, f13s, f2, f2s = (getattr(layer, name).data for name, _ in params)
    plan = EpPlan(owner, hot, me, dev)
    return Mxfp4W4A8EPExperts(f13, f13s, f2, f2s, swiglu_limit, plan=plan, **make_kwargs)
