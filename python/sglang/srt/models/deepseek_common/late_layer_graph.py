"""Breakable CUDA graph for the late layers of a DeepSeek-V4 prefill under decoder SWA
bounded replay (``SGLANG_OPT_DSV4_LATE_LAYER_GRAPH``, default off).

Under bounded replay the layers after the last kv_source layer run over each request's
last ``SWA_WINDOW`` extend rows only (at most max_running_requests x SWA_WINDOW rows).
Their GPU work is a few ms per layer, while launching them eagerly costs ~5 ms of CPU per
layer: once the launch queue's lead from the early layers is spent, the GPU waits on
Python. This runner captures layers ``late_layer_start .. end_layer`` once per row bucket
as a breakable CUDA graph and replays it.

What stays eager (graph breaks that read the live batch when they run):

- the window-KV store: its cache locations, and under prefill CP the gather of every
  rank's rows into the global token order;
- the low-ratio sources (the index-source layers' compressor/indexer, a per-request loop);
- the attention core (the tail's sparse metadata).

Everything else in those layers is row-local -- projections, RoPE, mHC, norms, the MoE
with its CP gather / reduce-scatter or TP all-reduce -- and is captured on a fixed-size
row block. Rows past the live count are zero on entry; the graph breaks only see the
live rows, and the rows past them are dropped on exit.

The first step that needs a bucket runs the late layers eagerly on that bucket's block
(that run is the step's result) and then captures the bucket; later steps replay it.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models import deepseek_v4_mhc as mhc

logger = logging.getLogger(__name__)

# Rows per late-layer block (this rank's tail rows, CP padding included). A step uses
# the smallest bucket that holds its rows; more rows than the largest bucket run eager.
_DEFAULT_BUCKETS = (
    32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024,
    1280, 1536, 1792, 2048, 2304, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192,
)  # fmt: skip


def parse_buckets(spec: str) -> Tuple[int, ...]:
    if not spec.strip():
        return _DEFAULT_BUCKETS
    buckets = sorted({int(x) for x in spec.split(",") if x.strip()})
    if not buckets or buckets[0] <= 0:
        raise ValueError(f"SGLANG_OPT_DSV4_LATE_LAYER_GRAPH_BUCKETS={spec!r}")
    return tuple(buckets)


def select_bucket(buckets: Tuple[int, ...], rows: int) -> Optional[int]:
    for bucket in buckets:
        if rows <= bucket:
            return bucket
    return None


# ---------------------------------------------------------------------------
# The live step: what the graph breaks read when they run
# ---------------------------------------------------------------------------


@dataclass
class _LiveStep:
    forward_batch: ForwardBatch
    # Live rows at the head of every late-layer row tensor this step.
    rows: int


_LIVE: Optional[_LiveStep] = None


def active() -> bool:
    """Whether the late layers run on a graph row block right now (eager warm-up,
    capture, replay, verify): the call sites then route through the breaks below."""
    return _LIVE is not None


def _live() -> _LiveStep:
    assert _LIVE is not None, "late-layer graph break outside a late-layer block"
    return _LIVE


def _set_live(step: Optional[_LiveStep]) -> None:
    global _LIVE
    _LIVE = step


@contextmanager
def _on_block(step: _LiveStep, num_token_non_padded: Optional[torch.Tensor]):
    """Run the late layers on a row block: the breaks see the live step, and the
    MoE's padded-row bound reads a runner-owned tensor (the captured graph keeps its
    address; the batch's own tensor dies with the batch)."""
    fb = step.forward_batch
    saved = getattr(fb, "num_token_non_padded", None)
    if saved is not None and num_token_non_padded is not None:
        fb.num_token_non_padded = num_token_non_padded
    _set_live(step)
    try:
        yield
    finally:
        _set_live(None)
        if saved is not None and num_token_non_padded is not None:
            fb.num_token_non_padded = saved


def _rows(t: Optional[torch.Tensor], n: int) -> Optional[torch.Tensor]:
    return None if t is None else t[:n]


def _noop(*args, **kwargs) -> None:
    # Capture stub: nothing a break produces is read while the segments are only
    # recorded, and skipping the body keeps rank-coupled breaks (the CP KV gather)
    # out of the capture.
    return None


# ---------------------------------------------------------------------------
# Graph breaks. Each takes the block's tensors (fixed addresses inside the graph) and
# runs the eager code on their live rows against the live batch.
# ---------------------------------------------------------------------------


@eager_on_graph(capture_stub=_noop)
def store_window_kv(layer, x, positions, qkv_a, q_rope) -> None:
    """MqaAttention._compute_kv_to_cache on the live rows (no prefill CP)."""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    step = _live()
    n = step.rows
    layer._compute_kv_to_cache(
        x[:n],
        positions[:n],
        step.forward_batch,
        get_attn_backend(),
        qkv_a=_rows(qkv_a, n),
        q_rope=_rows(q_rope, n),
    )


@eager_on_graph(capture_stub=_noop)
def cp_store_window_kv(layer, kv) -> None:
    """MqaAttention._cp_store_window_kv on the live rows (prefill CP gathers them)."""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    step = _live()
    layer._cp_store_window_kv(kv[: step.rows], step.forward_batch, get_attn_backend())


@eager_on_graph(capture_stub=_noop)
def low_ratio_sources(layer, x, q_lora, positions) -> None:
    from sglang.srt.model_executor.forward_context import get_attn_backend

    step = _live()
    n = step.rows
    get_attn_backend().forward_low_ratio_sources(
        layer=layer,
        x=x[:n],
        q_lora=_rows(q_lora, n),
        positions=positions[:n],
        forward_batch=step.forward_batch,
    )


@eager_on_graph(capture_stub=_noop)
def attention_core(
    attn_layer, q, k, output, compress_ratio, attn_sink, save_kv_cache
) -> None:
    """The backend's attention over the live rows; rows past them come out zero."""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    step = _live()
    n = step.rows
    ret = get_attn_backend().forward(
        q=q[:n],
        k=k[:n],
        v=k[:n],
        layer=attn_layer,
        forward_batch=step.forward_batch,
        compress_ratio=compress_ratio,
        attn_sink=attn_sink,
        save_kv_cache=save_kv_cache,
    )
    assert output[:n].numel() == ret.numel(), (output.shape, n, ret.shape)
    output[:n].view(ret.shape).copy_(ret)
    output[n:].zero_()


# ---------------------------------------------------------------------------
# The mHC state at the late-layer entry, flattened to fixed-address row tensors
# ---------------------------------------------------------------------------


def _flatten_state(state: mhc.HcState) -> Tuple[tuple, List[torch.Tensor]]:
    from sglang.srt.models import deepseek_v4_mhc as mhc

    if not isinstance(state.streams, torch.Tensor):
        raise _Unsupported("HcState streams is an outstanding post")
    names, tensors = ["streams"], [state.streams]
    if state.pre is not None:
        names.append("pre")
        tensors.append(state.pre)
    if state.input is not None:
        if not isinstance(state.input, mhc.HcNormed):
            raise _Unsupported(f"HcState input {type(state.input).__name__}")
        names.append("input")
        tensors.append(state.input.rows)
    if state.stats is not None:
        for j, t in enumerate(state.stats):
            names.append(f"stats{j}")
            tensors.append(t)
    sig = tuple(
        (name, tuple(t.shape[1:]), t.dtype) for name, t in zip(names, tensors)
    )
    return sig, tensors


def _unflatten_state(sig: tuple, tensors: List[torch.Tensor]) -> mhc.HcState:
    from sglang.srt.models import deepseek_v4_mhc as mhc

    by_name = {name: t for (name, _, _), t in zip(sig, tensors)}
    stats = None
    if "stats0" in by_name:
        stats = tuple(by_name[f"stats{j}"] for j in range(3))
    return mhc.HcState(
        by_name["streams"],
        by_name.get("pre"),
        mhc.HcNormed(by_name["input"]) if "input" in by_name else None,
        stats,
    )


class _Unsupported(Exception):
    """This step's late layers cannot use the graph (they run eagerly)."""


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    graph: BreakableCUDAGraph
    bucket: int
    # Views of the shared output buffers: (residual, pre or None, [aux ...]).
    outputs: Tuple[Optional[torch.Tensor], ...]
    # The MoE's padded-row bound the graph reads (the bucket: no row is masked,
    # as in the eager late layers, whose bound is the whole extend's count).
    num_token_non_padded: Optional[torch.Tensor] = None
    replays: int = 0


class LateLayerGraphRunner:
    """Captures and replays ``model._forward_late_layers`` per row bucket."""

    def __init__(self, model, max_rows: Optional[int] = None) -> None:
        self.model = model
        buckets = parse_buckets(envs.SGLANG_OPT_DSV4_LATE_LAYER_GRAPH_BUCKETS.get())
        # The shared row buffers are sized for the largest bucket this server can
        # reach, not the list's largest.
        top = None if max_rows is None else select_bucket(buckets, max_rows)
        self.buckets = buckets if top is None else tuple(b for b in buckets if b <= top)
        self.verify = envs.SGLANG_OPT_DSV4_LATE_LAYER_GRAPH_VERIFY.get()
        self._entries: Dict[tuple, _Entry] = {}
        # Per input signature: max-bucket buffers for the state, positions, ids.
        self._inputs: Dict[tuple, List[Optional[torch.Tensor]]] = {}
        self._outputs: Dict[tuple, List[Optional[torch.Tensor]]] = {}
        self._pool = None
        self._stream: Optional[torch.cuda.Stream] = None
        self._disabled: Optional[str] = None
        self._ntnp: Dict[int, torch.Tensor] = {}
        self._logged_unsupported: set = set()
        self.stats = {"replay": 0, "capture": 0, "eager": 0}

    # -- public --------------------------------------------------------------

    def run(
        self,
        *,
        state: mhc.HcState,
        positions: torch.Tensor,
        input_ids: Optional[torch.Tensor],
        input_ids_global: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        capture_dspark: bool,
        aux_out: List[torch.Tensor],
    ) -> Optional[mhc.HcState]:
        """Run the late layers on a graph block; None means "run them eagerly".

        On success the returned state is the materialized state after the last
        layer, and the DSpark aux rows of the late capture layers were appended to
        ``aux_out``.
        """
        if self._disabled is not None:
            return None
        rows = int(state.residual.shape[0]) if state.streams is not None else 0
        bucket = select_bucket(self.buckets, rows)
        if rows == 0 or bucket is None:
            self.stats["eager"] += 1
            return None
        try:
            sig, state_tensors = _flatten_state(state)
        except _Unsupported as e:
            self._note_unsupported(str(e))
            return None
        ids = (input_ids, input_ids_global)
        for t in (positions, *ids):
            if t is not None and t.shape[0] != rows:
                self._note_unsupported(f"row tensor of {t.shape[0]} rows, block {rows}")
                return None
        key_sig = (
            sig,
            positions.dtype,
            tuple(None if t is None else t.dtype for t in ids),
            capture_dspark,
        )
        inputs = self._input_buffers(key_sig, state_tensors, positions, ids)
        self._fill_inputs(inputs, [*state_tensors, positions, *ids], rows, bucket)
        entry = self._entries.get((bucket, key_sig))
        step = _LiveStep(forward_batch=forward_batch, rows=rows)
        if entry is None:
            result = self._warm_up_and_capture(
                key_sig, bucket, inputs, step, capture_dspark, ids
            )
        else:
            result = self._replay(entry, step)
            if self.verify:
                result = self._verify(
                    entry, key_sig, inputs, step, capture_dspark, ids, result,
                    state, positions, input_ids, input_ids_global,
                )  # fmt: skip
        residual, pre, aux = result
        aux_out.extend(aux)
        from sglang.srt.models import deepseek_v4_mhc as mhc

        return mhc.HcState(residual, pre)

    # -- buffers -------------------------------------------------------------

    def _input_buffers(self, key_sig, state_tensors, positions, ids):
        bufs = self._inputs.get(key_sig)
        if bufs is None:
            cap = self.buckets[-1]
            bufs = [
                None if t is None else t.new_zeros((cap, *t.shape[1:]))
                for t in (*state_tensors, positions, *ids)
            ]
            self._inputs[key_sig] = bufs
        return bufs

    @staticmethod
    def _fill_inputs(bufs, live, rows: int, bucket: int) -> None:
        for buf, t in zip(bufs, live):
            if buf is None:
                continue
            buf[:rows].copy_(t)
            if bucket > rows:
                buf[rows:bucket].zero_()

    def _block(self, key_sig, inputs, bucket: int, capture_dspark: bool, ids):
        """The body's arguments: views of the input buffers at the bucket size."""
        sig = key_sig[0]
        n_state = len(sig)
        views = [None if b is None else b[:bucket] for b in inputs]
        state = _unflatten_state(sig, views[:n_state])
        positions = views[n_state]
        input_ids, input_ids_global = views[n_state + 1], views[n_state + 2]
        return state, positions, input_ids, input_ids_global

    def _body(self, key_sig, inputs, bucket, step, capture_dspark, ids):
        state, positions, input_ids, input_ids_global = self._block(
            key_sig, inputs, bucket, capture_dspark, ids
        )
        state, aux = self.model._forward_late_layers(
            state=state,
            positions=positions,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
            forward_batch=step.forward_batch,
            capture_dspark=capture_dspark,
        )
        return state.residual, state.pre, aux

    def _num_token_non_padded(self, step: _LiveStep, bucket: int):
        live = getattr(step.forward_batch, "num_token_non_padded", None)
        if live is None:
            return None
        t = self._ntnp.get(bucket)
        if t is None or t.dtype != live.dtype or t.shape != live.shape:
            t = torch.full_like(live, bucket)
            self._ntnp[bucket] = t
        return t

    # -- warm-up, capture, replay ------------------------------------------------

    def _warm_up_and_capture(self, key_sig, bucket, inputs, step, capture_dspark, ids):
        # The warm-up is this step's result: the same eager code the graph records,
        # on the same block, so every lazily initialized kernel is ready for capture.
        ntnp = self._num_token_non_padded(step, bucket)
        with _on_block(step, ntnp):
            residual, pre, aux = self._body(
                key_sig, inputs, bucket, step, capture_dspark, ids
            )
            result = self._live_rows((residual, pre, aux), step.rows)
        del residual, pre, aux
        self.stats["eager"] += 1
        _drop_lazy_metadata_buffers()
        ok, err = self._capture(key_sig, bucket, inputs, step, capture_dspark, ids)
        if not self._all_ranks_ok(ok):
            self._disable(
                f"capture of bucket {bucket} failed"
                + (f": {err}" if err else " on another rank")
            )
            self._entries.pop((bucket, key_sig), None)
        return result

    def _capture(self, key_sig, bucket, inputs, step, capture_dspark, ids):
        from sglang.srt.distributed.parallel_state import graph_capture

        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
            self._stream = torch.cuda.Stream()
        t0 = time.perf_counter()
        graph = BreakableCUDAGraph()
        ntnp = self._num_token_non_padded(step, bucket)
        try:
            with _on_block(step, ntnp), graph_capture(stream=self._stream) as ctx:
                with BreakableCUDAGraphCapture(
                    cuda_graph=graph,
                    pool=self._pool,
                    stream=ctx.stream,
                    # Other threads (PD transfer, HiCache) keep using CUDA while a
                    # bucket is captured mid-serving.
                    capture_error_mode="thread_local",
                ):
                    residual, pre, aux = self._body(
                        key_sig, inputs, bucket, step, capture_dspark, ids
                    )
                    outputs = self._output_views(key_sig, bucket, residual, pre, aux)
                    for dst, src in zip(outputs, (residual, pre, *aux)):
                        if dst is not None:
                            dst.copy_(src)
        except Exception as e:  # noqa: BLE001 -- any capture failure falls back to eager
            logger.warning(
                "late-layer graph: capture of bucket %d failed: %s: %s",
                bucket,
                type(e).__name__,
                e,
            )
            return False, f"{type(e).__name__}: {e}"
        finally:
            # Buffers the capture cached on the live metadata are graph-pool memory.
            _drop_lazy_metadata_buffers()
        torch.cuda.current_stream().wait_stream(self._stream)
        self._entries[(bucket, key_sig)] = _Entry(
            graph=graph,
            bucket=bucket,
            outputs=tuple(outputs),
            num_token_non_padded=ntnp,
        )
        self.stats["capture"] += 1
        logger.info(
            "late-layer graph: captured bucket %d (%d live rows, %d segments) in %.2f s",
            bucket,
            step.rows,
            len(graph._segments),
            time.perf_counter() - t0,
        )
        return True, None

    def _output_views(self, key_sig, bucket, residual, pre, aux):
        bufs = self._outputs.get(key_sig)
        if bufs is None:
            cap = self.buckets[-1]
            bufs = [
                None if t is None else torch.empty(
                    (cap, *t.shape[1:]), dtype=t.dtype, device=t.device
                )
                for t in (residual, pre, *aux)
            ]  # fmt: skip
            self._outputs[key_sig] = bufs
        return [None if b is None else b[:bucket] for b in bufs]

    def _replay(self, entry: _Entry, step: _LiveStep):
        _set_live(step)
        try:
            entry.graph.replay()
        finally:
            _set_live(None)
        entry.replays += 1
        self.stats["replay"] += 1
        residual, pre, *aux = entry.outputs
        return self._live_rows((residual, pre, aux), step.rows)

    @staticmethod
    def _live_rows(result, rows: int):
        # Copies: the block is overwritten by the next step, and these rows can
        # still be read after it (DSpark draft prefill, PD hand-off).
        residual, pre, aux = result
        return (
            residual[:rows].clone(),
            None if pre is None else pre[:rows].clone(),
            [a[:rows].clone() for a in aux],
        )

    # -- verify (SGLANG_OPT_DSV4_LATE_LAYER_GRAPH_VERIFY) ------------------------

    def _verify(
        self, entry, key_sig, inputs, step, capture_dspark, ids, graph_result,
        state, positions, input_ids, input_ids_global,
    ):  # fmt: skip
        """Replay vs the same block run eagerly (must be bit-identical), and vs the
        unpadded eager late layers (padding effects only). Returns the unpadded
        eager result, so a verify run serves exactly what the default path serves.
        """
        # Re-fill: a replay may have written into its input rows in place.
        _, state_tensors = _flatten_state(state)
        self._fill_inputs(
            inputs,
            [*state_tensors, positions, input_ids, input_ids_global],
            step.rows,
            entry.bucket,
        )
        with _on_block(step, entry.num_token_non_padded):
            block_eager = self._live_rows(
                self._body(key_sig, inputs, entry.bucket, step, capture_dspark, ids),
                step.rows,
            )
        _drop_lazy_metadata_buffers()
        unpadded, aux = self.model._forward_late_layers(
            state=state,
            positions=positions,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
            forward_batch=step.forward_batch,
            capture_dspark=capture_dspark,
        )
        unpadded_result = (unpadded.residual, unpadded.pre, aux)
        logger.info(
            "late-layer graph verify: bucket %d rows %d | graph vs block eager %s | "
            "graph vs unpadded eager %s",
            entry.bucket,
            step.rows,
            _compare(graph_result, block_eager),
            _compare(graph_result, unpadded_result),
        )
        return unpadded_result

    # -- failure handling ----------------------------------------------------

    @staticmethod
    def _all_ranks_ok(ok: bool) -> bool:
        """Every rank must take the same path: the graphs hold collectives."""
        if not torch.distributed.is_initialized():
            return ok
        from sglang.srt.runtime_context import get_parallel

        group = get_parallel().tp_group
        flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
        torch.distributed.all_reduce(
            flag, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
        )
        return bool(flag.item())

    def _disable(self, reason: str) -> None:
        self._disabled = reason
        logger.warning("late-layer graph disabled, late layers run eagerly: %s", reason)

    def _note_unsupported(self, reason: str) -> None:
        self.stats["eager"] += 1
        if reason not in self._logged_unsupported:
            self._logged_unsupported.add(reason)
            logger.warning("late-layer graph: eager late layers (%s)", reason)


def _drop_lazy_metadata_buffers() -> None:
    """Forget per-step scratch the attention caches on the live tail metadata
    (``q_pad_buffer``, the zero-padded query heads under attention TP). The warm-up
    allocates it eagerly; a capture that found it would record that eager tensor's
    address instead of allocating (and zeroing) its own inside the graph."""
    try:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        meta = getattr(get_attn_backend(), "forward_metadata", None)
    except AssertionError:
        return
    if meta is not None and getattr(meta, "q_pad_buffer", None) is not None:
        meta.q_pad_buffer = None


def _compare(a, b) -> str:
    """Bitwise equality and max |diff| of (residual, pre, aux...)."""
    ta = [a[0], a[1], *a[2]]
    tb = [b[0], b[1], *b[2]]
    parts = []
    for name, x, y in zip(
        ["residual", "pre"] + [f"aux{j}" for j in range(len(a[2]))], ta, tb
    ):
        if x is None or y is None:
            continue
        if x.shape != y.shape:
            parts.append(f"{name} shape {tuple(x.shape)} vs {tuple(y.shape)}")
            continue
        equal = torch.equal(x, y)
        diff = (x.float() - y.float()).abs()
        parts.append(
            f"{name} equal={equal} max={diff.max().item():.3g} "
            f"mean={diff.mean().item():.3g}"
        )
    return "; ".join(parts)


def make_runner(model) -> Optional[LateLayerGraphRunner]:
    """The model's runner when the switch is on and its late layers are supported."""
    if not envs.SGLANG_OPT_DSV4_LATE_LAYER_GRAPH.get():
        return None
    reason = _unsupported_model_reason(model)
    if reason is not None:
        logger.warning("SGLANG_OPT_DSV4_LATE_LAYER_GRAPH ignored: %s", reason)
        return None
    runner = LateLayerGraphRunner(model, max_rows=_max_late_rows())
    logger.info(
        "late-layer graph on: layers %d..%d, row buckets %s",
        model.late_layer_start,
        model.end_layer - 1,
        runner.buckets,
    )
    return runner


def _max_late_rows() -> Optional[int]:
    """Most tail rows one rank can hold: every running request's SWA_WINDOW tail,
    split over the prefill-CP ranks (up to one more row per request on a rank)."""
    from sglang.srt.layers.attention.deepseek_v4_backend import SWA_WINDOW
    from sglang.srt.runtime_context import get_parallel, get_schedule

    try:
        max_reqs = get_schedule().max_running_requests
        cp = get_parallel().attn_cp_size
    except Exception:  # noqa: BLE001 -- no server context (unit tests)
        return None
    if not max_reqs:
        return None
    rows = int(max_reqs) * SWA_WINDOW
    if cp > 1:
        rows = -(-rows // cp) + int(max_reqs)
    return rows


def _unsupported_model_reason(model) -> Optional[str]:
    from sglang.srt.utils import is_cuda

    if model.late_layer_start is None:
        return "--enable-decoder-swa-bounded-replay is off"
    if not is_cuda():
        return "CUDA only"
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
        is_unified_kv_triton,
    )

    if is_unified_kv_triton():
        return "unified KV attention has its own store and attention call"
    for i in range(model.late_layer_start, model.end_layer):
        layer = model.layers[i]
        if getattr(layer, "engram", None) is not None:
            return f"layer {i} has an Engram"
        if getattr(layer, "hc_boundary_fused", False):
            return f"layer {i} uses the HIP fused mHC boundary"
    return None
