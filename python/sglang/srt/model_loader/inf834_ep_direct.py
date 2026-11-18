"""INF-834 SGLANG_OPT_INF834_EP_DIRECT_EXPERTS: read an expert-parallel rank's routed experts straight
from the checkpoint files into pinned staging and copy them to the device, beside the weight stream.

Why: on the EP8 decoder every rank stores 52 of the 416 physical expert slots per MoE layer. Through the
selective reader those tensors arrive as views over a file mapping and FusedMoE.weight_loader copies each
one host-to-device from pageable memory, in the stream's order; the stream spent 11.5 of 12.2 s waiting
on the iterator. The blockwise loader (EXPERT_BULK_LOAD) cannot take these layers: a logical expert with
two physical copies (32 redundant slots per layer with eplb32) makes offer() decline, and a block with a
declined slot is handed back whole, so on the decoder every tensor fell back to the per-tensor loader.

What this does, for each MoE layer the selective reader filtered (selective read must be on):
1. The model's read filter drops that layer's routed-expert tensors from the stream.
2. Before the stream, plan() reads the safetensors headers (the index gives the shard of each name) and
   resolves every consumed checkpoint tensor through the same name remap and expert_params_mapping as
   load_weights, so each piece lands in the parameter, shard and local slot FusedMoE.weight_loader picks:
   checkpoint expert e goes to every local slot of every physical copy of e
   (logical_to_all_physical + _map_global_expert_id_to_local_expert_id, as _weight_loader_physical),
   w1 to rows [0, half) and w3 to rows [half, 2*half) of w13, w2 to the whole w2 slot (moe_tp_size 1).
3. A thread pool runs one task per (parameter, block of local slots): pread each piece from its file
   offset into its place in a pinned staging buffer, then one host-to-device copy of the block.
4. A parameter this path does not cover exactly (FusedMoE not on the plain loader path, tp > 1, a dtype
   or shape it does not expect, a local slot no checkpoint expert fills, a piece missing) goes through
   FusedMoE.weight_loader with the same views the stream would have given it.

Same bytes in the same places; no device memory is allocated (copies into existing parameters). Host:
pinned staging per worker thread, released at the end.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import struct
import threading
import time
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# DeepSeek-V4 checkpoint names of routed-expert tensors: layers.<L>.ffn.experts.<E>.<w1|w2|w3>.<leaf>
EXPERT_NAME = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.")

_ST_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": getattr(torch, "float8_e8m0fnu", None),
}


def enabled() -> bool:
    return (
        envs.SGLANG_OPT_INF834_EP_DIRECT_EXPERTS.get()
        and envs.SGLANG_OPT_INF834_SELECTIVE_READ.get()
    )


class _Src:
    __slots__ = ("name", "path", "offset", "nbytes", "dtype", "shape", "data_start", "begin")

    def __init__(self, name, path, data_start, begin, end, dtype, shape):
        self.name = name
        self.path = path
        self.data_start = data_start
        self.begin = begin
        self.offset = data_start + begin
        self.nbytes = end - begin
        self.dtype = dtype
        self.shape = tuple(shape)


class _Group:
    """One destination parameter: its pieces (local slot, row offset, source)."""

    __slots__ = ("param", "layer", "kind", "resolved", "entries", "pieces", "reason", "num_slots", "plan")

    def __init__(self, param, layer, kind, resolved):
        self.param = param
        self.layer = layer
        self.kind = kind
        self.resolved = resolved
        self.entries: List[tuple] = []  # (src, resolved_name, shard_id, expert_id)
        self.pieces: List[Tuple[int, int, _Src]] = []  # (local slot, row offset, src)
        self.reason: Optional[str] = None  # set: fall back to weight_loader
        self.num_slots = 0
        self.plan = None


def _read_header(path: str):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return 8 + n, header


def list_sources(model_path: str, wanted: Dict[int, Set[int]]) -> List[_Src]:
    """The checkpoint routed-expert tensors of the wanted (layer -> logical experts), ordered by
    layer, expert, name."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map: Dict[str, str] = json.load(f)["weight_map"]
    by_file: Dict[str, List[str]] = {}
    for name, fn in weight_map.items():
        m = EXPERT_NAME.match(name)
        if m is None:
            continue
        local = wanted.get(int(m.group(1)))
        if local is None or int(m.group(2)) not in local:
            continue
        by_file.setdefault(fn, []).append(name)
    out: List[_Src] = []
    for fn, names in by_file.items():
        path = os.path.join(model_path, fn)
        data_start, header = _read_header(path)
        for name in names:
            info = header[name]
            begin, end = info["data_offsets"]
            out.append(
                _Src(name, path, data_start, begin, end, _ST_DTYPES.get(info["dtype"]), info["shape"])
            )

    def order(s: _Src):
        m = EXPERT_NAME.match(s.name)
        return (int(m.group(1)), int(m.group(2)), s.name)

    out.sort(key=order)
    return out


def make_resolver(remap: Callable[[str], str], expert_params_mapping, params_dict):
    """resolve(ckpt_name) -> (param, resolved_name, shard_id, expert_id, remapped_name) or None:
    the stream's steps for a routed-expert name in DeepseekV4 load_weights (name remap, then the
    first expert_params_mapping entry whose weight_name is in the name)."""
    from sglang.srt.model_loader.inf834_expert_bulk import make_expert_route

    route = make_expert_route(expert_params_mapping)

    def resolve(ckpt_name: str):
        name = remap(ckpt_name)
        for param_name, weight_name, expert_id, shard_id in (
            expert_params_mapping if route is None else route(name)
        ):
            if weight_name not in name:
                continue
            resolved = name.replace(weight_name, param_name)
            if resolved not in params_dict:
                return None
            return params_dict[resolved], resolved, shard_id, expert_id, name
        return None

    return resolve


class Inf834EpDirectExperts:
    """plan() resolves the pieces, start() runs them on worker threads, join() waits and logs."""

    def __init__(
        self,
        sources: List[_Src],
        resolve: Callable[[str], Optional[tuple]],
        threads: Optional[int] = None,
        block_slots: Optional[int] = None,
    ):
        # resolve(ckpt_name) -> (param, resolved_name, shard_id, expert_id, remapped_name) or None
        self.sources = sources
        self.resolve = resolve
        self.threads = max(1, threads or envs.SGLANG_OPT_INF834_EP_DIRECT_THREADS.get())
        self.block_slots = max(1, block_slots or envs.SGLANG_OPT_INF834_EP_DIRECT_BLOCK.get())
        self.groups: List[_Group] = []
        self.loaded_names: Set[str] = set()
        self.weight_names: List[str] = []
        self.unresolved: List[str] = []
        self._pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._futures: List[concurrent.futures.Future] = []
        self._lock = threading.Lock()
        self._tls = threading.local()
        self._fds: Dict[str, int] = {}
        self._storages: Dict[str, torch.UntypedStorage] = {}
        self._pinned = []
        self.bytes_direct = 0
        self.tasks_direct = 0
        self.tensors_fallback = 0
        self.read_secs = 0.0
        self.copy_secs = 0.0
        self.t_start = None
        self.t_plan = 0.0

    # ------------------------------------------------------------------ plan
    def plan(self) -> None:
        from sglang.srt.model_loader.inf834_expert_bulk import Inf834ExpertBulkLoader

        tic = time.perf_counter()
        groups: Dict[int, _Group] = {}
        for src in self.sources:
            r = self.resolve(src.name)
            if r is None:
                self.unresolved.append(src.name)
                continue
            param, resolved, shard_id, expert_id, remapped = r
            self.weight_names.append(remapped)
            self.loaded_names.add(resolved)
            g = groups.get(id(param))
            if g is None:
                plan = Inf834ExpertBulkLoader._plan_for(param, resolved)
                layer = getattr(getattr(param, "weight_loader", None), "__self__", None)
                g = groups[id(param)] = _Group(param, layer, plan.kind if plan else None, resolved)
                if plan is None:
                    g.reason = "not on the plain FusedMoE loader path"
                elif plan.tp_size != 1:
                    g.reason = f"moe_tp_size {plan.tp_size}"
                else:
                    g.plan = plan
            g.entries.append((src, resolved, shard_id, expert_id))
        for g in groups.values():
            if g.reason is None:
                self._place(g)
        self.groups = sorted(
            groups.values(),
            key=lambda g: (getattr(g.layer, "layer_id", -1), g.kind or "", g.resolved),
        )
        self.t_plan = time.perf_counter() - tic

    def _place(self, g: _Group) -> None:
        """The local slot and row offset of every piece, as FusedMoE.weight_loader would place it;
        g.reason is set when this group does not fit that picture exactly."""
        layer, data = g.layer, g.param.data
        meta = g.plan.meta
        n_slots = min(layer.num_local_experts, data.shape[0])
        g.num_slots = n_slots
        half = data.shape[1] // 2
        filled: Dict[Tuple[int, int], int] = {}
        for src, resolved, shard_id, expert_id in g.entries:
            if src.dtype is None or src.dtype != data.dtype or len(src.shape) != 2:
                g.reason = f"{src.name}: dtype/rank {src.dtype} {src.shape} vs {data.dtype}"
                return
            if g.kind == "w13":
                if shard_id not in ("w1", "w3") or src.shape != (half, data.shape[2]):
                    g.reason = f"{src.name}: shard {shard_id} shape {src.shape}"
                    return
                row = 0 if shard_id == "w1" else half
            else:
                if shard_id != "w2" or src.shape != tuple(data.shape[1:]):
                    g.reason = f"{src.name}: shard {shard_id} shape {src.shape}"
                    return
                row = 0
            physical = (
                [expert_id]
                if meta is None
                else list(meta.logical_to_all_physical(layer.layer_id, expert_id, False))
            )
            for p in physical:
                local = layer._map_global_expert_id_to_local_expert_id(p)
                if meta is not None and (local < 0 or local >= layer.num_local_experts):
                    continue  # _weight_loader_physical returns
                if meta is None and local == -1:
                    continue  # weight_loader returns
                if local >= n_slots:
                    g.reason = f"{src.name}: local slot {local} beyond {n_slots}"
                    return
                key = (local, row)
                if key in filled:
                    g.reason = f"{src.name}: slot {local} row {row} written twice"
                    return
                filled[key] = 1
                g.pieces.append((local, row, src))
        rows_needed = (0, half) if g.kind == "w13" else (0,)
        missing = [(s, r) for s in range(n_slots) for r in rows_needed if (s, r) not in filled]
        if missing:
            g.reason = f"{len(missing)} slot pieces with no checkpoint tensor (first {missing[0]})"

    # ------------------------------------------------------------------- run
    def start(self) -> None:
        self.t_start = time.perf_counter()
        self._pool = concurrent.futures.ThreadPoolExecutor(
            self.threads, thread_name_prefix="inf834-ep-direct"
        )
        for g in self.groups:
            if g.reason is not None:
                self._futures.append(self._pool.submit(self._fallback, g))
                continue
            by_slot: Dict[int, List[Tuple[int, _Src]]] = {}
            for local, row, src in g.pieces:
                by_slot.setdefault(local, []).append((row, src))
            for first in range(0, g.num_slots, self.block_slots):
                count = min(self.block_slots, g.num_slots - first)
                pieces = [
                    (local - first, row, src)
                    for local in range(first, first + count)
                    for row, src in by_slot[local]
                ]
                self._futures.append(self._pool.submit(self._run_block, g, first, count, pieces))

    def _staging(self, nbytes: int, pin: bool) -> torch.Tensor:
        buf = getattr(self._tls, "buf", None)
        if buf is None or buf.numel() < nbytes or buf.is_pinned() != pin:
            buf = self._tls.buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pin)
            if pin:
                with self._lock:
                    self._pinned.append(buf)
        return buf[:nbytes]

    def _fd(self, path: str) -> int:
        with self._lock:
            fd = self._fds.get(path)
            if fd is None:
                fd = self._fds[path] = os.open(path, os.O_RDONLY)
            return fd

    @staticmethod
    def _pread(fd: int, dst: torch.Tensor, offset: int) -> None:
        view = memoryview(dst.numpy())
        done, total = 0, dst.numel()
        while done < total:
            got = os.preadv(fd, [view[done:]], offset + done)
            if got <= 0:
                raise OSError(f"INF834 ep direct: short read at {offset + done}")
            done += got

    def _run_block(self, g: _Group, first: int, count: int, pieces) -> None:
        data = g.param.data
        dst = data[first : first + count]
        if dst.device.type == "cuda":
            torch.cuda.set_device(dst.device)
        pin = dst.device.type == "cuda"
        slot_bytes = data[0].numel() * data.element_size()
        row_bytes = data.shape[2] * data.element_size()
        staging = self._staging(count * slot_bytes, pin)
        tic = time.perf_counter()
        for slot, row, src in pieces:
            pos = slot * slot_bytes + row * row_bytes
            self._pread(self._fd(src.path), staging[pos : pos + src.nbytes], src.offset)
        mid = time.perf_counter()
        dst.copy_(staging.view(dst.dtype).view(dst.shape))
        end = time.perf_counter()
        with self._lock:
            self.bytes_direct += count * slot_bytes
            self.tasks_direct += 1
            self.read_secs += mid - tic
            self.copy_secs += end - mid

    def _view(self, src: _Src) -> torch.Tensor:
        with self._lock:
            storage = self._storages.get(src.path)
            if storage is None:
                storage = self._storages[src.path] = torch.UntypedStorage.from_file(
                    src.path, shared=False, nbytes=os.path.getsize(src.path)
                )
        raw = torch.empty(0, dtype=torch.uint8).set_(storage, src.offset, (src.nbytes,), (1,))
        return raw.view(src.dtype).view(src.shape)

    def _fallback(self, g: _Group) -> None:
        """FusedMoE.weight_loader, as the stream would have called it, for a group this path
        does not place itself."""
        if g.param.data.device.type == "cuda":
            torch.cuda.set_device(g.param.data.device)
        for src, resolved, shard_id, expert_id in g.entries:
            g.param.weight_loader(
                g.param, self._view(src), resolved, shard_id=shard_id, expert_id=expert_id
            )
        with self._lock:
            self.tensors_fallback += len(g.entries)

    def launch(self) -> None:
        """plan() and start() on a background thread, so the weight stream starts at once;
        join() re-raises anything they raised."""
        self._launch_error = None

        def run():
            try:
                self.plan()
                self.start()
            except BaseException as e:  # re-raised by join()
                self._launch_error = e

        self._launcher = threading.Thread(target=run, name="inf834-ep-direct-plan", daemon=True)
        self._launcher.start()

    def join(self, label: str) -> None:
        launcher = getattr(self, "_launcher", None)
        if launcher is not None:
            launcher.join()
            if self._launch_error is not None:
                raise self._launch_error
        for f in concurrent.futures.as_completed(self._futures):
            f.result()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        wall = time.perf_counter() - (self.t_start or time.perf_counter())
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        self._storages.clear()
        pinned = sum(b.numel() for b in self._pinned)
        self._pinned.clear()
        self._tls = threading.local()
        if pinned:
            empty = getattr(torch._C, "_host_emptyCache", None)
            if empty is not None:
                empty()
        fallback_groups = [g for g in self.groups if g.reason is not None]
        logger.info(
            "INF834 ep direct experts [%s]: %d tensors, %d parameters direct (%d tasks, %.2f GB, "
            "pread %.1fs + device copy %.1fs summed over %d threads, block %d slots, pinned %.2f GB), "
            "%d parameters / %d tensors through weight_loader, %d unresolved; plan %.2fs, wall %.1fs",
            label,
            len(self.sources),
            len(self.groups) - len(fallback_groups),
            self.tasks_direct,
            self.bytes_direct / 1e9,
            self.read_secs,
            self.copy_secs,
            self.threads,
            self.block_slots,
            pinned / 1e9,
            len(fallback_groups),
            self.tensors_fallback,
            len(self.unresolved),
            self.t_plan,
            wall,
        )
        if fallback_groups:
            logger.info(
                "INF834 ep direct experts [%s]: weight_loader path for %s (first reason: %s)",
                label,
                fallback_groups[0].resolved,
                fallback_groups[0].reason,
            )
        if self.unresolved:
            raise RuntimeError(
                "INF834 ep direct experts: checkpoint expert tensors the model did not resolve "
                f"(first: {self.unresolved[:3]}); the stream no longer carries them"
            )


class LazyLaunch:
    """The model's read filter (and with it the list of experts this path reads) is built inside
    the weight iterator, at its first next(): DefaultModelLoader._get_all_weights is a generator.
    wrap() launches the direct reader right after the first tensor arrives, when the filter has
    decided; get_sources() returns the sources the filter left on the model (taking them), or None."""

    def __init__(self, get_sources: Callable[[], Optional[List[_Src]]], resolve):
        self.get_sources = get_sources
        self.resolve = resolve
        self.direct: Optional[Inf834EpDirectExperts] = None
        self._checked = False

    def _check(self) -> None:
        self._checked = True
        sources = self.get_sources()
        if sources:
            self.direct = Inf834EpDirectExperts(sources, self.resolve)
            self.direct.launch()

    def wrap(self, weights):
        for item in weights:
            if not self._checked:
                self._check()
            yield item
        if not self._checked:
            self._check()
