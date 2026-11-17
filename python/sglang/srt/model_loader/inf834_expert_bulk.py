"""INF-834 blockwise load of routed MoE experts (SGLANG_OPT_INF834_EXPERT_BULK_LOAD).

The default path hands every checkpoint expert tensor to FusedMoE.weight_loader on
its own: one Python task, a handful of tensor views and one host-to-device copy
from pageable (mmap) memory per tensor. A DeepSeek-V4.1 TP4 prefill rank does
that 92,160 times for 72 GB (1.5 MB weights, 92 KB scales), and the stream is
bound by that per-tensor work, not by the disk, the page cache or the device.

With the switch on, tensors that take the plain FusedMoE path are collected per
destination parameter in blocks of SGLANG_OPT_INF834_EXPERT_BULK_BLOCK experts.
When a block has all its pieces, one worker task gathers the TP slices into a
host staging buffer with a single torch.cat (the GIL is released for the whole
gather) and copies the block to the parameter with one copy_; the staging buffer
is pinned unless SGLANG_OPT_INF834_EXPERT_BULK_PIN=0.

The source slice and the destination of every piece are the ones
FusedMoE._load_w13 / _load_w2 use; a tensor or parameter outside the cases
listed in _plan_for / offer is left to the per-tensor loader, and so is every
piece of a block that never completes. The bytes that end up in the parameters
are the same; no device memory is allocated (copy_ into existing parameters).

SGLANG_OPT_INF834_DIRECT_READ (with SELECTIVE_READ): the selective reader leaves the
routed-expert tensors unpopulated and tags each with its file and offset. A block
whose pieces all carry that tag is filled by pread from the file into the staging
buffer instead of torch.cat from the mapping: a piece that is contiguous in the
file (w1/w3 rows of this rank) is read straight to its place; a strided piece (the
w2 column slice) is read as the covering byte range into a per-thread scratch
buffer and its columns copied out. Same bytes; the page tables are never filled.
That last case reads tp_size times the w2 bytes, which lost on a warm page cache, so
by default (SGLANG_OPT_INF834_DIRECT_READ_W2=0) only file-contiguous pieces are pread
and w2 blocks gather from the mapping, which the selective reader still populates.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Quant methods whose expert tensors FusedMoE._weight_loader_impl sends straight to
# _load_w13 / _load_w2 (no transpose, no per-method reshaping before the copy).
_PLAIN_METHODS = ("Mxfp4MarlinMoEMethod",)
_RUNAI_ATTR = "_sglang_runai_streamer_tensor"


class _Plan:
    """One eligible destination parameter: w13 (two row halves per expert, sharded
    on rows) or w2 (one piece per expert, sharded on columns)."""

    __slots__ = (
        "param", "layer", "kind", "pieces", "num_local", "tp_size", "tp_rank", "meta"
    )

    def __init__(self, param, layer, kind):
        self.param = param
        self.layer = layer
        self.kind = kind
        self.pieces = 2 if kind == "w13" else 1
        self.num_local = param.shape[0]
        self.tp_size = layer.moe_tp_size
        self.tp_rank = layer.moe_tp_rank
        # Expert-location metadata weight_loader routes through (None: plain mapping).
        self.meta = None


class _Block:
    __slots__ = ("plan", "first", "count", "slots", "srcs", "filled", "calls", "closed")

    def __init__(self, plan: _Plan, first: int, count: int):
        self.plan = plan
        self.first = first
        self.count = count
        self.slots: List[Optional[torch.Tensor]] = [None] * (count * plan.pieces)
        # DIRECT_READ: (path, file offset of the piece's first byte) per slot, or None.
        self.srcs: List[Optional[Tuple[str, int]]] = [None] * (count * plan.pieces)
        self.filled = 0
        # The original weight_loader arguments, in arrival order, for the fallback.
        self.calls: List[tuple] = []
        self.closed = False


class Inf834ExpertBulkLoader:
    """offer() takes a routed-expert tensor on its way to FusedMoE.weight_loader and
    returns True when this loader will place it; finish() hands whatever could not
    be placed blockwise back to the per-tensor loader."""

    def __init__(self, submit: Callable[..., None]):
        # submit(func, *args): run func(*args) on the model's load executor.
        self._submit = submit
        self.block_experts = max(1, envs.SGLANG_OPT_INF834_EXPERT_BULK_BLOCK.get())
        self._want_pin = envs.SGLANG_OPT_INF834_EXPERT_BULK_PIN.get()
        self._plans: Dict[int, Optional[_Plan]] = {}
        self._blocks: Dict[Tuple[int, int], _Block] = {}
        self._lock = threading.Lock()
        self._free: Dict[Tuple[int, bool], List[torch.Tensor]] = {}
        self._pinned_used = False
        self.tensors_bulk = 0
        self.tensors_noop = 0
        self.tensors_fallback = 0
        self.blocks_done = 0
        self.bytes_bulk = 0
        self.gather_secs = 0.0
        self.copy_secs = 0.0
        from sglang.srt.model_loader.inf834_selective_read import direct_read_enabled

        self._direct = direct_read_enabled()
        # Strided pieces (the w2 column slice) are pread only with DIRECT_READ_W2;
        # otherwise their block gathers from the (populated) mapping as before.
        self._direct_w2 = envs.SGLANG_OPT_INF834_DIRECT_READ_W2.get()
        self._fds: Dict[str, int] = {}
        self._tls = threading.local()
        self.blocks_direct = 0
        self.params_unplanned: List[str] = []
        self.bytes_direct_read = 0

    # ------------------------------------------------------------------ plan
    @staticmethod
    def _plan_for(param, weight_name: str) -> Optional[_Plan]:
        from sglang.srt.eplb.expert_location import (
            get_global_expert_location_metadata,
        )
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            FusedMoE,
            FusedMoeWeightScaleSupported,
        )

        loader = getattr(param, "weight_loader", None)
        layer = getattr(loader, "__self__", None)
        if (
            not isinstance(layer, FusedMoE)
            or getattr(loader, "__func__", None) is not FusedMoE.weight_loader
        ):
            return None
        # weight_loader with expert-location metadata (set for every target model,
        # also without EPLB) goes through _weight_loader_physical per physical copy;
        # offer() follows it when the logical expert has exactly one copy.
        meta = (
            get_global_expert_location_metadata()
            if layer.uses_expert_location
            else None
        )
        if getattr(param, "_sglang_require_global_experts", False):
            return None
        # _weight_loader_impl: nothing may touch the tensor before _load_w13/_load_w2.
        if layer.scheme is not None:
            return None
        if type(layer.quant_method).__name__ not in _PLAIN_METHODS:
            return None
        if (
            layer.use_padded_loading
            or layer.use_presharded_weights
            or layer.use_triton_kernels
            or layer.use_flashinfer_trtllm_moe
            or layer._has_fused_shared
            or not layer.moe_runner_config.is_gated
            or getattr(layer.quant_method, "load_up_proj_weight_first", False)
        ):
            return None
        if (
            getattr(param, "is_gguf_weight", False)
            or getattr(param, "is_gguf_weight_type", False)
            or getattr(param, "is_transposed", False)
        ):
            return None
        if param.data.dim() != 3:
            return None
        is_scale = None
        kind = None
        for attr, k, scale in (
            ("w13_weight", "w13", False),
            ("w2_weight", "w2", False),
            ("w13_weight_scale_inv", "w13", True),
            ("w2_weight_scale_inv", "w2", True),
        ):
            if getattr(layer, attr, None) is param:
                kind, is_scale = k, scale
        if kind is None:
            return None
        # The name decides the branch in _weight_loader_impl.
        if any(s in weight_name for s in ("input_scale", "g_idx")):
            return None
        if is_scale:
            if "scale" not in weight_name or getattr(
                param, "quant_method", None
            ) not in (
                FusedMoeWeightScaleSupported.GROUP.value,
                FusedMoeWeightScaleSupported.BLOCK.value,
            ):
                return None
        elif (
            any(s in weight_name for s in ("scale", "zero", "offset", "weight_shape"))
            or "weight" not in weight_name
        ):
            return None
        if kind == "w13" and param.shape[1] % 2:
            return None
        plan = _Plan(param, layer, kind)
        plan.meta = meta
        return plan

    # ----------------------------------------------------------------- offer
    def offer(
        self,
        param,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: Optional[int],
    ) -> bool:
        if (
            expert_id is None
            or loaded_weight.device.type != "cpu"
            or getattr(loaded_weight, _RUNAI_ATTR, False)
        ):
            return False
        key = id(param)
        if key not in self._plans:
            self._plans[key] = self._plan_for(param, weight_name)
            if self._plans[key] is None:
                self.params_unplanned.append(weight_name)
        plan = self._plans[key]
        if plan is None:
            return False
        layer = plan.layer
        if plan.meta is None:
            local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
            if local_id == -1:
                # weight_loader returns here too: the expert has no slot on this rank.
                self.tensors_noop += 1
                return True
            if not 0 <= local_id < plan.num_local:
                return False
        else:
            # weight_loader's metadata branch (no fused shared experts here, see
            # _plan_for): one _weight_loader_physical call per physical copy.
            physical = plan.meta.logical_to_all_physical(layer.layer_id, expert_id, False)
            if len(physical) != 1:
                return False  # several copies: the per-tensor loader places each
            local_id = layer._map_global_expert_id_to_local_expert_id(physical[0])
            if local_id < 0 or local_id >= layer.num_local_experts:
                self.tensors_noop += 1  # _weight_loader_physical returns here
                return True
            if local_id >= plan.num_local:
                return False
        data = param.data
        if loaded_weight.dtype != data.dtype or loaded_weight.dim() != 2:
            return False
        if plan.kind == "w13":
            if shard_id not in ("w1", "w3"):
                return False
            half = data.shape[1] // 2
            loaded_shard = loaded_weight.shape[0] // plan.tp_size
            src = loaded_weight.narrow(0, loaded_shard * plan.tp_rank, loaded_shard)
            if tuple(src.shape) != (half, data.shape[2]):
                return False
            piece = 1 if shard_id == "w3" else 0
        else:
            if shard_id != "w2":
                return False
            loaded_shard = loaded_weight.shape[1] // plan.tp_size
            src = loaded_weight.narrow(1, loaded_shard * plan.tp_rank, loaded_shard)
            if tuple(src.shape) != tuple(data.shape[1:]):
                return False
            piece = 0

        index = local_id // self.block_experts
        block = self._blocks.get((key, index))
        if block is None:
            first = index * self.block_experts
            block = _Block(plan, first, min(self.block_experts, plan.num_local - first))
            self._blocks[(key, index)] = block
        if block.closed:
            return False
        slot = (local_id - block.first) * plan.pieces + piece
        if block.slots[slot] is not None:
            # Not a checkpoint this path understands: give the whole block back.
            self._abandon(block)
            return False
        block.slots[slot] = src
        if self._direct and (src.is_contiguous() or self._direct_w2):
            tag = getattr(loaded_weight, "_inf834_file_src", None)
            if tag is not None:
                delta = src.data_ptr() - loaded_weight.data_ptr()
                block.srcs[slot] = (tag[0], tag[1] + delta)
        block.calls.append((param, loaded_weight, weight_name, shard_id, expert_id))
        block.filled += 1
        if block.filled == len(block.slots):
            block.closed = True
            self._submit(self._run_block, block)
        return True

    def _abandon(self, block: _Block) -> None:
        """Per-tensor loader for everything the block holds, in arrival order."""
        block.closed = True
        calls, block.calls, block.slots = block.calls, [], []
        for param, loaded_weight, weight_name, shard_id, expert_id in calls:
            self.tensors_fallback += 1
            self._submit(
                param.weight_loader,
                param,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
            )

    # ------------------------------------------------------------- block task
    def _staging(self, nbytes: int, pin: bool) -> torch.Tensor:
        with self._lock:
            free = self._free.get((nbytes, pin))
            if free:
                return free.pop()
        if pin:
            self._pinned_used = True
        return torch.empty(nbytes, dtype=torch.uint8, pin_memory=pin)

    def _run_block(self, block: _Block) -> None:
        plan = block.plan
        data = plan.param.data
        dst = data[block.first : block.first + block.count]
        pin = self._want_pin and data.device.type == "cuda"
        nbytes = dst.numel() * dst.element_size()
        staging = self._staging(nbytes, pin)
        tic = time.perf_counter()
        pieces = block.slots
        if self._direct and all(s is not None for s in block.srcs):
            self._read_block(block, staging)
            typed = staging.view(dst.dtype).view(dst.shape)
        elif dst.element_size() == 1:
            # Byte views: torch.cat does not cover every one-byte dtype (float8_*).
            pieces = [p.view(torch.uint8) for p in pieces]
            out = staging.view(-1, *pieces[0].shape[1:])
            torch.cat(pieces, dim=0, out=out)
            typed = staging.view(dst.dtype).view(dst.shape)
        else:
            typed = staging.view(dst.dtype).view(dst.shape)
            out = typed.view(-1, *pieces[0].shape[1:])
            torch.cat(pieces, dim=0, out=out)
        mid = time.perf_counter()
        dst.copy_(typed)
        end = time.perf_counter()
        n = len(block.slots)
        direct = self._direct and all(s is not None for s in block.srcs)
        block.slots, block.calls, block.srcs = [], [], []
        with self._lock:
            if direct:
                self.blocks_direct += 1
                self.bytes_direct_read += nbytes
            self._free.setdefault((nbytes, pin), []).append(staging)
            self.tensors_bulk += n
            self.blocks_done += 1
            self.bytes_bulk += nbytes
            self.gather_secs += mid - tic
            self.copy_secs += end - mid

    # ----------------------------------------------------------- direct read
    def _fd(self, path: str) -> int:
        with self._lock:
            fd = self._fds.get(path)
            if fd is None:
                fd = self._fds[path] = os.open(path, os.O_RDONLY)
            return fd

    def _scratch(self, nbytes: int) -> torch.Tensor:
        buf = getattr(self._tls, "buf", None)
        if buf is None or buf.numel() < nbytes:
            buf = self._tls.buf = torch.empty(nbytes, dtype=torch.uint8)
        return buf[:nbytes]

    @staticmethod
    def _pread(fd: int, dst: torch.Tensor, offset: int) -> None:
        """Fill the contiguous uint8 host tensor dst from the file at offset."""
        view = memoryview(dst.numpy())
        done, total = 0, dst.numel()
        while done < total:
            got = os.preadv(fd, [view[done:]], offset + done)
            if got <= 0:
                raise OSError(f"INF834 direct read: short read at {offset + done}")
            done += got

    def _read_block(self, block: _Block, staging: torch.Tensor) -> None:
        """staging (uint8) <- the pieces' bytes read from their files, in slot order:
        the layout torch.cat(block.slots) produces."""
        pos = 0
        for piece, (path, offset) in zip(block.slots, block.srcs):
            es = piece.element_size()
            nbytes = piece.numel() * es
            out = staging[pos : pos + nbytes]
            fd = self._fd(path)
            if piece.is_contiguous():
                self._pread(fd, out, offset)
            elif piece.dim() == 2 and piece.stride(1) == 1:
                rows, row_bytes = piece.shape[0], piece.shape[1] * es
                pitch = piece.stride(0) * es
                cover = self._scratch((rows - 1) * pitch + row_bytes)
                self._pread(fd, cover, offset)
                out.view(rows, row_bytes).copy_(
                    cover.as_strided((rows, row_bytes), (pitch, 1))
                )
            else:  # not a layout offer() produces; read it from the mapping
                out.copy_(piece.contiguous().view(-1).view(torch.uint8))
            pos += nbytes

    # ---------------------------------------------------------------- finish
    def finish(self) -> None:
        """Call after the last offer(), before waiting for the executor: blocks that
        never completed go to the per-tensor loader."""
        for block in list(self._blocks.values()):
            if not block.closed:
                self._abandon(block)

    def close(self, label: str) -> None:
        """Call after every submitted task finished: drop the staging buffers (and
        the pinned host cache they came from) and log what was done."""
        with self._lock:
            self._free.clear()
            fds, self._fds = list(self._fds.values()), {}
        for fd in fds:
            os.close(fd)
        if self._pinned_used:
            empty = getattr(torch._C, "_host_emptyCache", None)
            if empty is not None:
                empty()
        logger.info(
            "INF834 expert bulk load [%s]: %d tensors in %d blocks of up to %d experts "
            "(%.2f GB) through %s staging; %d left to the per-tensor loader, %d without "
            "a local slot; gather %.1fs, device copy %.1fs (summed over worker threads)",
            label,
            self.tensors_bulk,
            self.blocks_done,
            self.block_experts,
            self.bytes_bulk / 1e9,
            "pinned" if self._pinned_used else "pageable",
            self.tensors_fallback,
            self.tensors_noop,
            self.gather_secs,
            self.copy_secs,
        )
        if self.params_unplanned:
            logger.info(
                "INF834 expert bulk load [%s]: %d expert parameters not on the blockwise "
                "path (first: %s)",
                label,
                len(self.params_unplanned),
                ", ".join(self.params_unplanned[:3]),
            )
        if self._direct:
            logger.info(
                "INF834 direct read [%s]: %d of %d blocks (%.2f GB) read from the "
                "checkpoint files into staging with pread",
                label,
                self.blocks_direct,
                self.blocks_done,
                self.bytes_direct_read / 1e9,
            )


def make_expert_route(expert_params_mapping) -> Optional[Callable[[str], tuple]]:
    """The entries of FusedMoE.make_expert_params_mapping whose weight_name is a
    substring of a tensor name, without scanning all of them.

    Every weight_name is "experts.<id>.<proj>."; a name that holds exactly one such
    token can match that entry only, and a name that holds none matches no entry.
    Returns None (keep the scan) when the mapping has any other form; a name with
    several tokens gets the whole mapping back, i.e. the original scan.
    """
    import re

    token = re.compile(r"^experts\.(\d+)\.([A-Za-z0-9_]+)\.$")
    index: Dict[Tuple[int, str], tuple] = {}
    projs = set()
    for entry in expert_params_mapping:
        m = token.match(entry[1])
        if m is None or int(m.group(1)) != entry[2]:
            return None
        key = (entry[2], m.group(2))
        if key in index:
            return None
        index[key] = entry
        projs.add(m.group(2))
    if not index:
        return None
    # No projection name may contain another, or one token could satisfy two entries.
    if any(a != b and a in b for a in projs for b in projs):
        return None
    finder = re.compile(
        r"experts\.(\d+)\.(" + "|".join(sorted(map(re.escape, projs))) + r")\."
    )
    whole = tuple(expert_params_mapping)

    def route(name: str) -> tuple:
        found = finder.findall(name)
        if not found:
            return ()
        if len(found) > 1:
            return whole
        entry = index.get((int(found[0][0]), found[0][1]))
        # "experts.007.x." names expert 7 to the regex but not to the scan.
        return (entry,) if entry is not None and entry[1] in name else ()

    return route
