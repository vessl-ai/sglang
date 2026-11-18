"""INF-834 selective checkpoint read (SGLANG_OPT_INF834_SELECTIVE_READ for the target
model, SGLANG_OPT_INF834_DRAFT_SHARDS for the DSpark draft).

The default loader hands every rank every checkpoint tensor through safetensors'
get_tensor, which (safetensors 0.8, torch 2.14) reads the whole tensor
synchronously while holding the GIL. So every rank reads all ~510 GB of the
checkpoint (the Engram tables in full, not just its rows), serialized on one
lock, and routes all ~96k names through load_weights. With a filter from the
model this path:

1. skips shards the index says hold no kept tensor,
2. builds zero-touch views from the safetensors header for kept tensors only, so
   the model's load loop never sees the rest (non-local experts, the draft's
   tensors in the target, vision when unused),
3. pre-populates the kept byte ranges (row ranges where the model gives them) of
   that mapping with MADV_POPULATE_READ from a thread pool, outside the GIL,
   inside the loader's existing sliding window.

The filter only removes tensors the model's load_weights would not consume; the
row ranges only steer the prefetch and never change what is copied. Host memory
and file I/O only: no device allocation is added, removed or reordered.
"""

from __future__ import annotations

import collections
import concurrent.futures
import ctypes
import dataclasses
import itertools
import json
import logging
import mmap
import os
import re
import struct
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import safetensors
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_MADV_POPULATE_READ = 22  # Linux >= 5.14; not in Python's mmap module
_CHUNK_BYTES = 64 << 20
_PAGE = mmap.PAGESIZE


@dataclasses.dataclass(frozen=True)
class Inf834ReadFilter:
    """keep(name): the model consumes this checkpoint tensor on this rank.
    rows(name, shape): (first row, row count) along dim 0 that the model copies,
    or None for the whole tensor. Used only to size the prefetch."""

    label: str
    keep: Callable[[str], bool]
    rows: Callable[[str, Sequence[int]], Optional[Tuple[int, int]]] = (
        lambda name, shape: None
    )


class _Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.files_total = 0
        self.files_skipped_by_index = 0
        self.tensors_seen = 0
        self.tensors_kept = 0
        self.bytes_kept = 0
        self.bytes_prefetched = 0
        self.prefetch_secs = 0.0
        self.bytes_direct = 0

    def add(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)


_libc = ctypes.CDLL(None, use_errno=True)
_libc.madvise.restype = ctypes.c_int
_libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
_pool_lock = threading.Lock()
_populate_disabled = False


def _get_pool() -> concurrent.futures.ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = concurrent.futures.ThreadPoolExecutor(
                max(1, envs.SGLANG_OPT_INF834_PREFETCH_THREADS.get()),
                thread_name_prefix="inf834-prefetch",
            )
        return _pool


def _populate(addr: int, length: int) -> int:
    start = addr & ~(_PAGE - 1)
    rc = _libc.madvise(
        ctypes.c_void_p(start), ctypes.c_size_t(length + addr - start), _MADV_POPULATE_READ
    )
    return 0 if rc == 0 else ctypes.get_errno()


def populate_ranges(ranges: List[Tuple[int, int]]) -> int:
    """MADV_POPULATE_READ over (address, nbytes) ranges in 64 MiB chunks on the
    shared pool; returns the bytes requested. A hint only: errors are logged."""
    global _populate_disabled
    if _populate_disabled or not ranges:
        return 0
    chunks = [
        (addr + off, min(_CHUNK_BYTES, nbytes - off))
        for addr, nbytes in ranges
        for off in range(0, nbytes, _CHUNK_BYTES)
    ]
    errors = [e for e in _get_pool().map(lambda c: _populate(*c), chunks) if e]
    if errors:
        _populate_disabled = True
        logger.warning(
            "INF834 selective read: MADV_POPULATE_READ failed (errno %d, %d of %d "
            "chunks); prefetch disabled for the rest of this load",
            errors[0],
            len(errors),
            len(chunks),
        )
    return sum(n for _, n in chunks)


def files_with_kept_tensors(
    files: List[str], hf_folder: str, read_filter: Inf834ReadFilter, stats: _Stats
) -> List[str]:
    """Drop shards that, by the safetensors index, hold no tensor the filter keeps.
    Without an index every shard is kept (load_file_filtered still filters names)."""
    stats.add(files_total=len(files))
    index_path = os.path.join(hf_folder, "model.safetensors.index.json")
    try:
        with open(index_path) as f:
            weight_map: Dict[str, str] = json.load(f)["weight_map"]
    except (OSError, ValueError, KeyError):
        return files
    needed = {fn for name, fn in weight_map.items() if read_filter.keep(name)}
    indexed = set(weight_map.values())
    kept = [
        f
        for f in files
        if os.path.basename(f) in needed or os.path.basename(f) not in indexed
    ]
    stats.add(files_skipped_by_index=len(files) - len(kept))
    return kept


# safetensors dtype names -> torch dtypes for the zero-touch views below; any other
# dtype falls back to safetensors' own get_tensor.
_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": getattr(torch, "float8_e8m0fnu", None),
}


def _read_header(st_file: str) -> Tuple[int, Dict[str, dict]]:
    with open(st_file, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return 8 + n, header


# Routed-expert checkpoint tensors (DeepSeek "layers.N.ffn.experts.E.w1.weight" and
# HF "model.layers.N.mlp.experts.E.gate_proj.weight" forms): with DIRECT_READ the
# blockwise expert loader reads these itself, so they are not populated here.
_DIRECT_READ_NAME = re.compile(r"(^|\.)experts\.\d+\.")


def direct_read_enabled() -> bool:
    return (
        envs.SGLANG_OPT_INF834_DIRECT_READ.get()
        and envs.SGLANG_OPT_INF834_EXPERT_BULK_LOAD.get()
    )


def load_file_filtered(
    st_file: str, read_filter: Inf834ReadFilter, stats: _Stats
) -> Dict[str, torch.Tensor]:
    """Views of the kept tensors of one shard, with their bytes populated.

    The views are built from the safetensors header over one private file
    mapping (what safetensors itself maps), not with get_tensor: in safetensors
    0.8 + torch 2.14 get_tensor and get_slice read the whole tensor
    synchronously while holding the GIL (torch.asarray on the storage slice),
    so each rank would read every tensor it is handed, Engram tables included.
    Same dtype, shape, strides and bytes as get_tensor; reading happens only in
    populate_ranges (kept bytes, row ranges where given) and in the consumer."""
    data_start, header = _read_header(st_file)
    kept = [k for k in header if read_filter.keep(k)]
    direct_read = direct_read_enabled()
    result: Dict[str, torch.Tensor] = {}
    ranges, kept_bytes = [], 0
    if kept:
        storage = torch.UntypedStorage.from_file(
            st_file, shared=False, nbytes=os.path.getsize(st_file)
        )
        base = storage.data_ptr()
        fallback = []
        for name in kept:
            info = header[name]
            begin, end = info["data_offsets"]
            dtype = _DTYPES.get(info["dtype"])
            if dtype is None:
                fallback.append(name)
                continue
            raw = torch.empty(0, dtype=torch.uint8, device="cpu").set_(
                storage, data_start + begin, (end - begin,), (1,)
            )
            t = raw.view(dtype).view(info["shape"])
            result[name] = t
            nbytes = end - begin
            kept_bytes += nbytes
            if direct_read:
                # Where the bytes are in the file (SGLANG_OPT_INF834_DIRECT_READ).
                t._inf834_file_src = (st_file, data_start + begin)
                if _DIRECT_READ_NAME.search(name) and (
                    envs.SGLANG_OPT_INF834_DIRECT_READ_W2.get()
                    or (t.dim() > 0 and read_filter.rows(name, tuple(info["shape"])))
                ):
                    # This rank's bytes of the tensor are one contiguous file range
                    # (rows() gives them), or w2 is pread as well: the blockwise
                    # loader preads them, so no populate here.
                    stats.add(bytes_direct=nbytes)
                    continue
            addr = base + data_start + begin
            rows = read_filter.rows(name, tuple(info["shape"])) if t.dim() > 0 else None
            if rows is not None and t.shape[0] > 0:
                row_bytes = nbytes // t.shape[0]
                addr, nbytes = addr + rows[0] * row_bytes, rows[1] * row_bytes
            if nbytes > 0:
                ranges.append((addr, nbytes))
        if fallback:
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
                for name in fallback:
                    result[name] = f.get_tensor(name)
                    kept_bytes += result[name].numel() * result[name].element_size()
    tic = time.perf_counter()
    prefetched = populate_ranges(ranges)
    stats.add(
        tensors_seen=len(header),
        tensors_kept=len(result),
        bytes_kept=kept_bytes,
        bytes_prefetched=prefetched,
        prefetch_secs=time.perf_counter() - tic,
    )
    return result


def fused_moe_skips_nonlocal(experts) -> bool:
    """True when FusedMoE.weight_loader (unoverridden) decides which experts this
    rank stores, so the decision below mirrors it. Anything else keeps all experts."""
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    if not isinstance(experts, FusedMoE):
        return False
    if getattr(type(experts), "weight_loader", None) is not FusedMoE.weight_loader:
        return False
    if type(getattr(experts, "quant_method", None)).__name__ == "KTEPWrapperMethod":
        return False
    return not any(
        getattr(p, "_sglang_require_global_experts", False)
        for p in experts.parameters(recurse=False)
    )


def expert_consumed(experts, meta, logical_id: int) -> bool:
    """Whether FusedMoE.weight_loader copies checkpoint expert `logical_id` into a
    local slot on this rank: the same branches as weight_loader and
    _weight_loader_physical, without touching the tensor."""
    if meta is None:
        return experts._map_global_expert_id_to_local_expert_id(logical_id) != -1
    shared_id = (
        logical_id - meta.num_logical_experts if experts._has_fused_shared else -1
    )
    if 0 <= shared_id < experts.num_fused_shared_experts:
        return True
    for physical_id in meta.logical_to_all_physical(experts.layer_id, logical_id, False):
        local = experts._map_global_expert_id_to_local_expert_id(physical_id)
        if 0 <= local < experts.num_local_experts:
            return True
    return False


def iterate(
    files: List[str],
    max_workers: int,
    read_filter: Inf834ReadFilter,
    stats: _Stats,
    drop_after_load: Optional[Callable[[str], None]] = None,
):
    """The default loader's sliding window (max_workers loading + 1 ready), with
    load_file_filtered as the per-shard loader. Yield order is unchanged: shard
    order, names sorted within a shard."""
    tic = time.perf_counter()
    from sglang.srt.model_loader import inf834_cold_preread

    files = list(files)
    preread = inf834_cold_preread.maybe_start(files, read_filter.label)
    try:
        yield from _iterate_window(
            files, max_workers, read_filter, stats, drop_after_load, preread
        )
    finally:
        if preread is not None:
            preread.stop()
            preread.log(read_filter.label)
    log_stats(read_filter, stats, time.perf_counter() - tic)


def _iterate_window(files, max_workers, read_filter, stats, drop_after_load, preread):
    started = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        file_iter = iter(files)
        pending = collections.deque(
            (f, executor.submit(load_file_filtered, f, read_filter, stats))
            for f in itertools.islice(file_iter, max_workers + 1)
        )
        started = len(pending)
        if preread is not None:
            preread.advance(started)
        while pending:
            st_file, future = pending.popleft()
            state_dict = future.result()
            del future
            next_file = next(file_iter, None)
            if next_file is not None:
                pending.append(
                    (
                        next_file,
                        executor.submit(
                            load_file_filtered, next_file, read_filter, stats
                        ),
                    )
                )
                started += 1
                if preread is not None:
                    preread.advance(started)
            for name in sorted(state_dict.keys()):
                yield name, state_dict[name]
            del state_dict
            if drop_after_load is not None:
                drop_after_load(st_file)


def new_stats() -> _Stats:
    return _Stats()


def log_stats(read_filter: Inf834ReadFilter, stats: _Stats, wall_secs: float) -> None:
    logger.info(
        "INF834 selective read [%s]: shards %d of %d opened (%d skipped by index), "
        "tensors kept %d of %d in opened shards, kept %.2f GB, prefetched %.2f GB "
        "(%.1f s in loader threads), iterator wall %.1f s",
        read_filter.label,
        stats.files_total - stats.files_skipped_by_index,
        stats.files_total,
        stats.files_skipped_by_index,
        stats.tensors_kept,
        stats.tensors_seen,
        stats.bytes_kept / 1e9,
        stats.bytes_prefetched / 1e9,
        stats.prefetch_secs,
        wall_secs,
    )
    if stats.bytes_direct:
        logger.info(
            "INF834 direct read [%s]: %.2f GB of routed-expert tensors left unpopulated "
            "for the blockwise loader to pread",
            read_filter.label,
            stats.bytes_direct / 1e9,
        )


def _tensor_digest(t: torch.Tensor) -> str:
    """Order-sensitive digest of a tensor's bytes: a position-weighted byte sum,
    computed in 16 MiB chunks on the GPU when there is one (host tensors are
    copied chunk by chunk), else on the CPU."""
    if t.numel() == 0:
        return "empty"
    flat = t.detach().reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    raw = flat.view(torch.uint8)
    device = (
        raw.device
        if raw.device.type != "cpu"
        else torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else raw.device
    )
    total, step = 0, 1 << 24
    for start in range(0, raw.numel(), step):
        chunk = raw[start : start + step].to(device).to(torch.int64)
        weight = torch.arange(start, start + chunk.numel(), device=device) % 65521 + 1
        total += int((chunk * weight).sum())
        del chunk, weight
    return f"w{total:x}"


def write_digest(model: torch.nn.Module, out_dir: str) -> None:
    """SGLANG_OPT_INF834_WEIGHT_DIGEST_DIR (diagnostic for the A/B proof): one JSON
    per rank and model class with a digest of every parameter, buffer and other
    tensor attribute after load and post-processing. Allocates temporary device
    memory (then empties the cache): never combine with a foundry SAVE/LOAD run."""
    import hashlib

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    tic = time.perf_counter()
    seen, digests = set(), {}
    for kind, items in (
        ("param", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, t in items:
            key = (t.data_ptr(), t.dtype, tuple(t.shape))
            seen.add(key)
            digests[f"{kind}:{name}"] = f"{t.dtype}|{tuple(t.shape)}|{t.device.type}|{_tensor_digest(t)}"
    for mod_name, mod in model.named_modules():
        # Weights a load-time transform moved off the parameters, kept as a tuple of tensors
        # on the module or on its quant method, e.g. the SM90 decode MoE kernels' packed
        # experts (Mxfp4MarlinMoEMethod._sm90_packed, a NamedTuple with an int field): digested
        # as buffers, so the byte gate covers the experts.
        holders = [("", vars(mod))]
        qm = getattr(mod, "quant_method", None)
        if qm is not None and hasattr(qm, "__dict__"):
            holders.append(("quant_method.", vars(qm)))
        for prefix, attrs in holders:
            for attr, value in attrs.items():
                if not isinstance(value, tuple) or not any(
                    isinstance(v, torch.Tensor) for v in value
                ):
                    continue
                names = getattr(value, "_fields", None) or [str(i) for i in range(len(value))]
                for field, v in zip(names, value):
                    if not isinstance(v, torch.Tensor):
                        continue
                    key = (v.data_ptr(), v.dtype, tuple(v.shape))
                    if key in seen:
                        continue
                    seen.add(key)
                    digests[f"buffer:{mod_name}.{prefix}{attr}.{field}"] = (
                        f"{v.dtype}|{tuple(v.shape)}|{v.device.type}|{_tensor_digest(v)}"
                    )
        for attr, value in vars(mod).items():
            if not isinstance(value, torch.Tensor) or attr.startswith("_"):
                continue
            key = (value.data_ptr(), value.dtype, tuple(value.shape))
            if key in seen:
                continue
            seen.add(key)
            digests[f"attr:{mod_name}.{attr}"] = (
                f"{value.dtype}|{tuple(value.shape)}|{value.device.type}|{_tensor_digest(value)}"
            )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    whole = hashlib.blake2b(
        "\n".join(f"{k}={v}" for k, v in sorted(digests.items()) if not k.startswith("attr:")).encode(),
        digest_size=12,
    ).hexdigest()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{type(model).__name__}-rank{rank}.json")
    with open(path, "w") as f:
        json.dump(digests, f, indent=0, sort_keys=True)
    logger.info(
        "INF834 weight digest %s rank=%d: %d tensors, params+buffers %s, %.1fs -> %s",
        type(model).__name__,
        rank,
        len(digests),
        whole,
        time.perf_counter() - tic,
        path,
    )
