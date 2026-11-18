"""INF-834 SGLANG_OPT_INF834_HOST_POOL_PREALLOC: populate the mooncake-allocated
HiCache host pools while the weights load.

With --hicache-storage-backend=mooncake the host pools come from mooncake's
ShmHelper::allocate (memfd + mmap + MADV_POPULATE_WRITE). Their sizes follow from
the device pool sizes, which are known only after the weights are loaded, so the
populate (4 KiB shmem pages, a few GB/s) sits on the boot's critical path.

The sizes do not change from boot to boot of the same configuration. With the
switch on, every allocation size is recorded under
SGLANG_CACHE_DIR/inf834_host_pools/, and the next boot of that configuration
allocates segments of the recorded sizes in a background thread started when
the weight load starts. MooncakeHostTensorAllocator.allocate() then takes a
ready segment when the size it is asked for is among them, and allocates as
before when it is not. finalize() frees segments nobody took and rewrites the
record when the sizes differed.

A segment is the same object either way (same allocator call, same size, zero
filled); only the time of the call moves. cudaHostRegister stays where it was.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
from typing import Dict, List, Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cv = threading.Condition(_lock)
_started = False
_finalized = False
_thread: Optional[threading.Thread] = None
_allocator = None
_pending: List[int] = []  # recorded sizes the thread has not allocated yet
_in_progress: Optional[int] = None
_ready: Dict[int, List[int]] = {}  # size -> pointers nobody took yet
_stop = False
_observed: List[int] = []  # sizes asked for in this process, in order
_stats = {
    "prealloc_bytes": 0,
    "prealloc_secs": 0.0,
    "taken": 0,
    "taken_bytes": 0,
    "waited": 0.0,
}


# Server arguments the host pool sizes follow from. Nothing pod specific (address,
# pod name, ports): the record must match a new pod of the same configuration.
_KEY_FIELDS = (
    "tp_size",
    "dp_size",
    "attn_cp_size",
    "mem_fraction_static",
    "max_running_requests",
    "max_total_tokens",
    "chunked_prefill_size",
    "page_size",
    "kv_cache_dtype",
    "swa_prefix_tails",
    "speculative_algorithm",
    "disaggregation_mode",
    "hicache_ratio",
    "hicache_size",
    "hicache_mem_layout",
    "hicache_io_backend",
    "hicache_storage_backend",
)


def _record_key() -> str:
    parts = []
    try:
        from sglang.srt.runtime_context import get_server_args

        args = get_server_args()
        parts.append(os.path.basename(str(args.model_path).rstrip("/")))
        parts += [f"{f}={getattr(args, f, None)}" for f in _KEY_FIELDS]
    except Exception:
        # No published server args (tests): the argument vector, minus the address.
        parts += [a for a in sys.argv[1:] if not a.startswith(("--host", "--port"))]
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            parts.append(f"{props.name}:{props.total_memory}")
    except Exception:
        pass
    return hashlib.sha1("\0".join(parts).encode()).hexdigest()[:16]


def _record_path() -> str:
    return os.path.join(
        os.path.expanduser(envs.SGLANG_CACHE_DIR.get()),
        "inf834_host_pools",
        f"host_pools-{_record_key()}.json",
    )


def _read_record() -> List[int]:
    try:
        with open(_record_path()) as f:
            sizes = json.load(f)["sizes"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    if not isinstance(sizes, list) or not all(
        isinstance(s, int) and s > 0 for s in sizes
    ):
        return []
    return sizes


def _write_record(sizes: List[int]) -> None:
    path = _record_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"sizes": sizes}, f)
        os.replace(tmp, path)
    except OSError as e:
        logger.warning("INF834 host pool prealloc: cannot write %s: %s", path, e)


def _run(alloc_fn) -> None:
    global _in_progress
    while True:
        with _cv:
            if _stop or not _pending:
                _in_progress = None
                _cv.notify_all()
                return
            size = _pending.pop(0)
            _in_progress = size
        tic = time.perf_counter()
        try:
            ptr = alloc_fn(size)
        except Exception as e:  # the foreground path allocates and reports instead
            logger.warning("INF834 host pool prealloc: alloc(%d) failed: %s", size, e)
            with _cv:
                _pending.clear()
                _in_progress = None
                _cv.notify_all()
            return
        with _cv:
            _ready.setdefault(size, []).append(ptr)
            _stats["prealloc_bytes"] += size
            _stats["prealloc_secs"] += time.perf_counter() - tic
            _in_progress = None
            _cv.notify_all()


def start(sizes: List[int], alloc_fn, free_fn) -> None:
    """Start the background allocation of `sizes` (tests pass their own alloc/free)."""
    global _started, _thread, _allocator
    with _cv:
        if _started:
            return
        _started = True
        _pending.extend(sizes)
        _allocator = free_fn
    _thread = threading.Thread(
        target=_run, args=(alloc_fn,), name="inf834-host-prealloc", daemon=True
    )
    _thread.start()


def maybe_start_host_pool_prealloc() -> None:
    """Called when the target model's weight load starts."""
    if _started or not envs.SGLANG_OPT_INF834_HOST_POOL_PREALLOC.get():
        return
    from sglang.srt.mem_cache.pool_host.common import get_allocator_type

    try:
        if get_allocator_type() != "mooncake":
            return
    except Exception:
        return
    sizes = _read_record()
    if not sizes:
        logger.info(
            "INF834 host pool prealloc: no record at %s; this boot allocates in the "
            "foreground and writes it",
            _record_path(),
        )
        return
    try:
        from mooncake.store import MooncakeHostMemAllocator
    except ImportError:
        return
    allocator = MooncakeHostMemAllocator()
    logger.info(
        "INF834 host pool prealloc: started for %d segments (%.2f GB) from %s",
        len(sizes),
        sum(sizes) / 1e9,
        _record_path(),
    )
    start(sizes, allocator.alloc, allocator.free)


def take(size: int) -> Optional[int]:
    """A ready segment of exactly `size` bytes, waiting for the background thread
    when it is about to produce one; None when no such segment is coming."""
    with _cv:
        _observed.append(size)
        if not _started or _finalized:
            return None
        tic = time.perf_counter()
        while True:
            ptrs = _ready.get(size)
            if ptrs:
                _stats["taken"] += 1
                _stats["taken_bytes"] += size
                _stats["waited"] += time.perf_counter() - tic
                return ptrs.pop()
            if _in_progress != size and size not in _pending:
                return None
            _cv.wait()


def unclaimed_bytes() -> int:
    """Bytes held by ready segments nobody took yet (for the host memory budget)."""
    with _cv:
        return sum(size * len(ptrs) for size, ptrs in _ready.items())


def finalize() -> None:
    """Called once the host pools exist: stop, free what was not taken, record."""
    global _finalized, _stop
    if not envs.SGLANG_OPT_INF834_HOST_POOL_PREALLOC.get():
        return
    with _cv:
        if _finalized:
            return
        _finalized = True
        _stop = True
        _pending.clear()
    if _thread is not None:
        _thread.join()
    with _cv:
        leftovers = [(size, ptr) for size, ptrs in _ready.items() for ptr in ptrs]
        _ready.clear()
        observed = list(_observed)
    for _, ptr in leftovers:
        try:
            _allocator(ptr)
        except Exception as e:
            logger.warning("INF834 host pool prealloc: free failed: %s", e)
    recorded = _read_record()
    if observed and observed != recorded:
        _write_record(observed)
    logger.info(
        "INF834 host pool prealloc: took %d of %d requested segments (%.2f GB) from "
        "the background thread (populated %.2f GB in %.1fs, foreground waited %.1fs); "
        "freed %d unclaimed (%.2f GB); record %s",
        _stats["taken"],
        len(observed),
        _stats["taken_bytes"] / 1e9,
        _stats["prealloc_bytes"] / 1e9,
        _stats["prealloc_secs"],
        _stats["waited"],
        len(leftovers),
        sum(size for size, _ in leftovers) / 1e9,
        "unchanged" if observed == recorded else "written",
    )


def _reset_for_tests() -> None:
    global _started, _finalized, _thread, _allocator, _in_progress, _stop
    with _cv:
        _started = _finalized = _stop = False
        _thread = _allocator = _in_progress = None
        _pending.clear()
        _ready.clear()
        _observed.clear()
        for k in _stats:
            _stats[k] = 0 if isinstance(_stats[k], int) else 0.0
