"""Keep the HiCache host pool out of a process checkpoint.

With ``SGLANG_CHECKPOINT_RELEASE_HOST_POOL=1`` a prepared engine holds no host
pool pages: the checkpoint (CRIU) then writes nothing for the pool, and the
CUDA restore has no registered host memory of the pool to pin again. The
pool's virtual addresses stay mapped, so every tensor, pointer table and
allocator built over them is unchanged; only the pages and the CUDA
registration go away and come back.

Prepare (``before_release``, before the KV cache is released):

1. The GPU is synchronized, so no device-to-host copy into the pool is still
   running, and the cache is flushed (``Scheduler.flush_cache``): the radix
   tree, device and host, is empty, so nothing indexes a page that restore
   brings back as zeros.
2. Every registered host-pool buffer is unregistered from CUDA with exactly
   the (pointer, size) ranges it was registered with at boot.
3. Its pages are returned: ``MADV_REMOVE`` on a shared mapping (the Mooncake
   allocator's memfd; this punches a hole in the file, which CRIU skips:
   criu/shmem.c ``do_dump_one_shmem`` dumps only the SEEK_DATA segments),
   ``MADV_DONTNEED`` on a private anonymous mapping (CRIU dumps only present
   pages). Any other mapping refuses the prepare.

Restore (``/checkpoint/restore``, before the L3 client is attached):

4. The pages are faulted in again in this process, several threads at once
   (``MADV_POPULATE_WRITE``). A criuctl post-criu hook
   (``hostpool-populate``) can do the same work from outside while the GPU
   is restored; then this step only walks page tables that are already full.
5. The same ranges are registered with CUDA again, one thread per buffer, and
   the kernel pointer tables of the pools are checked to be unchanged.

The L3 client (Mooncake store) registers the pool with its own transfer
engine when it is attached, so the pool must not be registered with an L3
client while it is released: the feature requires the deferred L3 client
(``SGLANG_CHECKPOINT_DEFER_HICACHE_STORAGE``) and refuses a prepare with an
attached one.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host.common import (
    _CUDA_HOST_REGISTERED_RANGES_ATTR,
    make_kernel_ptr_table,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

MADV_DONTNEED = 4
MADV_REMOVE = 9
MADV_POPULATE_WRITE = 23
PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
# A populate or register task covers at most this many bytes.
_CHUNK = 1 << 30
# Residency is sampled at this many pages per region (mincore of one page each).
_RESIDENCY_SAMPLES = 1024
# At or above this sampled residency the restore skips its own populate.
_POPULATED = 0.999


def releases_host_pool() -> bool:
    return bool(envs.SGLANG_CHECKPOINT_RELEASE_HOST_POOL.get())


def validate_host_pool_release(cfg: Any) -> None:
    """Refuse configurations the release does not cover."""
    if not envs.SGLANG_CHECKPOINT_RELEASE_HOST_POOL.get():
        return
    flag = "SGLANG_CHECKPOINT_RELEASE_HOST_POOL"
    if not cfg.enable_hierarchical_cache:
        return  # no host pool: nothing to release, a no-op
    if not cfg.checkpoint_prepare_on_startup:
        raise ValueError(f"{flag}=1 requires --checkpoint-prepare-on-startup")
    if (
        cfg.hicache_storage_backend is not None
        and not envs.SGLANG_CHECKPOINT_DEFER_HICACHE_STORAGE.get()
    ):
        # An attached L3 client keeps its own registration of the pool pages.
        raise ValueError(
            f"{flag}=1 with an L3 storage backend requires "
            "SGLANG_CHECKPOINT_DEFER_HICACHE_STORAGE=1"
        )


# ---------------------------------------------------------------------------
# The operations that touch the OS and CUDA, behind one object so that tests
# can run the release/re-register cycle with a fake allocator.
# ---------------------------------------------------------------------------


class HostOps:
    """Real operations: libc madvise/mincore, /proc/self/maps, cudart."""

    def __init__(self) -> None:
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        self._libc.madvise.restype = ctypes.c_int
        self._libc.mincore.argtypes = (
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_ubyte),
        )
        self._libc.mincore.restype = ctypes.c_int

    def madvise(self, addr: int, length: int, advice: int) -> int:
        """0, or the errno."""
        if self._libc.madvise(addr, length, advice) == 0:
            return 0
        return ctypes.get_errno()

    def resident_fraction(self, addr: int, length: int) -> float:
        """Fraction of sampled pages of [addr, addr+length) that are resident."""
        npages = length // PAGE
        if npages <= 0:
            return 1.0
        step = max(1, npages // _RESIDENCY_SAMPLES)
        vec = (ctypes.c_ubyte * 1)()
        seen = hit = 0
        for i in range(0, npages, step):
            if self._libc.mincore(addr + i * PAGE, PAGE, vec) != 0:
                continue
            seen += 1
            hit += vec[0] & 1
        return hit / seen if seen else 0.0

    def mapping_kinds(self, addrs: list[int]) -> list[str]:
        """'shared', 'private_anon', 'private_file' or 'none' for the VMA at each
        address (one read of /proc/self/maps)."""
        vmas = []
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split(None, 5)
                lo, hi = (int(x, 16) for x in parts[0].split("-"))
                path = parts[5].strip() if len(parts) > 5 else ""
                if parts[1][3] == "s":
                    kind = "shared"
                elif not path or path.startswith("["):
                    kind = "private_anon"
                else:
                    kind = "private_file"
                vmas.append((lo, hi, kind))
        out = []
        for addr in addrs:
            out.append(next((k for lo, hi, k in vmas if lo <= addr < hi), "none"))
        return out

    def mapping_kind(self, addr: int) -> str:
        return self.mapping_kinds([addr])[0]

    def cuda_register(self, ptr: int, size: int) -> int:
        return int(torch.cuda.cudart().cudaHostRegister(ptr, size, 0))

    def cuda_unregister(self, ptr: int) -> int:
        return int(torch.cuda.cudart().cudaHostUnregister(ptr))

    def cuda_error(self, rc: int) -> str:
        try:
            return str(torch.cuda.cudart().cudaGetErrorString(rc))
        except Exception:  # pragma: no cover - diagnostics only
            return "?"

    def synchronize(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def current_device(self) -> Optional[int]:
        return torch.cuda.current_device() if torch.cuda.is_available() else None

    def set_device(self, device: Optional[int]) -> None:
        if device is not None:
            torch.cuda.set_device(device)


_ops: HostOps | None = None


def get_ops() -> HostOps:
    global _ops
    if _ops is None:
        _ops = HostOps()
    return _ops


def set_ops(ops) -> None:
    """Tests only: replace the OS and CUDA operations."""
    global _ops
    _ops = ops


# ---------------------------------------------------------------------------
# What is released
# ---------------------------------------------------------------------------


@dataclass
class ReleasedBuffer:
    tensor: torch.Tensor  # keeps the buffer alive and carries its ranges again
    ranges: list[tuple[int, int]]  # its CUDA registration at boot, in order
    start: int  # the page-aligned span whose pages were returned
    end: int
    kind: str  # mapping kind: shared | private_anon

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass
class ReleasedHostPool:
    buffers: list[ReleasedBuffer] = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        return sum(b.nbytes for b in self.buffers)


def host_pools(tree_cache) -> list:
    """The host pools of a HiCache tree: the group's entries, else the one pool."""
    controller = getattr(tree_cache, "cache_controller", None)
    root = getattr(controller, "mem_pool_host", None)
    if root is None:
        return []
    entries = getattr(root, "entries", None)
    if entries is not None:
        return [e.host_pool for e in entries]
    return [root]


def _registered(t: Any) -> bool:
    return isinstance(t, torch.Tensor) and bool(
        getattr(t, _CUDA_HOST_REGISTERED_RANGES_ATTR, None)
    )


def registered_host_tensors(pools: list) -> list[torch.Tensor]:
    """Every CUDA-registered host tensor a pool holds (attributes and lists of
    them), once each. Registration ranges identify a buffer: data_refs alias
    kv_buffer elements, and two pools may share one domain."""
    out, seen = [], set()
    for pool in pools:
        for value in vars(pool).values():
            items = value if isinstance(value, (list, tuple)) else [value]
            for t in items:
                if not _registered(t):
                    continue
                key = tuple(getattr(t, _CUDA_HOST_REGISTERED_RANGES_ATTR))
                if key in seen:
                    continue
                seen.add(key)
                out.append(t)
    return out


def _page_span(t: torch.Tensor, ranges: list[tuple[int, int]]) -> tuple[int, int]:
    """The whole pages inside the buffer's registered bytes."""
    lo = min(p for p, _ in ranges)
    hi = max(p + s for p, s in ranges)
    start = (lo + PAGE - 1) // PAGE * PAGE
    end = hi // PAGE * PAGE
    return start, max(start, end)


def _log(stage: str, phase: str, seconds: float, **fields: Any) -> None:
    extra = "".join(f" {k}={v}" for k, v in fields.items())
    logger.info(f"Checkpoint {stage} phase={phase} seconds={seconds:.3f}{extra}")


def release_host_pool(scheduler: Scheduler) -> Optional[ReleasedHostPool]:
    """Prepare side; see the module docstring. Raises on anything that would
    leave the pool half released (the prepare then fails)."""
    if not releases_host_pool():
        return None
    tree_cache = scheduler.tree_cache
    pools = host_pools(tree_cache)
    if not pools:
        logger.info("Checkpoint prepare: no HiCache host pool to release")
        return None
    controller = tree_cache.cache_controller
    if getattr(controller, "enable_storage", False):
        raise RuntimeError(
            "SGLANG_CHECKPOINT_RELEASE_HOST_POOL: an L3 storage client is attached "
            "and holds the host pool registered; only the deferred L3 client "
            "(first prepare of a --checkpoint-prepare-on-startup boot) is covered"
        )
    ops = get_ops()
    tic = time.perf_counter()
    # No device-to-host copy (write-through) may still be writing into the pool.
    ops.synchronize()
    if not scheduler.flush_cache(empty_cache=False):
        raise RuntimeError("could not flush the cache before releasing the host pool")
    flushed = time.perf_counter() - tic

    tensors = registered_host_tensors(pools)
    released = ReleasedHostPool()
    spans = []
    for t in tensors:
        ranges = list(getattr(t, _CUDA_HOST_REGISTERED_RANGES_ATTR))
        spans.append((t, ranges, *_page_span(t, ranges)))
    kinds = ops.mapping_kinds([start for _, _, start, _ in spans])
    for (t, ranges, start, end), kind in zip(spans, kinds):
        if kind not in ("shared", "private_anon"):
            raise RuntimeError(
                f"host pool buffer at {start:#x} is a {kind} mapping; its pages "
                "cannot be returned without losing the mapping's meaning"
            )
        released.buffers.append(ReleasedBuffer(t, ranges, start, end, kind))

    for buf in released.buffers:
        for ptr, size in reversed(buf.ranges):
            rc = ops.cuda_unregister(ptr)
            if rc != 0:
                raise RuntimeError(
                    f"cudaHostUnregister({ptr:#x}, size={size}) returned {rc} "
                    f"({ops.cuda_error(rc)})"
                )
        setattr(buf.tensor, _CUDA_HOST_REGISTERED_RANGES_ATTR, [])
    unregistered = time.perf_counter() - tic

    for buf in released.buffers:
        if buf.nbytes == 0:
            continue
        advice = MADV_REMOVE if buf.kind == "shared" else MADV_DONTNEED
        err = ops.madvise(buf.start, buf.nbytes, advice)
        if err:
            raise RuntimeError(
                f"madvise({buf.start:#x}, {buf.nbytes}, "
                f"{'MADV_REMOVE' if advice == MADV_REMOVE else 'MADV_DONTNEED'}) "
                f"failed: {os.strerror(err)}"
            )
    resident = [ops.resident_fraction(b.start, b.nbytes) for b in released.buffers]
    left = max(resident) if resident else 0.0
    _log(
        "prepare",
        "host_pool_release",
        time.perf_counter() - tic,
        buffers=len(released.buffers),
        bytes=released.nbytes,
        kinds=",".join(sorted({b.kind for b in released.buffers})) or "-",
        flush_s=f"{flushed:.3f}",
        unregister_s=f"{unregistered - flushed:.3f}",
        max_resident_after=f"{left:.4f}",
    )
    if left > 0.01:
        raise RuntimeError(
            f"host pool pages still resident after release ({left:.1%} of a "
            "buffer): the checkpoint would carry them"
        )
    scheduler.released_host_pool = released
    return released


def _parallel(tasks: list[Callable[[], Any]], threads: int, device) -> list:
    ops = get_ops()

    def run(fn):
        ops.set_device(device)
        return fn()

    if threads <= 1 or len(tasks) <= 1:
        return [run(fn) for fn in tasks]
    with ThreadPoolExecutor(max_workers=min(threads, len(tasks))) as pool:
        return list(pool.map(run, tasks))


def restore_host_pool(scheduler: Scheduler) -> None:
    """Restore side; see the module docstring. A no-op when nothing was
    released. Raises (the restore then fails) on any registration failure or a
    pointer table that changed."""
    released: Optional[ReleasedHostPool] = getattr(
        scheduler, "released_host_pool", None
    )
    if released is None:
        return
    ops = get_ops()
    threads = max(1, envs.SGLANG_CHECKPOINT_HOST_POOL_THREADS.get())
    device = ops.current_device()

    tic = time.perf_counter()
    before = [ops.resident_fraction(b.start, b.nbytes) for b in released.buffers]
    # Already populated (hooks/hostpool-populate ran while the GPU was being
    # restored): walking the full page tables again would only cost time.
    skip = bool(before) and min(before) >= _POPULATED
    tasks, errors = [], []
    lock = threading.Lock()
    for buf in [] if skip else released.buffers:
        for a in range(buf.start, buf.end, _CHUNK):
            n = min(_CHUNK, buf.end - a)

            def populate(a=a, n=n):
                err = ops.madvise(a, n, MADV_POPULATE_WRITE)
                if err:
                    with lock:
                        errors.append(err)

            tasks.append(populate)
    _parallel(tasks, threads, device)
    # A kernel without MADV_POPULATE_WRITE (EINVAL) only costs speed:
    # cudaHostRegister faults the pages in itself.
    _log(
        "restore",
        "host_pool_populate",
        time.perf_counter() - tic,
        bytes=released.nbytes,
        resident_before=f"{min(before) if before else 1.0:.4f}",
        skipped=int(skip),
        threads=threads,
        errno=",".join(sorted({os.strerror(e) for e in errors})) or "-",
    )

    tic = time.perf_counter()

    def register(buf: ReleasedBuffer):
        done = []
        for ptr, size in buf.ranges:
            rc = ops.cuda_register(ptr, size)
            if rc != 0:
                for p, _ in reversed(done):
                    ops.cuda_unregister(p)
                return f"cudaHostRegister({ptr:#x}, {size}) returned {rc} ({ops.cuda_error(rc)})"
            done.append((ptr, size))
        setattr(buf.tensor, _CUDA_HOST_REGISTERED_RANGES_ATTR, list(buf.ranges))
        return None

    failures = [
        f
        for f in _parallel(
            [lambda b=b: register(b) for b in released.buffers], threads, device
        )
        if f
    ]
    if failures:
        raise RuntimeError(
            "re-registering the host pool failed: " + "; ".join(failures)
        )
    checked = _check_pointer_tables(host_pools(scheduler.tree_cache))
    scheduler.released_host_pool = None
    _log(
        "restore",
        "host_pool_register",
        time.perf_counter() - tic,
        buffers=len(released.buffers),
        bytes=released.nbytes,
        ranges=sum(len(b.ranges) for b in released.buffers),
        ptr_tables_checked=checked,
    )


def _check_pointer_tables(pools: list) -> int:
    """The device-side pointer tables built over the pool at boot must name the
    same addresses after the registration is redone (CUDA UVA keeps the host
    address as the device address; this verifies it instead of assuming it)."""
    n = 0
    for pool in pools:
        table = getattr(pool, "data_ptrs", None)
        refs = getattr(pool, "data_refs", None)
        if not isinstance(table, torch.Tensor) or not refs:
            continue
        fresh = make_kernel_ptr_table(
            refs,
            table.device,
            host_memory_registered=bool(getattr(pool, "pin_memory", True)),
        )
        # As int64 views: comparison kernels do not cover uint64 everywhere.
        if not torch.equal(
            fresh.cpu().view(torch.int64), table.cpu().view(torch.int64)
        ):
            raise RuntimeError(
                f"host pool {getattr(pool, 'pool_name', pool)!r}: the kernel pointer "
                "table differs after re-registration"
            )
        n += 1
    return n
