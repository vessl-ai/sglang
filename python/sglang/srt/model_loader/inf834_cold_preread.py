"""INF-834 SGLANG_OPT_INF834_COLD_PREREAD: fill the page cache with the checkpoint
shards ahead of the selective reader, with large sequential reads, each shard read
from disk by one rank of the node.

On a cold page cache the target weight stream of a TP4 DeepSeek-V4.1 prefill reads
the checkpoint through page faults: SELECTIVE_READ maps every shard in every rank
and populates the byte ranges it needs with MADV_POPULATE_READ, so the disk sees
fault-sized read-ahead requests from 4 x 8 threads, and the node read the 510 GB
at 5.5-6.9 GB/s (the volume does 20.7 GB/s with 16 threads of 16 MiB reads).

With the switch on, every rank starts a reader thread when the target stream
starts. The rank reads the shards whose position in the stream's file list is
congruent to its local rank modulo the local world size, in stream order, with
PREREAD_THREADS threads of 16 MiB pread into a scratch buffer whose contents are
dropped: the point is the page cache, which the four ranks share. A shard is
skipped when it is already resident (warm boot).

How far ahead it reads is bounded twice. At most PREREAD_AHEAD_FILES shards ahead
of the stream's front (the shards the loader has started). And within the pod's
memory cgroup: pre-read pages sit on the inactive file list until the loader
touches them, and that list is what reclaim takes first, so a read-ahead larger
than the memory the cgroup can still give to the page cache evicts its own unread
pages and they are read again (the prefill pod sits at its 480 GiB limit during
the weight load). A shard is read only while this rank's unconsumed read-ahead
plus the shard fits in

    (memory.max - anon - shmem - active_file - PREREAD_MARGIN_GB) / local ranks

from the cgroup's memory.stat; with no limit only the shard count applies.

Reads only: nothing the model loads changes; the populate and the copies that
follow find the pages in the page cache instead of reading the disk.
"""

from __future__ import annotations

import ctypes
import logging
import mmap
import os
import threading
import time
from typing import List, Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_CHUNK = 16 << 20
_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]


def resident_fraction(path: str, samples: int = 256) -> float:
    """Share of `samples` evenly spaced pages of the file that are in the page cache."""
    size = os.path.getsize(path)
    if size == 0:
        return 1.0
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = _libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            return 0.0
        try:
            page = mmap.PAGESIZE
            vec = (ctypes.c_ubyte * 1)()
            hit = 0
            n = max(1, min(samples, size // page))
            for i in range(n):
                off = (size // page * i // n) * page
                if _libc.mincore(ctypes.c_void_p(addr + off), page, vec) == 0 and vec[0] & 1:
                    hit += 1
            return hit / n
        finally:
            _libc.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)


def cgroup_headroom(cgroup_dir: str = "/sys/fs/cgroup") -> Optional[int]:
    """Bytes the memory cgroup can still give to new page cache without evicting
    active or unevictable memory: memory.max - anon - shmem - active_file. None
    when there is no limit or no cgroup v2 files."""
    try:
        with open(os.path.join(cgroup_dir, "memory.max")) as f:
            raw = f.read().strip()
        if raw == "max":
            return None
        limit = int(raw)
        stat = {}
        with open(os.path.join(cgroup_dir, "memory.stat")) as f:
            for line in f:
                k, _, v = line.partition(" ")
                stat[k] = int(v)
        used = stat.get("anon", 0) + stat.get("shmem", 0) + stat.get("active_file", 0)
        return limit - used
    except (OSError, ValueError):
        return None


def local_rank_and_size() -> tuple:
    """(local rank, ranks on this node) of the calling process; (0, 1) outside torch.distributed."""
    try:
        import torch
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            local = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
            local = min(local, dist.get_world_size())
            return dist.get_rank() % local, local
    except Exception:
        pass
    return 0, 1


class ColdPrereader:
    """Read this rank's share of `files` into the page cache, ahead of the stream."""

    def __init__(
        self,
        files: List[str],
        rank: int,
        nranks: int,
        threads: int,
        ahead: int,
        resident_skip: float = 0.9,
        margin_bytes: int = 16 << 30,
        cgroup_dir: str = "/sys/fs/cgroup",
    ):
        self.files = list(files)
        self.mine = [i for i in range(len(self.files)) if i % max(1, nranks) == rank]
        self.threads = max(1, threads)
        self.ahead = max(1, ahead)
        self.resident_skip = resident_skip
        self.nranks = max(1, nranks)
        self.margin = margin_bytes
        self.cgroup_dir = cgroup_dir
        self.sizes = {i: os.path.getsize(self.files[i]) for i in self.mine}
        self.files_late = 0  # skipped: no room before the loader reached them
        self.min_budget: Optional[int] = None
        self._done = set()
        self._front = 0  # number of files the stream has started
        self._cv = threading.Condition()
        self._stop = False
        self._thread: Optional[threading.Thread] = None
        self.bytes_read = 0
        self.files_read = 0
        self.files_skipped = 0
        self.waited_secs = 0.0
        self.read_secs = 0.0
        self.error: Optional[str] = None

    # -- stream side
    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="inf834-cold-preread", daemon=True
        )
        self._thread.start()

    def advance(self, started_files: int) -> None:
        with self._cv:
            self._front = max(self._front, started_files)
            self._cv.notify_all()

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join()

    # -- reader side
    def _budget(self) -> Optional[int]:
        room = cgroup_headroom(self.cgroup_dir)
        if room is None:
            return None
        budget = (room - self.margin) // self.nranks
        self.min_budget = budget if self.min_budget is None else min(self.min_budget, budget)
        return budget

    def _fits(self, i: int) -> bool:
        """This rank's unconsumed read-ahead (its shards the loader has not started)
        plus shard i fits in its share of the cgroup headroom."""
        budget = self._budget()
        if budget is None:
            return True
        ahead = sum(self.sizes[j] for j in self.mine if self._front <= j < i and j in self._done)
        return ahead + self.sizes[i] <= budget

    def _read_file(self, path: str, pool) -> int:
        size = os.path.getsize(path)
        fd = os.open(path, os.O_RDONLY)
        tls = threading.local()

        def chunk(off: int) -> int:
            buf = getattr(tls, "buf", None)
            if buf is None:
                buf = tls.buf = bytearray(_CHUNK)
            view = memoryview(buf)
            want = min(_CHUNK, size - off)
            done = 0
            while done < want:
                got = os.preadv(fd, [view[done:want]], off + done)
                if got <= 0:
                    break
                done += got
            return done

        try:
            return sum(pool.map(chunk, range(0, size, _CHUNK)))
        finally:
            os.close(fd)

    def _run(self) -> None:
        import concurrent.futures

        try:
            with concurrent.futures.ThreadPoolExecutor(
                self.threads, thread_name_prefix="inf834-preread"
            ) as pool:
                for i in self.mine:
                    tic = time.perf_counter()
                    late = False
                    with self._cv:
                        while not self._stop:
                            if i < self._front + self.ahead and self._fits(i):
                                break
                            if i < self._front:
                                late = True  # the loader is on it already; no room
                                break
                            self._cv.wait(0.5)
                        if self._stop:
                            return
                    self.waited_secs += time.perf_counter() - tic
                    if late:
                        self.files_late += 1
                        continue
                    path = self.files[i]
                    if resident_fraction(path) >= self.resident_skip:
                        self.files_skipped += 1
                        continue
                    tic = time.perf_counter()
                    self.bytes_read += self._read_file(path, pool)
                    self.read_secs += time.perf_counter() - tic
                    self.files_read += 1
                    self._done.add(i)
        except Exception as e:  # reads only: the stream reads the bytes itself anyway
            self.error = repr(e)

    def log(self, label: str) -> None:
        logger.info(
            "INF834 cold preread [%s]: %d of %d files read (%.2f GB, %.1f s reading at "
            "%.2f GB/s, %.1f s waiting for the window), %d already resident, %d skipped "
            "for lack of cgroup room (smallest per-rank budget %s)%s",
            label,
            self.files_read,
            len(self.mine),
            self.bytes_read / 1e9,
            self.read_secs,
            self.bytes_read / 1e9 / self.read_secs if self.read_secs else 0.0,
            self.waited_secs,
            self.files_skipped,
            self.files_late,
            "none" if self.min_budget is None else f"{self.min_budget / 1e9:.1f} GB",
            f"; stopped on {self.error}" if self.error else "",
        )


def maybe_start(files: List[str], label: str) -> Optional[ColdPrereader]:
    if not envs.SGLANG_OPT_INF834_COLD_PREREAD.get() or label != "target":
        return None
    rank, nranks = local_rank_and_size()
    reader = ColdPrereader(
        files,
        rank,
        nranks,
        envs.SGLANG_OPT_INF834_PREREAD_THREADS.get(),
        envs.SGLANG_OPT_INF834_PREREAD_AHEAD_FILES.get(),
        margin_bytes=int(envs.SGLANG_OPT_INF834_PREREAD_MARGIN_GB.get() * (1 << 30)),
    )
    logger.info(
        "INF834 cold preread [%s]: local rank %d of %d reads %d of %d files, %d threads, "
        "window %d files, cgroup margin %.0f GiB",
        label,
        rank,
        nranks,
        len(reader.mine),
        len(files),
        reader.threads,
        reader.ahead,
        reader.margin / (1 << 30),
    )
    reader.start()
    return reader
