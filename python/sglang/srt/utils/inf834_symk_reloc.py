"""INF-834: NCCL symmetric-memory kernels in persisted (Foundry) CUDA graphs.

Why this exists. A regular NCCL kernel takes a device pointer to the
communicator state, and each process fills that state itself, so a graph
restored in another process still reads valid state. A *symmetric* kernel
(``ncclSymkDevKernel_*``, chosen when both buffers are registered windows,
i.e. under ``--enable-symm-mem``) takes the state BY VALUE in its launch
argument buffer (``ncclSymkDevWorkArgs``: ``kcomm`` + per-work window
pointers). A captured graph therefore bakes in addresses that belong to the
capturing process and that Foundry's allocation region does not place:

* ``kcomm.devComm.abortFlag``: ``cudaHostAlloc``'d pinned host memory (a host
  mmap address, different in every process);
* ``kcomm.devComm.windowTable`` / ``resourceWindow`` and the works'
  ``inputWin`` / ``outputWin``: objects of NCCL's shadow pool, a
  ``cudaMemPoolCreate`` pool the driver places itself;
* ``lsaFlatBase`` / the multicast base: reservations of 2 GB and more, which
  Foundry's hook places by hint after the region (the same on SAVE and LOAD
  when the hint is honoured).

A LOAD process replays the SAVE process's values and faults at the first
symmetric collective (``CUDA error: an illegal memory access``).

What this does (``SGLANG_OPT_INF834_SYMK_RELOC=1``, default off). At the same
point on SAVE and LOAD, right after the symmetric-memory windows are prebuilt,
it captures one tiny throwaway CUDA graph per symmetric collective and group
with the raw driver API and reads the kernel nodes' argument buffers
("probe"). SAVE stores the probe next to the graphs. LOAD compares its own
probe with SAVE's word by word; every 8-byte word that differs and looks like
an address is a pointer that moved. LOAD then writes copies of the archive's
``.cugraph`` files with those words replaced in the argument buffers of the
NCCL symmetric kernels and points Foundry at the copies. Nothing else in the
graphs changes; when no word moved, the archive is used as it is.

``SGLANG_OPT_INF834_ADDR_AUDIT_DIR=<dir>`` (diagnostic, default off) writes
``<dir>/<mode>-rank<N>.json`` (owner ranges + probe) and the process's
``/proc/self/maps`` for ``graph_addr_audit.py``.
"""

from __future__ import annotations

import atexit
import ctypes
import json
import logging
import os
import shutil
import struct
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)

PROBE_FILE = "inf834_symk_probe.json"
# A word is treated as an address when 4 GiB <= w < 2^47.
_PTR_MIN = 1 << 32
_PTR_MAX = 1 << 47

_SECTION_STRING_TABLE = 0
_SECTION_NODE_TABLE = 1
_SECTION_ARG_BUFFER_DATA = 5
_NODE_ENTRY_SIZE = 168
_NODE_KERNEL = 0

_shadow_dirs: list[str] = []


def reloc_enabled() -> bool:
    return os.environ.get("SGLANG_OPT_INF834_SYMK_RELOC", "0") == "1"


def audit_dir() -> Optional[str]:
    return os.environ.get("SGLANG_OPT_INF834_ADDR_AUDIT_DIR") or None


def _ptr_like(word: int) -> bool:
    return _PTR_MIN <= word < _PTR_MAX


def _foundry_config():
    """(mode, cfg) of an active Foundry persistence, or (None, None)."""
    from sglang.srt.utils.foundry_adapter import get_foundry_adapter

    adapter = get_foundry_adapter()
    if not adapter.enabled or adapter.mode not in ("save", "load"):
        return None, None
    from foundry.integration.sglang.config import get_config

    cfg = get_config()
    if cfg is None or cfg.workspace_dir is None:
        return None, None
    return adapter.mode, cfg


# ---------------------------------------------------------------------------
# Probe: argument buffers of the NCCL kernels of one captured collective
# ---------------------------------------------------------------------------


def _check(result, what: str):
    """cuda.bindings returns (CUresult, ...); raise on error, return the rest."""
    err = result[0]
    if int(err) != 0:
        raise RuntimeError(f"INF834 symk probe: {what} failed with {err!r}")
    rest = result[1:]
    return rest[0] if len(rest) == 1 else rest


def _read_extra_buffer(extra_addr: int) -> Optional[bytes]:
    """The launch parameter buffer a kernel node carries in ``extra``:
    {BUFFER_POINTER(1), ptr, BUFFER_SIZE(2), &size, END(0)} in any order."""
    if not extra_addr:
        return None
    buf_ptr = None
    size = None
    for i in range(0, 16, 2):
        key = ctypes.c_void_p.from_address(extra_addr + 8 * i).value or 0
        if key == 0:
            break
        value = ctypes.c_void_p.from_address(extra_addr + 8 * (i + 1)).value or 0
        if key == 1:
            buf_ptr = value
        elif key == 2 and value:
            size = ctypes.c_size_t.from_address(value).value
        else:
            return None
    if not buf_ptr or not size or size > (1 << 20):
        return None
    return ctypes.string_at(buf_ptr, size)


def _capture_kernel_args(stream: torch.cuda.Stream, launch) -> list[tuple[str, bytes]]:
    """Run ``launch()`` under a raw stream capture and return (kernel name,
    argument buffer) of every kernel node that carries one. The graph is
    destroyed; nothing is executed."""
    from cuda.bindings import driver as cu

    handle = cu.CUstream(stream.cuda_stream)
    _check(
        cu.cuStreamBeginCapture(
            handle, cu.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED
        ),
        "cuStreamBeginCapture",
    )
    graph = None
    try:
        launch()
    finally:
        graph = _check(cu.cuStreamEndCapture(handle), "cuStreamEndCapture")
    out: list[tuple[str, bytes]] = []
    try:
        _nodes, count = _check(cu.cuGraphGetNodes(graph, 0), "cuGraphGetNodes(count)")
        if count:
            nodes, _n = _check(cu.cuGraphGetNodes(graph, count), "cuGraphGetNodes")
            for node in nodes[:count]:
                ntype = _check(cu.cuGraphNodeGetType(node), "cuGraphNodeGetType")
                if ntype != cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
                    continue
                params = _check(
                    cu.cuGraphKernelNodeGetParams(node), "cuGraphKernelNodeGetParams"
                )
                try:
                    name = _check(cu.cuFuncGetName(params.func), "cuFuncGetName")
                    name = name.decode(errors="replace") if isinstance(name, bytes) else str(name)
                except Exception:  # noqa: BLE001 - a nameless kernel is still recorded
                    name = "<unnamed>"
                blob = _read_extra_buffer(int(params.extra or 0))
                if blob is not None:
                    out.append((name, blob))
    finally:
        cu.cuGraphDestroy(graph)
    return out


def _symmetric_groups() -> list:
    from sglang.srt.distributed.parallel_state import _groups

    groups = []
    for name in sorted(_groups):
        group = _groups[name]()
        if (
            group is None
            or group.world_size <= 1
            or getattr(group, "pynccl_comm", None) is None
        ):
            continue
        groups.append((name, group))
    return groups


def run_probe(*, device: str, forward_stream: torch.cuda.Stream) -> dict:
    """{label: {"kernel": name, "args": hex}} for one all-gather, reduce-scatter
    and all-reduce on symmetric buffers of every pynccl group, small and large.
    Collective: every rank calls it at the same point."""
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        use_symmetric_memory,
    )

    probe: dict = {}
    small, large = 1024, 4 * 1024 * 1024  # bf16 elements per rank, as the prebuild
    with torch.get_device_module(device).stream(forward_stream):
        for name, group in _symmetric_groups():
            world = group.world_size
            comm = group.pynccl_comm
            for size_label, per_rank in (("small", small), ("large", large)):
                with use_symmetric_memory(group):
                    full = torch.zeros(per_rank * world, dtype=torch.bfloat16, device=device)
                    part = torch.zeros(per_rank, dtype=torch.bfloat16, device=device)
                forward_stream.synchronize()
                ops = (
                    ("all_gather", lambda: comm.all_gather(full, part)),
                    ("reduce_scatter", lambda: comm.reduce_scatter(part, full)),
                    ("all_reduce", lambda: comm.all_reduce(full)),
                )
                with comm.change_state(enable=True):
                    for op_label, launch in ops:
                        captured = _capture_kernel_args(forward_stream, launch)
                        for index, (kernel, blob) in enumerate(captured):
                            label = f"{name}/{op_label}/{size_label}/{index}"
                            probe[label] = {"kernel": kernel, "args": blob.hex()}
                del full, part
        forward_stream.synchronize()
    return probe


# ---------------------------------------------------------------------------
# SAVE vs LOAD probe -> moved words
# ---------------------------------------------------------------------------


def moved_words(save_probe: dict, load_probe: dict) -> tuple[dict[int, int], list[str]]:
    """({SAVE word: LOAD word} for address-like words that differ, problems).

    A problem that starts with "note:" does not stop the relocation: a word that
    is not an address on one side, or a SAVE word seen with two LOAD values, is
    an unused (uninitialised) argument field, never rewritten."""
    candidates: dict[int, set[int]] = {}
    problems: list[str] = []
    for label in sorted(set(save_probe) | set(load_probe)):
        a, b = save_probe.get(label), load_probe.get(label)
        if a is None or b is None:
            problems.append(f"{label}: only in {'SAVE' if b is None else 'LOAD'}")
            continue
        if a["kernel"] != b["kernel"]:
            problems.append(f"{label}: kernel {a['kernel']} on SAVE, {b['kernel']} on LOAD")
            continue
        ab, bb = bytes.fromhex(a["args"]), bytes.fromhex(b["args"])
        if len(ab) != len(bb):
            problems.append(f"{label}: argument size {len(ab)} on SAVE, {len(bb)} on LOAD")
            continue
        usable = len(ab) // 8 * 8
        for off in range(0, usable, 8):
            (wa,) = struct.unpack_from("<Q", ab, off)
            (wb,) = struct.unpack_from("<Q", bb, off)
            if wa == wb:
                continue
            if _ptr_like(wa) and _ptr_like(wb):
                candidates.setdefault(wa, set()).add(wb)
            else:
                problems.append(f"note: {label}+{off}: non-address word 0x{wa:x} -> 0x{wb:x}")
    moved: dict[int, int] = {}
    for wa, targets in candidates.items():
        if len(targets) == 1:
            moved[wa] = next(iter(targets))
        else:
            shown = ", ".join(f"0x{t:x}" for t in sorted(targets)[:4])
            problems.append(f"note: 0x{wa:x} differs with {len(targets)} LOAD values ({shown}); not rewritten")
    return moved, problems


# ---------------------------------------------------------------------------
# .cugraph relocation
# ---------------------------------------------------------------------------


def relocate_cugraph(data: bytes, moved: dict[int, int], kernel_filter: str) -> tuple[Optional[bytearray], int, int]:
    """(patched copy or None when nothing changed, nodes touched, words replaced)."""
    magic, _ver, _flags, n_nodes, _nd, _ng, n_sections, _ns = struct.unpack_from(
        "<8sIIIIIII", data, 0
    )
    if magic != b"CUGRAPH\0":
        raise ValueError("not a .cugraph file")
    sections = {}
    for i in range(n_sections):
        stype, _pad, off, size = struct.unpack_from("<IIQQ", data, 64 + 24 * i)
        sections[stype] = (off, size)
    if _SECTION_ARG_BUFFER_DATA not in sections or _SECTION_NODE_TABLE not in sections:
        return None, 0, 0
    node_off, _node_size = sections[_SECTION_NODE_TABLE]
    str_off, _str_size = sections.get(_SECTION_STRING_TABLE, (0, 0))
    arg_off, arg_size = sections[_SECTION_ARG_BUFFER_DATA]
    patched: Optional[bytearray] = None
    done: set[int] = set()
    nodes_touched = words = 0
    for i in range(n_nodes):
        base = node_off + _NODE_ENTRY_SIZE * i
        if data[base + 4] != _NODE_KERNEL:
            continue
        body = base + 8
        fn_off, fn_len = struct.unpack_from("<II", data, body + 36)
        a_off, a_size = struct.unpack_from("<II", data, body + 128)
        if a_off == 0xFFFFFFFF or a_size < 8 or a_off + a_size > arg_size:
            continue
        name = data[str_off + fn_off : str_off + fn_off + fn_len]
        if kernel_filter.encode() not in name:
            continue
        touched = False
        start = arg_off + a_off
        for off in range(start, start + a_size // 8 * 8, 8):
            if off in done:
                continue
            (w,) = struct.unpack_from("<Q", data, off)
            new = moved.get(w)
            if new is None:
                continue
            if patched is None:
                patched = bytearray(data)
            struct.pack_into("<Q", patched, off, new)
            done.add(off)
            words += 1
            touched = True
        nodes_touched += touched
    return patched, nodes_touched, words


def _cleanup_shadow_dirs() -> None:
    for path in _shadow_dirs:
        shutil.rmtree(path, ignore_errors=True)


def build_shadow_workspace(workspace_dir: str, moved: dict[int, int], kernel_filter: str) -> tuple[str, dict]:
    """A directory with the archive's files (symlinks) and relocated copies of
    the ``.cugraph`` files that carry a moved word."""
    root = os.environ.get("SGLANG_OPT_INF834_SYMK_RELOC_DIR") or "/dev/shm"
    shadow = os.path.join(root, f"inf834-symk-reloc-{os.getpid()}", os.path.basename(workspace_dir))
    shutil.rmtree(os.path.dirname(shadow), ignore_errors=True)
    os.makedirs(shadow)
    if not _shadow_dirs:
        atexit.register(_cleanup_shadow_dirs)
    _shadow_dirs.append(os.path.dirname(shadow))
    stats = {"files": 0, "files_patched": 0, "nodes": 0, "words": 0, "bytes": 0}
    for entry in sorted(os.listdir(workspace_dir)):
        src = os.path.join(workspace_dir, entry)
        dst = os.path.join(shadow, entry)
        if entry.endswith(".cugraph"):
            stats["files"] += 1
            with open(src, "rb") as f:
                data = f.read()
            patched, nodes, words = relocate_cugraph(data, moved, kernel_filter)
            if patched is not None:
                with open(dst, "wb") as f:
                    f.write(patched)
                stats["files_patched"] += 1
                stats["nodes"] += nodes
                stats["words"] += words
                stats["bytes"] += len(patched)
                continue
        os.symlink(src, dst)
    return shadow, stats


# ---------------------------------------------------------------------------
# Entry point (capture_state.prepare_capture_state)
# ---------------------------------------------------------------------------

_done = False


def _owner_ranges(cfg) -> list[dict]:
    ranges: list[dict] = []
    try:
        from foundry.allocation_region import parse_size

        ranges.append(
            {
                "name": "foundry_region",
                "kind": "foundry_region",
                "base": int(cfg.base_addr),
                "size": int(parse_size(cfg.region_size)),
            }
        )
    except Exception as e:  # noqa: BLE001 - diagnostic only
        logger.warning(f"INF834 addr audit: region range unavailable: {e!r}")
    try:
        from sglang.srt.distributed.device_communicators import pynccl_allocator

        pool = pynccl_allocator._mem_pool
        if pool is not None:
            for i, segment in enumerate(pool.snapshot()):
                ranges.append(
                    {
                        "name": f"nccl_symm_pool_segment_{i}",
                        "kind": "nccl_symm_pool",
                        "base": int(segment["address"]),
                        "size": int(segment["total_size"]),
                    }
                )
    except Exception as e:  # noqa: BLE001 - diagnostic only
        logger.warning(f"INF834 addr audit: symmetric pool ranges unavailable: {e!r}")
    return ranges


def _write_audit(mode: str, rank: int, cfg, probe: dict, extra: dict) -> None:
    directory = audit_dir()
    if not directory:
        return
    try:
        os.makedirs(directory, exist_ok=True)
        doc = {"mode": mode, "rank": rank, "pid": os.getpid(), "ranges": _owner_ranges(cfg), "probe": probe}
        doc.update(extra)
        with open(os.path.join(directory, f"{mode}-rank{rank}.json"), "w") as f:
            json.dump(doc, f)
        shutil.copyfile("/proc/self/maps", os.path.join(directory, f"{mode}-rank{rank}.maps"))
        logger.info(f"INF834 addr audit: wrote {directory}/{mode}-rank{rank}.json")
    except Exception as e:  # noqa: BLE001 - diagnostic only
        logger.warning(f"INF834 addr audit: not written: {e!r}")


def probe_and_relocate(*, device: str, forward_stream: torch.cuda.Stream) -> None:
    """SAVE: record the probe. LOAD: relocate the archive's NCCL symmetric
    kernels to this process's addresses. No-op unless Foundry persistence is
    active, symmetric memory is on and the switch (or the audit) is set."""
    global _done
    if _done or not (reloc_enabled() or audit_dir()):
        return
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        is_symmetric_memory_enabled,
    )

    if not is_symmetric_memory_enabled():
        return
    mode, cfg = _foundry_config()
    if mode is None:
        return
    _done = True
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    t0 = time.perf_counter()
    probe = run_probe(device=device, forward_stream=forward_stream)
    probe_s = time.perf_counter() - t0
    if not probe:
        raise RuntimeError(
            "INF834 symk probe: no NCCL kernel argument buffer was captured "
            "(no pynccl group, or the collectives did not run under capture)"
        )
    kernels = sorted({v["kernel"] for v in probe.values()})
    workspace_dir = cfg.workspace_dir
    if mode == "save":
        with open(os.path.join(workspace_dir, PROBE_FILE), "w") as f:
            json.dump({"version": 1, "probe": probe}, f)
        logger.info(
            f"INF834 symk probe (save): {len(probe)} kernel argument buffers of "
            f"{len(kernels)} kernels recorded in {probe_s:.3f}s -> {workspace_dir}/{PROBE_FILE}"
        )
        _write_audit(mode, rank, cfg, probe, {})
        return

    path = os.path.join(workspace_dir, PROBE_FILE)
    if not os.path.exists(path):
        raise RuntimeError(
            f"INF834 symk relocation: {path} is missing: the archive was saved "
            "without SGLANG_OPT_INF834_SYMK_RELOC=1; re-SAVE with it"
        )
    with open(path) as f:
        save_probe = json.load(f)["probe"]
    moved, problems = moved_words(save_probe, probe)
    for problem in problems[:16]:
        logger.warning(f"INF834 symk relocation: {problem}")
    structural = [p for p in problems if not p.startswith("note:")]
    if structural:
        raise RuntimeError(
            f"INF834 symk relocation: SAVE and LOAD probes do not line up ({len(structural)} "
            f"problems, first: {structural[0]}); the archive cannot be relocated"
        )
    moved_text = ", ".join(f"0x{a:x}->0x{b:x}" for a, b in sorted(moved.items())[:12])
    extra = {"moved": {f"0x{a:x}": f"0x{b:x}" for a, b in moved.items()}, "problems": problems}
    if not reloc_enabled():
        logger.info(
            f"INF834 symk probe (load, audit only): {len(moved)} address words differ "
            f"from SAVE [{moved_text}]"
        )
        _write_audit(mode, rank, cfg, probe, extra)
        return
    if not moved:
        logger.info(
            f"INF834 symk relocation: 0 address words moved ({len(probe)} buffers "
            f"compared in {probe_s:.3f}s); archive used as saved"
        )
        _write_audit(mode, rank, cfg, probe, extra)
        return
    t1 = time.perf_counter()
    kernel_filter = os.environ.get("SGLANG_OPT_INF834_SYMK_RELOC_KERNELS", "ncclSymk")
    shadow, stats = build_shadow_workspace(workspace_dir, moved, kernel_filter)
    cfg.workspace_dir = shadow
    logger.info(
        f"INF834 symk relocation: {len(moved)} address words moved [{moved_text}]; "
        f"rewrote {stats['words']} words in {stats['nodes']} kernel nodes of "
        f"{stats['files_patched']}/{stats['files']} graphs ({stats['bytes'] / 1e6:.1f} MB) "
        f"into {shadow} in {time.perf_counter() - t1:.3f}s (probe {probe_s:.3f}s)"
    )
    extra["relocation"] = stats
    _write_audit(mode, rank, cfg, probe, extra)
