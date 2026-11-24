"""INF-834 SGLANG_OPT_INF834_WEIGHT_SNAPSHOT: per-rank weight snapshot (node-prepared state).

Why: on a prepared node every boot re-derives each rank's weights from the safetensors checkpoint:
name remap, TP slicing, w2 column gathers, the streaming fp8 wo_a dequant and one Python step per
checkpoint tensor (93k on a prefill rank), with the GPU bytes going through pageable host-to-device
copies (decoder: 54 GB per rank at 47.6 GB/s across 8 GPUs). The result is a pure function of the
checkpoint, the arguments, the code and the rank. This module stores that result once per node and
rank and fills the parameters from it on later boots.

What is stored: every parameter and buffer of the model (deduplicated by storage), byte for byte, at
the end of the model's weight stream: after load_weights has consumed the checkpoint, before
load_weights' own post steps (DeepseekV4: post_load_weights, Engram finish_load, MHC prewarm) and
before process_weights_after_loading (repack). Those steps run unchanged after a restore. Engram host
tables are parameters (CPU, huge-page host memory) and are restored with pread straight into them.

Phase (SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_PHASE, part of the identity):
  stream_end  as above.
  repacked    the same file, except that the four raw expert parameters of every MoE layer that takes the
              EP decode repack (Mxfp4MarlinMoEMethod._prepare_sm90_decode_ep_weights) hold that layer's
              packed experts (_sm90_packed: the tensor fields concatenated, laid over w13_weight,
              w13_weight_scale_inv, w2_weight, w2_weight_scale_inv in that order; same byte count on the
              DSv4 shapes) instead of the raw bytes. The commit file records, per layer, the packed type,
              its fields (tensor shapes and dtypes, non-tensor values), the split and the scale maximum the
              repack checked. On restore the repack allocates the packed tensors as it would, copies them
              from the raw parameters on the device and skips the pack; everything else in
              process_weights_after_loading (kernel prepare, deletes, empty_cache) runs as before. The save
              writes the rest at stream end and each layer's packed bytes right after its pack; the commit
              happens after process_weights_after_loading (finish()), only if every expected layer delivered.

IO policy on restore (SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_DIRECT_IO): "auto" decides per 64 MiB chunk with
mincore right before reading it: resident -> buffered pread, otherwise O_DIRECT (the page cache can lose
part of the file between the boot's start and the read: on the decoder rig a file 97% resident 30 s
earlier was read 70% from the disk at 30 GB/s through the buffered path). After the restore the O_DIRECT
ranges are read again, buffered, by one background thread (SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_REWARM), so
the next boot is warm. "file": one decision for the whole file; "1"/"0": always O_DIRECT / buffered.

Engram host tables (SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_DEFER_PREFAULT): the per-rank tables skip the
pre-fault and the cudaHostRegister at model init; begin() registers them after the restore wrote (and so
faulted) every page, or pre-faults and registers them first when nothing is restored.

Files, under SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_DIR / <label>-<identity digest> /:
  rank<R>.bin      the bytes, every entry at a 4 KiB aligned offset, padded to 4 KiB
  rank<R>.commit   JSON: identity, entry table, per-64-MiB-chunk xxh3 checksums. Written last
                   (temp file, fsync, rename, directory fsync); a .bin without it is ignored.
  rank<R>.lock     flock held while saving (another process on the node saving the same rank skips)

Modes (SGLANG_OPT_INF834_WEIGHT_SNAPSHOT): "load" restores when a committed snapshot with the same
identity exists, else loads normally; "save" loads normally and writes the snapshot; "auto" does
"load" and, when nothing was restored, "save". Anything that does not match (identity, entry table,
file size, the repacked layers) falls back to the normal load before a parameter is touched. A checksum mismatch found
while filling makes the caller run the normal load over everything. Fallbacks after a committed
snapshot was found are logged as one WARNING per rank starting with "INF834 weight snapshot FALLBACK".
"""

from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import logging
import mmap
import os
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
ALIGN = 4096
CHUNK = 64 << 20
FALLBACK = "INF834 weight snapshot FALLBACK"
ALLOWED_MODELS = ("DeepseekV4ForCausalLM", "DeepseekV4ForCausalLMDSpark")
PHASES = ("stream_end", "repacked")
# The raw expert parameters a repacked layer's packed bytes are laid over, in this order.
REPACK_PARAMS = ("w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv")

try:
    import xxhash

    def _checksum(buf) -> str:
        return "x" + format(xxhash.xxh3_64_intdigest(buf), "016x")

except ImportError:  # pragma: no cover
    import zlib

    def _checksum(buf) -> str:
        return "c" + format(zlib.crc32(buf), "08x")


def mode() -> str:
    m = (envs.SGLANG_OPT_INF834_WEIGHT_SNAPSHOT.get() or "").strip().lower()
    return m if m in ("load", "save", "auto") else ""


def phase() -> str:
    p = (envs.SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_PHASE.get() or "stream_end").strip().lower()
    if p not in PHASES:
        raise ValueError(f"SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_PHASE must be one of {PHASES}, got {p!r}")
    return p


def defers_host_prefault() -> bool:
    """Engram host tables skip the pre-fault and the registration at init (WeightSnapshot.begin finishes them)."""
    return (
        envs.SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_DEFER_PREFAULT.get()
        and mode() in ("load", "auto")
        and bool(envs.SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_DIR.get())
    )


def _round_up(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) // a * a


# ---------------------------------------------------------------------------------------- identity

# Server args that cannot change weight bytes: left out of the identity, so a change in them (another
# master address, a timeout) still finds the snapshot. Explicit names only; any other argument, known or
# unknown, counts. Keys starting with "_" are ServerArgs bookkeeping, not arguments.
ARGS_NOT_IN_IDENTITY = frozenset(
    {
        # addresses, ports, endpoints, devices of the transfer path
        "host", "port", "nccl_port", "dist_init_addr", "grpc_port", "gated_launch_port",
        "smg_http_sidecar_port", "disaggregation_bootstrap_port", "engine_info_bootstrap_port",
        "encoder_bootstrap_port", "encoder_urls", "encoder_register_urls", "decoupled_spec_bind_endpoint",
        "decoupled_spec_connect_endpoints", "load_publish_endpoint", "otlp_traces_endpoint",
        "disaggregation_ib_device", "mooncake_ib_device", "kv_events_config",
        "hicache_storage_backend_extra_config",  # mooncake master / metadata server endpoints
        "remote_instance_weight_loader_seed_instance_service_port",
        "remote_instance_weight_loader_send_weights_group_ports",
        # logging, tracing
        "log_level", "log_level_http", "log_requests", "log_requests_format", "log_requests_level",
        "log_requests_target", "decode_log_interval", "enable_request_time_stats_logging", "stat_loggers",
        "uvicorn_access_log_exclude_prefixes", "crash_dump_folder", "enable_trace", "trace_modules",
        "otlp_service_name", "spec_trace_dir",
        # metrics
        "enable_metrics", "enable_metrics_for_all_schedulers", "enable_mfu_metrics",
        "enable_forward_pass_metrics", "export_metrics_to_file", "export_metrics_to_file_dir",
        "extra_metric_labels", "forward_pass_metrics_ipc_name", "forward_pass_metrics_worker_id",
        "metrics_ipc_name", "tokenizer_metrics_allowed_custom_labels",
        "tokenizer_metrics_custom_labels_header", "bucket_e2e_request_latency",
        "bucket_inter_token_latency", "bucket_time_to_first_token", "prompt_tokens_buckets",
        "generation_tokens_buckets", "prefill_delayer_forward_passes_buckets",
        "prefill_delayer_wait_seconds_buckets", "enable_cache_report",
        # timeouts
        "dist_timeout", "watchdog_timeout", "soft_watchdog_timeout", "grpc_response_timeout_secs",
        "dynamic_batch_tokenizer_batch_timeout", "elastic_ep_scale_timeout", "weight_cache_timeout",
        # serving front end
        "stream_interval", "api_key", "admin_api_key", "served_model_name", "ssl_ca_certs",
        "ssl_certfile", "ssl_keyfile", "ssl_keyfile_password", "enable_ssl_refresh",
        "tokenizer_worker_num", "skip_server_warmup", "warmups",
        # CUDA graph persistence (Foundry): "save" in the prepare boot, "load" in the cell, and its TOML path;
        # they select graph capture or replay, not weight bytes
        "cuda_graph_persistence", "cuda_graph_persistence_config",
        # communication buffers (symmetric-memory pool for collectives), not weights
        "enable_symm_mem",
        # process checkpoint lifecycle (INF-840): when the server prepares and serves, not what it loads; a
        # cold start without it (the fallback of a pod that restores) finds the prepare boot's snapshot
        "checkpoint_prepare_on_startup",
        # per process: GPU numbering of this unit, sampling seed
        "base_gpu_id", "gpu_id_step", "random_seed",
    }
)


_ENV_VOLATILE = re.compile(
    r"^(SGLANG_RUN_ID|SGLANG_OPT_INF834_WEIGHT_SNAPSHOT.*|SGLANG_OPT_INF834_STANDBY.*|"
    r"SGLANG_OPT_INF834_(EARLY_HTTP_WORKERS|FAST_WARMUP_POLL|EARLY_CUINIT|NUMA_PRECOMPUTE|"
    r"TOKENIZER_MEMO|TOKENIZER_PREFETCH|W6_PROFILE|LOAD_PROFILE|WEIGHT_DIGEST_DIR)|"
    r"SGLANG_JIT_DEEPGEMM_PRECOMPILE|"  # DeepGEMM kernel JIT at startup, not weights
    r"SGLANG_OPT_INF834_ENGRAM_FAULT_CAP_S|"  # Engram host-table fault time cap (huge-page fallback), not weights
    r"SGLANG_SYMM_MEM_PREALLOC_GB_SIZE|"  # symmetric-memory pool size, set when --enable-symm-mem is on
    # INF-840 process checkpoint: deferred PD transport and L3 client, KV region image, host pool release
    r"SGLANG_CHECKPOINT_(DEFER_PD_TRANSPORT|DEFER_HICACHE_STORAGE|RESTORE_KV_FILL|LOG_KV_REGIONS|"
    r"KV_REGION_IMAGE|KV_REGION_IMAGE_MAX_MB|RELEASE_HOST_POOL|HOST_POOL_THREADS|RESTORE_OVERLAP_L3)|"
    # INF-840 Engram host table backing (tmpfs file) and pin flag: where the bytes live, not the bytes
    r"SGLANG_DSV41_ENGRAM_HOST_TABLE_(DIR|REGISTER_READONLY)|"
    r"SGLANG_LOGGING.*|SGLANG_.*LOG_LEVEL.*|SGLANG_IMAGE_TAG|SGLANG_BUILD_.*)$"
)
_code_digest_cache: Optional[str] = None


def code_digest() -> str:
    """Digest of the sglang Python sources (all .py files under the package)."""
    global _code_digest_cache
    if _code_digest_cache is None:
        import sglang

        root = os.path.dirname(os.path.abspath(sglang.__file__))
        h = hashlib.sha256()
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for fn in sorted(filenames):
                if fn.endswith(".py"):
                    path = os.path.join(dirpath, fn)
                    h.update(os.path.relpath(path, root).encode())
                    with open(path, "rb") as f:
                        h.update(hashlib.sha256(f.read()).digest())
        _code_digest_cache = h.hexdigest()[:24]
    return _code_digest_cache


def _file_digest(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:24]


def _parallel_layout() -> Dict[str, object]:
    out: Dict[str, object] = {}
    try:
        from sglang.srt.runtime_context import get_parallel

        p = get_parallel()
        for name in (
            "tp_size", "tp_rank", "pp_size", "pp_rank", "moe_ep_size", "moe_ep_rank",
            "moe_tp_size", "moe_tp_rank", "attn_dp_size", "attn_dp_rank", "attn_cp_size",
            "attn_cp_rank", "attn_tp_size", "attn_tp_rank",
        ):
            try:
                out[name] = getattr(p, name)
            except Exception:
                out[name] = None
    except Exception:
        pass
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        out["world_rank"] = torch.distributed.get_rank()
        out["world_size"] = torch.distributed.get_world_size()
    return out


def _server_args_view() -> Tuple[Dict[str, object], Optional[str]]:
    """(normalized args, expert placement file) from the published server args."""
    try:
        from sglang.srt.runtime_context import get_server_args

        sa = get_server_args()
    except Exception:
        return {}, None
    raw = getattr(sa, "_raw_input", None)
    if not isinstance(raw, dict):
        raw = {k: v for k, v in vars(sa).items() if not k.startswith("_")}
    args = {}
    for k, v in sorted(raw.items()):
        if k.startswith("_") or k in ARGS_NOT_IN_IDENTITY:
            continue
        try:
            json.dumps(v)
            args[k] = v
        except TypeError:
            args[k] = repr(v)
    placement = raw.get("init_expert_location")
    if isinstance(placement, str) and os.path.isfile(placement):
        args.pop("init_expert_location", None)  # the content counts, not the path
    else:
        placement = None
    return args, placement


def identity(model: torch.nn.Module, label: str, model_path: str, rank: int, phase_: str = "stream_end") -> Dict[str, object]:
    files = []
    if os.path.isdir(model_path):
        for fn in sorted(os.listdir(model_path)):
            if fn.endswith((".safetensors", ".json")):
                st = os.stat(os.path.join(model_path, fn))
                files.append([fn, st.st_size, st.st_mtime_ns])
    args, placement = _server_args_view()
    env = {k: v for k, v in sorted(os.environ.items()) if k.startswith("SGLANG_") and not _ENV_VOLATILE.match(k)}
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    return {
        "format": FORMAT_VERSION,
        "phase": phase_,
        "label": label,
        "model_class": type(model).__name__,
        "rank": rank,
        "checkpoint": {"path": os.path.abspath(model_path), "files": files,
                       "index": _file_digest(os.path.join(model_path, "model.safetensors.index.json"))},
        "server_args": args,
        "placement": _file_digest(placement),
        "parallel": _parallel_layout(),
        "env": env,
        "code": code_digest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
    }


def identity_key(ident: Dict[str, object]) -> str:
    """Directory key: everything except the rank (all ranks of one engine share the directory)."""
    shared = {k: v for k, v in ident.items() if k not in ("rank", "parallel")}
    par = dict(ident.get("parallel") or {})
    for k in list(par):
        if k.endswith("_rank"):
            par.pop(k)
    shared["parallel_sizes"] = par
    return hashlib.sha256(json.dumps(shared, sort_keys=True, default=str).encode()).hexdigest()[:20]


# ------------------------------------------------------------------------------------------ entries

def entries(model: torch.nn.Module) -> List[Tuple[str, torch.Tensor]]:
    """(name, tensor) for every parameter and buffer, one per storage range, in a fixed order."""
    out, seen = [], set()
    for kind, items in (
        ("param", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, t in items:
            if t is None or t.numel() == 0 or t.device.type == "meta":
                continue
            key = (t.untyped_storage().data_ptr(), t.storage_offset(), t.dtype, tuple(t.shape))
            if key in seen:
                continue
            seen.add(key)
            out.append((f"{kind}:{name}", t))
    return out


def _noncontiguous(ents: List[Tuple[str, torch.Tensor]]) -> Optional[str]:
    for name, t in ents:
        if not t.is_contiguous():
            return name
    return None


def _table(ents: List[Tuple[str, torch.Tensor]]) -> List[Dict[str, object]]:
    table, off = [], 0
    for name, t in ents:
        nbytes = t.numel() * t.element_size()
        table.append({"name": name, "dtype": str(t.dtype), "shape": list(t.shape),
                      "device": t.device.type, "nbytes": nbytes, "offset": off})
        off += _round_up(nbytes)
    return table


def _table_matches(saved: List[Dict[str, object]], live: List[Dict[str, object]]) -> Optional[str]:
    if len(saved) != len(live):
        return f"entry count {len(saved)} vs {len(live)}"
    for a, b in zip(saved, live):
        for k in ("name", "dtype", "shape", "device", "nbytes", "offset"):
            if a.get(k) != b.get(k):
                return f"entry {b['name']}: {k} {a.get(k)!r} vs {b.get(k)!r}"
    return None


# -------------------------------------------------------------------------------------- IO engines

class _Staging:
    """Page-aligned host buffers, pinned when CUDA is available (pinned memory from cudaHostAlloc
    is page-aligned, which O_DIRECT needs; the anonymous mmap fallback is page-aligned too)."""

    def __init__(self, nbytes: int):
        self.pinned = torch.cuda.is_available()
        if self.pinned:
            self.t = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        else:
            self._mm = mmap.mmap(-1, nbytes)
            self.t = torch.frombuffer(self._mm, dtype=torch.uint8)

    def view(self, n: int) -> torch.Tensor:
        return self.t[:n]


def _flat_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.detach().reshape(-1).view(torch.uint8)


def _libc():
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long)
    libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
    return libc


class _Residency:
    """mincore over byte ranges of one file (read-only shared mapping; mapping faults nothing in)."""

    def __init__(self, path: str):
        import ctypes

        self._ct = ctypes
        self.size = os.path.getsize(path)
        self.libc = _libc()
        self.addr = None
        fd = os.open(path, os.O_RDONLY)
        try:
            addr = self.libc.mmap(None, max(1, self.size), mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        finally:
            os.close(fd)
        if addr not in (None, ctypes.c_void_p(-1).value):
            self.addr = addr

    def fraction(self, off: int, n: int) -> float:
        if self.addr is None:
            return 0.0
        n = min(n, self.size - off)
        if n <= 0:
            return 1.0
        pages = (n + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        vec = (self._ct.c_ubyte * pages)()
        if self.libc.mincore(self._ct.c_void_p(self.addr + off), n, vec) != 0:
            return 0.0
        return float((np.frombuffer(vec, dtype=np.uint8) & 1).mean())

    def close(self) -> None:
        if self.addr is not None:
            self.libc.munmap(self._ct.c_void_p(self.addr), max(1, self.size))
            self.addr = None


class PreadPinnedEngine:
    """(a) multi-threaded pread -> pinned staging -> device copy, one task per 64 MiB chunk.
    policy: "buffered", "direct" (O_DIRECT) or "chunk" (per chunk: buffered when every page of the chunk
    is resident, else O_DIRECT)."""

    name = "pread_pinned"

    def __init__(self, threads: int):
        self.threads = max(1, threads)
        self._tls = threading.local()
        self._lock = threading.Lock()
        self._bufs: List[_Staging] = []
        self.stats = {"buffered": 0, "direct": 0}
        self.direct_ranges: List[Tuple[int, int]] = []

    def _buf(self) -> _Staging:
        b = getattr(self._tls, "buf", None)
        if b is None:
            b = self._tls.buf = _Staging(CHUNK)
            with self._lock:
                self._bufs.append(b)
        return b

    def read(self, path: str, tasks, policy: str, verify) -> Tuple[int, float]:
        """tasks: (file offset, nbytes, padded nbytes, dest uint8 view, expected checksum)."""
        o_direct = getattr(os, "O_DIRECT", 0)
        fd_b = fd_d = -1
        res = None
        if policy in ("buffered", "chunk"):
            fd_b = os.open(path, os.O_RDONLY)
        if policy == "direct":
            fd_d = os.open(path, os.O_RDONLY | o_direct)
        elif policy == "chunk" and o_direct:
            try:
                fd_d = os.open(path, os.O_RDONLY | o_direct)
                res = _Residency(path)
            except OSError:  # a filesystem without O_DIRECT (tmpfs): buffered only
                fd_d = -1
        tic = time.perf_counter()
        bad = []

        def run(task):
            off, n, padded, dst, want = task
            if dst.device.type == "cuda":
                torch.cuda.set_device(dst.device)
            direct = fd_d >= 0 and (res is None or res.fraction(off, padded) < 1.0)
            st = self._buf()
            buf = st.view(padded if direct else n)
            mv = memoryview(buf.numpy())
            fd = fd_d if direct else fd_b
            done = 0
            while done < len(mv):
                got = os.preadv(fd, [mv[done:]], off + done)
                if got <= 0:
                    raise OSError(f"short read at {off + done} in {path}")
                done += got
            with self._lock:
                self.stats["direct" if direct else "buffered"] += n
                if direct:
                    self.direct_ranges.append((off, padded))
            if verify and _checksum(mv[:n]) != want:
                bad.append(off)
                return
            dst.copy_(buf[:n], non_blocking=False)

        try:
            with concurrent.futures.ThreadPoolExecutor(self.threads, thread_name_prefix="inf834-snap") as pool:
                for f in [pool.submit(run, t) for t in tasks]:
                    f.result()
        finally:
            for fd in (fd_b, fd_d):
                if fd >= 0:
                    os.close(fd)
            if res is not None:
                res.close()
        if bad:
            raise ChecksumError(f"{len(bad)} chunks failed the checksum (first at offset {min(bad)})")
        return sum(t[1] for t in tasks), time.perf_counter() - tic

    def close(self) -> None:
        self._bufs.clear()
        self._tls = threading.local()


class ChecksumError(RuntimeError):
    pass


def _resident_fraction(path: str, sample: int = 256) -> float:
    """mincore over evenly spaced 2 MiB windows of the file."""
    size = os.path.getsize(path)
    if size == 0:
        return 1.0
    res = _Residency(path)
    try:
        win = 2 << 20
        step = max(win, size // sample)
        fr = [res.fraction(off, win) for off in range(0, size, step)]
        return sum(fr) / len(fr) if fr else 1.0
    finally:
        res.close()


def _io_policy(path: str) -> str:
    knob = (envs.SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_DIRECT_IO.get() or "auto").lower()
    if knob in ("1", "true", "on"):
        return "direct"
    if knob in ("0", "false", "off"):
        return "buffered"
    if knob == "file":
        try:
            return "direct" if _resident_fraction(path) < 0.5 else "buffered"
        except Exception:
            return "buffered"
    return "chunk"


def _merge(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[List[int]] = []
    for off, n in sorted(ranges):
        if out and off <= out[-1][0] + out[-1][1]:
            out[-1][1] = max(out[-1][1], off + n - out[-1][0])
        else:
            out.append([off, n])
    return [(a, b) for a, b in out]


def _prepares_checkpoint() -> bool:
    """Whether this boot ends in a process checkpoint (--checkpoint-prepare-on-startup)."""
    try:
        from sglang.srt.runtime_context import get_serving

        return bool(get_serving().checkpoint_prepare_on_startup)
    except Exception:  # no published server args (tests, offline tools)
        return False


def _rewarm(path: str, ranges: List[Tuple[int, int]], label: str, rank: int) -> Optional[threading.Thread]:
    """Background buffered reads of the ranges this boot read with O_DIRECT, so they are in the page cache
    for the next boot (POSIX_FADV_WILLNEED is not used: on the rig's xfs it brought in about 1% of a range)."""
    ranges = _merge(ranges)
    if not ranges:
        return None

    def run():
        tic = time.perf_counter()
        done = 0
        try:
            fd = os.open(path, os.O_RDONLY)
            buf = bytearray(CHUNK)
            mv = memoryview(buf)
            try:
                for off, n in ranges:
                    end = off + n
                    while off < end:
                        got = os.preadv(fd, [mv[: min(CHUNK, end - off)]], off)
                        if got <= 0:
                            break
                        off += got
                        done += got
            finally:
                os.close(fd)
            logger.info("INF834 weight snapshot [%s rank %d]: rewarm read %.2f GB into the page cache in %.2f s",
                        label, rank, done / 1e9, time.perf_counter() - tic)
        except OSError as e:
            logger.info("INF834 weight snapshot [%s rank %d]: rewarm stopped after %.2f GB (%r)", label, rank, done / 1e9, e)

    t = threading.Thread(target=run, name="inf834-snap-rewarm", daemon=True)
    t.start()
    return t


# --------------------------------------------------------------------------------- repacked layers

def _repack_candidates(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    """Modules whose quant method will take a repack the "repacked" phase stores (EP decode experts)."""
    out = {}
    for name, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        check = getattr(qm, "inf834_snapshot_repack_candidate", None)
        if check is None or not all(isinstance(getattr(mod, p, None), torch.Tensor) for p in REPACK_PARAMS):
            continue
        if check(mod):
            out[name] = mod
    return out


_DTYPES = {str(d): d for d in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16,
                               torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.bool)}


def _import_type(qualname: str):
    import importlib

    module, _, name = qualname.partition(":")
    obj = importlib.import_module(module)
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def _packed_record(prefix: str, packed, split, scale_max: int, mode_: str) -> Dict[str, object]:
    cls = type(packed)
    names = getattr(packed, "_fields", None)
    if names is None:
        raise TypeError(f"{prefix}: packed experts of type {cls.__name__} are not a NamedTuple")
    fields = []
    for name, v in zip(names, packed):
        if isinstance(v, torch.Tensor):
            if not v.is_contiguous():
                raise ValueError(f"{prefix}: packed field {name} is not contiguous")
            fields.append({"name": name, "shape": list(v.shape), "dtype": str(v.dtype)})
        else:
            json.dumps(v)  # non-tensor fields must be plain values
            fields.append({"name": name, "value": v})
    return {"layer": prefix, "type": f"{cls.__module__}:{cls.__qualname__}", "fields": fields,
            "split": split, "scale_max": int(scale_max), "mode": mode_}


def _record_nbytes(rec: Dict[str, object]) -> int:
    n = 0
    for f in rec["fields"]:
        if "shape" in f:
            numel = 1
            for d in f["shape"]:
                numel *= int(d)
            n += numel * torch.empty(0, dtype=_DTYPES[f["dtype"]]).element_size()
    return n


def _copy_stream(srcs: List[torch.Tensor], dsts: List[torch.Tensor]) -> None:
    """Copy the concatenated bytes of srcs into the concatenated bytes of dsts until dsts are full."""
    si = so = 0
    for d in dsts:
        do = 0
        while do < d.numel():
            if si >= len(srcs):
                raise ValueError("source stream shorter than the destination")
            n = min(d.numel() - do, srcs[si].numel() - so)
            d[do:do + n].copy_(srcs[si][so:so + n])
            do += n
            so += n
            if so == srcs[si].numel():
                si, so = si + 1, 0


def restore_packed(layer: torch.nn.Module, rec: Dict[str, object]):
    """Called by the repack in place of the pack: the packed experts restored into this layer's raw
    expert parameters, copied into freshly allocated tensors (same order and sizes as the pack's).
    Returns (packed NamedTuple, split)."""
    cls = _import_type(rec["type"])
    regions = [_flat_bytes(getattr(layer, p).data) for p in REPACK_PARAMS]
    device = regions[0].device
    values, tensors = [], []
    for f in rec["fields"]:
        if "shape" in f:
            t = torch.empty(tuple(f["shape"]), dtype=_DTYPES[f["dtype"]], device=device)
            tensors.append(t)
            values.append(t)
        else:
            values.append(f["value"])
    _copy_stream(regions, [_flat_bytes(t) for t in tensors])
    del regions
    layer.__dict__.pop("_inf834_snapshot_packed", None)
    return cls(*values), rec["split"]


# ------------------------------------------------------------------------------------------ save job

class _SaveJob:
    """One snapshot file being written: temp file, per-chunk checksums, commit."""

    def __init__(self, snap: "WeightSnapshot", ents, table, size: int):
        self.snap = snap
        self.ents = ents
        self.table = table
        self.size = size
        self.index = {name: i for i, (name, _) in enumerate(ents)}
        self.first_chunk, k = [], 0
        for row in table:
            self.first_chunk.append(k)
            k += max(1, -(-row["nbytes"] // CHUNK)) if row["nbytes"] else 0
        self.sums: List[Optional[str]] = [None] * k
        self.records: Dict[str, Dict[str, object]] = {}
        self.pending: Dict[str, List[int]] = {}
        self.by_module: Dict[int, str] = {}
        self.secs = 0.0
        self.tmp_dir = os.path.join(snap.dir, ".tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.tmp_bin = os.path.join(self.tmp_dir, f"rank{snap.rank}.bin.{os.getpid()}")
        # O_DIRECT: the snapshot (54-135 GB per rank) does not go through this pod's page cache and
        # memory cgroup. Falls back to buffered writes where the filesystem refuses O_DIRECT.
        self.direct = (envs.SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_DIRECT_IO.get() or "auto").lower() not in ("0", "false", "off")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        self.fd = -1
        if self.direct and hasattr(os, "O_DIRECT"):
            try:
                self.fd = os.open(self.tmp_bin, flags | os.O_DIRECT, 0o644)
            except OSError:
                self.fd = -1
        if self.fd < 0:
            self.direct = False
            self.fd = os.open(self.tmp_bin, flags, 0o644)
        os.ftruncate(self.fd, size)
        self.pool = concurrent.futures.ThreadPoolExecutor(snap.threads, thread_name_prefix="inf834-snap-save")
        self.tls = threading.local()

    def _write(self, chunks) -> None:
        """chunks: (file offset, nbytes, checksum index, fill(buf uint8 view of nbytes))."""
        tic = time.perf_counter()

        def run(c):
            off, n, k, fill = c
            st = getattr(self.tls, "buf", None)
            if st is None:
                st = self.tls.buf = _Staging(CHUNK)
            buf = st.view(n)
            fill(buf)
            self.sums[k] = _checksum(memoryview(buf.numpy()))
            # O_DIRECT writes whole 4 KiB blocks: the tail of the last block is entry padding.
            mv = memoryview(st.view(_round_up(n) if self.direct else n).numpy())
            done = 0
            while done < len(mv):
                done += os.pwrite(self.fd, mv[done:], off + done)

        for f in [self.pool.submit(run, c) for c in chunks]:
            f.result()
        self.secs += time.perf_counter() - tic

    def write_entries(self, idxs: List[int]) -> None:
        chunks = []
        for i in idxs:
            row, flat = self.table[i], _flat_bytes(self.ents[i][1])
            for j, start in enumerate(range(0, row["nbytes"], CHUNK)):
                n = min(CHUNK, row["nbytes"] - start)
                src = flat[start:start + n]

                def fill(buf, src=src):
                    if src.device.type == "cuda":
                        torch.cuda.set_device(src.device)
                    buf.copy_(src, non_blocking=False)

                chunks.append((row["offset"] + start, n, self.first_chunk[i] + j, fill))
        self._write(chunks)

    def write_stream(self, idxs: List[int], fields: List[torch.Tensor]) -> None:
        """The concatenated bytes of fields, laid over the entries idxs in order; zeros after the end."""
        srcs = [_flat_bytes(f) for f in fields]
        bounds, acc = [], 0
        for s_ in srcs:
            bounds.append((acc, s_))
            acc += s_.numel()
        total = acc
        chunks, base = [], 0
        for i in idxs:
            row = self.table[i]
            for j, start in enumerate(range(0, row["nbytes"], CHUNK)):
                n = min(CHUNK, row["nbytes"] - start)
                lo = base + start

                def fill(buf, lo=lo, n=n):
                    pos = 0
                    for b0, src in bounds:
                        a, b = max(lo, b0), min(lo + n, b0 + src.numel())
                        if a < b:
                            if src.device.type == "cuda":
                                torch.cuda.set_device(src.device)
                            buf[a - lo:b - lo].copy_(src[a - b0:b - b0], non_blocking=False)
                            pos = max(pos, b - lo)
                    if lo + n > total:
                        buf[max(0, total - lo):].zero_()

                chunks.append((row["offset"] + start, n, self.first_chunk[i] + j, fill))
            base += row["nbytes"]
        if total > base:
            raise ValueError(f"packed bytes {total} exceed the raw expert parameters {base}")
        self._write(chunks)

    def commit(self) -> None:
        snap = self.snap
        self.pool.shutdown()
        try:
            os.fsync(self.fd)
        finally:
            os.close(self.fd)
            self.fd = -1
        os.replace(self.tmp_bin, snap.bin)
        meta = {"identity": snap.ident, "table": self.table, "size": self.size, "chunk": CHUNK, "checksums": self.sums,
                "saved_at": time.time(), "algorithm": "xxh3_64" if self.sums and self.sums[0].startswith("x") else "crc32"}
        if snap.phase == "repacked":
            meta["repacked"] = [self.records[k] for k in sorted(self.records)]
        tmp_commit = os.path.join(self.tmp_dir, f"rank{snap.rank}.commit.{os.getpid()}")
        with open(tmp_commit, "w") as f:
            json.dump(meta, f, default=str)
            f.flush()
            os.fsync(f.fileno())
        dfd = os.open(snap.dir, os.O_RDONLY)
        try:
            os.fsync(dfd)  # the .bin rename is durable before the commit appears
            os.replace(tmp_commit, snap.commit)
            os.fsync(dfd)
        finally:
            os.close(dfd)
        logger.info(
            "INF834 weight snapshot [%s rank %d]: saved %d entries (%s, %d repacked layers), %.2f GB, %.2f s blocking "
            "the boot (%s writes, %d threads) to %s",
            snap.label, snap.rank, len(self.table), snap.phase, len(self.records), self.size / 1e9, self.secs,
            "O_DIRECT" if self.direct else "buffered", snap.threads, snap.bin,
        )

    def abort(self, why: str) -> None:
        self.pool.shutdown(wait=True)
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
        try:
            os.remove(self.tmp_bin)
        except OSError:
            pass
        logger.warning("INF834 weight snapshot [%s rank %d]: save abandoned (%s); nothing committed",
                       self.snap.label, self.snap.rank, why)


# ------------------------------------------------------------------------------------ save / load

class WeightSnapshot:
    """One model on one rank. begin() before the weight stream, end() after it, finish() after
    process_weights_after_loading (the model loader calls it through finish_model)."""

    def __init__(self, model: torch.nn.Module, label: str, model_path: Optional[str] = None):
        self.model = model
        self.label = label
        self.mode = mode()
        self.restored = False
        self.reason = ""
        self.staged: List[str] = []
        self._job: Optional[_SaveJob] = None
        self._lock = None
        self.supported = type(model).__name__ in ALLOWED_MODELS
        root = envs.SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_DIR.get() or ""
        self.enabled = bool(self.mode) and self.supported and bool(root)
        self.rank = 0
        if not self.enabled:
            return
        self.phase = phase()
        if model_path is None:
            from sglang.srt.runtime_context import get_model

            model_path = get_model().model_path
        self.rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        self.ident = identity(model, label, model_path, self.rank, self.phase)
        self.dir = os.path.join(root, f"{label}-{identity_key(self.ident)}")
        self.bin = os.path.join(self.dir, f"rank{self.rank}.bin")
        self.commit = os.path.join(self.dir, f"rank{self.rank}.commit")
        self.threads = max(1, envs.SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_THREADS.get())
        model._inf834_weight_snapshot = self  # finish_model() finds it after the repack

    # ---------------------------------------------------------------------------------- load
    def begin(self) -> bool:
        """Restore if a committed snapshot with this identity exists. True: the parameters hold
        the snapshot and the weight stream must be skipped. False: run the normal stream."""
        restored = self._begin()
        tables = [m.host_table for m in self.model.modules()
                  if getattr(getattr(m, "host_table", None), "deferred", False)]
        if tables:
            tic = time.perf_counter()
            for t in tables:
                t.complete(prefault=not restored)
            logger.info("INF834 weight snapshot [%s rank %d]: %d deferred engram host tables registered (%s) in %.2f s",
                        self.label, self.rank, len(tables),
                        "pages written by the restore, no pre-fault" if restored else "pre-faulted first",
                        time.perf_counter() - tic)
        return restored

    def _begin(self) -> bool:
        if not self.enabled or self.mode not in ("load", "auto"):
            return False
        if not os.path.exists(self.commit):
            logger.info("INF834 weight snapshot [%s rank %d]: none at %s, loading the checkpoint", self.label, self.rank, self.dir)
            return False
        try:
            with open(self.commit) as f:
                meta = json.load(f)
        except (OSError, ValueError) as e:
            return self._fallback(f"unreadable commit file ({e!r})", touched=False)
        if meta.get("identity") != json.loads(json.dumps(self.ident, default=str)):
            diff = sorted(k for k in self.ident if meta.get("identity", {}).get(k) != json.loads(json.dumps(self.ident[k], default=str)))
            return self._fallback(f"identity differs in {diff}", touched=False)
        ents = entries(self.model)
        nc = _noncontiguous(ents)
        if nc:
            return self._fallback(f"non-contiguous entry {nc}", touched=False)
        live = _table(ents)
        why = _table_matches(meta.get("table", []), live)
        if why:
            return self._fallback(f"entry table differs: {why}", touched=False)
        size = live[-1]["offset"] + _round_up(live[-1]["nbytes"]) if live else 0
        if not os.path.exists(self.bin) or os.path.getsize(self.bin) != size or meta.get("size") != size:
            return self._fallback(f"file size {os.path.getsize(self.bin) if os.path.exists(self.bin) else 'missing'} vs {size}", touched=False)
        records: Dict[str, Dict[str, object]] = {}
        cands: Dict[str, torch.nn.Module] = {}
        if self.phase == "repacked":
            records = {r["layer"]: r for r in meta.get("repacked", [])}
            cands = _repack_candidates(self.model)
            if set(records) != set(cands):
                return self._fallback(f"repacked layers differ: {len(records)} stored vs {len(cands)} live "
                                      f"(e.g. {sorted(set(records) ^ set(cands))[:2]})", touched=False)
            index = {row["name"]: row for row in live}
            for prefix, rec in records.items():
                try:
                    _import_type(rec["type"])
                    room = sum(index[f"param:{prefix}.{p}"]["nbytes"] for p in REPACK_PARAMS)
                    need = _record_nbytes(rec)
                except Exception as e:
                    return self._fallback(f"repacked layer {prefix}: unusable record ({e!r})", touched=False)
                if need > room:
                    return self._fallback(f"repacked layer {prefix}: {need} packed bytes > {room}", touched=False)
        sums = meta.get("checksums", [])
        tasks, i = [], 0
        for (name, t), row in zip(ents, live):
            flat = _flat_bytes(t)
            for start in range(0, row["nbytes"], CHUNK):
                n = min(CHUNK, row["nbytes"] - start)
                tasks.append((row["offset"] + start, n, _round_up(n), flat[start:start + n], sums[i] if i < len(sums) else None))
                i += 1
        if i != len(sums):
            return self._fallback(f"checksum count {len(sums)} vs {i}", touched=False)
        engine = PreadPinnedEngine(self.threads)
        policy = _io_policy(self.bin)
        try:
            nbytes, secs = engine.read(self.bin, tasks, policy, verify=True)
        except ChecksumError as e:
            return self._fallback(f"checksum: {e}", touched=True)
        except OSError as e:
            return self._fallback(f"read failed: {e!r}", touched=True)
        finally:
            engine.close()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._mark_host_tables_dirty()
        for prefix, rec in records.items():
            cands[prefix]._inf834_snapshot_packed = rec
        self.staged = sorted(records)
        self.restored = True
        st = engine.stats
        logger.info(
            "INF834 weight snapshot [%s rank %d]: restored %d entries (%s, %d repacked layers), %.2f GB in %.2f s "
            "(%.1f GB/s, %s, policy %s: %.2f GB buffered + %.2f GB O_DIRECT, %d threads) from %s",
            self.label, self.rank, len(live), self.phase, len(records), nbytes / 1e9, secs,
            nbytes / 1e9 / max(secs, 1e-9), engine.name, policy, st["buffered"] / 1e9, st["direct"] / 1e9,
            self.threads, self.bin,
        )
        if policy == "chunk" and engine.direct_ranges and envs.SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_REWARM.get():
            if _prepares_checkpoint():
                # The rewarm thread holds the snapshot file open for minutes; a process checkpoint taken
                # meanwhile records that descriptor, and restoring it needs the same file at the same path.
                logger.info("INF834 weight snapshot [%s rank %d]: rewarm skipped (--checkpoint-prepare-on-startup)",
                            self.label, self.rank)
            else:
                self._rewarm_thread = _rewarm(self.bin, engine.direct_ranges, self.label, self.rank)
        return True

    def _mark_host_tables_dirty(self) -> None:
        # Engram host tables: their loaders set dirty so finish_load runs (huge-page collapse, log).
        for mod in self.model.modules():
            ht = getattr(mod, "host_table", None)
            if ht is not None and hasattr(ht, "dirty"):
                ht.dirty = True

    def _fallback(self, reason: str, touched: bool) -> bool:
        self.reason = reason
        logger.warning(
            "%s [%s rank %d]: %s; %s",
            FALLBACK, self.label, self.rank, reason,
            "the normal checkpoint load overwrites every parameter (and the repack runs)" if touched
            else "loading the checkpoint, no parameter was touched",
        )
        return False

    # ---------------------------------------------------------------------------------- save
    def end(self) -> None:
        """After the weight stream: write the snapshot (mode save, or auto when nothing was restored).
        Blocking: the parameters change in the steps that follow. Phase "repacked": writes everything but
        the expert parameters of the repacked layers now, those layers' packed bytes at their repack
        (save_packed), and commits in finish()."""
        if not self.enabled or self.restored or self.mode not in ("save", "auto"):
            return
        os.makedirs(self.dir, exist_ok=True)
        lock = open(os.path.join(self.dir, f"rank{self.rank}.lock"), "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info("INF834 weight snapshot [%s rank %d]: another process is saving %s; skipped", self.label, self.rank, self.bin)
            lock.close()
            return
        job = None
        try:
            if os.path.exists(self.commit):
                return  # committed meanwhile by another process with the same identity
            ents = entries(self.model)
            nc = _noncontiguous(ents)
            if nc:
                raise RuntimeError(f"non-contiguous entry {nc}: not supported")
            table = _table(ents)
            size = table[-1]["offset"] + _round_up(table[-1]["nbytes"]) if table else 0
            job = _SaveJob(self, ents, table, size)
            skip = set()
            if self.phase == "repacked":
                for prefix in _repack_candidates(self.model):
                    idxs = [job.index[f"param:{prefix}.{p}"] for p in REPACK_PARAMS]
                    job.pending[prefix] = idxs
                    skip.update(idxs)
            job.write_entries([i for i in range(len(ents)) if i not in skip])
            if job.pending:
                # By module identity: post steps can register aliases (DeepseekV4 post_load_weights sets
                # layers[i]._next_layer), after which named_modules() names a layer by another path.
                for prefix, mod in _repack_candidates(self.model).items():
                    mod._inf834_snapshot_save = self
                    job.by_module[id(mod)] = prefix
                self._job, self._lock = job, lock
                lock = None  # held until finish()
                logger.info("INF834 weight snapshot [%s rank %d]: stream-end part written (%.2f s); waiting for the "
                            "repack of %d layers", self.label, self.rank, job.secs, len(job.pending))
                return
            job.commit()
        except Exception as e:  # a failed save never fails the boot
            if job is not None:
                job.abort(repr(e))
            else:
                logger.warning("INF834 weight snapshot [%s rank %d]: save failed (%r); nothing committed", self.label, self.rank, e)
        finally:
            if lock is not None:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock.close()

    def save_packed(self, layer: torch.nn.Module, packed, split, scale_max: int, mode_: str) -> None:
        """Called by the repack right after it packed `layer` (before it deletes the raw parameters)."""
        layer.__dict__.pop("_inf834_snapshot_save", None)
        job = self._job
        if job is None:
            return
        prefix = job.by_module.get(id(layer))
        try:
            if prefix not in job.pending:
                raise RuntimeError(f"repack of an unexpected layer {prefix}")
            rec = _packed_record(prefix, packed, split, scale_max, mode_)
            job.write_stream(job.pending[prefix], [v for v in packed if isinstance(v, torch.Tensor)])
            job.records[prefix] = rec
            del job.pending[prefix]
        except Exception as e:
            self._abandon(f"layer {prefix}: {e!r}")

    def _abandon(self, why: str) -> None:
        if self._job is not None:
            self._job.abort(why)
            self._job = None
        for m in self.model.modules():
            m.__dict__.pop("_inf834_snapshot_save", None)
        self._release()

    def _release(self) -> None:
        if self._lock is not None:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None

    def finish(self) -> None:
        """After process_weights_after_loading. Restore: every staged layer must have been consumed by its
        repack (else the raw parameters hold packed bytes: fail the boot). Save: commit if every expected
        layer delivered its packed bytes."""
        if self.restored:
            left = sorted(n for n, m in self.model.named_modules() if "_inf834_snapshot_packed" in m.__dict__)
            if left:
                raise RuntimeError(
                    f"INF834 weight snapshot [{self.label} rank {self.rank}]: {len(left)} layers hold restored packed "
                    f"experts their repack did not consume (first {left[0]}); the model is not usable"
                )
            return
        job = self._job
        if job is None:
            return
        if job.pending:
            self._abandon(f"{len(job.pending)} layers were not repacked (first {sorted(job.pending)[0]})")
            return
        try:
            job.commit()
        except Exception as e:
            job.abort(repr(e))
        finally:
            self._job = None
            self._release()


def finish_model(model: torch.nn.Module) -> None:
    """Model loader hook, after process_weights_after_loading."""
    snap = getattr(model, "_inf834_weight_snapshot", None)
    if snap is not None:
        snap.finish()
