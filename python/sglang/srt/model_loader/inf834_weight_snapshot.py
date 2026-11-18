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
The phase is recorded ("stream_end"); a post-repack variant ("post_process") would enumerate what
write_digest enumerates (parameters, buffers, tensors held in tuples such as _sm90_packed) and would
also skip the repack; it is not implemented.

Files, under SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_DIR / <label>-<identity digest> /:
  rank<R>.bin      the bytes, every entry at a 4 KiB aligned offset, padded to 4 KiB
  rank<R>.commit   JSON: identity, entry table, per-64-MiB-chunk xxh3 checksums. Written last
                   (temp file, fsync, rename, directory fsync); a .bin without it is ignored.
  rank<R>.lock     flock held while saving (another process on the node saving the same rank skips)

Modes (SGLANG_OPT_INF834_WEIGHT_SNAPSHOT): "load" restores when a committed snapshot with the same
identity exists, else loads normally; "save" loads normally and writes the snapshot; "auto" does
"load" and, when nothing was restored, "save". Anything that does not match (identity, entry table,
file size) falls back to the normal load before a parameter is touched. A checksum mismatch found
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

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
ALIGN = 4096
CHUNK = 64 << 20
FALLBACK = "INF834 weight snapshot FALLBACK"
ALLOWED_MODELS = ("DeepseekV4ForCausalLM", "DeepseekV4ForCausalLMDSpark")

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
        # per process: GPU numbering of this unit, sampling seed
        "base_gpu_id", "gpu_id_step", "random_seed",
    }
)


_ENV_VOLATILE = re.compile(
    r"^(SGLANG_RUN_ID|SGLANG_OPT_INF834_WEIGHT_SNAPSHOT.*|SGLANG_OPT_INF834_STANDBY.*|"
    r"SGLANG_OPT_INF834_(EARLY_HTTP_WORKERS|FAST_WARMUP_POLL|EARLY_CUINIT|NUMA_PRECOMPUTE|"
    r"TOKENIZER_MEMO|TOKENIZER_PREFETCH|W6_PROFILE|LOAD_PROFILE|WEIGHT_DIGEST_DIR)|"
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


def identity(model: torch.nn.Module, label: str, model_path: str, rank: int) -> Dict[str, object]:
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
        "phase": "stream_end",
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


class PreadPinnedEngine:
    """(a) multi-threaded pread -> pinned staging -> device copy, one task per 64 MiB chunk.
    O_DIRECT when the file is not resident in the page cache (cold node), buffered otherwise."""

    name = "pread_pinned"

    def __init__(self, threads: int):
        self.threads = max(1, threads)
        self._tls = threading.local()
        self._lock = threading.Lock()
        self._bufs: List[_Staging] = []

    def _buf(self) -> _Staging:
        b = getattr(self._tls, "buf", None)
        if b is None:
            b = self._tls.buf = _Staging(CHUNK)
            with self._lock:
                self._bufs.append(b)
        return b

    def read(self, path: str, tasks, direct: bool, verify) -> Tuple[int, float]:
        """tasks: (file offset, nbytes, padded nbytes, dest uint8 view, expected checksum)."""
        flags = os.O_RDONLY | (getattr(os, "O_DIRECT", 0) if direct else 0)
        fd = os.open(path, flags)
        tic = time.perf_counter()
        bad = []

        def run(task):
            off, n, padded, dst, want = task
            if dst.device.type == "cuda":
                torch.cuda.set_device(dst.device)
            st = self._buf()
            buf = st.view(padded if direct else n)
            mv = memoryview(buf.numpy())
            done = 0
            while done < len(mv):
                got = os.preadv(fd, [mv[done:]], off + done)
                if got <= 0:
                    raise OSError(f"short read at {off + done} in {path}")
                done += got
            if verify and _checksum(mv[:n]) != want:
                bad.append(off)
                return
            dst.copy_(buf[:n], non_blocking=False)

        try:
            with concurrent.futures.ThreadPoolExecutor(self.threads, thread_name_prefix="inf834-snap") as pool:
                for f in [pool.submit(run, t) for t in tasks]:
                    f.result()
        finally:
            os.close(fd)
        if bad:
            raise ChecksumError(f"{len(bad)} chunks failed the checksum (first at offset {bad[0]})")
        return sum(t[1] for t in tasks), time.perf_counter() - tic

    def close(self) -> None:
        self._bufs.clear()
        self._tls = threading.local()


class ChecksumError(RuntimeError):
    pass


def _resident_fraction(path: str, sample: int = 256) -> float:
    """mincore over evenly spaced 2 MiB windows of the file (libc mmap, read-only)."""
    import ctypes

    size = os.path.getsize(path)
    if size == 0:
        return 1.0
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long)
    libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            return 0.0
        try:
            win = 2 << 20
            step = max(win, size // sample)
            vec = (ctypes.c_ubyte * (win // mmap.PAGESIZE))()
            res = tot = 0
            for off in range(0, size, step):
                n = min(win, size - off)
                if libc.mincore(ctypes.c_void_p(addr + off), n, vec) != 0:
                    return 0.0
                pages = (n + mmap.PAGESIZE - 1) // mmap.PAGESIZE
                res += sum(1 for i in range(pages) if vec[i] & 1)
                tot += pages
            return res / tot if tot else 1.0
        finally:
            libc.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)


def _direct_io(path: str) -> bool:
    knob = (envs.SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_DIRECT_IO.get() or "auto").lower()
    if knob in ("1", "true", "on"):
        return True
    if knob in ("0", "false", "off"):
        return False
    try:
        return _resident_fraction(path) < 0.5
    except Exception:
        return False


# ------------------------------------------------------------------------------------ save / load

class WeightSnapshot:
    """One model on one rank. begin() before the weight stream, end() after it."""

    def __init__(self, model: torch.nn.Module, label: str, model_path: Optional[str] = None):
        self.model = model
        self.label = label
        self.mode = mode()
        self.restored = False
        self.reason = ""
        self.supported = type(model).__name__ in ALLOWED_MODELS
        root = envs.SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_DIR.get() or ""
        self.enabled = bool(self.mode) and self.supported and bool(root)
        if not self.enabled:
            return
        if model_path is None:
            from sglang.srt.runtime_context import get_model

            model_path = get_model().model_path
        self.rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        self.ident = identity(model, label, model_path, self.rank)
        self.dir = os.path.join(root, f"{label}-{identity_key(self.ident)}")
        self.bin = os.path.join(self.dir, f"rank{self.rank}.bin")
        self.commit = os.path.join(self.dir, f"rank{self.rank}.commit")
        self.threads = max(1, envs.SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_THREADS.get())

    # ---------------------------------------------------------------------------------- load
    def begin(self) -> bool:
        """Restore if a committed snapshot with this identity exists. True: the parameters hold
        the snapshot and the weight stream must be skipped. False: run the normal stream."""
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
        direct = _direct_io(self.bin)
        try:
            nbytes, secs = engine.read(self.bin, tasks, direct, verify=True)
        except ChecksumError as e:
            return self._fallback(f"checksum: {e}", touched=True)
        except OSError as e:
            return self._fallback(f"read failed: {e!r}", touched=True)
        finally:
            engine.close()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._mark_host_tables_dirty()
        self.restored = True
        logger.info(
            "INF834 weight snapshot [%s rank %d]: restored %d entries, %.2f GB in %.2f s (%.1f GB/s, %s, %s, %d threads) from %s",
            self.label, self.rank, len(live), nbytes / 1e9, secs, nbytes / 1e9 / max(secs, 1e-9),
            engine.name, "O_DIRECT" if direct else "buffered", self.threads, self.bin,
        )
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
            "the normal checkpoint load overwrites every parameter" if touched else "loading the checkpoint, no parameter was touched",
        )
        return False

    # ---------------------------------------------------------------------------------- save
    def end(self) -> None:
        """After the weight stream: write the snapshot (mode save, or auto when nothing was restored).
        Blocking: the parameters change in the steps that follow."""
        if not self.enabled or self.restored or self.mode not in ("save", "auto"):
            return
        os.makedirs(self.dir, exist_ok=True)
        lock = open(os.path.join(self.dir, f"rank{self.rank}.lock"), "w")
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logger.info("INF834 weight snapshot [%s rank %d]: another process is saving %s; skipped", self.label, self.rank, self.bin)
                return
            if os.path.exists(self.commit):
                return  # committed meanwhile by another process with the same identity
            self._save()
        except Exception as e:  # a failed save never fails the boot
            logger.warning("INF834 weight snapshot [%s rank %d]: save failed (%r); nothing committed", self.label, self.rank, e)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()

    def _save(self) -> None:
        tic = time.perf_counter()
        ents = entries(self.model)
        nc = _noncontiguous(ents)
        if nc:
            raise RuntimeError(f"non-contiguous entry {nc}: not supported")
        table = _table(ents)
        size = table[-1]["offset"] + _round_up(table[-1]["nbytes"]) if table else 0
        tmp_dir = os.path.join(self.dir, ".tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_bin = os.path.join(tmp_dir, f"rank{self.rank}.bin.{os.getpid()}")
        # O_DIRECT: the snapshot (54-135 GB per rank) does not go through this pod's page cache and
        # memory cgroup. Falls back to buffered writes where the filesystem refuses O_DIRECT.
        direct = (envs.SGLANG_OPT_INF834_WEIGHT_SNAPSHOT_DIRECT_IO.get() or "auto").lower() not in ("0", "false", "off")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = -1
        if direct and hasattr(os, "O_DIRECT"):
            try:
                fd = os.open(tmp_bin, flags | os.O_DIRECT, 0o644)
            except OSError:
                fd = -1
        if fd < 0:
            direct = False
            fd = os.open(tmp_bin, flags, 0o644)
        os.ftruncate(fd, size)
        chunks = []
        for (name, t), row in zip(ents, table):
            flat = _flat_bytes(t)
            for start in range(0, row["nbytes"], CHUNK):
                n = min(CHUNK, row["nbytes"] - start)
                chunks.append((row["offset"] + start, flat[start:start + n]))
        sums: List[Optional[str]] = [None] * len(chunks)
        tls = threading.local()

        def run(i):
            off, src = chunks[i]
            if src.device.type == "cuda":
                torch.cuda.set_device(src.device)
            st = getattr(tls, "buf", None)
            if st is None:
                st = tls.buf = _Staging(CHUNK)
            n = src.numel()
            buf = st.view(n)
            buf.copy_(src, non_blocking=False)
            sums[i] = _checksum(memoryview(buf.numpy()))
            # O_DIRECT writes whole 4 KiB blocks: the tail of the last block is entry padding.
            mv = memoryview(st.view(_round_up(n) if direct else n).numpy())
            done = 0
            while done < len(mv):
                done += os.pwrite(fd, mv[done:], off + done)

        try:
            with concurrent.futures.ThreadPoolExecutor(self.threads, thread_name_prefix="inf834-snap-save") as pool:
                for f in [pool.submit(run, i) for i in range(len(chunks))]:
                    f.result()
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_bin, self.bin)
        meta = {"identity": self.ident, "table": table, "size": size, "chunk": CHUNK, "checksums": sums,
                "saved_at": time.time(), "algorithm": "xxh3_64" if sums and sums[0].startswith("x") else "crc32"}
        tmp_commit = os.path.join(tmp_dir, f"rank{self.rank}.commit.{os.getpid()}")
        with open(tmp_commit, "w") as f:
            json.dump(meta, f, default=str)
            f.flush()
            os.fsync(f.fileno())
        dfd = os.open(self.dir, os.O_RDONLY)
        try:
            os.fsync(dfd)  # the .bin rename is durable before the commit appears
            os.replace(tmp_commit, self.commit)
            os.fsync(dfd)
        finally:
            os.close(dfd)
        secs = time.perf_counter() - tic
        logger.info(
            "INF834 weight snapshot [%s rank %d]: saved %d entries, %.2f GB in %.2f s (blocking the boot, %s writes, %d threads) to %s",
            self.label, self.rank, len(table), size / 1e9, secs, "O_DIRECT" if direct else "buffered", self.threads, self.bin,
        )
