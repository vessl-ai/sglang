"""Keep the PD transport and the HiCache L3 client out of a process checkpoint.

A process checkpoint (CRIU) cannot carry an RDMA device, and the process is
restored into a new pod with a new IP. With ``--checkpoint-prepare-on-startup``
two opt-in env flags make an engine boot without that state and build it in
``/checkpoint/restore``, after the KV cache is mapped again:

* ``SGLANG_CHECKPOINT_DEFER_PD_TRANSPORT`` (PD prefill or decode, Mooncake):
  the KV managers boot on the fake transfer backend, so the scheduler never
  creates a Mooncake TransferEngine (RDMA devices, handshake listener), never
  registers KV memory with it, binds no ZMQ socket to the pod IP and does not
  register to the prefill bootstrap server. Warmup already uses fake transfers.
  Restore creates the TransferEngine with the new IP, then a Mooncake KV
  manager per queue, which registers the KV, state and metadata buffers, binds
  its sockets and, on prefill, registers to the bootstrap server.
* ``SGLANG_CHECKPOINT_DEFER_HICACHE_STORAGE`` (HiCache with the Mooncake L3
  backend): HiCache boots with its device and host pools, laid out and
  allocated for the configured backend, but the L3 client is not created.
  Restore attaches it through the same path as ``PUT /hicache/storage-backend``
  with the original ``--hicache-storage-backend*`` arguments and the new IP as
  the client's local hostname.

Every registration or connection failure raises, which leaves the checkpoint
manager FAILED and makes ``/checkpoint/restore`` return an error. Each restore
phase logs one line with its duration.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.environ import envs
from sglang.srt.runtime_context import (
    get_device,
    get_disagg,
    get_exec,
    get_memory,
    get_serving,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

CHECKPOINT_REJECT_MESSAGE = (
    "This engine is preparing for or holding a process checkpoint; requests are "
    "served again after /checkpoint/restore."
)
DEFERRED_TRANSPORT_REJECT_MESSAGE = (
    "The PD transport of this engine is not built yet; it is built by "
    "/checkpoint/restore. Only warmup and health requests are served before."
)


def checkpoint_refuses_new_requests(scheduler: Scheduler) -> bool:
    """With an abort-mode prepare the scheduler keeps running, so a request that
    reaches it while it drains for the KV release, or after it, would run
    without a KV cache: refuse it. A retract pause holds new requests in the
    waiting queue instead, which is safe."""
    from sglang.srt.managers.io_struct import CheckpointState

    manager = getattr(scheduler, "checkpoint_manager", None)
    return (
        manager is not None
        and manager.state != CheckpointState.RUNNING
        and not getattr(scheduler, "_engine_paused", False)
    )


def defers_pd_transport() -> bool:
    """Whether this engine boots without its PD transport."""
    return (
        envs.SGLANG_CHECKPOINT_DEFER_PD_TRANSPORT.get()
        and bool(get_serving().checkpoint_prepare_on_startup)
        and get_disagg().disaggregation_mode != "null"
    )


def defers_hicache_storage() -> bool:
    """Whether this engine boots HiCache without its L3 storage client."""
    memory = get_memory()
    return (
        envs.SGLANG_CHECKPOINT_DEFER_HICACHE_STORAGE.get()
        and bool(get_serving().checkpoint_prepare_on_startup)
        and bool(memory.enable_hierarchical_cache)
        and memory.hicache_storage_backend is not None
    )


def validate_checkpoint_deferral(cfg: Any) -> None:
    """Refuse configurations the deferral does not cover. A flag whose subject
    is absent (no PD mode, no L3 backend) is a no-op and passes."""
    from sglang.srt.mem_cache.host_pool_checkpoint import validate_host_pool_release

    validate_host_pool_release(cfg)
    defer_pd = (
        envs.SGLANG_CHECKPOINT_DEFER_PD_TRANSPORT.get()
        and cfg.disaggregation_mode != "null"
    )
    defer_l3 = (
        envs.SGLANG_CHECKPOINT_DEFER_HICACHE_STORAGE.get()
        and cfg.enable_hierarchical_cache
        and cfg.hicache_storage_backend is not None
    )
    if not (defer_pd or defer_l3):
        return

    def refuse(flag: str, why: str) -> None:
        raise ValueError(f"{flag}=1 {why}")

    if defer_pd:
        flag = "SGLANG_CHECKPOINT_DEFER_PD_TRANSPORT"
        if not cfg.checkpoint_prepare_on_startup:
            # Without it nothing would ever build the transport.
            refuse(flag, "requires --checkpoint-prepare-on-startup")
        if cfg.disaggregation_transfer_backend != "mooncake":
            refuse(flag, "supports only --disaggregation-transfer-backend mooncake")
        if cfg.enable_pd_role_switch:
            refuse(flag, "does not support --enable-pd-role-switch")
        if envs.SGLANG_DISAGG_STAGING_BUFFER.get():
            refuse(flag, "does not support SGLANG_DISAGG_STAGING_BUFFER")
        if envs.SGLANG_RUST_SERVER.get():
            refuse(flag, "does not support SGLANG_RUST_SERVER")
        # Other users of the shared TransferEngine would need it at boot.
        if getattr(cfg, "elastic_ep_backend", None) is not None or getattr(
            cfg, "enable_elastic_expert_backup", False
        ):
            refuse(flag, "does not support elastic EP")
        if (
            getattr(cfg, "encoder_only", False) or getattr(cfg, "language_only", False)
        ) and getattr(cfg, "encoder_transfer_backend", None) == "mooncake":
            refuse(flag, "does not support Mooncake encoder disaggregation")
        if (
            cfg.enable_hierarchical_cache
            and cfg.hicache_storage_backend == "mooncake"
            and not envs.SGLANG_CHECKPOINT_DEFER_HICACHE_STORAGE.get()
        ):
            # The L3 client opens its own RDMA context and master connection.
            refuse(
                flag,
                "with the Mooncake HiCache backend also requires "
                "SGLANG_CHECKPOINT_DEFER_HICACHE_STORAGE=1",
            )

    if defer_l3:
        flag = "SGLANG_CHECKPOINT_DEFER_HICACHE_STORAGE"
        if not cfg.checkpoint_prepare_on_startup:
            refuse(flag, "requires --checkpoint-prepare-on-startup")
        if cfg.hicache_storage_backend != "mooncake":
            refuse(flag, "supports only --hicache-storage-backend mooncake")
        if getattr(cfg, "disaggregation_decode_enable_offload_kvcache", False):
            # The decode offload manager builds its own storage client at boot.
            refuse(
                flag, "does not support --disaggregation-decode-enable-offload-kvcache"
            )


def resolve_restore_host_ip(requested: Optional[str]) -> str:
    """The IP of the pod this process now runs in.

    The process environment is the one captured at checkpoint time, so the
    SGLANG_HOST_IP / HOST_IP that get_local_ip_auto() reads first may name the
    old pod. Use the IP the restore request names, else detect it from the
    live network namespace the way get_local_ip_auto() does without the env.
    """
    from sglang.srt.utils.network import (
        get_local_ip_by_nic,
        get_local_ip_by_remote,
    )

    if requested:
        return str(ipaddress.ip_address(requested))
    ip = get_local_ip_by_nic() or get_local_ip_by_remote()
    if not ip:
        raise RuntimeError(
            "Cannot detect this pod's IP; pass /checkpoint/restore?host_ip=<pod IP>"
        )
    return ip


def _point_env_at(host_ip: str, *, hicache: bool) -> None:
    """Make later IP lookups in this process return the restored pod's IP."""
    names = ["SGLANG_HOST_IP"]
    if hicache:
        # Takes precedence over local_hostname in the L3 extra config.
        names.append("MOONCAKE_LOCAL_HOSTNAME")
    for name in names:
        old = os.environ.get(name)
        if old != host_ip:
            logger.info(
                "Checkpoint restore sets %s=%s (was %s)", name, host_ip, old or "unset"
            )
        os.environ[name] = host_ip


def _log_phase(phase: str, seconds: float, **fields: Any) -> None:
    extra = "".join(f" {k}={v}" for k, v in fields.items())
    logger.info(f"Checkpoint restore phase={phase} seconds={seconds:.3f}{extra}")


def _pd_queue(scheduler: Scheduler):
    """The queue that owns this scheduler's KV manager."""
    from sglang.srt.disaggregation.utils import DisaggregationMode

    if scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
        return scheduler.disagg_prefill_bootstrap_queue
    if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
        return scheduler.disagg_decode_prealloc_queue
    return None


# ---------------------------------------------------------------------------
# Diagnostics for "the restored engine reads KV the native one does not":
# the registered regions, compared across boot and restore, and a fill of the
# KV memory after resume. memory saver resume maps fresh physical memory and
# does not clear it, while a fresh boot allocates the KV pools with
# torch.zeros, so anything a kernel reads but no transfer writes differs.
# ---------------------------------------------------------------------------

KV_FILL_MODES = ("none", "zero", "poison")
_KV_FILL_BYTE = {"zero": 0x00, "poison": 0xFF}


def _cuda_driver():
    import ctypes

    return ctypes.CDLL("libcuda.so.1")


def _cu_memset_d8(ptr: int, value: int, nbytes: int) -> None:
    import ctypes

    fn = _cuda_driver().cuMemsetD8_v2
    fn.argtypes = [ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t]
    fn.restype = ctypes.c_int
    rc = fn(ptr, value, nbytes)
    if rc != 0:
        raise RuntimeError(f"cuMemsetD8({ptr:#x}, {value}, {nbytes}) returned {rc}")


def _cu_allocation_of(ptr: int) -> Optional[tuple]:
    """(base, size) of the CUDA allocation that contains ptr, or None."""
    import ctypes

    try:
        fn = _cuda_driver().cuMemGetAddressRange_v2
    except OSError:
        return None
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_uint64,
    ]
    fn.restype = ctypes.c_int
    base, size = ctypes.c_uint64(), ctypes.c_size_t()
    if fn(ctypes.byref(base), ctypes.byref(size), ptr) != 0:
        return None
    return base.value, size.value


def _gpu_regions(kv_args) -> dict:
    """Transfer-layout regions per kind: {kind: [(ptr, len, item_len), ...]}.
    Draft entries are the tail of kv_data (num_draft_entries)."""
    ptrs = list(getattr(kv_args, "kv_data_ptrs", None) or [])
    lens = list(getattr(kv_args, "kv_data_lens", None) or [])
    items = list(getattr(kv_args, "kv_item_lens", None) or [])
    n_draft = int(getattr(kv_args, "num_draft_entries", 0) or 0)
    split = len(ptrs) - n_draft
    out = {
        "kv": list(zip(ptrs[:split], lens[:split], items[:split])),
        "draft_kv": list(zip(ptrs[split:], lens[split:], items[split:])),
    }
    state_ptrs = getattr(kv_args, "state_data_ptrs", None) or []
    state_lens = getattr(kv_args, "state_data_lens", None) or []
    state_items = getattr(kv_args, "state_item_lens", None) or []
    state_types = getattr(kv_args, "state_types", None) or []
    for i, (p, l) in enumerate(zip(state_ptrs, state_lens)):
        name = (
            getattr(state_types[i], "value", state_types[i])
            if i < len(state_types)
            else i
        )
        it = state_items[i] if i < len(state_items) else []
        out[f"state{i}_{name}"] = list(zip(p, l, it or [0] * len(p)))
    return out


def _host_regions(kv_args) -> list:
    return list(
        zip(
            getattr(kv_args, "aux_data_ptrs", None) or [],
            getattr(kv_args, "aux_data_lens", None) or [],
            getattr(kv_args, "aux_item_lens", None) or [],
        )
    )


def _digest(values) -> str:
    import hashlib

    return hashlib.sha256(repr(list(values)).encode()).hexdigest()[:12]


def log_kv_regions(tag: str, kv_args, *, gpu: bool = True) -> None:
    """With SGLANG_CHECKPOINT_LOG_KV_REGIONS=1: one line per region kind with
    count, bytes, digests of lengths, item lengths and offsets relative to the
    first region (comparable across processes), the first pointer (comparable
    within one process), and how many regions are not inside one CUDA
    allocation (mooncake then exports the requested range for dma-buf)."""
    if not envs.SGLANG_CHECKPOINT_LOG_KV_REGIONS.get() or kv_args is None:
        return
    kinds = dict(_gpu_regions(kv_args))
    kinds["aux_host"] = _host_regions(kv_args)
    for kind, regions in kinds.items():
        if not regions:
            continue
        ptrs = [int(p) for p, _, _ in regions]
        lens = [int(l) for _, l, _ in regions]
        not_one_alloc = "-"
        if gpu and kind != "aux_host":
            n = 0
            for p, l in zip(ptrs, lens):
                if l == 0:
                    continue
                alloc = _cu_allocation_of(p)
                if alloc is None or p + l > alloc[0] + alloc[1]:
                    n += 1
            not_one_alloc = n
        logger.info(
            f"KV regions tag={tag} kind={kind} n={len(regions)} bytes={sum(lens)} "
            f"lens={_digest(lens)} items={_digest(int(i) for _, _, i in regions)} "
            f"rel={_digest(p - ptrs[0] for p in ptrs)} first={ptrs[0]:#x} "
            f"not_one_alloc={not_one_alloc}"
        )


def kv_args_differences(a, b) -> list:
    """Names of KVArgs fields whose values differ (fields of both only)."""
    if a is None or b is None:
        return ["<missing>"]
    va, vb = vars(a), vars(b)
    diff = []
    for key in sorted(set(va) & set(vb)):
        try:
            same = bool(va[key] == vb[key])
        except Exception:
            same = repr(va[key]) == repr(vb[key])
        if not same:
            diff.append(key)
    return diff


def fill_kv_region(scheduler: Scheduler, mode: str) -> None:
    """Fill the KV transfer regions (KV, draft KV, state) after resume, and
    zero the request-to-token table in either fill mode. "zero" restores what
    a fresh boot's torch.zeros pools hold; "poison" (0xFF bytes) makes every
    read of KV memory that no transfer wrote produce NaN."""
    if mode not in KV_FILL_MODES:
        raise ValueError(f"kv_fill={mode!r}; expected one of {KV_FILL_MODES}")
    if mode == "none":
        return
    import torch

    tic = time.perf_counter()
    value = _KV_FILL_BYTE[mode]
    queue = _pd_queue(scheduler)
    kv_args = getattr(getattr(queue, "kv_manager", None), "kv_args", None)
    if getattr(scheduler, "enable_hisparse", False):
        # Its KV transfer regions are host memory.
        kv_args = None
    nbytes = nregions = 0
    if kv_args is not None:
        for regions in _gpu_regions(kv_args).values():
            for ptr, length, _ in regions:
                if length:
                    _cu_memset_d8(int(ptr), value, int(length))
                    nbytes += int(length)
                    nregions += 1
    req_to_token = getattr(
        getattr(scheduler, "req_to_token_pool", None), "req_to_token", None
    )
    if req_to_token is not None:
        req_to_token.zero_()
    torch.cuda.synchronize()
    _log_phase(
        "kv_fill",
        time.perf_counter() - tic,
        mode=mode,
        regions=nregions,
        bytes=nbytes,
        req_to_token_zeroed=req_to_token is not None,
    )


def capture_kv_region_image(scheduler: Scheduler) -> None:
    """Before the first KV release: keep the region's startup bytes (see
    mem_cache/kv_region_image.py). Later prepares reuse the first image, so
    every restore returns the region to its startup state."""
    if not envs.SGLANG_CHECKPOINT_KV_REGION_IMAGE.get():
        return
    if getattr(scheduler, "kv_region_image", None) is not None:
        return
    from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
    from sglang.srt.mem_cache.kv_region_image import capture_kv_region

    scheduler.kv_region_image = capture_kv_region(
        GPU_MEMORY_TYPE_KV_CACHE,
        max_bytes=envs.SGLANG_CHECKPOINT_KV_REGION_IMAGE_MAX_MB.get() << 20,
    )


def before_kv_release(scheduler: Scheduler) -> None:
    """The checkpoint manager's before_release: the KV region image, then (with
    SGLANG_CHECKPOINT_RELEASE_HOST_POOL) the HiCache host pool release."""
    from sglang.srt.mem_cache.host_pool_checkpoint import release_host_pool

    capture_kv_region_image(scheduler)
    release_host_pool(scheduler)


def apply_kv_region_image(scheduler: Scheduler) -> None:
    image = getattr(scheduler, "kv_region_image", None)
    if image is None:
        return
    from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
    from sglang.srt.mem_cache.kv_region_image import apply_kv_region

    tic = time.perf_counter()
    apply_kv_region(GPU_MEMORY_TYPE_KV_CACHE, image)
    _log_phase(
        "kv_region_image",
        time.perf_counter() - tic,
        segments=len(image.segments),
        image_bytes=image.nbytes,
    )


def _kv_pools(scheduler: Scheduler) -> list:
    pools = []
    allocator = getattr(scheduler, "token_to_kv_pool_allocator", None)
    if allocator is not None:
        pools.append(allocator.get_kvcache())
    draft_worker = getattr(scheduler, "draft_worker", None)
    if draft_worker is not None:
        pools.append(getattr(draft_worker, "primary_draft_kv_pool", None))
    return [p for p in pools if p is not None]


def restore_kv_initial_state(scheduler: Scheduler) -> None:
    """Re-write what the KV pools initialize once at construction and kernels
    rely on: memory saver resume maps fresh physical pages, so these rows are
    whatever those pages held. DeepSeek-V4 compressor state: the C4 -1 row
    (padding sentinel, score -inf) and the request-scoped rings; the
    request-to-token padding row 0, which padded CUDA-graph rows read."""
    tic = time.perf_counter()
    n = 0
    for pool in _kv_pools(scheduler):
        for name in ("compress_state_pools", "indexer_compress_state_pools"):
            for state_pool in getattr(pool, name, None) or []:
                reset = getattr(state_pool, "reset_initial_rows", None)
                if reset is not None:
                    reset()
                    n += 1
    req_to_token = getattr(
        getattr(scheduler, "req_to_token_pool", None), "req_to_token", None
    )
    if req_to_token is not None:
        req_to_token[0].zero_()
    _log_phase(
        "kv_reinit",
        time.perf_counter() - tic,
        compress_state_pools=n,
        req_to_token_row0=req_to_token is not None,
    )


def check_pd_registration(scheduler: Scheduler) -> None:
    """Raise if the KV manager reports a failed registration (C0). Covers the
    in-place restore, where resume re-registers the KV memory itself."""
    queue = _pd_queue(scheduler)
    kv_manager = getattr(queue, "kv_manager", None)
    if kv_manager is None:
        return
    if getattr(kv_manager, "buffers_registered", None) is False:
        raise RuntimeError(
            "Mooncake did not register the KV, state or metadata buffers; "
            "see the 'batch memory registration failed' log line"
        )
    if getattr(kv_manager, "bootstrap_registered", None) is False:
        raise RuntimeError("The prefill could not register to its bootstrap server")


def _build_pd_transport(scheduler: Scheduler, host_ip: str) -> None:
    from sglang.srt.disaggregation.utils import TransferBackend
    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
        init_mooncake_transfer_engine,
    )

    tic = time.perf_counter()
    engine = init_mooncake_transfer_engine(
        hostname=host_ip,
        gpu_id=get_device().gpu_id,
        ib_device=(
            get_disagg().disaggregation_ib_device or get_exec().moe.mooncake_ib_device
        ),
    )
    _log_phase(
        "transport_init",
        time.perf_counter() - tic,
        session=engine.get_session_id(),
        ib_device=engine.get_ib_device() or "-",
    )

    queue = _pd_queue(scheduler)
    backend = TransferBackend(get_disagg().disaggregation_transfer_backend)
    boot_kv_args = getattr(queue.kv_manager, "kv_args", None)
    log_kv_regions("boot-fake", boot_kv_args)
    tic = time.perf_counter()
    # The queue's buffers and pools are the ones the fake manager was built on.
    queue.transfer_backend = backend
    queue.kv_manager = queue._init_kv_manager()
    total = time.perf_counter() - tic
    kv_manager = queue.kv_manager
    # The fake manager got its kv_args from the same _init_kv_manager at boot,
    # so any difference is state that changed between boot and restore.
    differences = kv_args_differences(
        boot_kv_args, getattr(kv_manager, "kv_args", None)
    )
    _log_phase(
        "mr_registration",
        getattr(kv_manager, "buffer_registration_seconds", float("nan")),
        ok=getattr(kv_manager, "buffers_registered", None),
        regions=getattr(kv_manager, "num_registered_regions", None),
    )
    if hasattr(kv_manager, "bootstrap_registered"):
        _log_phase(
            "bootstrap_registration",
            getattr(kv_manager, "bootstrap_registration_seconds", float("nan")),
            ok=kv_manager.bootstrap_registered,
        )
    _log_phase(
        "kv_manager",
        total,
        rank_ip=getattr(kv_manager, "local_ip", None),
        rank_port=getattr(kv_manager, "rank_port", None),
        kv_args_same_as_boot=not differences,
    )
    if differences:
        logger.warning(f"Checkpoint restore kv_args differ from boot in {differences}")
    check_pd_registration(scheduler)
    scheduler.transfer_backend = backend
    scheduler.pd_transport_deferred = False


def _attach_hicache_storage(scheduler: Scheduler) -> None:
    from sglang.srt.managers.io_struct import AttachHiCacheStorageReqInput

    memory = get_memory()
    tic = time.perf_counter()
    out = scheduler.attach_hicache_storage_wrapped(
        AttachHiCacheStorageReqInput(
            hicache_storage_backend=memory.hicache_storage_backend,
            hicache_storage_backend_extra_config_json=(
                memory.hicache_storage_backend_extra_config
            ),
            hicache_storage_prefetch_policy=memory.hicache_storage_prefetch_policy,
            hicache_write_policy=memory.hicache_write_policy,
        )
    )
    _log_phase(
        "l3_attach",
        time.perf_counter() - tic,
        ok=out.success,
        backend=memory.hicache_storage_backend,
    )
    if not out.success:
        raise RuntimeError(
            f"Attaching the HiCache storage backend failed: {out.message}"
        )
    scheduler.hicache_storage_deferred = False


def _overlap_refusal() -> Optional[str]:
    """Why the L3 attach may not run beside the PD transport build, or None.
    The attach creates gloo groups (world collectives) on its thread; the PD
    build must then run none, which holds for one node without the Rust
    server (common/conn.py: the bootstrap port sync and topology gather)."""
    from sglang.srt.runtime_context import get_parallel

    if get_parallel().nnodes != 1:
        return "multi-node engine (the PD build broadcasts the bootstrap port)"
    if envs.SGLANG_RUST_SERVER.get():
        return "SGLANG_RUST_SERVER (the PD build gathers the topology)"
    return None


def _build_overlapped(scheduler: Scheduler, ip: str) -> None:
    """The host pool registration and the L3 attach on a second thread while
    this thread builds the PD transport (two Mooncake transfer engines are
    created at once, one per thread)."""
    import threading

    import torch

    from sglang.srt.mem_cache.host_pool_checkpoint import restore_host_pool

    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    result: dict = {}

    def side() -> None:
        t0 = time.perf_counter()
        try:
            if device is not None:
                torch.cuda.set_device(device)
            restore_host_pool(scheduler)
            _attach_hicache_storage(scheduler)
        except BaseException as e:  # re-raised on the restore thread
            result["error"] = e
        result["seconds"] = time.perf_counter() - t0

    tic = time.perf_counter()
    thread = threading.Thread(target=side, name="checkpoint-restore-l3", daemon=True)
    thread.start()
    pd_error = None
    try:
        _build_pd_transport(scheduler, ip)
    except BaseException as e:
        pd_error = e
    pd_seconds = time.perf_counter() - tic
    thread.join()
    _log_phase(
        "overlap_l3",
        time.perf_counter() - tic,
        pd_seconds=f"{pd_seconds:.3f}",
        l3_seconds=f"{result.get('seconds', float('nan')):.3f}",
        ok=pd_error is None and "error" not in result,
    )
    if pd_error is not None:
        raise pd_error
    if "error" in result:
        raise result["error"]


def build_transport_on_restore(
    scheduler: Scheduler,
    host_ip: Optional[str],
    kv_fill: Optional[str] = None,
    overlap_l3: Optional[bool] = None,
) -> None:
    """Run by the checkpoint manager after the KV cache is mapped again.
    `kv_fill` (request) or SGLANG_CHECKPOINT_RESTORE_KV_FILL (env) fills the
    KV memory first (see fill_kv_region); the rows the pools initialize at
    construction are then written again (restore_kv_initial_state). A HiCache
    host pool released at prepare (SGLANG_CHECKPOINT_RELEASE_HOST_POOL) is
    faulted in and registered again before the L3 client is attached.
    `overlap_l3` (request) or SGLANG_CHECKPOINT_RESTORE_OVERLAP_L3 (env) runs
    that and the L3 attach on a second thread beside the PD transport build."""
    from sglang.srt.mem_cache.host_pool_checkpoint import restore_host_pool

    fill_kv_region(scheduler, kv_fill or envs.SGLANG_CHECKPOINT_RESTORE_KV_FILL.get())
    apply_kv_region_image(scheduler)
    restore_kv_initial_state(scheduler)
    defer_pd = getattr(scheduler, "pd_transport_deferred", False)
    defer_l3 = getattr(scheduler, "hicache_storage_deferred", False)
    if not (defer_pd or defer_l3):
        restore_host_pool(scheduler)
        check_pd_registration(scheduler)
        return

    tic = time.perf_counter()
    ip = resolve_restore_host_ip(host_ip)
    _point_env_at(ip, hicache=defer_l3)
    _log_phase("host_ip", time.perf_counter() - tic, host_ip=ip)
    overlap = (
        overlap_l3
        if overlap_l3 is not None
        else envs.SGLANG_CHECKPOINT_RESTORE_OVERLAP_L3.get()
    )
    if overlap and defer_pd and defer_l3:
        refusal = _overlap_refusal()
        if refusal is not None:
            logger.warning(f"Checkpoint restore: overlap_l3 not used: {refusal}")
            overlap = False
    else:
        overlap = False
    if overlap:
        _build_overlapped(scheduler, ip)
    else:
        restore_host_pool(scheduler)
        if defer_pd:
            _build_pd_transport(scheduler, ip)
        if defer_l3:
            _attach_hicache_storage(scheduler)
    _log_phase("total", time.perf_counter() - tic, overlap_l3=int(overlap))
