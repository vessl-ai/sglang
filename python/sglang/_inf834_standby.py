"""INF-834 SGLANG_OPT_INF834_STANDBY_CHILDREN=<n>: standby scheduler children.

Without the switch the engine spawns its scheduler processes (and the DP controller) only after the
main process has imported sglang and parsed the arguments; every child then imports sglang again
(about 9 s), so the two imports run one after the other. With the switch, `import sglang` in the
main process of `python -m sglang.launch_server` first starts n standby
scheduler children (one per GPU 0..n-1, each spawned behind the same `numactl` wrapper the engine
would use for that GPU) and one standby DP controller. They import the scheduler modules while the
main process imports and parses. When the engine reaches its spawn loop it hands each standby child
the exact (target, args, kwargs) it would have given mp.Process, plus the final os.environ.

A handoff is refused, and the engine spawns that child the normal way, when
- the numactl arguments the engine resolves for the GPU differ from the ones the standby child was
  started with, or
- an environment variable that changed between the standby start and the handoff was read by the
  standby child's Python code during its imports (recorded), or is a CUDA_* variable (the CUDA driver
  was initialized during the imports and reads those itself).
So a child that runs its target sees the same binding, arguments and environment as a normal child.
Every decision is logged ("INF834 standby: ...").

This module is imported by sglang/__init__.py before anything heavy and must stay stdlib-only.
"""

import os
import sys
import threading
import time

_ENV_COUNT = "SGLANG_OPT_INF834_STANDBY_CHILDREN"
_ENV_FORCE = "SGLANG_OPT_INF834_STANDBY_FORCE_MAIN"  # tests: treat any main script as the server
PRELOAD = (
    "sglang.launch_server",
    "sglang.srt.managers.scheduler",
    "sglang.srt.managers.data_parallel_controller",
)

_PROC_NVIDIA_GPUS = "/proc/driver/nvidia/gpus"  # module globals so a test can point them elsewhere
_SYS_PCI_DEVICES = "/sys/bus/pci/devices"
_lock = threading.Lock()
_pool = {}  # key -> (process, parent connection, numactl args or None)
_ready = threading.Event()
_t0 = time.perf_counter()
_stats = {"started": 0, "used": 0, "refused": 0}


def _log(msg: str) -> None:
    sys.stderr.write(f"INF834 standby: {msg}\n")
    sys.stderr.flush()


def _count() -> int:
    try:
        return int(os.environ.get(_ENV_COUNT, "0") or 0)
    except ValueError:
        return 0


def _is_server_main() -> bool:
    import multiprocessing
    import multiprocessing.process

    # A spawned child imports sglang while it bootstraps (parent_process() is still None then).
    if (
        multiprocessing.parent_process() is not None
        or getattr(multiprocessing.process.current_process(), "_inheriting", False)
        or "--multiprocessing-fork" in getattr(sys, "orig_argv", [])
    ):
        return False
    if os.environ.get(_ENV_FORCE, "") == "1":
        return True
    argv = list(getattr(sys, "orig_argv", []))
    return any(
        a == "-m" and i + 1 < len(argv) and argv[i + 1] == "sglang.launch_server"
        for i, a in enumerate(argv)
    )


def maybe_start() -> None:
    n = _count()
    if n <= 0 or not _is_server_main():
        return
    # Spawned from the main thread, synchronously: a background thread starts them only after the
    # main thread's imports release the GIL (seconds late), and PR_SET_PDEATHSIG, which the
    # children set, fires when the *thread* that spawned a child exits.
    _launcher(n)


# ---------------------------------------------------------------------------------------------
# NUMA binding as sglang.srt.utils.numa_utils.configure_subprocess resolves it (stdlib + pynvml).
# The engine compares its own result with these strings at the handoff; any difference refuses.


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _node_cpus(node: int) -> set:
    try:
        text = open(f"/sys/devices/system/node/node{node}/cpulist").read().strip()
    except OSError:
        return set()
    cpus = set()
    for part in text.split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            cpus.update(range(int(a), int(b) + 1))
        else:
            cpus.add(int(part))
    return cpus


def _probe(args: str) -> bool:
    import subprocess

    try:
        return (
            subprocess.run(
                ["numactl", *args.split(), "true"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            ).returncode
            == 0
        )
    except Exception:
        return False


def _probed(args: str):
    if _probe(args):
        return args
    if "--membind=" in args:
        pref = args.replace("--membind=", "--preferred=")
        if _probe(pref):
            return pref
    cpu_only = " ".join(t for t in args.split() if not t.startswith("--membind"))
    if cpu_only and cpu_only != args and _probe(cpu_only):
        return cpu_only
    return None


def _numactl_args_for(gpu_ids):
    """{gpu_id: numactl args or None}. None also when anything is unusual (the engine's own
    resolution then decides, and a mismatch refuses the handoff)."""
    import glob
    import math
    import shutil

    out = {g: None for g in gpu_ids}
    if not (_env_bool("SGLANG_NUMA_BIND_V2", True) and _env_bool("SGLANG_AUTO_NUMA_BIND", True)):
        return out
    if not os.path.isdir("/sys/devices/system/node/node1") or not shutil.which("numactl"):
        return out
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return out  # logical -> NVML index mapping left to the engine
    try:
        nodes, how = _gpu_nodes(gpu_ids)
    except Exception as e:
        _log(f"numa lookup failed ({e!r}); standby children start without a binding")
        return out
    _stats["numa_source"] = how
    try:
        allowed = os.sched_getaffinity(0)
        for g in gpu_ids:
            node = nodes.get(g)
            if node is None:
                continue
            node_cpus = _node_cpus(node)
            if not node_cpus:
                args = f"--cpunodebind={node} --membind={node}"
            else:
                target = node_cpus & allowed
                if not target:
                    continue
                if target == node_cpus:
                    args = f"--cpunodebind={node} --membind={node}"
                else:
                    args = f"--physcpubind={','.join(str(c) for c in sorted(target))} --membind={node}"
            out[g] = _probed(args)
    except Exception as e:
        _log(f"numa binding failed ({e!r}); standby children start without a binding")
        out = {g: None for g in gpu_ids}
    return out


def _gpu_nodes_sysfs(gpu_ids):
    """SGLANG_OPT_INF834_STANDBY_SYSFS_NUMA: the NUMA node of each GPU from sysfs, without the
    NVIDIA driver. NVML numbers GPUs in PCI bus-id order, so GPU i is the i-th entry of
    /proc/driver/nvidia/gpus sorted by bus id; its node is /sys/bus/pci/devices/<bus id>/numa_node.
    nvmlInit runs behind the CUDA driver initialization (cuInit) in this process, which the
    early-cuInit thread has just started, so the NVML path costs seconds here. Returns None when
    sysfs does not give one node per GPU; the engine's NVML result is compared at the handoff."""
    root = _PROC_NVIDIA_GPUS
    if not os.path.isdir(root):
        return None
    bus_ids = sorted(os.listdir(root), key=lambda b: b.lower())
    if not bus_ids or max(gpu_ids) >= len(bus_ids):
        return None
    nodes = {}
    for g in gpu_ids:
        try:
            node = int(open(f"{_SYS_PCI_DEVICES}/{bus_ids[g].lower()}/numa_node").read().strip())
        except (OSError, ValueError):
            return None
        nodes[g] = node if node >= 0 else None
    return nodes


def _gpu_nodes(gpu_ids):
    """({gpu_id: NUMA node or None}, source)."""
    if os.environ.get("SGLANG_OPT_INF834_STANDBY_SYSFS_NUMA", "").lower() in ("1", "true"):
        nodes = _gpu_nodes_sysfs(gpu_ids)
        if nodes is not None:
            return nodes, "sysfs"
    import glob
    import math

    import pynvml

    pynvml.nvmlInit()
    try:
        n_nodes = len(glob.glob("/sys/devices/system/node/node[0-9]*"))
        bits = 64
        size = max(1, math.ceil(n_nodes / bits))
        nodes = {}
        for g in gpu_ids:
            h = pynvml.nvmlDeviceGetHandleByIndex(g)
            mask = pynvml.nvmlDeviceGetMemoryAffinity(h, size, pynvml.NVML_AFFINITY_SCOPE_NODE)
            found = [i for i in range(n_nodes) if mask[i // bits] & (1 << (i % bits))]
            nodes[g] = found[0] if found else None
        return nodes, "nvml"
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------------------------
# Main process side


def _spawn(ctx, key, numactl_args):
    import multiprocessing.spawn as mspawn
    import random

    parent_conn, child_conn = ctx.Pipe(duplex=True)
    proc = ctx.Process(target=_standby_main, args=(key, child_conn), name=f"inf834-standby-{key[0]}-{key[1]}")
    if numactl_args:
        old = os.fsdecode(mspawn.get_executable())
        path = f"/tmp/sglang_temp_file_{time.time()}_{random.randrange(0, 10000000)}.sh"
        with open(path, "w") as f:
            f.write(f'#!/bin/sh\nexec numactl {numactl_args} {old} "$@"')
        os.chmod(path, 0o777)
        mspawn.set_executable(path)
        try:
            proc.start()
        finally:
            mspawn.set_executable(old)
    else:
        proc.start()
    child_conn.close()
    return proc, parent_conn


def _launcher(n: int) -> None:
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    try:
        bindings = _numactl_args_for(list(range(n)))
        t_bind = time.perf_counter() - _t0
        with _lock:
            for g in range(n):
                proc, conn = _spawn(ctx, ("scheduler", g), bindings[g])
                _pool[("scheduler", g)] = (proc, conn, bindings[g])
            proc, conn = _spawn(ctx, ("dp_controller", None), None)
            _pool[("dp_controller", None)] = (proc, conn, None)
            _stats["started"] = len(_pool)
        _log(
            f"started {len(_pool)} standby children in {time.perf_counter() - _t0:.2f} s "
            f"(NUMA lookup {t_bind:.2f} s via {_stats.get('numa_source', '-')}; bindings: {bindings})"
        )
    except Exception as e:
        _log(f"could not start standby children ({e!r})")
    finally:
        _ready.set()


def take(kind: str, gpu_id, numactl_args, target, args=(), kwargs=None):
    """Hand (target, args, kwargs) to the standby child for (kind, gpu_id). Returns the started
    multiprocessing.Process, or None (spawn normally). numactl_args: what the engine resolved."""
    if _count() <= 0:
        return None
    if not _ready.wait(timeout=120):
        return None
    with _lock:
        entry = _pool.pop((kind, gpu_id), None)
    if entry is None:
        return None
    proc, conn, standby_args = entry
    if standby_args != numactl_args:
        _refuse_entry(proc, conn, f"{kind} {gpu_id}: numactl args differ (standby {standby_args!r}, engine {numactl_args!r})")
        return None
    tic = time.perf_counter()
    try:
        conn.send((target, tuple(args), dict(kwargs or {}), dict(os.environ)))
        status, detail = conn.recv()
    except Exception as e:
        _refuse_entry(proc, conn, f"{kind} {gpu_id}: handoff failed ({e!r})")
        return None
    if status != "ok":
        _refuse_entry(proc, conn, f"{kind} {gpu_id}: {detail}")
        return None
    conn.close()
    _stats["used"] += 1
    _log(
        f"{kind} {gpu_id} handed to standby pid {proc.pid} (waited {time.perf_counter() - tic:.2f} s; "
        f"{detail})"
    )
    return proc


def _refuse_entry(proc, conn, reason: str) -> None:
    _stats["refused"] += 1
    _log(f"not used, spawning normally: {reason}")
    try:
        conn.send(None)
    except Exception:
        pass
    conn.close()
    proc.join(timeout=30)
    if proc.is_alive():
        proc.kill()
        proc.join()


def release_unused() -> None:
    """End every standby child that was not handed a target (call after the spawn phase)."""
    if _count() <= 0 or not _ready.is_set():
        return
    with _lock:
        left = list(_pool.items())
        _pool.clear()
    for (kind, g), (proc, conn, _) in left:
        try:
            conn.send(None)
        except Exception:
            pass
        conn.close()
    for (_, _), (proc, _, _) in left:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()
            proc.join()
    _log(f"spawn phase done: used {_stats['used']}, refused {_stats['refused']}, released {len(left)} unused")


# ---------------------------------------------------------------------------------------------
# Standby child side


def _standby_main(key, conn) -> None:
    import ctypes
    import signal

    try:  # die with the main process, as kill_itself_when_parent_died does in the real child
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)
    except Exception:
        pass
    read_keys = set()
    environ_cls = type(os.environ)
    orig_getitem = environ_cls.__getitem__

    def recording_getitem(self, k):
        read_keys.add(k)
        return orig_getitem(self, k)

    environ_cls.__getitem__ = recording_getitem
    tic = time.perf_counter()
    try:
        import importlib

        for m in PRELOAD:
            importlib.import_module(m)
    finally:
        environ_cls.__getitem__ = orig_getitem
    t_import = time.perf_counter() - tic
    msg = conn.recv()
    if msg is None:
        return
    target, args, kwargs, env = msg
    # Compare with this process's environment after its own imports: the import of sglang sets
    # some variables itself (cache dirs, library paths) exactly as it did in the main process.
    mine = dict(os.environ)
    changed = {k for k in set(env) | set(mine) if env.get(k) != mine.get(k)}
    blocked = sorted(k for k in changed if k in read_keys or k.startswith("CUDA_"))
    if blocked:
        conn.send(("refuse", f"env changed after the standby import and read by it: {blocked}"))
        conn.close()
        return
    for k in changed:
        if k in env:
            os.environ[k] = env[k]
        else:
            os.environ.pop(k, None)
    conn.send(("ok", f"standby import {t_import:.2f} s, {len(changed)} env keys updated: {sorted(changed)}"))
    conn.close()
    target(*args, **kwargs)
