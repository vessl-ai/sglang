"""Keep a host-bound scheduler thread on a fast CPU clock (Linux utilization clamp).

Under the ``schedutil`` cpufreq governor a CPU's clock follows the frequency-invariant
utilization of the tasks on it and is sized so that the CPU is busy about 80% of the
time. A DP-attention decode scheduler whose iteration is bound by its own host loop
sleeps in every iteration's cross-rank CPU collectives for as long as the slowest rank
needs, so its busy fraction stays near 60% whatever the clock is. The governor then
keeps lowering the clock (0.8-1.5 GHz measured on a 4.1 GHz part), which makes the host
loop slower still: a stable slow state. A scheduler that waits for the GPU instead spins
in the CUDA synchronize call, stays near 100% busy and runs at the top clock.

A per-task minimum utilization clamp (``sched_setattr`` with
``SCHED_FLAG_UTIL_CLAMP_MIN``) makes schedutil request the clamped utilization whenever
the task is runnable, independent of its duty cycle. Threads created afterwards inherit
the clamp, so applying it before the process starts its runtime threads (gloo, NCCL,
ZMQ) covers them too. Raising a task's own clamp needs no capability; it needs a kernel
built with ``CONFIG_UCLAMP_TASK`` and is bounded by the cgroup's ``cpu.uclamp.*``.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import sys
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# linux/sched.h
SCHED_FLAG_KEEP_POLICY = 0x08
SCHED_FLAG_KEEP_PARAMS = 0x10
SCHED_FLAG_UTIL_CLAMP_MIN = 0x20

UCLAMP_MAX_VALUE = 1024

# syscall numbers of sched_setattr / sched_getattr
_SYSCALL_NR = {
    "x86_64": (314, 315),
    "aarch64": (274, 275),
}


class SchedAttr(ctypes.Structure):
    """``struct sched_attr`` (linux/sched/types.h), SCHED_ATTR_SIZE_VER1 = 56 bytes."""

    _fields_ = [
        ("size", ctypes.c_uint32),
        ("sched_policy", ctypes.c_uint32),
        ("sched_flags", ctypes.c_uint64),
        ("sched_nice", ctypes.c_int32),
        ("sched_priority", ctypes.c_uint32),
        ("sched_runtime", ctypes.c_uint64),
        ("sched_deadline", ctypes.c_uint64),
        ("sched_period", ctypes.c_uint64),
        ("sched_util_min", ctypes.c_uint32),
        ("sched_util_max", ctypes.c_uint32),
    ]


def _syscall_numbers() -> Optional[tuple]:
    if not sys.platform.startswith("linux"):
        return None
    return _SYSCALL_NR.get(platform.machine())


def _libc():
    return ctypes.CDLL(None, use_errno=True)


def get_current_thread_sched_attr() -> SchedAttr:
    """``sched_getattr`` of the calling thread; raises OSError where unsupported."""
    nrs = _syscall_numbers()
    if nrs is None:
        raise OSError(
            f"sched_getattr is not available on {sys.platform}/{platform.machine()}"
        )
    attr = SchedAttr()
    ret = _libc().syscall(
        ctypes.c_long(nrs[1]),
        ctypes.c_int(0),
        ctypes.byref(attr),
        ctypes.c_uint(ctypes.sizeof(SchedAttr)),
        ctypes.c_uint(0),
    )
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"sched_getattr: {os.strerror(err)}")
    return attr


def set_current_thread_uclamp_min(value: int) -> int:
    """Set the calling thread's requested minimum utilization clamp; returns the value
    read back. Policy, nice value and the maximum clamp are kept. Raises OSError when
    the kernel or the platform does not support it, ValueError for a bad value."""
    if not 0 <= value <= UCLAMP_MAX_VALUE:
        raise ValueError(f"uclamp min must be in [0, {UCLAMP_MAX_VALUE}], got {value}")
    nrs = _syscall_numbers()
    if nrs is None:
        raise OSError(
            f"sched_setattr is not available on {sys.platform}/{platform.machine()}"
        )
    # Start from the current attributes: the kernel compares sched_nice with the
    # thread's nice value even with KEEP_PARAMS, so a zero there would read as a
    # priority raise for a thread with a positive nice value.
    attr = get_current_thread_sched_attr()
    attr.size = ctypes.sizeof(SchedAttr)
    attr.sched_flags = (
        SCHED_FLAG_KEEP_POLICY | SCHED_FLAG_KEEP_PARAMS | SCHED_FLAG_UTIL_CLAMP_MIN
    )
    attr.sched_util_min = value
    ret = _libc().syscall(
        ctypes.c_long(nrs[0]), ctypes.c_int(0), ctypes.byref(attr), ctypes.c_uint(0)
    )
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"sched_setattr(util_min={value}): {os.strerror(err)}")
    return int(get_current_thread_sched_attr().sched_util_min)


def maybe_set_scheduler_uclamp_min(value: Optional[int] = None) -> Optional[int]:
    """Apply ``SGLANG_OPT_SCHED_UCLAMP_MIN`` (or ``value``) to the calling thread.

    Off when the value is negative (the default). Never raises: an unsupported kernel
    or platform only logs a warning, because the clamp is a performance hint and the
    scheduler must run the same without it. Returns the value read back, or None.
    """
    if value is None:
        from sglang.srt.environ import envs

        value = envs.SGLANG_OPT_SCHED_UCLAMP_MIN.get()
    if value is None or value < 0:
        return None
    try:
        got = set_current_thread_uclamp_min(value)
    except (OSError, ValueError) as e:
        logger.warning(
            "SGLANG_OPT_SCHED_UCLAMP_MIN=%s not applied (%s); the scheduler runs "
            "without a utilization clamp.",
            value,
            e,
        )
        return None
    logger.info(
        "Scheduler thread %d utilization clamp min = %d (SGLANG_OPT_SCHED_UCLAMP_MIN).",
        threading.get_native_id(),
        got,
    )
    return got
