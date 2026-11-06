"""Unit tests for srt/utils/sched_uclamp.py (scheduler utilization clamp) — no server, no GPU."""

import ctypes
import os
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

from sglang.srt.utils import sched_uclamp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _kernel_has_uclamp() -> bool:
    try:
        with open("/proc/self/sched") as f:
            return "uclamp.min" in f.read()
    except OSError:
        return False


class TestSchedAttrLayout(CustomTestCase):
    def test_struct_is_sched_attr_ver1(self):
        # SCHED_ATTR_SIZE_VER1: the util clamps sit at offsets 48 and 52.
        self.assertEqual(ctypes.sizeof(sched_uclamp.SchedAttr), 56)
        self.assertEqual(sched_uclamp.SchedAttr.sched_util_min.offset, 48)
        self.assertEqual(sched_uclamp.SchedAttr.sched_util_max.offset, 52)


class TestMaybeSetSchedulerUclampMin(CustomTestCase):
    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_OPT_SCHED_UCLAMP_MIN", None)
            with mock.patch.object(
                sched_uclamp, "set_current_thread_uclamp_min"
            ) as setter:
                self.assertIsNone(sched_uclamp.maybe_set_scheduler_uclamp_min())
                setter.assert_not_called()

    def test_env_value_is_applied(self):
        with mock.patch.dict(os.environ, {"SGLANG_OPT_SCHED_UCLAMP_MIN": "1024"}):
            with mock.patch.object(
                sched_uclamp, "set_current_thread_uclamp_min", return_value=1024
            ) as setter:
                self.assertEqual(sched_uclamp.maybe_set_scheduler_uclamp_min(), 1024)
                setter.assert_called_once_with(1024)

    def test_failure_is_a_warning_not_an_error(self):
        # An unsupported kernel/platform must not stop the scheduler.
        for exc in (OSError(95, "Operation not supported"), ValueError("bad")):
            with mock.patch.object(
                sched_uclamp, "set_current_thread_uclamp_min", side_effect=exc
            ):
                with self.assertLogs(sched_uclamp.logger, level="WARNING"):
                    self.assertIsNone(sched_uclamp.maybe_set_scheduler_uclamp_min(512))

    def test_out_of_range_rejected_before_syscall(self):
        with self.assertRaises(ValueError):
            sched_uclamp.set_current_thread_uclamp_min(1025)
        with self.assertRaises(ValueError):
            sched_uclamp.set_current_thread_uclamp_min(-1)

    @unittest.skipIf(sys.platform.startswith("linux"), "non-Linux path only")
    def test_non_linux_raises_oserror(self):
        with self.assertRaises(OSError):
            sched_uclamp.set_current_thread_uclamp_min(1024)


@unittest.skipUnless(
    sys.platform.startswith("linux") and _kernel_has_uclamp(),
    "needs Linux with CONFIG_UCLAMP_TASK and /proc/<pid>/sched",
)
class TestUclampOnLinux(CustomTestCase):
    """Runs in a child process so the test runner's own threads keep their clamp."""

    SCRIPT = textwrap.dedent(
        """
        import threading
        from sglang.srt.utils import sched_uclamp

        def uclamp_min(tid):
            with open(f"/proc/self/task/{tid}/sched") as f:
                for line in f:
                    if line.startswith("uclamp.min"):
                        return int(line.split(":")[1])
            raise RuntimeError("no uclamp.min line")

        before = {}
        started = threading.Event()
        stop = threading.Event()

        def early():
            before["tid"] = threading.get_native_id()
            started.set()
            stop.wait()

        t0 = threading.Thread(target=early)
        t0.start()
        started.wait()
        nice = sched_uclamp.get_current_thread_sched_attr().sched_nice
        got = sched_uclamp.maybe_set_scheduler_uclamp_min(1024)
        late = {}
        t1 = threading.Thread(target=lambda: late.setdefault("min", uclamp_min(threading.get_native_id())))
        t1.start(); t1.join()
        attr = sched_uclamp.get_current_thread_sched_attr()
        print(got, uclamp_min(threading.get_native_id()), uclamp_min(before["tid"]),
              late["min"], attr.sched_nice == nice)
        stop.set(); t0.join()
        """
    )

    def test_set_read_back_and_inherited_by_later_threads(self):
        out = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            capture_output=True,
            text=True,
            check=True,
            env=dict(os.environ),
        ).stdout.split()
        got, main_min, early_min, late_min, nice_kept = out
        self.assertEqual(int(got), 1024)
        self.assertEqual(int(main_min), 1024)
        # Threads started afterwards (gloo, NCCL, ZMQ in the scheduler) inherit it.
        self.assertEqual(int(late_min), 1024)
        # A thread started before the call keeps its own clamp (hence: call early).
        self.assertEqual(int(early_min), 0)
        self.assertEqual(nice_kept, "True")


if __name__ == "__main__":
    unittest.main()
