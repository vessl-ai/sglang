"""INF-834 SGLANG_OPT_INF834_EARLY_CUINIT: run cuInit on a background thread at import time.

Kept free of heavy imports (no torch): it must start before them to overlap them.
"""

import ctypes
import os
import sys
import threading
import time


def _run(t_start: float) -> None:
    try:
        rc = ctypes.CDLL("libcuda.so.1").cuInit(0)
    except OSError as e:  # no driver (GPU-less pod): nothing to overlap
        rc = f"no libcuda ({e})"
    sys.stderr.write(
        f"INF834 early cuInit: pid {os.getpid()} rc={rc} took "
        f"{time.perf_counter() - t_start:.2f} s\n"
    )
    sys.stderr.flush()


def start() -> None:
    threading.Thread(
        target=_run,
        args=(time.perf_counter(),),
        name="inf834-early-cuinit",
        daemon=True,
    ).start()
