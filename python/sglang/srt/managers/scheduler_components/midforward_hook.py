"""Scheduler work run from inside a long prefill forward, at fixed layer indices.

A full prefill step launches thousands of kernels. Once the CUDA launch queue of
the forward stream is full, the host thread spends most of the step blocked in
launch calls (INF-681 k-gap: ~200 ms per 65k-token CP4 step), while the
scheduler's own work (the previous batch's result, KV-transfer polling, request
intake) runs after the forward returns, with the GPU idle. The scheduler arms
callbacks keyed by layer index before ``run_batch``; the model calls
``maybe_fire(layer_id)`` at the top of each layer and the callback runs right
there, while the GPU drains the queued kernels.

Rules for callbacks:
- Every rank of the group must arm the same layers for the same batch, because
  callbacks run CPU collectives (all ranks reach layer ``i`` in the same order).
- Callbacks must not enqueue work on the forward stream; they switch to their
  own stream.
- A callback fires at most once per arm; ``disarm`` returns the layers that fired
  so the caller can run whatever did not fire after the forward instead.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch

_callbacks: Optional[Dict[int, Callable[[], None]]] = None
_fired: List[int] = []


def parse_layers(spec: str) -> List[int]:
    """'4,12' -> [4, 12] (sorted, unique, non-negative)."""
    layers = sorted({int(x) for x in spec.split(",") if x.strip()})
    if any(layer < 0 for layer in layers):
        raise ValueError(f"mid-forward hook layers must be >= 0, got {spec!r}")
    return layers


def arm(callbacks: Dict[int, Callable[[], None]]) -> None:
    global _callbacks
    assert _callbacks is None, "mid-forward hooks are already armed"
    _callbacks = dict(callbacks)
    _fired.clear()


def disarm() -> List[int]:
    """Drop the callbacks that did not fire; return the layers that fired."""
    global _callbacks
    _callbacks = None
    fired = list(_fired)
    _fired.clear()
    return fired


def is_armed() -> bool:
    return _callbacks is not None


def maybe_fire(layer_id: int) -> None:
    """Called by the model at the top of layer ``layer_id``; no-op unless armed."""
    if torch.compiler.is_compiling() or _callbacks is None:
        # A compiled region never runs the callbacks: the caller does the work
        # after the forward instead.
        return
    callback = _callbacks.pop(layer_id, None)
    if callback is None:
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        # Never run scheduler work inside a graph capture; the caller runs it after
        # the forward instead (it is not marked as fired).
        return
    _fired.append(layer_id)
    callback()
