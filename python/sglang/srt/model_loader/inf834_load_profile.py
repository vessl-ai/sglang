"""INF-834 diagnostic (SGLANG_OPT_INF834_LOAD_PROFILE): where one load_weights call
spends its wall time. Not a lever: it only reads the clock.

"iterator wait" is the time the consumer loop spent inside the weight iterator
(shard loading, prefetch, dequant generators); the rest of the stream phase is
the consumer's own work in the main thread (name routing, submitting or running
the per-tensor loaders).
"""

from __future__ import annotations

import logging
import time
from typing import Iterable, Iterator, List, Tuple

logger = logging.getLogger(__name__)


class Inf834LoadProfile:
    def __init__(self, label: str):
        self.label = label
        self.start = time.perf_counter()
        self.last = self.start
        self.phases: List[Tuple[str, float]] = []
        self.iterator_secs = 0.0
        self.items = 0

    def wrap_iterator(self, weights: Iterable) -> Iterator:
        it = iter(weights)
        clock = time.perf_counter
        while True:
            tic = clock()
            try:
                item = next(it)
            except StopIteration:
                self.iterator_secs += clock() - tic
                return
            self.iterator_secs += clock() - tic
            self.items += 1
            yield item

    def mark(self, phase: str) -> None:
        now = time.perf_counter()
        self.phases.append((phase, now - self.last))
        self.last = now

    def log(self, routed: int) -> None:
        stream = dict(self.phases).get("stream", 0.0)
        logger.info(
            "INF834 load profile [%s]: total %.1fs; %s; stream = iterator wait %.1fs + "
            "consumer main thread %.1fs; %d tensors from the iterator, %d routed",
            self.label,
            time.perf_counter() - self.start,
            ", ".join(f"{name} {secs:.1f}s" for name, secs in self.phases),
            self.iterator_secs,
            max(0.0, stream - self.iterator_secs),
            self.items,
            routed,
        )
