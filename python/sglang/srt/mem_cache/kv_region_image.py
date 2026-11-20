"""The startup bytes of the memory-saver KV region, kept across a KV release.

`/checkpoint/prepare` releases the KV region (memory saver pause) and
`/checkpoint/restore` maps fresh physical pages in its place (resume). A fresh
engine starts from what the pool constructors and warmup wrote there: zeros
(`torch.zeros` pools, request-to-token table), sentinel and padding rows with
other values (for example the DeepSeek-V4 compressor's score -inf row), and the
few pages warmup used. Kernels read some of those startup bytes and no request
ever writes them, so they must survive the release.

Rather than re-running every constructor, `KVRegionImage` captures the region
once, before the first release, as the chunks that are not all zero (startup
memory is almost all zeros), and puts it back after a resume: zero the region,
then copy those chunks in. The region then holds the same bytes as when it was
captured, for every pool in it, without a list of pools.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

# (address, size in bytes, device index) of one allocation of the region.
Segment = Tuple[int, int, int]

CHUNK_BYTES = 64 << 10  # divides the 2 MiB CUDA VMM granularity
SLICE_BYTES = 128 << 20  # bounds the temporary memory of one scan step


def memory_saver_segments(tag: str) -> List[Segment]:
    """The allocations torch_memory_saver made under `tag`: every region()
    allocates from a torch MemPool per (tag, ...) key, one segment each."""
    from torch_memory_saver import torch_memory_saver as tms

    impl = getattr(tms, "_impl", None)
    if impl is None:
        return []
    segments: List[Segment] = []
    for key, pool in list(impl._mem_pools.items()):
        if (key[0] if isinstance(key, tuple) else key) != tag:
            continue
        try:
            snapshot = pool.snapshot(include_traces=False)
        except TypeError:
            snapshot = pool.snapshot()
        for seg in snapshot:
            segments.append(
                (int(seg["address"]), int(seg["total_size"]), int(seg["device"]))
            )
    return sorted(segments)


def cuda_view(address: int, size: int, device: int) -> torch.Tensor:
    """A uint8 tensor over device memory this tensor does not own."""
    dev = torch.device("cuda", device)
    storage = torch._C._construct_storage_from_data_pointer(address, dev, size)
    return torch.empty(0, dtype=torch.uint8, device=dev).set_(storage, 0, (size,), (1,))


@dataclass
class _Piece:
    segment: Segment
    offset: int  # start of the slice within the segment
    chunks: torch.Tensor  # int64 chunk indices within the slice (host)
    data: torch.Tensor  # the chunks' bytes (host, uint8, [n, CHUNK_BYTES])


@dataclass
class KVRegionImage:
    segments: List[Segment]
    pieces: List[_Piece] = field(default_factory=list)
    tails: List[Tuple[Segment, torch.Tensor]] = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        return sum(p.data.numel() for p in self.pieces) + sum(
            t.numel() for _, t in self.tails
        )

    @classmethod
    def capture(
        cls,
        segments: Sequence[Segment],
        view: Callable[[int, int, int], torch.Tensor] = cuda_view,
        max_bytes: Optional[int] = None,
    ) -> KVRegionImage:
        image = cls(segments=list(segments))
        for seg in image.segments:
            t = view(*seg)
            full = seg[1] - seg[1] % CHUNK_BYTES
            for off in range(0, full, SLICE_BYTES):
                part = t[off : min(off + SLICE_BYTES, full)].view(-1, CHUNK_BYTES)
                # Compare 8 bytes at a time: the temporary is 1/8 of the slice.
                nonzero = part.view(torch.int64).ne(0).any(dim=1)
                idx = nonzero.nonzero().squeeze(1)
                if idx.numel():
                    image.pieces.append(
                        _Piece(seg, off, idx.cpu(), part.index_select(0, idx).cpu())
                    )
            if full < seg[1] and bool(t[full:].ne(0).any()):
                image.tails.append((seg, t[full:].cpu().clone()))
            if max_bytes is not None and image.nbytes > max_bytes:
                raise RuntimeError(
                    f"KV region image exceeds {max_bytes} bytes: the region is "
                    "not mostly zero (capture it before any request is served)"
                )
        return image

    def apply(
        self,
        segments: Sequence[Segment],
        view: Callable[[int, int, int], torch.Tensor] = cuda_view,
    ) -> None:
        if sorted(segments) != sorted(self.segments):
            raise RuntimeError(
                "The KV region's allocations differ from the captured ones "
                f"({len(segments)} vs {len(self.segments)} segments)"
            )
        views = {seg: view(*seg) for seg in self.segments}
        for t in views.values():
            t.zero_()
        for p in self.pieces:
            t = views[p.segment]
            full = p.segment[1] - p.segment[1] % CHUNK_BYTES
            part = t[p.offset : min(p.offset + SLICE_BYTES, full)].view(-1, CHUNK_BYTES)
            part.index_copy_(
                0, p.chunks.to(part.device), p.data.to(part.device, non_blocking=True)
            )
        for seg, tail in self.tails:
            t = views[seg]
            t[seg[1] - tail.numel() :].copy_(tail.to(t.device))


def capture_kv_region(tag: str, max_bytes: int) -> KVRegionImage:
    tic = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    segments = memory_saver_segments(tag)
    if not segments:
        raise RuntimeError(f"No memory saver allocations under tag {tag!r}")
    image = KVRegionImage.capture(segments, max_bytes=max_bytes)
    logger.info(
        f"KV region image captured: segments={len(segments)} "
        f"region_bytes={sum(s[1] for s in segments)} image_bytes={image.nbytes} "
        f"seconds={time.perf_counter() - tic:.3f}"
    )
    return image


def apply_kv_region(tag: str, image: KVRegionImage) -> None:
    image.apply(memory_saver_segments(tag))
    torch.cuda.synchronize()
