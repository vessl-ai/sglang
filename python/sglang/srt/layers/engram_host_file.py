"""File backing for the per-rank engram host table (SGLANG_DSV41_ENGRAM_HOST_TABLE_DIR).

CRIU copies every present page of an anonymous private mapping into its image, but
for a MAP_SHARED mapping of a regular file it records only the mapping and reopens
the file by path on restore. Keeping each rank's shard in a file on a tmpfs mount
therefore takes the table out of a process checkpoint, provided the same file, with
the same bytes, is at the same path when the process is restored.

Each table file gets a JSON manifest beside it (written after the weights are
loaded) with the model fingerprint and a sampled content digest, so that a restore
step can refuse a missing, resized, rewritten or stale table before CRIU maps it:

    python -m sglang.srt.layers.engram_host_file verify --dir DIR \
        [--model-path MODEL] [--expected DIR_WITH_MANIFEST_COPIES]

This module imports only the standard library, so the restore step can run it
without the engine and without importing the sglang package (which is slow), by
path: `python3 -I .../sglang/srt/layers/engram_host_file.py verify ...`.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import mmap
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

MANIFEST_VERSION = 1
MANIFEST_SUFFIX = ".json"
TABLE_SUFFIX = ".bin"
TABLE_MODE = 0o600
# A file on a disk filesystem would be written back and Linux refuses a long-term
# writable pin of such a mapping; memfd and anonymous shared memory live on the
# kernel's internal mount, which CRIU dumps. Only a mounted tmpfs fits.
ALLOWED_FSTYPES = ("tmpfs",)
# 1024 samples of 64 KiB: 64 MiB read per check, spread evenly over the file.
DIGEST_SAMPLES = 1024
DIGEST_CHUNK = 64 * 1024
# The part of a safetensors file that names, types and places every tensor.
_SAFETENSORS_HEADER_MAX = 64 * 1024 * 1024


@dataclass(frozen=True)
class Mount:
    mount_point: str
    fstype: str
    super_options: str

    @property
    def huge(self) -> str:
        """Value of the tmpfs huge= option ("" when absent, which means never)."""
        for opt in self.super_options.split(","):
            if opt.startswith("huge="):
                return opt[len("huge=") :]
        return ""


def _unescape(field: str) -> str:
    # mountinfo escapes space, tab, newline and backslash as \ooo octal.
    out, i = [], 0
    while i < len(field):
        octal = field[i + 1 : i + 4]
        if field[i] == "\\" and len(octal) == 3 and octal.isdigit():
            out.append(chr(int(octal, 8)))
            i += 4
        else:
            out.append(field[i])
            i += 1
    return "".join(out)


def mount_of(path: str, mountinfo: str = "/proc/self/mountinfo") -> Mount:
    """The mount holding `path`: the longest mount point that prefixes its real path."""
    real = os.path.realpath(path)
    best: Optional[Mount] = None
    with open(mountinfo) as f:
        for line in f:
            left, sep, right = line.partition(" - ")
            if not sep:
                continue
            fields = left.split()
            if len(fields) < 5:
                continue
            point = _unescape(fields[4])
            inside = real == point or real.startswith(point.rstrip("/") + "/")
            if not inside:
                continue
            rfields = right.split()
            fstype = rfields[0] if rfields else ""
            sopts = rfields[2] if len(rfields) > 2 else ""
            # Later lines win on an equal mount point: they are mounted on top.
            if best is None or len(point) >= len(best.mount_point):
                best = Mount(point, fstype, sopts)
    if best is None:
        raise ValueError(f"no mount holds {real} in {mountinfo}")
    return best


def check_dir(path: str, mountinfo: str = "/proc/self/mountinfo") -> Mount:
    """Refuse a directory that CRIU would copy or that cannot hold a pinned table."""
    if not os.path.isdir(path):
        raise ValueError(f"engram host table directory {path!r} does not exist")
    mount = mount_of(path, mountinfo)
    if mount.fstype not in ALLOWED_FSTYPES:
        raise ValueError(
            f"engram host table directory {path!r} is on {mount.fstype} "
            f"({mount.mount_point}); it must be on a tmpfs mount"
        )
    return mount


def table_file(directory: str, name: str, tp_rank: int, tp_size: int) -> str:
    return os.path.join(directory, f"{name}.tp{tp_rank}-of-{tp_size}{TABLE_SUFFIX}")


def manifest_path(table_path: str) -> str:
    return table_path + MANIFEST_SUFFIX


def create_table_file(path: str, nbytes: int) -> int:
    """A new, zero-filled file of nbytes; returns an O_RDWR descriptor.

    An existing file is unlinked first, never truncated: a process that still maps
    it (for example an engine restored from a checkpoint) keeps its pages, while
    truncating would make its next access fault with SIGBUS."""
    for stale in (path, manifest_path(path)):
        try:
            os.unlink(stale)
        except FileNotFoundError:
            pass
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, TABLE_MODE)
    try:
        # Explicit, so the process umask cannot change what CRIU will compare.
        os.fchmod(fd, TABLE_MODE)
        os.ftruncate(fd, nbytes)
        # Reserve the pages now: on a full tmpfs this fails here with ENOSPC,
        # where touching an unreserved page later would kill the rank with
        # SIGBUS. (Not on macOS, which has no posix_fallocate.)
        if hasattr(os, "posix_fallocate") and nbytes > 0:
            try:
                os.posix_fallocate(fd, 0, nbytes)
            except OSError as e:
                os.unlink(path)
                raise OSError(
                    e.errno,
                    f"cannot reserve {nbytes} bytes for {path} "
                    f"(is its tmpfs large enough?): {e.strerror}",
                ) from e
    except BaseException:
        os.close(fd)
        raise
    return fd


def _safetensors_header_sha256(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            return ""
        n = int.from_bytes(raw, "little")
        if n > _SAFETENSORS_HEADER_MAX:
            return ""
        return hashlib.sha256(raw + f.read(n)).hexdigest()


def model_fingerprint(model_path: str) -> dict:
    """Cheap identity of a checkpoint directory: config hash, and name, size, mtime
    and header hash of every safetensors file. Copying the model without keeping
    mtimes makes tables look stale, which only costs a cold start."""
    files = []
    for p in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        st = os.stat(p)
        files.append(
            {
                "name": os.path.basename(p),
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
                "header_sha256": _safetensors_header_sha256(p),
            }
        )
    config = os.path.join(model_path, "config.json")
    config_sha = ""
    if os.path.exists(config):
        with open(config, "rb") as f:
            config_sha = hashlib.sha256(f.read()).hexdigest()
    return {"config_sha256": config_sha, "safetensors": files}


def sample_digest(buf, nbytes: int) -> str:
    """sha256 over DIGEST_SAMPLES evenly spaced DIGEST_CHUNK-byte windows of buf
    (the whole buffer when it is smaller). Catches a missing, rewritten or
    different table; not a full integrity check."""
    h = hashlib.sha256()
    h.update(nbytes.to_bytes(8, "little"))
    if nbytes <= DIGEST_SAMPLES * DIGEST_CHUNK:
        h.update(buf[:nbytes])
        return h.hexdigest()
    stride = nbytes // DIGEST_SAMPLES
    for i in range(DIGEST_SAMPLES):
        start = i * stride
        h.update(buf[start : start + DIGEST_CHUNK])
    # The last bytes too, where the scale block of the table ends.
    h.update(buf[nbytes - DIGEST_CHUNK : nbytes])
    return h.hexdigest()


def file_sample_digest(path: str) -> str:
    size = os.path.getsize(path)
    if size == 0:
        return sample_digest(b"", 0)
    with open(path, "rb") as f, mmap.mmap(f.fileno(), size, prot=mmap.PROT_READ) as m:
        return sample_digest(m, size)


def write_manifest(table_path: str, fields: dict) -> str:
    """Write the manifest atomically beside the table file; returns its path."""
    body = {
        "version": MANIFEST_VERSION,
        "file": os.path.basename(table_path),
        "written_at": time.time(),
        "hostname": socket.gethostname(),
        **fields,
    }
    out = manifest_path(table_path)
    tmp = f"{out}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(body, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    return out


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def verify_dir(
    directory: str,
    model_path: Optional[str] = None,
    expected_dir: Optional[str] = None,
) -> tuple[int, list[str]]:
    """Check every table that a manifest names. Manifests come from expected_dir
    (copies kept with a checkpoint) when given, otherwise from directory itself.
    Returns (tables checked, problems); no problem means a restore may map them."""
    source = expected_dir or directory
    manifests = sorted(
        glob.glob(os.path.join(source, f"*{TABLE_SUFFIX}{MANIFEST_SUFFIX}"))
    )
    current_model = model_fingerprint(model_path) if model_path else None

    def check(mpath: str) -> list[str]:
        try:
            m = _load(mpath)
        except (OSError, ValueError) as e:
            return [f"{mpath}: unreadable manifest ({e})"]
        if m.get("version") != MANIFEST_VERSION:
            return [f"{mpath}: manifest version {m.get('version')!r}"]
        table = os.path.join(directory, m["file"])
        if not os.path.exists(table):
            return [f"{table}: missing"]
        st = os.stat(table)
        if st.st_size != m["nbytes"]:
            return [f"{table}: {st.st_size} bytes, expected {m['nbytes']}"]
        out = []
        # CRIU refuses a restore when the mode differs from the dumped one.
        if st.st_mode & 0o7777 != TABLE_MODE:
            out.append(f"{table}: mode {st.st_mode & 0o7777:o}, expected 600")
        if expected_dir is not None:
            local = os.path.join(directory, os.path.basename(mpath))
            if not os.path.exists(local):
                out.append(f"{local}: missing (the table was never finished)")
        if current_model is not None and m.get("model") != current_model:
            out.append(f"{table}: built from a different model than {model_path}")
        if file_sample_digest(table) != m.get("sample_digest"):
            out.append(f"{table}: content differs from {mpath}")
        return out

    # One thread per table: hashlib releases the GIL, so the 16 sampled reads of
    # an EP8 engine (64 MiB each) overlap instead of adding up before a restore.
    problems: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, min(16, len(manifests)))) as pool:
        for found in pool.map(check, manifests):
            problems.extend(found)
    if not manifests:
        problems.append(f"{source}: no table manifests")
    return len(manifests), problems


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sglang.srt.layers.engram_host_file")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="check table files before a restore maps them")
    v.add_argument("--dir", required=True, help="directory holding the table files")
    v.add_argument("--model-path", help="model directory the tables must come from")
    v.add_argument(
        "--expected",
        help="directory with the manifests saved with the checkpoint "
        "(default: the manifests beside the tables)",
    )
    args = ap.parse_args(argv)
    n, problems = verify_dir(args.dir, args.model_path, args.expected)
    for p in problems:
        print(f"engram host table: {p}", file=sys.stderr)
    if not problems:
        print(f"engram host table: {n} table(s) in {args.dir} verified")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
