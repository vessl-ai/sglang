"""Unit tests for the INF-834 relocation of NCCL symmetric kernels in a Foundry archive."""

import ctypes
import os
import struct
import tempfile
import unittest

from sglang.srt.utils import inf834_symk_reloc as reloc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

WINDOW_TABLE, RESOURCE_WIN, ABORT_FLAG, WINDOW = (
    0x9220000000,
    0x922000D000,
    0x791D6FA00000,
    0x922000D160,
)
LSA_BASE = 0x612400000000
NEW = {
    WINDOW_TABLE: 0x9240000000,
    RESOURCE_WIN: 0x924000D000,
    ABORT_FLAG: 0x7A0000001000,
    WINDOW: 0x924000D160,
}


def _args(
    table, resource, abort, window, handle=7, n_elts=0x5000, garbage=0x792262C84F80
) -> bytes:
    # Shared (kcomm-like) part: magic, windowTable, resourceWindow, lsaFlatBase, abortFlag, handle.
    # Then one work: an uninitialised word, nElts, inputWin, outputWin.
    return struct.pack(
        "<10Q",
        0x59DFCAFEBEEF,
        table,
        resource,
        LSA_BASE,
        abort,
        handle,
        garbage,
        n_elts,
        window,
        window,
    )


def _probe(table, resource, abort, window, handle=7, garbage=0x792262C84F80) -> dict:
    probe = {}
    for op, n_elts in (("all_gather", 0x5000), ("reduce_scatter", 0xA000)):
        blob = _args(
            table,
            resource,
            abort,
            window,
            handle=handle,
            n_elts=n_elts,
            garbage=garbage + n_elts,
        )
        probe[f"tp/{op}/small/0"] = {
            "kernel": f"ncclSymkDevKernel_{op}",
            "args": blob.hex(),
        }
    return probe


def _cugraph(nodes: list[tuple[str, bytes]]) -> bytes:
    """Minimal .cugraph: string table, node table, argument buffers."""
    strings = b""
    args = b""
    table = b""
    for i, (name, blob) in enumerate(nodes):
        body = bytearray(160)
        struct.pack_into("<II", body, 36, len(strings), len(name))
        struct.pack_into("<IIII", body, 120, 0, 0, len(args), len(blob))
        table += struct.pack("<IB3x", i, 0) + bytes(body)
        strings += name.encode()
        args += blob
    sections = [(0, strings), (1, table), (5, args)]
    header = struct.pack(
        "<8sIIIIIII", b"CUGRAPH\0", 3, 0, len(nodes), 0, 0, len(sections), len(nodes)
    ) + bytes(28)
    directory = b""
    offset = 64 + 24 * len(sections)
    for stype, payload in sections:
        directory += struct.pack("<IIQQ", stype, 0, offset, len(payload))
        offset += len(payload)
    return header + directory + b"".join(payload for _, payload in sections)


class TestSymkRelocation(CustomTestCase):
    def test_moved_words_finds_addresses_and_shared_state(self):
        save = _probe(WINDOW_TABLE, RESOURCE_WIN, ABORT_FLAG, WINDOW)
        load = _probe(
            *[NEW[x] for x in (WINDOW_TABLE, RESOURCE_WIN, ABORT_FLAG, WINDOW)],
            handle=9,
            garbage=0x7A2222222222,
        )
        moved, positional, problems = reloc.moved_words(save, load)
        for old, new in NEW.items():
            self.assertEqual(moved[old], new)
        self.assertNotIn(LSA_BASE, moved)
        # The handle (a small integer in the shared part) moves by position.
        self.assertEqual(positional, {(40, 7): 9})
        self.assertTrue(all(p.startswith("note:") for p in problems), problems)

    def test_probes_that_do_not_line_up_are_structural(self):
        save = _probe(WINDOW_TABLE, RESOURCE_WIN, ABORT_FLAG, WINDOW)
        load = dict(save)
        load.pop("tp/all_gather/small/0")
        _, _, problems = reloc.moved_words(save, load)
        self.assertTrue(any(not p.startswith("note:") for p in problems))

    def test_relocate_rewrites_only_symmetric_kernel_arguments(self):
        blob = _args(WINDOW_TABLE, RESOURCE_WIN, ABORT_FLAG, WINDOW)
        data = _cugraph(
            [("ncclSymkDevKernel_AllGather_STMC", blob), ("other_kernel", blob)]
        )
        patched, nodes, words = reloc.relocate_cugraph(
            data, NEW, "ncclSymk", {(40, 7): 9}
        )
        self.assertEqual((nodes, len(patched)), (1, len(data)))
        self.assertEqual(words, 6)  # table, resource, abort, handle, window (x2)
        first = bytes(patched)[-2 * len(blob) : -len(blob)]
        self.assertEqual(
            first,
            _args(
                *[NEW[x] for x in (WINDOW_TABLE, RESOURCE_WIN, ABORT_FLAG, WINDOW)],
                handle=9,
            ),
        )
        self.assertEqual(bytes(patched)[-len(blob) :], blob)
        again, _, words_again = reloc.relocate_cugraph(
            bytes(patched), NEW, "ncclSymk", {(40, 7): 9}
        )
        self.assertIsNone(again)
        self.assertEqual(words_again, 0)

    def test_shadow_workspace_links_everything_else(self):
        blob = _args(WINDOW_TABLE, RESOURCE_WIN, ABORT_FLAG, WINDOW)
        with tempfile.TemporaryDirectory() as root:
            rank = os.path.join(root, "rank_0")
            os.makedirs(rank)
            with open(os.path.join(rank, "graph_0.cugraph"), "wb") as f:
                f.write(_cugraph([("ncclSymkDevKernel_AllGather_STMC", blob)]))
            with open(os.path.join(rank, "graph_1.cugraph"), "wb") as f:
                f.write(_cugraph([("other_kernel", blob)]))
            with open(os.path.join(rank, "region_layout.json"), "w") as f:
                f.write("{}")
            os.environ["SGLANG_OPT_INF834_SYMK_RELOC_DIR"] = os.path.join(
                root, "shadow"
            )
            try:
                shadow, stats = reloc.build_shadow_workspace(rank, NEW, "ncclSymk")
            finally:
                del os.environ["SGLANG_OPT_INF834_SYMK_RELOC_DIR"]
            self.assertEqual(
                (stats["files"], stats["files_patched"], stats["nodes"]), (2, 1, 1)
            )
            self.assertFalse(os.path.islink(os.path.join(shadow, "graph_0.cugraph")))
            self.assertTrue(os.path.islink(os.path.join(shadow, "graph_1.cugraph")))
            self.assertTrue(os.path.islink(os.path.join(shadow, "region_layout.json")))
            reloc._cleanup_shadow_dirs()
            self.assertFalse(os.path.exists(shadow))

    def test_read_extra_buffer(self):
        payload = bytes(range(200))
        buf = ctypes.create_string_buffer(payload, len(payload))
        size = ctypes.c_size_t(len(payload))
        arr = (ctypes.c_void_p * 5)(
            1, ctypes.addressof(buf), 2, ctypes.addressof(size), 0
        )
        self.assertEqual(reloc._read_extra_buffer(ctypes.addressof(arr)), payload)
        self.assertIsNone(reloc._read_extra_buffer(0))


if __name__ == "__main__":
    unittest.main()
