"""SGLANG_DSV41_ENGRAM_HOST_TABLE_DIR: the per-rank engram host table as a MAP_SHARED
file on tmpfs (so CRIU records the mapping, not the pages), its directory check, the
replace-not-truncate rule and the manifest that a restore step verifies. Runs on CPU;
the _HostTable case stubs cudaHostRegister and needs torch."""

import importlib.util
import json
import mmap
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.layers import engram_host_file as H
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MOUNTINFO = """\
22 1 8:3 / / rw,relatime shared:1 - ext4 /dev/sda3 rw
30 22 0:25 / /dev/shm rw,nosuid,nodev shared:4 - tmpfs tmpfs rw,size=1048576k
41 22 253:0 / /data rw,relatime shared:9 - xfs /dev/mapper/vg-data rw,sunit=512
52 41 0:60 / /data/engram rw,relatime shared:12 - tmpfs tmpfs rw,size=268435456k,huge=within_size,mode=700
53 22 0:61 / /mnt/with\\040space rw - tmpfs tmpfs rw
"""


class MountTest(unittest.TestCase):
    def setUp(self):
        fd, self.info = tempfile.mkstemp()
        with os.fdopen(fd, "w") as f:
            f.write(MOUNTINFO)
        self.addCleanup(os.unlink, self.info)

    def test_longest_mount_point_wins(self):
        m = H.mount_of("/data/engram/x.bin", self.info)
        self.assertEqual(
            (m.mount_point, m.fstype, m.huge),
            ("/data/engram", "tmpfs", "within_size"),
        )
        self.assertEqual(H.mount_of("/data/models/a", self.info).fstype, "xfs")
        self.assertEqual(H.mount_of("/dataX", self.info).mount_point, "/")

    def test_escaped_mount_point_and_absent_huge(self):
        m = H.mount_of("/mnt/with space/t", self.info)
        self.assertEqual((m.mount_point, m.huge), ("/mnt/with space", ""))

    def test_check_dir_refuses_disk_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(
                H, "mount_of", return_value=H.Mount("/data", "xfs", "")
            ):
                with self.assertRaisesRegex(ValueError, "must be on a tmpfs"):
                    H.check_dir(d)
            with mock.patch.object(
                H, "mount_of", return_value=H.Mount("/x", "tmpfs", "huge=advise")
            ):
                self.assertEqual(H.check_dir(d).huge, "advise")
        with self.assertRaisesRegex(ValueError, "does not exist"):
            H.check_dir("/nonexistent/engram")


class TableFileTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def test_name_and_zero_fill(self):
        path = H.table_file(self.dir, "sglang_engram_3", 2, 8)
        self.assertTrue(path.endswith("sglang_engram_3.tp2-of-8.bin"))
        fd = H.create_table_file(path, 3 * mmap.PAGESIZE + 5)
        os.close(fd)
        with open(path, "rb") as f:
            self.assertEqual(f.read(), bytes(3 * mmap.PAGESIZE + 5))

    def test_replace_keeps_a_live_mapping_intact(self):
        path = H.table_file(self.dir, "t", 0, 1)
        fd = H.create_table_file(path, mmap.PAGESIZE)
        old = mmap.mmap(
            fd,
            mmap.PAGESIZE,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        os.close(fd)
        old[:4] = b"OLD!"
        H.write_manifest(path, {"nbytes": mmap.PAGESIZE, "sample_digest": "x"})
        # A second engine on the same directory replaces the file; the first one,
        # still mapping the old inode, must not see zeros or take a SIGBUS.
        fd = H.create_table_file(path, 2 * mmap.PAGESIZE)
        os.close(fd)
        self.assertEqual(old[:4], b"OLD!")
        self.assertEqual(os.path.getsize(path), 2 * mmap.PAGESIZE)
        self.assertFalse(
            os.path.exists(H.manifest_path(path)), "a stale manifest must go"
        )
        old.close()


class ManifestTest(unittest.TestCase):
    NBYTES = H.DIGEST_SAMPLES * H.DIGEST_CHUNK * 2 + 12345  # takes the sampled path

    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.dir = os.path.join(self.root.name, "tables")
        self.model = os.path.join(self.root.name, "model")
        os.makedirs(self.dir)
        os.makedirs(self.model)
        header = json.dumps(
            {
                "engram.weight": {
                    "dtype": "F8_E4M3",
                    "shape": [4, 4],
                    "data_offsets": [0, 16],
                }
            }
        ).encode()
        with open(os.path.join(self.model, "model-00001.safetensors"), "wb") as f:
            f.write(len(header).to_bytes(8, "little") + header + bytes(16))
        with open(os.path.join(self.model, "config.json"), "w") as f:
            f.write('{"engram_layer_ids": [3, 7]}')
        self.path = H.table_file(self.dir, "sglang_engram_3", 0, 8)
        fd = H.create_table_file(self.path, self.NBYTES)
        with mmap.mmap(
            fd,
            self.NBYTES,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        ) as m:
            m[:] = bytes((i * 7) & 0xFF for i in range(256)) * (
                self.NBYTES // 256
            ) + bytes(self.NBYTES % 256)
            digest = H.sample_digest(m, self.NBYTES)
        os.close(fd)
        H.write_manifest(
            self.path,
            {
                "nbytes": self.NBYTES,
                "model": H.model_fingerprint(self.model),
                "sample_digest": digest,
            },
        )

    def _poke(self, offset: int):
        with open(self.path, "r+b") as f:
            f.seek(offset)
            b = f.read(1)
            f.seek(offset)
            f.write(bytes([b[0] ^ 0xFF]))

    def test_fresh_table_verifies(self):
        self.assertEqual(H.verify_dir(self.dir, self.model), (1, []))
        self.assertEqual(
            H.file_sample_digest(self.path),
            json.load(open(H.manifest_path(self.path)))["sample_digest"],
        )

    def test_sampled_bytes_detect_rewrites(self):
        self._poke(0)
        n, problems = H.verify_dir(self.dir)
        self.assertEqual(n, 1)
        self.assertRegex(problems[0], "content differs")

    def test_tail_is_sampled(self):
        self._poke(self.NBYTES - 1)
        self.assertRegex(H.verify_dir(self.dir)[1][0], "content differs")

    def test_size_missing_and_model_change(self):
        with open(os.path.join(self.model, "config.json"), "w") as f:
            f.write('{"engram_layer_ids": [3, 8]}')
        self.assertRegex(H.verify_dir(self.dir, self.model)[1][0], "different model")
        self.assertEqual(os.stat(self.path).st_mode & 0o7777, 0o600)
        os.chmod(self.path, 0o644)  # CRIU would refuse this file ("bad mode")
        self.assertRegex(H.verify_dir(self.dir)[1][0], "mode 644, expected 600")
        os.truncate(self.path, self.NBYTES - 1)
        self.assertRegex(H.verify_dir(self.dir)[1][0], "bytes, expected")
        os.unlink(self.path)
        self.assertRegex(H.verify_dir(self.dir)[1][0], "missing")

    def test_expected_manifests_from_the_checkpoint(self):
        saved = os.path.join(self.root.name, "ckpt")
        os.makedirs(saved)
        with (
            open(H.manifest_path(self.path)) as src,
            open(
                os.path.join(saved, os.path.basename(H.manifest_path(self.path))), "w"
            ) as dst,
        ):
            dst.write(src.read())
        self.assertEqual(H.verify_dir(self.dir, None, saved), (1, []))
        # A later cold start on this node rewrote the table with other bytes.
        fd = H.create_table_file(self.path, self.NBYTES)
        os.close(fd)
        H.write_manifest(
            self.path,
            {"nbytes": self.NBYTES, "sample_digest": H.file_sample_digest(self.path)},
        )
        self.assertEqual(H.verify_dir(self.dir)[1], [], "self-consistent on its own")
        self.assertRegex(H.verify_dir(self.dir, None, saved)[1][0], "content differs")
        # A cold start that died before finish_load leaves no manifest beside the table.
        os.unlink(H.manifest_path(self.path))
        self.assertTrue(
            any("never finished" in p for p in H.verify_dir(self.dir, None, saved)[1])
        )

    def test_cli(self):
        self.assertEqual(
            H.main(["verify", "--dir", self.dir, "--model-path", self.model]), 0
        )
        empty = os.path.join(self.root.name, "empty")
        os.makedirs(empty)
        self.assertEqual(H.main(["verify", "--dir", empty]), 1)


@unittest.skipUnless(sys.platform.startswith("linux"), "reads /proc/self/mountinfo")
class LinuxTmpfsTest(unittest.TestCase):
    """The real kernel: /dev/shm passes the check, the root filesystem does not, and
    a table file on tmpfs takes MADV_HUGEPAGE as a MAP_SHARED mapping."""

    def test_real_mounts(self):
        if H.mount_of("/dev/shm").fstype != "tmpfs":
            self.skipTest("/dev/shm is not tmpfs here")
        if H.mount_of("/").fstype == "tmpfs":
            self.skipTest("the root filesystem is tmpfs here")
        with self.assertRaisesRegex(ValueError, "must be on a tmpfs"):
            H.check_dir("/")
        with tempfile.TemporaryDirectory(dir="/dev/shm") as d:
            self.assertEqual(H.check_dir(d).fstype, "tmpfs")
            path = H.table_file(d, "t", 0, 1)
            n = 4 * 1024 * 1024
            fd = H.create_table_file(path, n)
            m = mmap.mmap(
                fd, n, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
            )
            os.close(fd)
            m.madvise(mmap.MADV_HUGEPAGE)
            m[n - 3 :] = b"end"
            m.close()
            with open(path, "rb") as f:
                f.seek(n - 3)
                self.assertEqual(f.read(), b"end")


@unittest.skipIf(importlib.util.find_spec("torch") is None, "needs torch")
class HostTableFileTest(unittest.TestCase):
    """_HostTable with file_dir: a MAP_SHARED file (not anonymous memory), loader
    writes reach the file, finish_load writes a manifest that verifies."""

    def setUp(self):
        from sglang.srt.layers import engram as E

        self.E = E
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.registered = []
        cudart = SimpleNamespace(
            cudaHostRegister=lambda ptr, n, flags: (
                self.registered.append((ptr, n, flags)) or 0
            )
        )
        for p in (
            mock.patch.object(E.torch.cuda, "cudart", return_value=cudart),
            mock.patch.object(E, "_drop_page_cache_once"),
            mock.patch.object(E, "drop_checkpoint_page_cache"),
            mock.patch.object(E._HostTable, "_advise_hugepage"),
            mock.patch.object(E._HostTable, "_collapse"),
            mock.patch.object(E, "_model_path", return_value=None),
            mock.patch.object(
                E.engram_host_file,
                "check_dir",
                return_value=H.Mount("/t", "tmpfs", "huge=within_size"),
            ),
        ):
            p.start()
            self.addCleanup(p.stop)
        self.group = SimpleNamespace(rank_in_group=3, world_size=8)

    def test_file_backed_shard(self):
        nbytes = 5 * mmap.PAGESIZE + 3
        t = self.E._HostTable(
            "per_rank",
            nbytes,
            "sglang_engram_7",
            self.group,
            file_dir=self.dir.name,
            describe={"layer_id": 7},
        )
        self.assertEqual(
            t.path, os.path.join(self.dir.name, "sglang_engram_7.tp3-of-8.bin")
        )
        self.assertEqual(self.registered, [(t.bytes.data_ptr(), nbytes, 0)])
        t.bytes[:4] = self.E.torch.tensor([1, 2, 3, 4], dtype=self.E.torch.uint8)
        t.bytes[-1] = 9
        t.dirty = True
        t.finish_load("layer 7")
        with open(t.path, "rb") as f:
            data = f.read()
        self.assertEqual(
            (data[:4], data[-1], len(data)), (b"\x01\x02\x03\x04", 9, nbytes)
        )
        m = json.load(open(H.manifest_path(t.path)))
        self.assertEqual(
            (m["layer_id"], m["tp_rank"], m["tp_size"], m["nbytes"]), (7, 3, 8, nbytes)
        )
        self.assertEqual(H.verify_dir(self.dir.name), (1, []))

    @unittest.skipUnless(sys.platform == "linux", "MAP_FIXED_NOREPLACE is Linux only")
    def test_file_backed_shard_at_fixed_address_under_graph_persistence(self):
        """INF-834 CUDA graph persistence maps the table at a fixed address; with
        file_dir that mapping is the table file, and the manifest digest reads it
        through self.bytes (self.mm is then a ctypes array)."""
        base = 0x5F0000000000
        nbytes = 3 * mmap.PAGESIZE + 5
        with (
            mock.patch.object(self.E, "_graphs_persisted", return_value=True),
            mock.patch.object(self.E, "_madvise_hugepage"),
            mock.patch.object(self.E, "_fixed_addr_cursor", base),
        ):
            t = self.E._HostTable(
                "per_rank", nbytes, "sglang_engram_9", self.group, file_dir=self.dir.name
            )
        self.assertEqual(t.bytes.data_ptr(), base)
        self.assertTrue(os.path.exists(t.path))
        # device="cpu": other suites in this directory leave the default device at cuda.
        t.bytes[-2:] = self.E.torch.tensor(
            [7, 8], dtype=self.E.torch.uint8, device="cpu"
        )
        t.dirty = True
        t.finish_load("layer 9")
        with open(t.path, "rb") as f:
            self.assertEqual(f.read()[-2:], b"\x07\x08")
        self.assertEqual(H.verify_dir(self.dir.name), (1, []))

    def test_deferred_registration_keeps_the_read_only_flag(self):
        """A table whose pre-fault and registration INF-834 defers to the weight
        snapshot is registered later with the same flags as at init."""
        with (
            mock.patch.object(self.E, "_inf834_defers_prefault", return_value=True),
            mock.patch.dict(
                os.environ, {"SGLANG_DSV41_ENGRAM_HOST_TABLE_REGISTER_READONLY": "1"}
            ),
        ):
            t = self.E._HostTable(
                "per_rank",
                2 * mmap.PAGESIZE,
                "sglang_engram_2",
                self.group,
                file_dir=self.dir.name,
            )
            self.assertTrue(t.deferred)
            self.assertEqual(self.registered, [])
            t.complete(prefault=True)
        self.assertEqual(
            self.registered, [(t.bytes.data_ptr(), 2 * mmap.PAGESIZE, 0x08)]
        )

    def test_default_stays_anonymous_and_shared_refuses_a_dir(self):
        t = self.E._HostTable("per_rank", mmap.PAGESIZE, "sglang_engram_1", self.group)
        self.assertIsNone(t.path)
        self.assertEqual(os.listdir(self.dir.name), [])
        with self.assertRaisesRegex(ValueError, "LAYOUT=per_rank"):
            self.E._HostTable(
                "shared", mmap.PAGESIZE, "x", self.group, file_dir=self.dir.name
            )


if __name__ == "__main__":
    unittest.main()
