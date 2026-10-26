import contextlib
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class FakeReplicateConfig:
    def __init__(self):
        self.group_ids = None
        self.preferred_segments = []


def _fake_store_class():
    class FakeMooncakeDistributedStore:
        instances = []

        def __init__(self):
            self.batch_put_calls = []
            self.existing_keys = set()
            self.objects = {}
            self.short_results = False
            type(self).instances.append(self)

        def setup(self, *args, **kwargs):
            return 0

        def register_buffer(self, *args, **kwargs):
            return 0

        def put(self, key, value, *args):
            self.objects[key] = value
            return 0

        def is_exist(self, key):
            return 1 if key in self.objects else 0

        def get(self, key):
            return self.objects.get(key)

        def batch_is_exist(self, keys):
            return [1 if key in self.existing_keys else 0 for key in keys]

        def _record(self, method, keys, ptrs, sizes, args):
            self.batch_put_calls.append(
                {
                    "method": method,
                    "keys": list(keys),
                    "ptrs": list(ptrs),
                    "sizes": list(sizes),
                    "args": args,
                    "preferred": (list(args[0].preferred_segments) if args else None),
                    "group_ids": (
                        list(args[0].group_ids)
                        if args and args[0].group_ids is not None
                        else None
                    ),
                }
            )
            self.existing_keys.update(keys)
            if self.short_results:
                return [0] * (len(keys) - 1)
            return [0] * len(keys)

        def batch_put_from(self, keys, ptrs, sizes, *args):
            return self._record("batch_put_from", keys, ptrs, sizes, args)

        def batch_put_from_multi_buffers(self, keys, ptrs, sizes, *args):
            return self._record("batch_put_from_multi_buffers", keys, ptrs, sizes, args)

    return FakeMooncakeDistributedStore


def _fake_modules(fake_store_cls):
    mooncake = types.ModuleType("mooncake")
    mooncake_store = types.ModuleType("mooncake.store")
    mooncake_store.MooncakeDistributedStore = fake_store_cls
    mooncake_store.ReplicateConfig = FakeReplicateConfig

    pool_host = types.ModuleType("sglang.srt.mem_cache.pool_host")

    class HostKVCache:
        pass

    class HostTensorAllocator:
        pass

    pool_host.HostKVCache = HostKVCache
    pool_host.HostTensorAllocator = HostTensorAllocator
    pool_host_mla = types.ModuleType("sglang.srt.mem_cache.pool_host.mla")

    class MLATokenToKVPoolHost:
        pass

    pool_host_mla.MLATokenToKVPoolHost = MLATokenToKVPoolHost
    return {
        "mooncake": mooncake,
        "mooncake.store": mooncake_store,
        "sglang.srt.mem_cache.pool_host": pool_host,
        "sglang.srt.mem_cache.pool_host.mla": pool_host_mla,
    }


class FakeHostKVCache:
    def __init__(self, objects_per_page):
        self.objects_per_page = objects_per_page
        self.kv_buffer = torch.empty((1024,), dtype=torch.uint8)
        self.layout = "page_first"
        self.page_size = 1

    def get_ksize_per_token(self):
        return 1

    def get_page_buffer_meta(self, indices):
        ptrs, sizes = [], []
        for page_idx in range(len(indices)):
            for object_idx in range(self.objects_per_page):
                ptrs.append(1000 + page_idx * 100 + object_idx)
                sizes.append(8)
        return ptrs, sizes

    def get_split_heads_page_buffer_meta(self, indices, split_factor):
        ptrs, sizes = [], []
        for page_idx in range(len(indices)):
            for object_idx in range(2 * split_factor):
                ptrs.append(2000 + page_idx * 100 + object_idx)
                sizes.append(8)
        return ptrs, sizes


class FakeMambaPool:
    """Two objects per page (temporal + one conv state), like a hybrid SSM."""

    page_size = 1
    temporal_state_elem_size = 1

    def __init__(self):
        self.conv_buffer = [torch.empty((64,), dtype=torch.uint8)]

    def get_hybrid_pool_buffer(self):
        return list(self.conv_buffer)

    def get_page_buffer_meta(self, indices):
        ptrs, sizes = [], []
        for i in range(len(indices)):
            ptrs.extend([5000 + i * 10, 5001 + i * 10])
            sizes.extend([8, 8])
        return ptrs, sizes


class FakeMultiBufferPool:
    page_size = 1

    def __init__(self):
        self.buffers = [torch.empty((128,), dtype=torch.uint8)] * 2

    def get_hybrid_pool_buffer(self):
        return self.buffers

    def get_page_buffer_meta(self, indices):
        ptrs, sizes = [], []
        for i in range(len(indices)):
            ptrs.extend([4000 + i * 10, 4001 + i * 10])
            sizes.extend([8, 16])
        return ptrs, sizes


def _make_store(
    *,
    placement=True,
    anchor_pages=None,
    enable_group_semantics=False,
    extra_backend_tag=None,
    model_name=None,
    is_mla_model=False,
    should_split_heads=False,
    tp_rank=0,
    tp_size=1,
    tp_lcm_size=None,
):
    fake_store_cls = _fake_store_class()
    extra_config = {
        "master_server_address": "127.0.0.1:50051",
        "check_server": False,
        "global_segment_size": 1024 * 1024,
        "enable_group_semantics": enable_group_semantics,
    }
    if extra_backend_tag is not None:
        extra_config["extra_backend_tag"] = extra_backend_tag
    cfg = HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=is_mla_model,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=model_name,
        tp_lcm_size=tp_lcm_size,
        should_split_heads=should_split_heads,
        extra_config=extra_config,
    )
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.dict("sys.modules", _fake_modules(fake_store_cls)))
        if placement:
            stack.enter_context(
                envs.SGLANG_HICACHE_MOONCAKE_PLACEMENT_KEY.override(True)
            )
        if anchor_pages is not None:
            stack.enter_context(
                envs.SGLANG_HICACHE_MOONCAKE_PLACEMENT_ANCHOR_PAGES.override(
                    anchor_pages
                )
            )
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStore,
        )

        store = MooncakeStore(cfg)
    return store, fake_store_cls.instances[-1]


def _chain(n):
    return [f"h{i}" for i in range(n)]


class TestChainAnchorKeys(CustomTestCase):
    def setUp(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            chain_anchor_keys,
        )

        self.anchors = chain_anchor_keys

    def test_head_pages_use_the_root_and_later_pages_the_anchor_page(self):
        chain = _chain(70)
        # Batch of pages 62..65 after a 62-page prefix; anchor page 64 is in
        # the batch itself.
        self.assertEqual(
            self.anchors(chain[:62], chain[62:66], 64),
            ["h0", "h0", "h64", "h64"],
        )
        # A later batch: the anchor page is in the prefix.
        self.assertEqual(self.anchors(chain[:66], chain[66:68], 64), ["h64", "h64"])

    def test_chain_start_and_root_only(self):
        chain = _chain(8)
        self.assertEqual(self.anchors([], chain[:3], 2), ["h0", "h0", "h2"])
        self.assertEqual(self.anchors(chain[:5], chain[5:8], 0), ["h0"] * 3)
        self.assertEqual(self.anchors(chain[:5], chain[5:8], -1), ["h0"] * 3)


class TestMooncakePlacementKey(CustomTestCase):
    def test_flag_off_keeps_the_three_arg_put(self):
        store, fake = _make_store(placement=False)
        store.register_mem_pool_host(FakeHostKVCache(objects_per_page=2))
        chain = _chain(4)

        result = store.batch_set_v1(
            chain[2:4],
            torch.tensor([0, 1]),
            HiCacheStorageExtraInfo(prefix_keys=chain[:2]),
        )

        self.assertEqual(result, [True, True])
        self.assertEqual(len(fake.batch_put_calls), 1)
        self.assertEqual(fake.batch_put_calls[0]["args"], ())

    def test_batch_across_the_anchor_is_split_in_two_puts(self):
        store, fake = _make_store(extra_backend_tag="tag", model_name="org/m")
        store.register_mem_pool_host(FakeHostKVCache(objects_per_page=2))
        chain = _chain(66)

        result = store.batch_set_v1(
            chain[62:66],
            torch.tensor([0, 1, 2, 3]),
            HiCacheStorageExtraInfo(prefix_keys=chain[:62]),
        )

        self.assertEqual(result, [True] * 4)
        calls = fake.batch_put_calls
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["preferred"], ["placement-key:tag_org-m_h0"])
        self.assertEqual(
            calls[0]["keys"],
            [
                "tag_org-m_h62_0_k",
                "tag_org-m_h62_0_v",
                "tag_org-m_h63_0_k",
                "tag_org-m_h63_0_v",
            ],
        )
        self.assertEqual(calls[0]["ptrs"], [1000, 1001, 1100, 1101])
        self.assertEqual(calls[1]["preferred"], ["placement-key:tag_org-m_h64"])
        self.assertEqual(calls[1]["ptrs"], [1200, 1201, 1300, 1301])

    def test_every_tp_rank_uses_the_same_key_for_a_page(self):
        chain = _chain(3)
        preferred = []
        for rank in (0, 1):
            store, fake = _make_store(
                should_split_heads=True, tp_rank=rank, tp_size=2, tp_lcm_size=4
            )
            store.register_mem_pool_host(FakeHostKVCache(objects_per_page=4))
            store.batch_set_v1(
                chain[2:3],
                torch.tensor([0]),
                HiCacheStorageExtraInfo(prefix_keys=chain[:2]),
            )
            (call,) = fake.batch_put_calls
            self.assertEqual(len(call["keys"]), 4)
            preferred.append(call["preferred"])
        self.assertEqual(preferred, [["placement-key:h0"]] * 2)

    def test_mla_and_root_only_anchor(self):
        store, fake = _make_store(is_mla_model=True, anchor_pages=0)
        store.register_mem_pool_host(FakeHostKVCache(objects_per_page=1))
        chain = _chain(200)

        store.batch_set_v1(
            chain[150:152],
            torch.tensor([0, 1]),
            HiCacheStorageExtraInfo(prefix_keys=chain[:150]),
        )

        (call,) = fake.batch_put_calls
        self.assertEqual(call["keys"], ["h150__k", "h151__k"])
        self.assertEqual(call["preferred"], ["placement-key:h0"])

    def test_without_prefix_keys_the_put_keeps_the_default_placement(self):
        store, fake = _make_store()
        store.register_mem_pool_host(FakeHostKVCache(objects_per_page=2))

        with self.assertLogs(
            "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store", "WARNING"
        ) as logs:
            store.batch_set_v1(["a"], torch.tensor([0]))
            store.batch_set_v1(["b"], torch.tensor([0]), HiCacheStorageExtraInfo())

        self.assertEqual(len(logs.records), 1)
        self.assertEqual([c["args"] for c in fake.batch_put_calls], [(), ()])

    def test_existing_keys_and_group_ids_stay_aligned(self):
        store, fake = _make_store(enable_group_semantics=True, anchor_pages=1)
        store.register_mem_pool_host(FakeHostKVCache(objects_per_page=2))
        fake.existing_keys.update({"h0_0_v", "h1_0_k"})
        chain = _chain(3)

        result = store.batch_set_v1(
            chain, torch.tensor([0, 1, 2]), HiCacheStorageExtraInfo(prefix_keys=[])
        )

        self.assertEqual(result, [True] * 3)
        calls = fake.batch_put_calls
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["keys"], ["h0_0_k"])
        self.assertEqual(calls[0]["preferred"], ["placement-key:h0"])
        self.assertEqual(calls[0]["group_ids"], ["sglang-hicache:h0"])
        self.assertEqual(calls[1]["keys"], ["h1_0_v", "h2_0_k", "h2_0_v"])
        self.assertEqual(calls[1]["preferred"], ["placement-key:h1"])
        self.assertEqual(
            calls[1]["group_ids"],
            ["sglang-hicache:h1", "sglang-hicache:h2", "sglang-hicache:h2"],
        )

    def test_short_put_result_fails_the_page(self):
        store, fake = _make_store()
        store.register_mem_pool_host(FakeHostKVCache(objects_per_page=2))
        fake.short_results = True

        result = store.batch_set_v1(
            ["h0"], torch.tensor([0]), HiCacheStorageExtraInfo(prefix_keys=[])
        )

        self.assertEqual(result, [False])

    def test_v2_mamba_trailing_pages_follow_their_chain_position(self):
        store, fake = _make_store(anchor_pages=4)
        store.register_mem_host_pool_v2(FakeMambaPool(), PoolName.MAMBA)
        chain = _chain(6)

        # A TRAILING_PAGES sidecar carries only the last pages; the
        # controller's extra_info prefix makes prefix_keys + keys contiguous.
        result = store.batch_set_v2(
            [
                PoolTransfer(
                    name=PoolName.MAMBA,
                    keys=chain[3:5],
                    host_indices=torch.tensor([0, 1]),
                )
            ],
            extra_info=HiCacheStorageExtraInfo(prefix_keys=chain[:3]),
        )

        self.assertEqual(result[PoolName.MAMBA], [True, True])
        calls = fake.batch_put_calls
        self.assertEqual(
            [(c["keys"], c["preferred"]) for c in calls],
            [
                (["h3_0_temporal", "h3_0_conv_0"], ["placement-key:h0"]),
                (["h4_0_temporal", "h4_0_conv_0"], ["placement-key:h4"]),
            ],
        )

    def test_controller_chain_groups_give_kv_and_mamba_the_same_key(self):
        from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            _trailing_chain_groups,
        )

        store, fake = _make_store(anchor_pages=4)
        store.register_mem_pool_host(FakeHostKVCache(objects_per_page=2))
        store.register_mem_host_pool_v2(FakeMambaPool(), PoolName.MAMBA)
        chain = _chain(7)
        prefix, span = chain[:3], chain[3:7]
        mamba = PoolTransfer(
            name=PoolName.MAMBA,
            keys=span[-1:],
            host_indices=torch.tensor([0]),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )

        for extra_info, transfers in _trailing_chain_groups(prefix, span, [mamba]):
            store.batch_set_v2(transfers, extra_info=extra_info)
        store.batch_set_v1(
            span,
            torch.tensor([0, 1, 2, 3]),
            HiCacheStorageExtraInfo(prefix_keys=prefix),
        )

        preferred = {
            key: call["preferred"][0]
            for call in fake.batch_put_calls
            for key in call["keys"]
        }
        self.assertEqual(preferred["h6_0_temporal"], "placement-key:h4")
        self.assertEqual(preferred["h6_0_k"], preferred["h6_0_temporal"])
        self.assertEqual(preferred["h3_0_k"], "placement-key:h0")

    def test_v2_multi_buffer_put_passes_the_key(self):
        store, fake = _make_store(is_mla_model=True)
        store.register_mem_host_pool_v2(FakeMultiBufferPool(), PoolName.DEEPSEEK_V4_C4)

        store.batch_set_v2(
            [
                PoolTransfer(
                    name=PoolName.DEEPSEEK_V4_C4,
                    keys=["h0", "h1"],
                    host_indices=torch.tensor([0, 1]),
                )
            ],
            extra_info=HiCacheStorageExtraInfo(prefix_keys=[]),
        )

        (call,) = fake.batch_put_calls
        self.assertEqual(call["method"], "batch_put_from_multi_buffers")
        self.assertEqual(call["ptrs"], [[4000, 4001], [4010, 4011]])
        self.assertEqual(call["preferred"], ["placement-key:h0"])


if __name__ == "__main__":
    unittest.main(verbosity=3)
