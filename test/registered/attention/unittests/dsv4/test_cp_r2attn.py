"""INF-874 R2ATTN: the prefill-CP compressor and window-KV switches keep the cache bytes.

SGLANG_OPT_DSV4_CP_FUSED_LOW_RATIO_COMPRESS: `_forward_low_ratio_sources_cp` writes the
ratio-1/2 kv_source state with Triton launches instead of `_low_ratio_compress_torch`. The
compressed V4 cache, the fp4 index-K cache, the pair-state ring and the latent must be the
same bytes with the switch on and off (the slot-0 dump of rows that publish nothing aside).

SGLANG_OPT_DSV4_CP_WINDOW_KV_FP8_ALLGATHER: each CP rank quantizes its own window-KV rows
before the all-gather and copies the gathered rows into the SWA pages. Every emulated rank's
SWA cache must equal the current path's (bf16 all-gather, token reorder, store of all rows),
for an equal split, an uneven split, a gather index (the late-layer tail layout) and rows
without an SWA slot. The all-gather is replaced by laying the ranks' buffers side by side.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

# Imported first: the kit pins a single-rank parallel context at import, so the CP
# overrides below must nest inside it.
from sglang.test.kits.attention_unittest.attention_methods import (  # noqa: E402,F401
    dsv4_attention as kit,
)
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEV = "cuda"
D = 512
RAW_PAGE = 256
EPS = 1e-20


def _freqs(max_pos):
    from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis

    return precompute_freqs_cis(
        dim=64, seqlen=max_pos, original_seq_len=65536, base=160000,
        factor=16, beta_fast=32, beta_slow=1,
    ).to(DEV)


def _slots(prefix, extend, seed):
    """Raw slots of every extend token on shuffled pages (page 0 is the padding page)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    seq = [p + e for p, e in zip(prefix, extend)]
    max_ctx = (max(seq) + RAW_PAGE - 1) // RAW_PAGE * RAW_PAGE
    per_req = max_ctx // RAW_PAGE
    num_pages = len(seq) * per_req + 2
    perm = torch.randperm(num_pages - 1, device=DEV, generator=g)[: len(seq) * per_req] + 1
    page_of = perm.view(len(seq), per_req).to(torch.int64)
    pos_all = torch.arange(max_ctx, device=DEV)
    slot_of = page_of[:, pos_all // RAW_PAGE] * RAW_PAGE + (pos_all % RAW_PAGE)[None, :]
    pos = torch.cat([torch.arange(p, s, device=DEV) for p, s in zip(prefix, seq)])
    req_row = torch.cat(
        [torch.full((e,), r, dtype=torch.int64, device=DEV) for r, e in enumerate(extend)]
    )
    return slot_of[req_row, pos], pos, req_row, num_pages, max(seq)


class _Pool:
    """The pool surface the two compressor paths touch, on the production store kernels."""

    def __init__(self, ratio, ring, num_pages, req_slots, state_init):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import INDEX_K_SLOT_BYTES
        from sglang.kernels.ops.attention.dsv4.low_ratio_compress_triton import (
            v4_page_bytes,
        )
        from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool

        self.kv_page = RAW_PAGE // ratio
        self.idx_page = 64
        self.uniform_fp8 = False
        self.kv_cache = torch.zeros(
            num_pages, v4_page_bytes(self.kv_page), dtype=torch.uint8, device=DEV
        )
        idx_pages = num_pages * self.kv_page // self.idx_page + 1
        self.idx_cache = torch.zeros(
            idx_pages, self.idx_page * INDEX_K_SLOT_BYTES, dtype=torch.uint8, device=DEV
        )
        self.state = None
        if ratio == 2:
            self.state = CompressStatePool(
                size=req_slots * ring, ring_size=ring, overlap=False, head_dim=D,
                dtype=torch.float32, device=DEV, enable_memory_saver=False, ratio=2,
                request_scoped=True,
            )
            if state_init is not None:
                self.state.kv_score_buffer.kv_score.copy_(state_init)

    def get_attention_compress_states(self, layer_id):
        return self.state

    def get_extra_key_layout(self, layer_id):
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

        return KVLayout.V4

    def get_extra_key_buffer(self, layer_id):
        return self.kv_cache

    def get_extra_key_page_size(self, layer_id):
        return self.kv_page

    def low_ratio_index_k_is_split(self, layer_id):
        return False

    def low_ratio_index_k_rne(self, layer_id):
        return True

    def get_index_k_with_scale_buffer(self, layer_id):
        return self.idx_cache

    def set_extra_key_buffer_fused(self, layer_id, loc, cache_k, freqs_cis=None):
        from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

        fused_store_cache(
            input=cache_k, cache=self.kv_cache, indices=loc, page_size=self.kv_page,
            type="flashmla", layout=KVLayout.V4,
        )

    def set_index_k_fp4(self, layer_id, loc, cache_k):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import store_fp4_index_k_cache

        store_fp4_index_k_cache(
            input=cache_k, cache=self.idx_cache, loc=loc, page_size=self.idx_page, rne=True
        )


def _backend_cls():
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend as B

    class Backend:
        _forward_low_ratio_sources_cp = B._forward_low_ratio_sources_cp
        _cp_fused_low_ratio_compress_ok = B._cp_fused_low_ratio_compress_ok
        _low_ratio_compress_cp_fused = B._low_ratio_compress_cp_fused
        _low_ratio_compress_torch = B._low_ratio_compress_torch
        _low_ratio_pair_partners = B._low_ratio_pair_partners
        _low_ratio_write_group = B._low_ratio_write_group

        def __init__(self, pool, core):
            self.token_to_kv_pool = pool
            self.forward_metadata = SimpleNamespace(core_metadata=core, late_layer_tail=None)

    return Backend


def _layer(ratio, hidden, seed):
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
        DeepseekV41Compressor,
        DeepseekV41Indexer,
        RMSNorm,
    )

    class Indexer(torch.nn.Module):
        forward_wk = DeepseekV41Indexer.forward_wk
        index_keys = DeepseekV41Indexer.index_keys

        def __init__(self):
            super().__init__()
            self.owns_k = True
            self.index_head_dim = 128
            self.rope_head_dim = 64
            self.wk = torch.nn.Linear(D, 128, bias=False, dtype=torch.bfloat16)
            self.k_norm = RMSNorm(128, EPS)

    torch.manual_seed(seed)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)  # the engine builds the model under bf16
    try:
        comp = DeepseekV41Compressor(hidden, D, ratio, EPS, fused_compress=False)
        idx = Indexer()
    finally:
        torch.set_default_dtype(prev)
    with torch.no_grad():
        for w in [comp.wkv.weight, idx.wk.weight] + (
            [comp.wgate.weight] if ratio == 2 else []
        ):
            w.normal_(0, hidden**-0.5 if w.shape[1] == hidden else D**-0.5)
        comp.norm.weight.uniform_(0.2, 2.0)
        idx.k_norm.weight.uniform_(0.2, 2.0)
    return SimpleNamespace(
        layer_id=2, compress_ratio=ratio, compressor=comp.to(DEV), indexer=idx.to(DEV),
        rope_head_dim=64,
    )


class TestCpFusedLowRatioCompress(CustomTestCase):
    HIDDEN = 256

    def _case(self, ratio, prefix, extend, ring, seed=0):
        import sglang.srt.layers.attention.deepseek_v4_backend as backend_mod
        from sglang.srt.runtime_context import get_parallel

        raw, pos, req_row, num_pages, max_seq = _slots(prefix, extend, seed)
        T = pos.shape[0]
        req_slots = torch.arange(len(extend), device=DEV) * 3 + 1  # req_pool_indices
        layer = _layer(ratio, self.HIDDEN, seed)
        layer.freqs_cis = _freqs(max_seq + 2)
        x = torch.randn(T, self.HIDDEN, device=DEV).to(torch.bfloat16)
        if ratio == 2:
            out_loc = torch.where((pos + 1) % 2 == 0, raw // 2, -1)
        else:
            out_loc = raw.clone()
        core = SimpleNamespace(
            raw_out_loc=raw,
            c2_out_loc=out_loc if ratio == 2 else None,
            c1_out_loc=out_loc if ratio == 1 else None,
        )
        req_count = int(req_slots.max()) + 1
        state_init = None
        if ratio == 2:
            rows = (req_count * ring + ring + 1 + 1) // 2 * 2
            g = torch.Generator(device=DEV).manual_seed(seed + 7)
            state_init = torch.randn(rows, 2 * D, device=DEV, generator=g) * 3
            state_init[-1, :D] = 0
            state_init[-1, D:] = float("-inf")
        batch = SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(total_seq_lens=T),
            extend_seq_lens_cpu=list(extend),
            extend_seq_lens=torch.tensor(extend, dtype=torch.int32, device=DEV),
            req_pool_indices=req_slots.to(torch.int32),
            forward_mode=ForwardMode.EXTEND,
            positions=pos.clone(),
        )
        latents = {}
        results = {}
        for on in (False, True):
            pool = _Pool(ratio, ring, num_pages, req_count, state_init)
            backend = _backend_cls()(pool, core)
            captured = {}
            orig_finish = layer.compressor.finish

            def finish(kv, _orig=orig_finish, _cap=captured):
                out = _orig(kv)
                _cap["latent"] = out
                return out

            layer.compressor.finish = finish
            with envs.SGLANG_OPT_DSV4_CP_FUSED_LOW_RATIO_COMPRESS.override(on), patch.object(
                backend_mod, "cp_materialize_global_token_order", lambda x, fb, s: x
            ), get_parallel().override(attn_cp_size=4, attn_cp_rank=0):
                self.assertEqual(backend._cp_fused_low_ratio_compress_ok(layer), on)
                if on:
                    # the fused path returns its latent; grab it through forward_wk
                    orig_wk = layer.indexer.forward_wk

                    def wk(lat, _orig=orig_wk, _cap=captured):
                        _cap["latent"] = lat
                        return _orig(lat)

                    layer.indexer.forward_wk = wk
                backend._forward_low_ratio_sources_cp(
                    layer=layer, x=x, q_lora=None, positions=pos, forward_batch=batch,
                    run_compressor=True, run_indexer=False,
                )
                torch.cuda.synchronize()
            layer.compressor.finish = orig_finish
            if on:
                del layer.indexer.forward_wk
            results[on] = pool
            latents[on] = captured["latent"]
        off, fused = results[False], results[True]
        # page 0 holds only the reserved slot 0, where the torch path dumps rows that
        # publish nothing
        self.assertTrue(torch.equal(off.kv_cache[1:], fused.kv_cache[1:]), "compressed KV")
        self.assertTrue(torch.equal(off.idx_cache[1:], fused.idx_cache[1:]), "index-K")
        self.assertGreater(int((off.kv_cache[1:] != 0).sum()), 0)
        if ratio == 2:
            a = off.state.kv_score_buffer.kv_score
            b = fused.state.kv_score_buffer.kv_score
            self.assertTrue(torch.equal(a.view(torch.int32), b.view(torch.int32)), "ring")
        publish = out_loc > 0
        self.assertTrue(
            torch.equal(
                latents[False][publish].view(torch.int16),
                latents[True][publish].view(torch.int16),
            )
        )

    def test_ratio2_ring8(self):
        # odd and even chunk starts, a 1-row chunk, chunks shorter and longer than the ring
        self._case(2, prefix=[1463, 400, 128, 8, 3, 0], extend=[37, 301, 5, 1, 9, 64], ring=8)

    def test_ratio2_ring2(self):
        self._case(2, prefix=[1463, 400, 127, 8], extend=[37, 301, 6, 3], ring=2, seed=1)

    def test_ratio2_long_chunk(self):
        self._case(2, prefix=[70001, 68000], extend=[3001, 2500], ring=8, seed=2)

    def test_ratio1(self):
        self._case(1, prefix=[1463, 400, 0], extend=[37, 301, 5], ring=0, seed=3)

    def test_gate(self):
        layer = SimpleNamespace(
            layer_id=2, compress_ratio=2, rope_head_dim=64,
            indexer=SimpleNamespace(owns_k=True, index_head_dim=128),
        )
        pool = _Pool(2, 2, 4, 2, None)
        backend = _backend_cls()(pool, None)
        with envs.SGLANG_OPT_DSV4_CP_FUSED_LOW_RATIO_COMPRESS.override(False):
            self.assertFalse(backend._cp_fused_low_ratio_compress_ok(layer))
        with envs.SGLANG_OPT_DSV4_CP_FUSED_LOW_RATIO_COMPRESS.override(True):
            self.assertTrue(backend._cp_fused_low_ratio_compress_ok(layer))
            pool.low_ratio_index_k_rne = lambda layer_id: False
            self.assertFalse(backend._cp_fused_low_ratio_compress_ok(layer))


class TestCpWindowKvFp8Allgather(CustomTestCase):
    CP = 4

    def _case(self, total, per_rank_actual, logical, gather_index, swa_hole=False, seed=0):
        import sglang.srt.layers.cp.interleave as interleave_mod
        import sglang.srt.layers.cp.window_kv_fp8 as fp8_mod
        from sglang.kernels.ops.attention.dsv4.elementwise import fused_k_norm_rope_flashmla
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
        from sglang.kernels.ops.attention.dsv4.low_ratio_compress_triton import (
            v4_page_bytes,
        )
        from sglang.srt.layers.cp.base import get_cp_strategy, init_cp_strategy
        from sglang.srt.layers.cp.interleave import InterleaveContextParallelMetadata
        from sglang.srt.runtime_context import get_parallel

        CP = self.CP
        g = torch.Generator(device=DEV).manual_seed(seed)
        phys = max(per_rank_actual)
        swa_page = 256
        num_pages = total // swa_page + 4
        swa_loc = torch.randperm(num_pages * swa_page - swa_page, device=DEV, generator=g)[:total]
        swa_loc = (swa_loc + swa_page).to(torch.int32)
        if swa_hole:
            swa_loc[::7] = -1
        pos = torch.randint(0, 200000, (total,), device=DEV, generator=g)
        kv_global = (torch.randn(total, D, device=DEV, generator=g) * 3).to(torch.bfloat16)
        weight = (torch.rand(D, device=DEV, generator=g) + 0.5).to(torch.bfloat16)
        freqs = _freqs(200001)
        src = (
            gather_index
            if gather_index is not None
            else (torch.arange(total, device=DEV) % CP) * phys + torch.arange(total, device=DEV) // CP
        )
        kv_local, pos_local = [], []
        for r in range(CP):
            t_r = torch.nonzero((src // phys) == r).flatten()
            t_r = t_r[torch.argsort(src[t_r] % phys)]
            assert t_r.shape[0] == logical[r]
            pad = per_rank_actual[r] - logical[r]
            kv_local.append(torch.cat([kv_global[t_r], torch.randn(pad, D, device=DEV, generator=g).to(torch.bfloat16)]))
            pos_local.append(torch.cat([pos[t_r], torch.zeros(pad, dtype=pos.dtype, device=DEV)]))
        meta = InterleaveContextParallelMetadata(
            per_rank_actual_token=list(per_rank_actual),
            max_rank_len=[phys] * CP,
            total_seq_lens=total,
            bs=1,
            per_rank_logical_token=list(logical),
            gather_index=gather_index,
        )
        batch = SimpleNamespace(attn_cp_metadata=meta)

        # the current path's result: the store over all rows in token order
        ref = torch.zeros(num_pages, v4_page_bytes(swa_page), dtype=torch.uint8, device=DEV)
        fused_k_norm_rope_flashmla(
            kv=kv_global, kv_weight=weight, eps=EPS, freqs_cis=freqs, positions=pos,
            out_loc=swa_loc, kvcache=ref, page_size=swa_page, layout=KVLayout.V4,
        )
        bufs = {}
        state = {}

        def fake_ag(gathered, local):
            view = gathered.view(CP, *local.shape)
            for r in range(CP):
                view[r].copy_(local if r == state["rank"] else bufs[state["kind"]][r])

        pages = fp8_mod.send_buffer_pages(phys)
        bufs["fp8"] = []
        for r in range(CP):
            b = torch.zeros(pages, fp8_mod.SEND_PAGE_BYTES, dtype=torch.uint8, device=DEV)
            fp8_mod.quantize_local_window_kv(kv_local[r], weight, EPS, freqs, pos_local[r], logical[r], phys, b)
            bufs["fp8"].append(b)
        bufs["bf16"] = []
        for r in range(CP):
            p = kv_local[r].new_zeros(phys, D)
            p[: logical[r]] = kv_local[r][: logical[r]]
            bufs["bf16"].append(p)

        init_cp_strategy(enable_prefill_cp=True, cp_size=CP, cp_strategy="interleave")
        for r in range(CP):
            state["rank"] = r
            with get_parallel().override(
                attn_tp_size=1, attn_tp_rank=0, attn_cp_size=CP, attn_cp_rank=r, attn_cp_group=None
            ), patch.object(fp8_mod, "attn_cp_all_gather_into_tensor", fake_ag), patch.object(
                fp8_mod, "is_allocation_symmetric", lambda: False
            ), patch.object(interleave_mod, "attn_cp_all_gather_into_tensor", fake_ag), patch.object(
                interleave_mod, "is_allocation_symmetric", lambda: False
            ):
                # the current path as the engine runs it on rank r
                state["kind"] = "bf16"
                cur = torch.zeros_like(ref)
                kv = get_cp_strategy().gather_kv_cache(kv_local[r], batch)
                fused_k_norm_rope_flashmla(
                    kv=kv, kv_weight=weight, eps=EPS, freqs_cis=freqs, positions=pos,
                    out_loc=swa_loc, kvcache=cur, page_size=swa_page, layout=KVLayout.V4,
                )
                state["kind"] = "fp8"
                new = torch.zeros_like(ref)
                fp8_mod.cp_store_window_kv_fp8(
                    kv=kv_local[r], kv_weight=weight, eps=EPS, freqs_cis=freqs,
                    local_positions=pos_local[r], swa_loc=swa_loc, swa_cache=new,
                    swa_page_size=swa_page, forward_batch=batch,
                )
                torch.cuda.synchronize()
            self.assertTrue(torch.equal(cur, ref), f"rank {r}: current path vs reference")
            self.assertTrue(torch.equal(new, ref), f"rank {r}: fp8 all-gather path")
        self.assertGreater(int((ref != 0).sum()), 0)

    def test_equal_split(self):
        self._case(4096, [1024] * 4, [1024] * 4, None)

    def test_uneven_split(self):
        # 4,099 tokens: ranks 0-2 hold 1,025 rows, rank 3 1,024 (+1 padding row)
        self._case(4099, [1025] * 4, [1025, 1025, 1025, 1024], None, seed=1)

    def test_small_with_holes(self):
        self._case(7, [2] * 4, [2, 2, 2, 1], None, swa_hole=True, seed=2)

    def test_gather_index(self):
        # the late-layer tail layout: owner = token % cp over a subset of tokens
        tokens = torch.cat([torch.arange(s - 128, s) for s in (5000, 9001, 12345)]).to(DEV)
        owner = tokens % self.CP
        counts = torch.bincount(owner, minlength=self.CP).tolist()
        max_local = max(counts)
        order = torch.argsort(owner, stable=True)
        starts = torch.tensor([sum(counts[:r]) for r in range(self.CP)], device=DEV)
        slot = torch.empty_like(owner)
        slot[order] = torch.arange(tokens.shape[0], device=DEV) - starts[owner[order]]
        gi = owner * max_local + slot
        self._case(tokens.shape[0], [max_local] * self.CP, counts, gi, swa_hole=True, seed=3)


if __name__ == "__main__":
    unittest.main()
