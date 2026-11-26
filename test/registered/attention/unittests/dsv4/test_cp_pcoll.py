"""INF-874 PCOLL: prefill collectives removed without changing a bit.

P1 SGLANG_OPT_DSV4_CP_REPLICATED_EMBEDDING: the replicated token embedding gives the
bits of the vocab-parallel lookup + bf16 all-reduce (-0 comes out +0), and so does its
replaced path (enabled("embed") False) on the replicated table.

P3-bit SGLANG_OPT_DSV4_CP_KV_SOURCE_PROJECTION_ALLGATHER: the kv_source compressor's
projections all-gathered from each rank's own rows equal the projections of the
all-gathered hidden rows bit for bit (production shapes included), and the compressed
caches written by `_forward_low_ratio_sources_cp` are the same bytes. Ranks are emulated
one after another; the all-gather is the interleave reorder of the ranks' rows.
"""

import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

# Imported first: the kit pins a single-rank parallel context at import.
from sglang.test.kits.attention_unittest.attention_methods import (  # noqa: E402,F401
    dsv4_attention as kit,
)
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEV = "cuda"
CP = 4


def _bits(t):
    t = t.contiguous()
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


# ---- P1 ---------------------------------------------------------------------------


class TestReplicatedEmbedding(CustomTestCase):
    V, D = 1003, 96

    def _table(self):
        g = torch.Generator().manual_seed(0)
        w = torch.randn(self.V, self.D, generator=g).to(torch.bfloat16)
        w[3, :5] = -0.0
        w[7, 1] = 0.0
        w[11, :] = -0.0
        w[13, 2] = torch.finfo(torch.bfloat16).tiny * -1
        return w

    def _module(self, rank, table):
        from sglang.srt.layers.cp import pcoll

        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            m = pcoll.ReplicatedVocabEmbedding(self.V, self.D, tp_size=CP, tp_rank=rank)
        finally:
            torch.set_default_dtype(prev)
        m.weight.weight_loader(m.weight, table)
        m = m.to(DEV)
        m.finish_load()
        return m

    def _reference(self, table, ids):
        """Vocab-parallel lookup: each rank its id range, zeros elsewhere, bf16 sum."""
        acc = torch.zeros(*ids.shape, self.D, dtype=torch.float32, device=DEV)
        t = table.to(DEV)
        for r in range(CP):
            lo, hi = self.V * r // CP, self.V * (r + 1) // CP
            owned = (ids >= lo) & (ids < hi)
            part = torch.where(owned[..., None], t[torch.where(owned, ids, 0)], 0)
            acc = acc + part.float()
        return acc.to(torch.bfloat16)

    def test_replicated_equals_vocab_parallel_allreduce(self):
        from sglang.srt.layers.cp import pcoll

        table = self._table()
        g = torch.Generator(device=DEV).manual_seed(1)
        ids = torch.randint(0, self.V, (4099,), device=DEV, generator=g)
        ids[:6] = torch.tensor([3, 7, 11, 13, 0, self.V - 1], device=DEV)
        ref = self._reference(table, ids)
        self.assertTrue(bool((_bits(ref[2]) == 0).all()))  # -0 rows come out +0
        mods = [self._module(r, table) for r in range(CP)]
        for r, m in enumerate(mods):
            out = m(ids)
            self.assertTrue(torch.equal(_bits(out), _bits(ref)), f"rank {r}")
        # The replaced path on the replicated table (measurement toggle off).
        parts = []
        with patch.object(pcoll, "enabled", lambda name: False), patch.object(
            pcoll, "tensor_model_parallel_all_reduce", lambda t: t
        ):
            for m in mods:
                parts.append(m(ids))
        acc = sum(p.float() for p in parts).to(torch.bfloat16)
        self.assertTrue(torch.equal(_bits(acc), _bits(ref)))

    def test_gate(self):
        from sglang.srt.layers.cp import pcoll

        cfg = SimpleNamespace(tie_word_embeddings=False)
        self.assertFalse(envs.SGLANG_OPT_DSV4_CP_REPLICATED_EMBEDDING.get())
        self.assertFalse(pcoll.replicated_embedding_wanted(cfg, None, 4))
        with envs.SGLANG_OPT_DSV4_CP_REPLICATED_EMBEDDING.override(True):
            self.assertTrue(pcoll.replicated_embedding_wanted(cfg, None, 4))
            self.assertFalse(pcoll.replicated_embedding_wanted(cfg, None, 1))
            self.assertFalse(pcoll.replicated_embedding_wanted(cfg, object(), 4))
            tied = SimpleNamespace(tie_word_embeddings=True)
            self.assertFalse(pcoll.replicated_embedding_wanted(tied, None, 4))


# ---- P3-bit -----------------------------------------------------------------------


def _meta(total):
    from sglang.srt.layers.cp.interleave import InterleaveContextParallelMetadata

    logical = [len(range(r, total, CP)) for r in range(CP)]
    phys = max(logical)
    return InterleaveContextParallelMetadata(
        per_rank_actual_token=[phys] * CP,
        max_rank_len=[phys] * CP,
        total_seq_lens=total,
        bs=1,
        per_rank_logical_token=logical,
    )


def _compressor(hidden, ratio, seed):
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Compressor

    torch.manual_seed(seed)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        comp = DeepseekV41Compressor(hidden, 512, ratio, 1e-20, fused_compress=False)
    finally:
        torch.set_default_dtype(prev)
    with torch.no_grad():
        for w in [comp.wkv.weight] + ([comp.wgate.weight] if ratio == 2 else []):
            w.normal_(0, hidden**-0.5)
    return comp.to(DEV)


def _local_rows(x_global, total, rank, seed):
    """Rank's interleaved rows, padded to the physical length with garbage."""
    phys = max(len(range(r, total, CP)) for r in range(CP))
    mine = x_global[rank::CP]
    g = torch.Generator(device=DEV).manual_seed(seed + rank)
    pad = torch.randn(phys - mine.shape[0], x_global.shape[1], device=DEV, generator=g)
    return torch.cat([mine, pad.to(x_global.dtype)])


def _emulated_projection_allgather(comp, x_global, total):
    """Run the helper on every rank: pass 1 records what each rank sends, pass 2 hands
    every rank the interleave-ordered union. Returns rank -> (kv, score)."""
    from sglang.srt.layers.cp import pcoll

    batch = SimpleNamespace(
        attn_cp_metadata=_meta(total), positions=torch.zeros(1, device=DEV)
    )
    xs = [_local_rows(x_global, total, r, 5) for r in range(CP)]
    sent = {}

    def capture(rank):
        def g(t):
            sent[rank] = t.clone()
            return t.new_empty(total, t.shape[1])

        return g

    def assembled(t):
        out = t.new_empty(total, t.shape[1])
        for r in range(CP):
            n = len(range(r, total, CP))
            out[r::CP] = sent[r][:n]
        return out

    for r in range(CP):
        pcoll.kv_source_projection_allgather(
            comp, xs[r], batch, cp_rank=r, cp_size=CP, gather=capture(r)
        )
    return {
        r: pcoll.kv_source_projection_allgather(
            comp, xs[r], batch, cp_rank=r, cp_size=CP, gather=assembled
        )
        for r in range(CP)
    }, batch, xs


class TestKvSourceProjectionAllgather(CustomTestCase):
    def _check(self, total, hidden, ratio, seed=0):
        comp = _compressor(hidden, ratio, seed)
        g = torch.Generator(device=DEV).manual_seed(seed)
        x_global = torch.randn(total, hidden, device=DEV, generator=g).to(torch.bfloat16)
        ref_kv, ref_score = comp.project(x_global)
        got, _, _ = _emulated_projection_allgather(comp, x_global, total)
        for r, (kv, score) in got.items():
            self.assertEqual(kv.shape, ref_kv.shape)
            self.assertTrue(torch.equal(_bits(kv), _bits(ref_kv)), f"kv rank {r}")
            if ratio == 2:
                self.assertTrue(torch.equal(_bits(score), _bits(ref_score)), f"score {r}")
            else:
                self.assertIsNone(score)

    def test_production_shape_ratio2(self):
        self._check(64968, 5120, 2)

    def test_production_shape_ratio1(self):
        self._check(64968, 5120, 1, seed=1)

    def test_uneven_and_small(self):
        for total in (1, 5, 4099, 65001):
            for ratio in (1, 2):
                with self.subTest(total=total, ratio=ratio):
                    self._check(total, 5120 if total > 4096 else 256, ratio, seed=total)

    def test_gather_index_falls_back(self):
        from sglang.srt.layers.cp import pcoll

        meta = _meta(8)
        meta.gather_index = torch.arange(8, device=DEV)
        batch = SimpleNamespace(attn_cp_metadata=meta, positions=torch.zeros(1, device=DEV))
        self.assertIsNone(
            pcoll.kv_source_projection_allgather(
                None, torch.zeros(2, 4, device=DEV), batch, cp_rank=0, cp_size=CP,
                gather=lambda t: t,
            )
        )

    def test_off_by_default(self):
        self.assertFalse(envs.SGLANG_OPT_DSV4_CP_KV_SOURCE_PROJECTION_ALLGATHER.get())
        self.assertFalse(envs.SGLANG_OPT_DSV4_CP_KV_SOURCE_PROJECTION_LOCAL_M.get())


def _r2attn():
    path = os.path.join(os.path.dirname(__file__), "test_cp_r2attn.py")
    spec = importlib.util.spec_from_file_location("_r2attn_harness", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestKvSourceProjectionCache(CustomTestCase):
    """`_forward_low_ratio_sources_cp` with the switch on writes the same compressed
    KV, index-K, ring state as with it off (fused and torch compressor paths)."""

    HIDDEN = 256

    def _case(self, ratio, prefix, extend, ring, fused, seed=0):
        import sglang.srt.layers.attention.deepseek_v4_backend as backend_mod
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.runtime_context import get_parallel

        h = _r2attn()
        raw, pos, req_row, num_pages, max_seq = h._slots(prefix, extend, seed)
        T = pos.shape[0]
        req_slots = torch.arange(len(extend), device=DEV) * 3 + 1
        layer = h._layer(ratio, self.HIDDEN, seed)
        layer.freqs_cis = h._freqs(max_seq + 2)
        x_global = torch.randn(T, self.HIDDEN, device=DEV).to(torch.bfloat16)
        out_loc = torch.where((pos + 1) % 2 == 0, raw // 2, -1) if ratio == 2 else raw.clone()
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
            state_init = torch.randn(rows, 2 * 512, device=DEV, generator=g) * 3
            state_init[-1, :512] = 0
            state_init[-1, 512:] = float("-inf")
        got, meta_batch, xs = _emulated_projection_allgather(layer.compressor, x_global, T)
        batch = SimpleNamespace(
            attn_cp_metadata=meta_batch.attn_cp_metadata,
            extend_seq_lens_cpu=list(extend),
            extend_seq_lens=torch.tensor(extend, dtype=torch.int32, device=DEV),
            req_pool_indices=req_slots.to(torch.int32),
            forward_mode=ForwardMode.EXTEND,
            positions=pos.clone(),
        )
        width = 512
        results = {}
        for on in (False, True):
            pool = h._Pool(ratio, ring, num_pages, req_count, state_init)
            Backend = h._backend_cls()
            Backend._cp_kv_source_projections = backend_mod.DeepseekV4AttnBackend._cp_kv_source_projections
            backend = Backend(pool, core)
            kv0, score0 = got[0]
            union = kv0 if score0 is None else torch.cat([kv0, score0], dim=1)

            def gather(t, fb, s, _union=union):
                if t.shape[1] == self.HIDDEN:  # the hidden all-gather (switch off)
                    return x_global
                self.assertEqual(t.shape[1], width if ratio == 1 else 2 * width)
                return _union

            with envs.SGLANG_OPT_DSV4_CP_FUSED_LOW_RATIO_COMPRESS.override(
                fused
            ), envs.SGLANG_OPT_DSV4_CP_KV_SOURCE_PROJECTION_ALLGATHER.override(
                on
            ), patch.object(
                backend_mod, "cp_materialize_global_token_order", gather
            ), get_parallel().override(attn_cp_size=CP, attn_cp_rank=0):
                backend._forward_low_ratio_sources_cp(
                    layer=layer, x=xs[0], q_lora=None, positions=pos[0::CP],
                    forward_batch=batch, run_compressor=True, run_indexer=False,
                )
                torch.cuda.synchronize()
            results[on] = pool
        off, new = results[False], results[True]
        self.assertTrue(torch.equal(off.kv_cache, new.kv_cache), "compressed KV")
        self.assertTrue(torch.equal(off.idx_cache, new.idx_cache), "index-K")
        self.assertGreater(int((off.kv_cache[1:] != 0).sum()), 0)
        if ratio == 2:
            a = off.state.kv_score_buffer.kv_score
            b = new.state.kv_score_buffer.kv_score
            self.assertTrue(torch.equal(a.view(torch.int32), b.view(torch.int32)), "ring")

    def test_ratio2_fused(self):
        self._case(2, prefix=[1463, 400, 128, 8, 3, 0], extend=[37, 301, 5, 1, 9, 64], ring=8, fused=True)

    def test_ratio2_torch(self):
        self._case(2, prefix=[1463, 400, 127, 8], extend=[37, 301, 6, 3], ring=2, fused=False, seed=1)

    def test_ratio1_fused(self):
        self._case(1, prefix=[1463, 400, 0], extend=[37, 301, 5], ring=0, fused=True, seed=3)


if __name__ == "__main__":
    unittest.main()
