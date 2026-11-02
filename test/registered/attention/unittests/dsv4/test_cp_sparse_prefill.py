"""INF-681 k-attnp: prefill CP (interleave) through the FlashMLA sparse prefill kernel.

SGLANG_OPT_DSV4_CP_SPARSE_PREFILL: each CP rank attends its own interleaved query rows with
flash_mla_sparse_fwd over a bf16 dequant of the whole chunk's history, instead of the sparse
decode kernel. One attention layer of a real backend runs on one GPU with the parallel context
set to CP rank r of 4 and the interleave strategy bound, so init_forward_metadata builds the
engine's CP-reindexed metadata. Checked against the decode-kernel path and an fp32 reference, for
every rank, ratios 0 / 1 / 2, and a request with fewer extend rows than CP ranks.

Also: SGLANG_OPT_DSV4_SPARSE_PREFILL_SHARED_SOURCE_DEQUANT (reuse of a kv_source's dequant),
SGLANG_OPT_DSV4_SPARSE_PREFILL_FAST_DEQUANT and SGLANG_OPT_DSV4_ROPE_WIDE are bit-identical.
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
from sglang.test.kits.attention_unittest.attention_methods import (  # noqa: E402
    dsv4_attention as kit,
)
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE = 256
W = 128
HEADS = 64
D = 512
TOPK = 512


def _build_layer(ratio, prefix, extend, cp_size, cp_rank, seed):
    """One attention layer of CP rank ``cp_rank``: backend, batch, q, sink, row facts."""
    from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
        quant_to_nope_fp8_rope_bf16_pack_triton,
    )
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    from sglang.srt.layers.cp.base import init_cp_strategy
    from sglang.srt.layers.cp.utils import is_cp_active, prepare_cp_forward
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    dev = "cuda"
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    seq = [p + e for p, e in zip(prefix, extend)]
    R, T = len(seq), sum(extend)
    max_ctx = (max(seq) + PAGE - 1) // PAGE * PAGE
    pages_per_req = max_ctx // PAGE
    num_pages = R * pages_per_req + 2
    swa_size = num_pages * PAGE
    init_cp_strategy(enable_prefill_cp=True, cp_size=cp_size, cp_strategy="interleave")
    case = kit.DSV4AttentionCase(
        name=f"cp_sparse_c{ratio}",
        backend="dsv4",
        forward_mode=ForwardMode.EXTEND,
        num_heads=HEADS,
        page_size=PAGE,
        prefix_lens=tuple(prefix),
        extend_lens=tuple(extend),
        compress_ratio=ratio,
    )
    runner = kit.MockDSV4ModelRunner(
        case=case,
        model_config=kit.TinyDSV4ModelConfig(
            num_heads=HEADS, context_len=max_ctx, compression_ratios=[ratio]
        ),
        dtype=torch.bfloat16,
        device=dev,
        max_context_len=max_ctx,
        swa_size=swa_size,
        compression_ratios=[ratio],
    )
    backend = ATTENTION_BACKENDS["dsv4"](runner)
    pool = runner.token_to_kv_pool

    # Page-aligned slots on shuffled pages.
    perm = torch.randperm(num_pages - 1, device=dev)[: R * pages_per_req] + 1
    page_of = perm.view(R, pages_per_req).long()
    pos_all = torch.arange(max_ctx, device=dev)
    slot_of = page_of[:, pos_all // PAGE] * PAGE + (pos_all % PAGE)[None, :]
    rtt = runner.req_to_token_pool.req_to_token
    rtt[:R, :max_ctx] = slot_of.to(rtt.dtype)

    positions = torch.cat(
        [torch.arange(prefix[r], seq[r], device=dev) for r in range(R)]
    )
    req_of_tok = torch.cat(
        [torch.full((extend[r],), r, device=dev, dtype=torch.long) for r in range(R)]
    )
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=R,
        input_ids=torch.arange(T, dtype=torch.int64, device=dev),
        req_pool_indices=torch.arange(R, dtype=torch.int32, device=dev),
        seq_lens=torch.tensor(seq, dtype=torch.int32, device=dev),
        seq_lens_cpu=torch.tensor(seq, dtype=torch.int32),
        out_cache_loc=slot_of[req_of_tok, positions].clone(),
        seq_lens_sum=sum(seq),
        positions=positions.clone(),
    )
    runner.kv_index_translator.bind_own_plan(batch)
    ext = torch.tensor(extend, dtype=torch.int32, device=dev)
    batch.extend_prefix_lens = torch.tensor(prefix, dtype=torch.int32, device=dev)
    batch.extend_prefix_lens_cpu = list(prefix)
    batch.extend_seq_lens = ext
    batch.extend_seq_lens_cpu = list(extend)
    batch.extend_start_loc = torch.zeros_like(ext)
    batch.extend_start_loc[1:] = torch.cumsum(ext[:-1], 0)
    batch.extend_num_tokens = T
    assert is_cp_active(batch)
    prepare_cp_forward(batch)
    cpm = batch.attn_cp_metadata

    gidx = torch.arange(cp_rank, T, cp_size, device=dev)
    n_real = gidx.shape[0]
    n_rows = cpm.per_rank_actual_token[cp_rank]
    loc_req, loc_pos = req_of_tok[gidx], positions[gidx]

    def rand_k(n):
        return torch.randn(n, D, dtype=torch.bfloat16, device=dev, generator=gen)

    swa_slots = torch.cat(
        [slot_of[r, max(0, prefix[r] - W + 1) : seq[r]] for r in range(R)]
    )
    swa_k = rand_k(swa_slots.shape[0])
    pool.set_swa_key_buffer_radix(
        layer_id=0,
        swa_loc=pool.translate_loc_from_full_to_swa(swa_slots),
        cache_nope_fp8_rope_bf16_pack=quant_to_nope_fp8_rope_bf16_pack_triton(swa_k),
    )
    c_tab = None
    if ratio in (1, 2):
        c_len = [s // ratio for s in seq]
        c_tab = torch.full((R, max(c_len)), -1, dtype=torch.long, device=dev)
        for r in range(R):
            c_tab[r, : c_len[r]] = (
                slot_of[r, torch.arange(c_len[r], device=dev) * ratio] // ratio
            )
        c_all = torch.cat([c_tab[r, : c_len[r]] for r in range(R)])
        pool.set_extra_key_buffer(
            layer_id=0,
            loc=c_all,
            cache_nope_fp8_rope_bf16_pack=quant_to_nope_fp8_rope_bf16_pack_triton(
                rand_k(c_all.shape[0])
            ),
        )

    q = torch.randn(n_rows, HEADS, D, dtype=torch.bfloat16, device=dev, generator=gen)
    q[n_real:] = 0
    sink = torch.randn(HEADS, dtype=torch.float32, device=dev, generator=gen)
    layer = RadixAttention(
        num_heads=HEADS,
        head_dim=D,
        scaling=D**-0.5,
        num_kv_heads=1,
        layer_id=0,
        v_head_dim=D,
    )
    return SimpleNamespace(
        backend=backend,
        pool=pool,
        batch=batch,
        q=q,
        sink=sink,
        layer=layer,
        ratio=ratio,
        n_real=n_real,
        loc_req=loc_req,
        loc_pos=loc_pos,
        slot_of=slot_of,
        c_tab=c_tab,
        gen=gen,
    )


def _seed_topk(f):
    """The indexer's output for the local rows: distinct causal positions."""
    md = f.backend.forward_metadata.core_metadata
    ratio = f.ratio
    raw_buf = md.sparse_raw_indices(ratio)
    raw = torch.full_like(raw_buf, -1)
    pidx = torch.full_like(raw_buf, -1)
    lens = torch.ones(raw.shape[0], dtype=torch.int32, device=raw.device)
    n_vis = (f.loc_pos + 1) // ratio
    cmax = max(int(n_vis.max()), 1)
    k = min(TOPK, cmax)
    score = torch.rand(f.n_real, cmax, device=raw.device, generator=f.gen)
    score.masked_fill_(
        torch.arange(cmax, device=raw.device)[None, :] >= n_vis[:, None], -1.0
    )
    val, idx = score.topk(k, dim=1)
    idx = torch.where(val >= 0, idx, -1).to(torch.int32)
    raw[: f.n_real, :k] = idx
    sl = f.c_tab[f.loc_req[:, None], idx.clamp_min(0).long()]
    pidx[: f.n_real, :k] = torch.where(idx >= 0, sl, -1).to(torch.int32)
    lens[: f.n_real] = torch.clamp(n_vis, min=1, max=TOPK).to(torch.int32)
    md.set_sparse_topk(ratio, page_indices=pidx, topk_lengths=lens, raw_indices=raw)
    return raw


def _reference(f, raw):
    """fp32 attention over the dequantized cache, every real local row."""
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
        dequantize_k_cache_paged,
    )

    def dq(buf, slots, page_size, layout):
        out = torch.empty(
            slots.numel(), 1, D, dtype=torch.bfloat16, device=slots.device
        )
        dequantize_k_cache_paged(
            buf,
            slots.reshape(-1).to(torch.int32),
            page_size=page_size,
            out=out,
            layout=layout,
        )
        return out.view(*slots.shape, D).float()

    pool = f.pool
    off = torch.arange(W, device=f.q.device)
    sp = f.loc_pos[:, None] - (W - 1) + off[None, :]
    valid = sp >= 0
    keys = dq(
        pool.get_swa_key_buffer_radix(0),
        pool.translate_loc_from_full_to_swa(
            f.slot_of[f.loc_req[:, None], sp.clamp_min(0)]
        ),
        pool.get_swa_key_page_size(),
        pool.get_swa_key_layout(),
    )
    if raw is not None:
        ri = raw[: f.n_real].long()
        keys = torch.cat(
            [
                keys,
                dq(
                    pool.get_extra_key_buffer(0),
                    f.c_tab[f.loc_req[:, None], ri.clamp_min(0)],
                    pool.get_extra_key_page_size(0),
                    pool.get_extra_key_layout(0),
                ),
            ],
            1,
        )
        valid = torch.cat([valid, ri >= 0], 1)
    sc = torch.einsum("nhd,nkd->nhk", f.q[: f.n_real].float(), keys) * D**-0.5
    sc = sc.masked_fill(~valid[:, None, :], float("-inf"))
    sc = torch.cat([sc, f.sink.view(1, -1, 1).expand(f.n_real, -1, 1)], -1)
    return torch.einsum("nhk,nkd->nhd", torch.softmax(sc, -1)[..., :-1], keys)


def _run(f, sparse, shared=False, fast=False):
    with (
        envs.SGLANG_OPT_DSV4_CP_SPARSE_PREFILL.override(sparse),
        envs.SGLANG_OPT_DSV4_SPARSE_PREFILL_SHARED_SOURCE_DEQUANT.override(shared),
        envs.SGLANG_OPT_DSV4_SPARSE_PREFILL_FAST_DEQUANT.override(fast),
    ):
        return f.backend.forward(
            q=f.q,
            k=f.q,
            v=f.q,
            layer=f.layer,
            forward_batch=f.batch,
            compress_ratio=f.ratio,
            save_kv_cache=False,
            attn_sink=f.sink,
        ).clone()


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestCPSparsePrefill(CustomTestCase):
    CP = 4
    SHAPES = (
        # (prefix, extend): mixed lengths, one request shorter than the CP size
        ((1463, 400, 128, 900), (37, 300, 5, 1)),
        ((5000,), (1031,)),
    )

    def tearDown(self):
        from sglang.srt.layers.cp.base import init_cp_strategy

        init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="zigzag")

    def _check(self, ratio, prefix, extend, rank):
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.runtime_context import get_parallel

        with (
            get_parallel().override(
                # The kit's runner installs tp_size 1 server args; keep the topology
                # consistent for a CP4 rank whose MoE runs TP4.
                tp_size=self.CP,
                moe_tp_size=self.CP,
                attn_tp_size=1,
                attn_tp_rank=0,
                attn_cp_size=self.CP,
                attn_cp_rank=rank,
            ),
            torch.no_grad(),
        ):
            f = _build_layer(ratio, prefix, extend, self.CP, rank, seed=11 + rank)
            with forward_context(ForwardContext(attn_backend=f.backend)):
                f.backend.init_forward_metadata(f.batch)
                md = f.backend.forward_metadata.core_metadata
                self.assertTrue(
                    torch.equal(md.positions_casual[: f.n_real].long(), f.loc_pos),
                    "metadata rows are not this rank's interleaved rows",
                )
                raw = _seed_topk(f) if ratio in (1, 2) else None
                o_dec = _run(f, sparse=False)
                self.assertIsNone(f.backend.forward_metadata.sparse_prefill_cache)
                o_sp = _run(f, sparse=True)
                self.assertIsNotNone(
                    f.backend.forward_metadata.sparse_prefill_cache,
                    "the CP layer did not take the sparse prefill kernel",
                )
                o_shared_fill = _run(f, sparse=True, shared=True)
                o_shared_reuse = _run(f, sparse=True, shared=True)
                o_fast = _run(f, sparse=True, fast=True)
                ref = _reference(f, raw)

        n = f.n_real
        for o in (o_dec, o_sp):
            err = (o[:n].float() - ref).norm() / ref.norm()
            self.assertLess(float(err), 4e-3)  # bf16 output rounding
        torch.testing.assert_close(o_sp[:n], o_dec[:n], atol=1.6e-2, rtol=1.6e-2)
        self.assertTrue(torch.equal(o_shared_fill[:n], o_sp[:n]))
        self.assertTrue(torch.equal(o_shared_reuse[:n], o_sp[:n]))
        self.assertTrue(torch.equal(o_fast[:n], o_sp[:n]))

    def test_cp_sparse_matches_decode_kernel_and_reference(self):
        for ratio in (2, 1, 0):
            for prefix, extend in self.SHAPES:
                for rank in range(self.CP):
                    with self.subTest(ratio=ratio, extend=extend, rank=rank):
                        self._check(ratio, prefix, extend, rank)


class TestCPSparsePrefillGate(CustomTestCase):
    def test_gate(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        for cp_active in (False, True):
            for switch in (False, True):
                for min_rows in (0, 4096):
                    for rows in (0, 1728, 4096, 16304):
                        with (
                            patch(
                                "sglang.srt.layers.attention.deepseek_v4_backend.is_cp_active",
                                return_value=cp_active,
                            ),
                            patch(
                                "sglang.srt.layers.attention.deepseek_v4_backend.get_platform",
                                return_value=SimpleNamespace(is_sm120=False),
                            ),
                            envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(True),
                            envs.SGLANG_OPT_DSV4_CP_SPARSE_PREFILL.override(switch),
                            envs.SGLANG_OPT_DSV4_CP_SPARSE_PREFILL_MIN_ROWS.override(
                                min_rows
                            ),
                        ):
                            got = DeepseekV4AttnBackend._use_sparse_prefill(
                                None, batch, num_qo_tokens=rows
                            )
                        want = not cp_active or (
                            switch and rows > 0 and rows >= min_rows
                        )
                        self.assertEqual(got, want, (cp_active, switch, min_rows, rows))


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestBitIdenticalKernels(CustomTestCase):
    def test_rope_wide(self):
        from sglang.kernels.ops.attention.dsv4.elementwise import fused_rope_inplace

        torch.manual_seed(0)
        freqs = torch.polar(
            torch.ones(1 << 17, 32, device="cuda"),
            torch.rand(1 << 17, 32, device="cuda") * 6.2831853,
        )
        for rows, heads in ((1, 64), (300, 64), (4099, 64), (777, 16), (65, 128)):
            for inverse in (False, True):
                for pos_dtype in (torch.int32, torch.int64):
                    base = torch.randn(
                        rows, heads, 512, dtype=torch.bfloat16, device="cuda"
                    )
                    pos = torch.randint(
                        0, 1 << 17, (rows,), device="cuda", dtype=pos_dtype
                    )
                    a, b = base.clone(), base.clone()
                    with envs.SGLANG_OPT_DSV4_ROPE_WIDE.override(False):
                        fused_rope_inplace(
                            a[..., -64:], None, freqs, pos, inverse=inverse
                        )
                    with envs.SGLANG_OPT_DSV4_ROPE_WIDE.override(True):
                        fused_rope_inplace(
                            b[..., -64:], None, freqs, pos, inverse=inverse
                        )
                    self.assertTrue(
                        torch.equal(a.view(torch.int16), b.view(torch.int16))
                    )
                    self.assertTrue(torch.equal(b[..., :-64], base[..., :-64]))

    def test_fast_dequant(self):
        from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
            NOPE_ROPE_BYTES,
            PADDED_SCALE_PER_TOKEN,
            dequantize_k_cache_paged,
        )

        torch.manual_seed(0)
        bpp = (PAGE * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN) + 575) // 576 * 576
        buf = torch.randint(0, 256, (64, bpp), dtype=torch.uint8, device="cuda")
        nope = buf[:, : PAGE * NOPE_ROPE_BYTES].view(64, PAGE, NOPE_ROPE_BYTES)[
            :, :, :448
        ]
        nope[(nope & 0x7F) == 0x7F] = 0x3C
        scales = buf[:, PAGE * NOPE_ROPE_BYTES : PAGE * (NOPE_ROPE_BYTES + 8)]
        scales.copy_(
            torch.randint(110, 141, scales.shape, dtype=torch.uint8, device="cuda")
        )
        for n in (1, 7, 8, 9, 1000, 12345):
            ids = torch.randint(0, 64 * PAGE, (n,), device="cuda", dtype=torch.int32)
            with envs.SGLANG_OPT_DSV4_SPARSE_PREFILL_FAST_DEQUANT.override(False):
                ref = dequantize_k_cache_paged(buf, ids, page_size=PAGE)
            ws = torch.full((n + 3, 1, 512), 2.0, dtype=torch.bfloat16, device="cuda")
            with envs.SGLANG_OPT_DSV4_SPARSE_PREFILL_FAST_DEQUANT.override(True):
                dequantize_k_cache_paged(buf, ids, page_size=PAGE, out=ws[3:])
            self.assertTrue(
                torch.equal(ws[3:].view(torch.int16), ref.view(torch.int16))
            )
            self.assertTrue(bool((ws[:3] == 2.0).all()))


if __name__ == "__main__":
    unittest.main()
