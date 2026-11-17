"""CPU-only byte tests for SGLANG_OPT_INF834_EP_DIRECT_EXPERTS (no server, no GPU).

The direct reader must leave every expert parameter of an EP rank byte-identical to what
FusedMoE.weight_loader leaves when the stream hands it the same checkpoint tensors, with the
DeepSeek-V4 name remap and expert mapping, for the EP8 decoder's shape of the problem:
384 logical experts, 416 physical slots (32 redundant copies, as eplb32), 52 slots per rank.
The placement comes from INF834_PLACEMENT (the real eplb32_at_a.json) when that file exists,
otherwise from a synthetic one with the same counts.
"""

import json
import os
import random
import struct
import tempfile
import types
import unittest
from unittest import mock

import torch
from torch import nn

from sglang.srt.layers.moe.fused_moe_triton.layer import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)
from sglang.srt.model_loader import inf834_ep_direct as epd
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

E8M0 = torch.float8_e8m0fnu
INTER, HID_BYTES, HID, INTER_BYTES = 16, 12, 20, 8
SCALE_COLS_13, SCALE_COLS_2 = 3, 2
N_LOGICAL, N_PHYSICAL, EP = 384, 416, 8
N_LOCAL = N_PHYSICAL // EP
PLACEMENT = os.environ.get("INF834_PLACEMENT", "/floor-dec/placements/eplb32_at_a.json")


class Mxfp4MarlinMoEMethod:  # the class name is what the plain-path check looks at
    pass


def make_rank_experts(layer_id, rank, tp_size=1):
    e = object.__new__(FusedMoE)
    nn.Module.__init__(e)
    e.layer_id = layer_id
    e.uses_expert_location = True
    e.num_fused_shared_experts = 0
    e._has_fused_shared = False
    e._num_global_routed = N_PHYSICAL
    e._expert_storage_rank = rank
    e._num_local_routed = N_LOCAL
    e.num_local_experts = N_LOCAL
    e.moe_tp_size, e.moe_tp_rank = tp_size, 0
    e.moe_ep_size = EP
    e.use_triton_kernels = False
    e.use_flashinfer_trtllm_moe = False
    e.use_presharded_weights = False
    e.__dict__["use_padded_loading"] = False
    e.scheme = None
    e.quant_method = Mxfp4MarlinMoEMethod()
    e.quant_config = None
    e.moe_runner_config = types.SimpleNamespace(is_gated=True)
    inter = INTER // tp_size
    params = {
        "w13_weight": torch.zeros(N_LOCAL, 2 * inter, HID_BYTES, dtype=torch.int8),
        "w2_weight": torch.zeros(N_LOCAL, HID, INTER_BYTES // tp_size, dtype=torch.int8),
        "w13_weight_scale_inv": torch.zeros(N_LOCAL, 2 * inter, SCALE_COLS_13, dtype=torch.uint8).view(E8M0),
        "w2_weight_scale_inv": torch.zeros(N_LOCAL, HID, SCALE_COLS_2 * 4 // tp_size, dtype=torch.uint8).view(E8M0),
    }
    for name, value in params.items():
        p = nn.Parameter(value, requires_grad=False)
        p.weight_loader = e.weight_loader
        if "scale" in name:
            p.quant_method = FusedMoeWeightScaleSupported.BLOCK.value
        e.register_parameter(name, p)
    return e


class Meta:
    def __init__(self, p2l):
        self.p2l = p2l
        self.num_logical_experts = N_LOGICAL

    def logical_to_all_physical(self, layer_id, logical_id, require_global=False):
        return [p for p, l in enumerate(self.p2l[layer_id]) if l == logical_id]


def placement(layers):
    if os.path.exists(PLACEMENT):
        p2l = json.load(open(PLACEMENT))["physical_to_logical_map"]
        moe = [row for row in p2l if len(row) == N_PHYSICAL]
        return {layer: moe[i % len(moe)] for i, layer in enumerate(layers)}, "eplb32_at_a.json"
    rng = random.Random(7)
    out = {}
    for layer in layers:
        row = list(range(N_LOGICAL)) + [rng.randrange(N_LOGICAL) for _ in range(N_PHYSICAL - N_LOGICAL)]
        rng.shuffle(row)
        out[layer] = row
    return out, "synthetic"


_ST = {torch.int8: "I8", torch.uint8: "U8", E8M0: "F8_E8M0"}


def write_shards(folder, named, shards=3):
    """A sharded safetensors checkpoint with an index, written by hand."""
    names = sorted(named)
    weight_map = {}
    for i in range(shards):
        part = names[i::shards]
        header, blobs, pos = {}, [], 0
        for n in part:
            t = named[n]
            raw = t.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
            header[n] = {"dtype": _ST[t.dtype], "shape": list(t.shape), "data_offsets": [pos, pos + len(raw)]}
            blobs.append(raw)
            pos += len(raw)
        text = json.dumps(header).encode()
        text += b" " * (-len(text) % 8)
        fn = f"model-{i:05d}-of-{shards:05d}.safetensors"
        with open(os.path.join(folder, fn), "wb") as f:
            f.write(struct.pack("<Q", len(text)))
            f.write(text)
            for b in blobs:
                f.write(b)
        for n in part:
            weight_map[n] = fn
    with open(os.path.join(folder, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)


def checkpoint(layers, seed=11):
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape):
        return torch.randint(0, 255, shape, generator=gen, dtype=torch.uint8)

    named = {}
    for layer in layers:
        for e in range(N_LOGICAL):
            pre = f"layers.{layer}.ffn.experts.{e}"
            named[f"{pre}.w1.weight"] = rnd(INTER, HID_BYTES).view(torch.int8)
            named[f"{pre}.w3.weight"] = rnd(INTER, HID_BYTES).view(torch.int8)
            named[f"{pre}.w2.weight"] = rnd(HID, INTER_BYTES).view(torch.int8)
            named[f"{pre}.w1.scale"] = rnd(INTER, SCALE_COLS_13).view(E8M0)
            named[f"{pre}.w3.scale"] = rnd(INTER, SCALE_COLS_13).view(E8M0)
            named[f"{pre}.w2.scale"] = rnd(HID, SCALE_COLS_2 * 4).view(E8M0)
    return named


def model_params(layers, rank, tp_size=1):
    experts = {layer: make_rank_experts(layer, rank, tp_size) for layer in layers}
    params = {}
    for layer, e in experts.items():
        for name, p in e.named_parameters():
            params[f"model.layers.{layer}.mlp.experts.{name}"] = p
    return experts, params


def resolver(params):
    mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=N_LOGICAL,
    )
    return epd.make_resolver(
        lambda n: DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format(n, is_nextn=False, num_hidden_layers=64),
        mapping,
        params,
    )


def state(experts):
    return {
        f"{layer}.{k}": v.data.view(torch.uint8).clone()
        for layer, e in experts.items()
        for k, v in e.named_parameters()
    }


class TestEpDirectExperts(CustomTestCase):
    LAYERS = (0, 5)

    def _run(self, rank, tp_size=1, p2l=None, threads=4, block=16, drop_slot=None):
        from sglang.srt.model_loader.inf834_selective_read import expert_consumed

        p2l = p2l or placement(self.LAYERS)[0]
        meta = Meta(p2l)
        named = checkpoint(self.LAYERS)
        targets = (
            "sglang.srt.layers.moe.fused_moe_triton.layer.get_global_expert_location_metadata",
            "sglang.srt.eplb.expert_location.get_global_expert_location_metadata",
        )
        with tempfile.TemporaryDirectory() as folder, mock.patch(targets[0], lambda: meta), mock.patch(
            targets[1], lambda: meta
        ):
            write_shards(folder, named)
            # Reference: the stream order and FusedMoE.weight_loader for every tensor.
            want, params_w = model_params(self.LAYERS, rank, tp_size)
            resolve_w = resolver(params_w)
            for name in sorted(named):
                r = resolve_w(name)
                self.assertIsNotNone(r, name)
                param, resolved, shard_id, expert_id, _ = r
                param.weight_loader(param, named[name], resolved, shard_id=shard_id, expert_id=expert_id)
            # Direct reader, with the filter's consumed sets.
            got, params_g = model_params(self.LAYERS, rank, tp_size)
            wanted = {
                layer: {e for e in range(N_LOGICAL) if expert_consumed(got[layer], meta, e)}
                for layer in self.LAYERS
            }
            if drop_slot is not None:  # a slot no checkpoint expert fills
                layer, e = drop_slot
                wanted[layer].discard(e)
            sources = epd.list_sources(folder, wanted)
            d = epd.Inf834EpDirectExperts(sources, resolver(params_g), threads=threads, block_slots=block)
            d.launch()
            d.join("test")
            if drop_slot is not None:
                # the reference with that expert left out
                want, params_w = model_params(self.LAYERS, rank, tp_size)
                resolve_w = resolver(params_w)
                for name in sorted(named):
                    if name.startswith(f"layers.{drop_slot[0]}.ffn.experts.{drop_slot[1]}."):
                        continue
                    param, resolved, shard_id, expert_id, _ = resolve_w(name)
                    param.weight_loader(param, named[name], resolved, shard_id=shard_id, expert_id=expert_id)
        a, b = state(want), state(got)
        self.assertEqual(list(a), list(b))
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)
        return d, wanted

    def test_every_rank_byte_identical(self):
        p2l, source = placement(self.LAYERS)
        for rank in range(EP):
            d, wanted = self._run(rank, p2l=p2l)
            self.assertEqual(d.tensors_fallback, 0, rank)
            self.assertEqual(len(d.unresolved), 0)
            self.assertEqual(len(d.sources), sum(len(v) for v in wanted.values()) * 6)
            self.assertGreater(d.tasks_direct, 0)
        print(f"placement: {source}; ranks 0-{EP - 1} identical; per layer and rank logical experts "
              f"{sorted(len(v) for v in wanted.values())} for {N_LOCAL} slots")

    def test_redundant_copies_present(self):
        p2l, _ = placement(self.LAYERS)
        dup = sum(1 for layer in self.LAYERS for e in set(p2l[layer]) if p2l[layer].count(e) > 1)
        self.assertGreater(dup, 0)

    def test_block_and_threads_do_not_matter(self):
        for threads, block in ((1, 1), (3, 7), (8, 64)):
            self._run(3, threads=threads, block=block)

    def test_unfilled_slot_falls_back_whole_parameter(self):
        p2l, _ = placement(self.LAYERS)
        rank = 2
        lo = rank * N_LOCAL
        e = p2l[5][lo]  # the logical expert in this rank's first slot of layer 5
        if p2l[5].count(e) > 1:
            self.skipTest("first slot holds a redundant expert")
        d, _ = self._run(rank, p2l=p2l, drop_slot=(5, e))
        self.assertGreater(d.tensors_fallback, 0)

    def test_tp_greater_than_one_falls_back(self):
        d, _ = self._run(1, tp_size=2)
        self.assertEqual(d.tasks_direct, 0)
        self.assertGreater(d.tensors_fallback, 0)


if __name__ == "__main__":
    unittest.main()
