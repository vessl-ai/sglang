"""The Solar-Open2 serving contract, asserted in the tree instead of an image.

These three axes were once checked by a gate script baked into a serving image
as its ``ENTRYPOINT``. That put a model's contract into a container layer, where
it could only ever be checked for the one model the image was built for, and
where a caller who overrode ``command`` skipped it entirely. The checks belong
here: they are facts about this tree, so this is where they can be checked for
every model at once and where no deployment can opt out.

Each axis is a silent-failure mode -- the engine boots, serves and returns 200s
whether or not it holds:

* registration -- an arch that fails to import is not registered, and
  ``import_model_classes`` swallows the import error (``strict=False``), so the
  first sign is a model-load failure far from the cause.
* KDA beta scale -- the scale has to be applied wherever the sigmoid is, and
  that is three sites. Missing only the packed-decode site produces a cell that
  scores like the unscaled-beta regime with every other knob correct.
* FSM verify path -- DSpark's verify never goes through ``layers/sampler.py``,
  where the FSM hook normally lives, so a tree with the FSM but without the
  verify-path hook silently ignores the reasoning budget.

A fourth axis, the router's output dtype, is not from that gate script -- it is
added here for the same reason: it is a fact about this tree that no deployment
can opt out of.

The wiring axes are read structurally with ``ast`` rather than by importing
the DSpark worker or matching source substrings: a substring is true of a
comment, while a parsed keyword argument or call is the wiring itself.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import ast
import os
import unittest
from pathlib import Path

import sglang.srt
from sglang.test.test_utils import CustomTestCase

_BETA_SCALE_ENV = "SOLAR_KDA_BETA_SCALE"
_BETA_SCALE_CONST = "_SOLAR_KDA_BETA_SCALE"
_SRT_ROOT = Path(next(iter(sglang.srt.__path__)))


def _parse(path):
    return ast.parse(Path(path).read_text())


def _passes_beta_scale_to_a_kernel(tree):
    """Is the module constant handed to a kernel as its ``BETA_SCALE`` arg?"""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if (
                keyword.arg == "BETA_SCALE"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == _BETA_SCALE_CONST
            ):
                return True
    return False


def _multiplies_by_beta_scale(tree):
    """Is the module constant applied as a multiply in Python?"""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
            continue
        for side in (node.left, node.right):
            if isinstance(side, ast.Name) and side.id == _BETA_SCALE_CONST:
                return True
    return False


def _imported_alias(tree, module, name):
    """The local name a ``from <module> import <name> [as alias]`` bound."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            for alias in node.names:
                if alias.name == name:
                    return alias.asname or alias.name
    return None


def _methods_called_on(tree, receiver):
    """Attribute names called on ``receiver`` anywhere in the module."""
    called = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == receiver
        ):
            called.add(node.func.attr)
    return called


class TestSolarOpen2Registration(CustomTestCase):
    """Gate: this tree really carries Solar-Open2, and registration ran.

    Every assertion here imports the thing it is about, so it proves the
    registration executed rather than that a line of source mentioning it
    exists.
    """

    def test_config_type_is_registered(self):
        from sglang.srt.configs.solar_open2 import SolarOpen2Config
        from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY

        self.assertIs(_CONFIG_REGISTRY.get("solar_open2"), SolarOpen2Config)

    def test_model_module_imports_and_declares_its_entry_class(self):
        # A clean import is the axis: import_model_classes() walks srt/models,
        # imports each module and reads EntryClass, but swallows import errors
        # unless strict=True -- so a module that raises is simply absent from
        # the registry with nothing but a log line to say so.
        from sglang.srt.models import solar_open2 as solar_open2_model

        entry = solar_open2_model.EntryClass
        entries = entry if isinstance(entry, list) else [entry]
        self.assertIn(
            "SolarOpen2ForCausalLM",
            [cls.__name__ for cls in entries],
        )

    def test_tool_call_parser_is_registered(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        from sglang.srt.function_call.solar_open2_detector import SolarOpen2Detector

        self.assertIs(
            FunctionCallParser.ToolCallParserEnum.get("solar_open2"),
            SolarOpen2Detector,
        )

    def test_reasoning_parser_is_registered(self):
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        self.assertIn("solar_open2", ReasoningParser.DetectorMap)


class TestKdaBetaScaleWiring(CustomTestCase):
    """Gate: the KDA beta scale is read the same way at all three sites.

    The scale multiplies the sigmoid that produces ``beta``, and there are three
    places that sigmoid is taken: the prefill/extend multiply in Python, the
    decode kernel, and the packed-decode kernel. A site that reads a different
    env key, or defaults differently, or never passes the value into its kernel,
    leaves that path on the unscaled-beta accuracy defect while the other two
    are correct.
    """

    def _sites(self):
        from sglang.kernels.ops.attention.fla import (
            fused_recurrent,
            fused_sigmoid_gating_recurrent,
        )
        from sglang.srt.models import kimi_linear

        return (kimi_linear, fused_sigmoid_gating_recurrent, fused_recurrent)

    def test_all_three_sites_define_the_constant(self):
        for module in self._sites():
            with self.subTest(module=module.__name__):
                self.assertTrue(
                    hasattr(module, _BETA_SCALE_CONST),
                    f"{module.__name__} does not define {_BETA_SCALE_CONST}",
                )

    def test_all_three_sites_resolve_the_same_value(self):
        # Env-relative on purpose: what matters is that the three agree and
        # track one key, not which value this runner happens to have set.
        expected = float(os.environ.get(_BETA_SCALE_ENV, "1.0"))
        for module in self._sites():
            with self.subTest(module=module.__name__):
                self.assertEqual(getattr(module, _BETA_SCALE_CONST), expected)

    def test_the_python_site_applies_it_as_a_multiply(self):
        from sglang.srt.models import kimi_linear

        self.assertTrue(
            _multiplies_by_beta_scale(_parse(kimi_linear.__file__)),
            f"kimi_linear.py never multiplies by {_BETA_SCALE_CONST}",
        )

    def test_both_kernel_sites_pass_it_into_the_kernel(self):
        from sglang.kernels.ops.attention.fla import (
            fused_recurrent,
            fused_sigmoid_gating_recurrent,
        )

        for module in (fused_sigmoid_gating_recurrent, fused_recurrent):
            with self.subTest(module=module.__name__):
                self.assertTrue(
                    _passes_beta_scale_to_a_kernel(_parse(module.__file__)),
                    f"{module.__name__} never passes "
                    f"BETA_SCALE={_BETA_SCALE_CONST} to a kernel",
                )


class TestFsmWiredIntoDsparkVerify(CustomTestCase):
    """Gate: DSpark's verify path consults the reasoning-budget FSM.

    DSpark verify does not go through ``layers/sampler.py``, so the sampler hook
    that enforces the reasoning budget everywhere else does not cover it. Read
    structurally: importing the worker would pull in the CUDA-graph machinery
    this suite has no device for.
    """

    WORKER = _SRT_ROOT / "speculative" / "dspark_components" / "dspark_worker_v2.py"

    def test_the_worker_source_is_where_we_think_it_is(self):
        self.assertTrue(self.WORKER.is_file(), f"{self.WORKER} is missing")

    def test_the_worker_imports_the_fsm_and_calls_both_hooks(self):
        tree = _parse(self.WORKER)
        alias = _imported_alias(tree, "sglang.srt.sampling", "solar_open2_fsm")
        self.assertIsNotNone(
            alias,
            "dspark_worker_v2.py does not import solar_open2_fsm; the reasoning "
            "budget is unenforced on the DSpark verify path",
        )
        called = _methods_called_on(tree, alias)
        # plan_gate decides before the target launch whether the folded in-graph
        # accept path has to be left; plan_verify builds the mask itself. One
        # without the other is a budget that is either never enforced or
        # enforced into a buffer nothing reads.
        for hook in ("plan_gate", "plan_verify"):
            with self.subTest(hook=hook):
                self.assertIn(
                    hook,
                    called,
                    f"dspark_worker_v2.py never calls {alias}.{hook}()",
                )


class TestRouterLogitsAreFloat32(CustomTestCase):
    """The router GEMM emits float32 logits, and the ranking bias is float32.

    A fourth silent-failure axis: bf16 here serves 200s and scores plausibly.
    `moe_fused_gate` already upcasts and ranks in fp32, so the only precision
    actually lost is the rounding of this GEMM's output -- which is why the fix
    is `out_dtype`, not fp32 parameters.

    The GEMM is checked by running the real `forward`, not by reading it: an
    AST test here was loose where it mattered (a dead fp32 `mm` inside a live
    helper passed) and tight where it did not (it failed on a keyword argument,
    a type annotation, or a module-level constant). The bias is set in
    `__init__`, which needs a distributed process group, so it stays structural
    -- scoped to this class, and rejecting a downcast applied to the result.
    """

    MODEL = _SRT_ROOT / "models" / "solar_open2.py"

    def test_the_routing_logits_handed_to_topk_are_float32(self):
        # Unbound call on a stand-in `self`: `__init__` would need a process
        # group. `torch.mm` is stubbed so the test needs no GPU and does not
        # depend on which builds implement the `out_dtype` kwarg.
        # ponytail: hoisting the GEMM into a helper means adding that method to
        # the stand-in too -- the failure is an AttributeError, not a silent pass.
        import unittest.mock
        from types import SimpleNamespace

        import torch

        from sglang.srt.models.solar_open2 import SolarOpen2MoE

        seen = {}

        def fake_mm(a, b, *, out_dtype=None):
            seen["out_dtype"] = out_dtype
            return torch.zeros(a.shape[0], b.shape[1], dtype=out_dtype or a.dtype)

        moe = SimpleNamespace(
            shared_experts=None,
            tp_size=1,
            gate=SimpleNamespace(weight=torch.zeros(3, 4, dtype=torch.bfloat16)),
            topk=lambda h, rl: seen.setdefault("routing_dtype", rl.dtype),
            experts=lambda h, topk_output: h,
        )
        with unittest.mock.patch("torch.mm", fake_mm):
            SolarOpen2MoE.forward(moe, torch.zeros(2, 4, dtype=torch.bfloat16))

        self.assertIs(
            seen.get("routing_dtype"),
            torch.float32,
            "the logits handed to self.topk must be float32; in bf16 the 8th/9th "
            "expert boundary is decided on 8 mantissa bits",
        )
        # A bf16 mm widened afterwards -- `minimax_m3.py:508` spells exactly
        # that for NPU -- reaches topk as float32 having already rounded away
        # what this change exists to keep.
        self.assertIs(
            seen.get("out_dtype"),
            torch.float32,
            "the GEMM itself must emit float32, not a bf16 mm widened after",
        )

    def test_the_ranking_bias_is_float32(self):
        # The bias is added to the score inside `moe_fused_gate`; storing it in
        # bf16 would round it into the tie window the fp32 logits avoid.
        cls = next(
            (
                n
                for n in ast.walk(_parse(self.MODEL))
                if isinstance(n, ast.ClassDef) and n.name == "SolarOpen2MoE"
            ),
            None,
        )
        self.assertIsNotNone(cls, "SolarOpen2MoE is gone or was renamed")
        assigns = [
            n
            for n in ast.walk(cls)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Attribute) and t.attr == "e_score_correction_bias"
                for t in n.targets
            )
        ]
        self.assertEqual(
            len(assigns), 1, "expected one e_score_correction_bias assignment"
        )
        alloc = [
            call
            for call in ast.walk(assigns[0])
            if isinstance(call, ast.Call)
            and any(kw.arg == "dtype" for kw in call.keywords)
        ]
        self.assertEqual(len(alloc), 1, "expected one dtype= allocation")
        self.assertTrue(
            all(
                isinstance(kw.value, ast.Attribute) and kw.value.attr == "float32"
                for kw in alloc[0].keywords
                if kw.arg == "dtype"
            ),
            "e_score_correction_bias must be allocated float32",
        )
        # A float32 allocation says nothing if the result is cast back, and
        # `.to(bf16)` is only one spelling of that -- `.bfloat16()` is another.
        # Pinning the call count instead of blacklisting names covers both.
        self.assertEqual(
            len([c for c in ast.walk(assigns[0]) if isinstance(c, ast.Call)]),
            2,
            "the bias must be exactly nn.Parameter(torch.empty(...)) with no "
            "call wrapped around it",
        )


if __name__ == "__main__":
    unittest.main()
