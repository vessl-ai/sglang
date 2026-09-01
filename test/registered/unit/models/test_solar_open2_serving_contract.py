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

The two wiring axes are read structurally with ``ast`` rather than by importing
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

    @staticmethod
    def _names_from(tree, alias, method):
        """Names assigned from ``<alias>.<method>()``."""
        out = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                continue
            func = node.value.func
            if isinstance(func, ast.Attribute) and func.attr == method:
                out.update(t.id for t in node.targets if isinstance(t, ast.Name))
        return out

    @staticmethod
    def _calls_to(node, method):
        return [
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == method
        ]

    @staticmethod
    def _enclosing_function(tree, node):
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if any(n is node for n in ast.walk(fn)):
                    return fn
        return None

    def _guard_tests(self, tree, method):
        """Every expression that can decide whether `method` runs.

        Four forms, because the defect is about which condition decides and a
        reader can write that condition four ways: an enclosing ``if``, a
        ternary, the left side of an ``and``, and -- the one that is easy to
        miss -- an early ``return`` above the call in the same function, which
        moves the decision out of the call's own guard without changing it.
        """
        calls = self._calls_to(tree, method)
        tests = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.IfExp)) and self._calls_to(node, method):
                tests.append(node.test)
            elif isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
                if any(self._calls_to(v, method) for v in node.values[1:]):
                    tests.append(node.values[0])
        for call in calls:
            fn = self._enclosing_function(tree, call)
            if fn is None:
                continue
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.If)
                    and any(isinstance(n, ast.Return) for n in ast.walk(node))
                    and not self._calls_to(node, method)
                    and node.lineno < call.lineno
                ):
                    tests.append(node.test)
        return tests

    def test_the_eager_mask_consults_fsm_activity_not_only_the_budget_window(self):
        """INF-414, pinned: one boolean was answering two questions.

        ``plan_gate`` answers "is this step within 2*stride of the reasoning
        budget?", which is what the folded-accept escape needs. The call site
        read it a second time to decide whether to call ``plan_verify`` at all,
        so outside that window no row was masked, the EOS ids the mask forbids
        stayed live, and the model could end its turn inside the think block.

        Asserted over the **union** of everything that can decide whether
        ``plan_verify`` runs, not over each condition separately. Separately is
        wrong in both directions: it passes a defect that moves the gate into an
        early return or a ternary, and it fails a correct tree that splits the
        folded-accept escape into its own ``if _solar_fsm_gate:`` branch with the
        mask fallback in an ``elif`` -- a shape this worker's own comment
        describes as legitimate. What must hold is only that activity is
        consulted somewhere among them; a tree that never mentions it decides
        the mask on the budget window alone.

        A wholly unguarded ``plan_verify`` passes and should: masking on every
        step is the safe direction.
        """
        tree = _parse(self.WORKER)
        alias = _imported_alias(tree, "sglang.srt.sampling", "solar_open2_fsm")
        self.assertIsNotNone(
            alias, "dspark_worker_v2.py does not import solar_open2_fsm"
        )

        activity = self._names_from(tree, alias, "is_active")
        self.assertTrue(
            activity,
            f"no local is assigned from {alias}.is_active(); the eager mask "
            "cannot be gated on whether the FSM is active",
        )

        tests = self._guard_tests(tree, "plan_verify")
        if not tests:
            return  # unguarded: masked on every step, which is the safe side
        mentioned = {
            n.id for t in tests for n in ast.walk(t) if isinstance(n, ast.Name)
        }
        self.assertTrue(
            mentioned & activity,
            "nothing that decides whether plan_verify runs consults "
            f"{sorted(activity)} -- "
            + "; ".join(f"line {t.lineno}: `{ast.unparse(t)}`" for t in tests)
            + ". A step on which the FSM is active can be left unmasked.",
        )

    def test_the_in_graph_mask_is_wired_to_the_verify_epilogue(self):
        """The other half of the mask, which no test reached.

        The eager ``plan_verify`` path covers steps that do not replay the
        verify cuda graph. Steps that do are masked inside the graph instead:
        the worker pushes ``folded_mask_flags`` into the epilogue's
        ``set_fsm_rows``. Delete that call and the guard above still passes, the
        wiring test above still passes, and every reasoning row on the in-graph
        path goes unmasked -- INF-414 in full, with the suite green.
        """
        tree = _parse(self.WORKER)
        alias = _imported_alias(tree, "sglang.srt.sampling", "solar_open2_fsm")
        self.assertIsNotNone(
            alias, "dspark_worker_v2.py does not import solar_open2_fsm"
        )

        wired = [
            call
            for call in self._calls_to(tree, "set_fsm_rows")
            if any(
                isinstance(arg, ast.Call)
                and isinstance(arg.func, ast.Attribute)
                and arg.func.attr == "folded_mask_flags"
                for arg in call.args
            )
        ]
        self.assertTrue(
            wired,
            "no set_fsm_rows(...) call takes folded_mask_flags(...) as an argument; "
            "the in-graph verify path builds no reasoning mask",
        )


if __name__ == "__main__":
    unittest.main()
