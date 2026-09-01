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
    def _names_from(tree, source):
        """Locals assigned from ``<anything>.<source>``, call or attribute.

        `source` names the method or attribute on the right-hand side, e.g.
        ``verify_epilogue``. Read structurally so a rename of the local does not
        silently empty the set the assertions rest on.
        """
        out = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if isinstance(value, ast.Call):
                value = value.func
            if isinstance(value, ast.Attribute) and value.attr == source:
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

    def test_the_in_graph_mask_is_wired_to_the_verify_epilogue(self):
        """The half of the INF-414 mask that no test reached.

        Steps that replay the verify cuda graph are masked inside it: the worker
        pushes ``folded_mask_flags`` and the forbidden set into the verify
        epilogue's ``set_fsm_rows``. Delete that call, or empty either argument,
        and every reasoning row on the in-graph path goes unmasked while the FSM
        suite and the wiring test above stay green.

        Both arguments are checked, and the receiver with them. Passing
        ``folded_mask_flags`` while dropping ``CFG.reasoning_forbidden`` leaves
        the wiring intact and the mask inert, and a second call on some other
        path would otherwise satisfy a module-wide search after the real one was
        removed.

        **The eager path is deliberately not asserted here.** Its invariant --
        a step on which the FSM is active is never left unmasked -- is about
        which condition decides, and every structural form of that question
        tried so far was wrong in one direction or the other. Judging each
        condition separately passes a gate moved into an early ``return`` or a
        ternary, and fails the correct if/elif split this worker's comment at
        :695-697 describes. Judging their union fixes both and then passes
        ``if _solar_fsm_on and _solar_fsm_gate:`` -- INF-414 in full, with the
        activity name present. Requiring that no conjunct be a bare gate name
        catches that and brings the if/elif false positive back. A test that
        passes the defect while claiming to pin it is worse than no test, so
        what is asserted here is only what can be asserted soundly. The eager
        path's behaviour is covered at the FSM level by
        ``test/registered/unit/sampling/test_solar_open2_fsm_mask_gate.py``, and
        the defective tree is caught by this test regardless, because it has no
        ``set_fsm_rows`` call at all.
        """
        tree = _parse(self.WORKER)
        alias = _imported_alias(tree, "sglang.srt.sampling", "solar_open2_fsm")
        self.assertIsNotNone(
            alias, "dspark_worker_v2.py does not import solar_open2_fsm"
        )

        def _is_call_on(node, receiver, method):
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == method
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == receiver
            )

        epilogues = self._names_from(tree, "verify_epilogue")
        self.assertTrue(
            epilogues,
            "no local is assigned from a .verify_epilogue attribute; the in-graph "
            "mask has nowhere to be staged",
        )

        wired = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "set_fsm_rows"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in epilogues
            ):
                continue
            args = list(node.args) + [kw.value for kw in node.keywords]
            flags = any(_is_call_on(a, alias, "folded_mask_flags") for a in args)
            forbidden = any(
                isinstance(a, ast.Attribute) and a.attr == "reasoning_forbidden"
                for a in args
            )
            if flags and forbidden:
                wired.append(node)

        self.assertTrue(
            wired,
            "no set_fsm_rows(...) call on the verify epilogue passes both "
            f"{alias}.folded_mask_flags(...) and a reasoning_forbidden set; the "
            "in-graph verify path builds no reasoning mask, or builds one that "
            "forbids nothing",
        )


if __name__ == "__main__":
    unittest.main()
