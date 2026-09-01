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

    def test_the_eager_mask_covers_every_step_the_in_graph_mask_does_not(self):
        """INF-414, pinned semantically rather than syntactically.

        ``plan_gate`` answers "is this step within 2*stride of the reasoning
        budget?", which is what the folded-accept escape needs. The call site
        read it a second time to decide whether to call ``plan_verify`` at all,
        so outside that window no row was masked, the EOS ids the mask forbids
        stayed live, and the model could end its turn inside the think block.

        The invariant is a proposition, so it is decided by truth table and not
        by the shape of the expression: over every assignment of the guard's
        free names, **FSM active and not masked in-graph must imply the guard
        holds**. Nothing here inspects how the condition is written, which is
        what makes it both sound and refactor-tolerant -- an equivalent rewrite
        passes and a narrowing one fails, and that is the only distinction that
        matters. Name-mention heuristics over the same expression were tried
        first and are not sound: `_solar_fsm_on and _solar_fsm_gate` names the
        activity local, passes every such rule, and is INF-414 in full.

        The one thing it assumes is a single guard site, so the count is
        asserted too and a split into several branches fails loudly here rather
        than passing silently.
        """
        tree = _parse(self.WORKER)
        alias = _imported_alias(tree, "sglang.srt.sampling", "solar_open2_fsm")
        self.assertIsNotNone(
            alias, "dspark_worker_v2.py does not import solar_open2_fsm"
        )

        activity = self._names_from(tree, "is_active")
        self.assertTrue(
            activity,
            f"no local is assigned from {alias}.is_active(); the eager mask "
            "cannot know whether the FSM is active",
        )

        guards = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and any(
                isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and c.func.attr == "plan_verify"
                for c in ast.walk(n)
            )
            and {x.id for x in ast.walk(n.test) if isinstance(x, ast.Name)} & activity
        ]
        self.assertEqual(
            len(guards),
            1,
            "expected exactly one guard on plan_verify that consults FSM "
            f"activity, found {len(guards)}; with several the implication below "
            "has to be checked over their disjunction",
        )

        test = guards[0].test
        names = sorted({n.id for n in ast.walk(test) if isinstance(n, ast.Name)})
        in_graph = {n for n in names if "in_graph" in n}
        self.assertTrue(
            in_graph,
            f"the guard {ast.unparse(test)!r} names no in-graph flag, so this "
            "test cannot tell which steps the epilogue already covers",
        )
        code = compile(ast.Expression(test), "<guard>", "eval")

        counterexamples = []
        for bits in range(1 << len(names)):
            env = {n: bool(bits >> i & 1) for i, n in enumerate(names)}
            active = any(env[n] for n in names if n in activity)
            covered = any(env[n] for n in in_graph)
            if active and not covered and not eval(code, {}, dict(env)):
                counterexamples.append(env)
        self.assertFalse(
            counterexamples,
            f"guard {ast.unparse(test)!r} is false on {counterexamples[:3]}, "
            "where the FSM is active and the in-graph mask does not apply -- "
            "those steps run unmasked, which is INF-414",
        )

    def test_the_in_graph_mask_is_wired_to_the_verify_epilogue(self):
        """The half of the INF-414 mask that no test reached.

        Steps that replay the verify cuda graph are masked inside it: the worker
        stages ``folded_mask_flags`` and the forbidden set into the verify
        epilogue's ``set_fsm_rows``, before the target verify launch that
        consumes them. Delete that call, empty either argument, or move it after
        the launch, and every reasoning row on the in-graph path goes unmasked
        while the FSM suite stays green.

        Both arguments are checked with their receivers, and the staging call is
        required to sit in the same function as ``plan_verify`` and ahead of it,
        which is what rules out a call parked somewhere it never runs.

        **What this cannot see.** It reads structure, not reachability: a call
        wrapped in ``if False:`` in the right place still passes. And the
        forbidden argument is matched on the attribute name, so pointing it at
        some other object's ``reasoning_forbidden`` passes while an empty
        literal does not. Scoping the search to the function that calls
        ``plan_verify`` is what rules out a decoy elsewhere in the module --
        the receiver alone does not, since more than one local in this file is
        assigned from ``.verify_epilogue``.

        The eager path is asserted separately, by truth table, in the test
        above.
        """
        tree = _parse(self.WORKER)
        alias = _imported_alias(tree, "sglang.srt.sampling", "solar_open2_fsm")
        self.assertIsNotNone(
            alias, "dspark_worker_v2.py does not import solar_open2_fsm"
        )

        def _call_to(node, method, receiver=None):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == method
            ):
                return False
            if receiver is None:
                return True
            value = node.func.value
            if isinstance(value, ast.Name):
                return value.id in receiver
            # `self._verify_executor.verify_epilogue.set_fsm_rows(...)` with no
            # local is the same wiring written inline, so accept the attribute
            # the locals are themselves assigned from.
            return isinstance(value, ast.Attribute) and value.attr == "verify_epilogue"

        epilogues = self._names_from(tree, "verify_epilogue")
        self.assertTrue(
            epilogues,
            "no local is assigned from a .verify_epilogue attribute; the "
            "in-graph mask has nowhere to be staged",
        )

        host = None
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                _call_to(n, "plan_verify") for n in ast.walk(fn)
            ):
                host = fn
                break
        self.assertIsNotNone(
            host, f"no function calls {alias}.plan_verify; the verify path is gone"
        )
        launch = min(n.lineno for n in ast.walk(host) if _call_to(n, "plan_verify"))

        wired = []
        for node in ast.walk(host):
            if not _call_to(node, "set_fsm_rows", epilogues):
                continue
            args = list(node.args) + [kw.value for kw in node.keywords]
            flags = any(_call_to(a, "folded_mask_flags", {alias}) for a in args)
            forbidden = any(
                isinstance(a, ast.Attribute) and a.attr == "reasoning_forbidden"
                for a in args
            )
            if flags and forbidden and node.lineno < launch:
                wired.append(node)

        self.assertTrue(
            wired,
            "no set_fsm_rows(...) on the verify epilogue, in the function that "
            f"calls plan_verify and ahead of it (line {launch}), passes both "
            f"{alias}.folded_mask_flags(...) and a reasoning_forbidden set; the "
            "in-graph verify path builds no reasoning mask, builds one that "
            "forbids nothing, or stages it after the launch that reads it",
        )


if __name__ == "__main__":
    unittest.main()
