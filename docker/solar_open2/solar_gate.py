"""Boot gate for the Solar-Open2 W4AFP8 serving image.

Four preconditions, every one of which is a silent-failure mode: the engine
boots, serves, and returns 200s either way, and the only symptom is degraded
accuracy, an unenforced reasoning budget, or a dead speculative accept rate.
So they are asserted on every container start rather than documented and
hoped for.

  (0) the base image really is a build of this fork.
      The overlay takes BASE_IMAGE from the caller with no default, so the
      one mistake it cannot rule out by construction is being stacked on a
      stock upstream image. Nothing else in the boot path notices: the
      engine starts and then fails to resolve the ``solar_open2`` model
      type at load time, far from the cause.

  (1) SOLAR_KDA_BETA_SCALE == 2.0 AND wired at all THREE sites.
      The KDA beta scale has to be applied wherever the sigmoid is, which
      is three separate places: the prefill/extend Python multiply, the
      decode kernel, and the packed-decode kernel. Missing only the
      packed-decode site has happened: the cell booted, served, and scored
      like the unscaled-beta (beta_scale=1.0) regime while every other knob
      matched the intended configuration, and a whole round of speculative
      decoding measurements had to be thrown away.

  (2) draft config num_target_layers == the target's num_hidden_layers.
      SGLang reads this field as the TARGET model's layer count and
      validates target_layer_ids against it; a checkpoint published with
      vLLM semantics carries the tap COUNT here instead, which makes the
      draft model refuse to build. The fc width is len(target_layer_ids),
      so correcting this field does not change any weight shape.

  (3) every aux tap lands on a full-attention (GQA) layer.
      A tap on a KDA layer boots clean and only kills accept rate.

  (4) the FSM is wired into the DSpark verify path when spec is ON.
      ``layers/sampler.py``, where the FSM hook normally lives, is never
      reached by DSpark's verify, so a build that has the FSM but not the
      verify-path hook silently ignores the reasoning budget with spec ON.
      It boots and serves either way; only the output length differs.

Exit non-zero on any failure: a broken image must not serve.
"""

import json
import os
import sys


def fail(msg):
    print(f"[SOLAR-GATE] FAIL: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def sglang_root():
    import sglang

    return os.path.dirname(sglang.__file__)


def read_source(root, rel, label):
    path = os.path.join(root, rel)
    if not os.path.exists(path):
        fail(f"{label}: {rel} not found under {root}")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def arg_value(argv, name):
    """--name=value or --name value; returns None when absent."""
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def gate_base(root):
    """The installed sglang carries the Solar-Open2 sources and registration.

    This is the check that catches BASE_IMAGE pointing at a stock upstream
    image instead of one of our engine builds.
    """
    for rel in (
        "srt/configs/solar_open2.py",
        "srt/models/solar_open2.py",
        "srt/sampling/solar_open2_fsm.py",
        "srt/function_call/solar_open2_detector.py",
    ):
        if not os.path.exists(os.path.join(root, rel)):
            fail(
                f"base image: {rel} missing under {root}. BASE_IMAGE does not "
                "look like a build of this fork -- this overlay only adds the "
                "boot contract, it does not add the Solar-Open2 sources."
            )
    rel = "srt/utils/hf_transformers/common.py"
    src = read_source(root, rel, "base image")
    if '_CONFIG_REGISTRY["solar_open2"]' not in src:
        fail(
            f"base image: {rel} does not register the solar_open2 config. The "
            "engine would start and then fail to resolve the model type at "
            "load time."
        )
    print("[SOLAR-GATE] base image carries Solar-Open2 sources + config registration: OK")


def gate_beta(root):
    want = os.environ.get("SOLAR_KDA_BETA_SCALE")
    if want is None:
        fail(
            "SOLAR_KDA_BETA_SCALE is unset (image default is 2.0 -- something "
            "cleared it). The code default is 1.0, which reproduces the "
            "unscaled-beta accuracy defect."
        )
    if abs(float(want) - 2.0) > 1e-9:
        fail(
            f"SOLAR_KDA_BETA_SCALE={want!r}, the serving value is 2.0 (KDA "
            "accuracy fix). Override deliberately only for an A/B, never for "
            "serving."
        )

    sites = (
        (
            "kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py",
            (
                "BETA_SCALE: tl.constexpr",
                "b_beta = BETA_SCALE /",
                "BETA_SCALE=_SOLAR_KDA_BETA_SCALE",
            ),
            "site1 decode kernel",
        ),
        (
            "srt/models/kimi_linear.py",
            ("sigmoid() * _SOLAR_KDA_BETA_SCALE",),
            "site2 prefill/extend python",
        ),
        (
            "kernels/ops/attention/fla/fused_recurrent.py",
            (
                "BETA_SCALE: tl.constexpr = 1.0",
                "beta_val = (BETA_SCALE * tl.sigmoid(b_val))",
                "BETA_SCALE=_SOLAR_KDA_BETA_SCALE",
            ),
            "site3 packed-decode",
        ),
    )
    for rel, needles, label in sites:
        src = read_source(root, rel, label)
        for n in needles:
            if n not in src:
                fail(f"{label}: {rel} missing {n!r}")
        print(f"[SOLAR-GATE] beta {label}: OK")
    print(f"[SOLAR-GATE] beta 3/3 sites wired, SOLAR_KDA_BETA_SCALE={want}")


def gate_fsm_spec(root, argv):
    """FSM must be wired into the DSpark verify path when spec is ON."""
    if not arg_value(argv, "--speculative-draft-model-path"):
        print(
            "[SOLAR-GATE] no --speculative-draft-model-path -> spec OFF cell, "
            "skipping FSM verify-path gate (4)"
        )
        return
    if os.environ.get("SOLAR_FSM", "0") != "1":
        # spec ON with the FSM deliberately off is a legitimate A/B, so this
        # cannot be a hard failure. Say out loud that the gate did not run.
        print(
            "[SOLAR-GATE] SOLAR_FSM != 1 -> FSM verify-path gate (4) SKIPPED, "
            "the reasoning budget is NOT enforced in this configuration"
        )
        return
    rel = "srt/speculative/dspark_components/dspark_worker_v2.py"
    src = read_source(root, rel, "FSM verify gate")
    for needle in ("solar_open2_fsm", "_solar_fsm_plan"):
        if needle not in src:
            fail(
                f"FSM verify gate: {rel} missing {needle!r} -- with spec ON the "
                "reasoning budget would be silently unenforced."
            )
    print("[SOLAR-GATE] FSM wired into DSpark verify path: OK")


def gate_draft(argv):
    draft = arg_value(argv, "--speculative-draft-model-path")
    if not draft:
        print(
            "[SOLAR-GATE] no --speculative-draft-model-path -> spec OFF cell, "
            "skipping draft gates (2) and (3)"
        )
        return
    target = arg_value(argv, "--model-path")
    if not target:
        fail("--speculative-draft-model-path given without --model-path")

    with open(os.path.join(draft, "config.json"), encoding="utf-8") as fh:
        dcfg = json.load(fh)
    with open(os.path.join(target, "config.json"), encoding="utf-8") as fh:
        tcfg = json.load(fh)

    n_target = tcfg.get("num_hidden_layers")
    ntl = dcfg.get("num_target_layers")
    if ntl != n_target:
        fail(
            f"draft num_target_layers={ntl!r} but target num_hidden_layers="
            f"{n_target!r}. SGLang validates target_layer_ids against this "
            "field, and a checkpoint published with vLLM semantics carries the "
            "tap count here instead. Point --speculative-draft-model-path at a "
            "local copy whose config.json has the field set to the target's "
            "layer count."
        )

    taps = dcfg.get("target_layer_ids") or []
    dtaps = ((dcfg.get("dflash_config") or {}).get("target_layer_ids")) or []
    if dtaps and list(dtaps) != list(taps):
        fail(f"target_layer_ids {taps} != dflash_config.target_layer_ids {dtaps}")
    gqa = tcfg.get("gqa_layers")
    if gqa is None:
        fail("target config has no gqa_layers; cannot verify aux taps")
    bad = [t for t in taps if t not in gqa]
    if bad:
        fail(
            f"aux taps {bad} are not full-attention layers. A KDA tap boots "
            f"clean and only kills accept rate. gqa_layers={sorted(gqa)}"
        )
    print(f"[SOLAR-GATE] draft num_target_layers={ntl} == target layers OK")
    print(
        f"[SOLAR-GATE] aux taps {taps} all full-attention OK "
        f"(fc width = {len(taps)} x {tcfg.get('hidden_size')})"
    )


def main():
    argv = sys.argv[1:]
    root = sglang_root()
    print(f"[SOLAR-GATE] sglang root = {root}", flush=True)
    gate_base(root)
    gate_beta(root)
    gate_draft(argv)
    gate_fsm_spec(root, argv)
    print("[SOLAR-GATE] ALL PRECONDITIONS PASS", flush=True)


if __name__ == "__main__":
    main()
