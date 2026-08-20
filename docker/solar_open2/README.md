# docker/solar_open2 — Solar-Open2 serving image boot contract

The Solar-Open2 config, model, tool-call detector and reasoning-budget FSM are
native to this fork, so an engine image built from this tree already carries all
of the source. It does not carry the image's **boot contract**: the two ENV
defaults the serving configuration depends on, and a gate that proves the wiring
is intact before the engine starts. This directory is that boot contract,
packaged as a thin overlay on top of an engine image.

The overlay adds nothing else. It pins no upstream base image and installs no
transfer-engine wheel — which engine build and which client libraries end up in
the image is decided one layer up, by the build chain that produces
`BASE_IMAGE`.

## Files

| file | role |
|---|---|
| `Dockerfile` | the overlay: `COPY` the gate and entrypoint in, set the executable bit, set the two ENV defaults, run the gate as a build-time check, set `ENTRYPOINT` |
| `solar_gate.py` | the gate; runs at build time and again on every container start |
| `solar-entrypoint.sh` | `ENTRYPOINT`; defaults `SOLAR_FSM_TOKENIZER_DIR`, runs the gate against the real launch args, then `exec`s them |

## Environment

| variable | value | why |
|---|---|---|
| `SOLAR_KDA_BETA_SCALE` | `2.0` (baked) | KDA beta scale used for serving. The code default is `1.0`, which is the unscaled-beta accuracy defect regime. Load-bearing: the gate refuses to boot if this is unset or not `2.0`. Override only for a deliberate A/B, never for serving. |
| `SOLAR_FSM` | `1` (baked) | Enables the reasoning-budget FSM. Without it decode can fall into an exact repetition loop. |
| `SOLAR_FSM_TOKENIZER_DIR` | defaulted by the entrypoint | The FSM needs a tokenizer directory to resolve the `<\|think:start\|>` / `<\|think:end\|>` ids. The entrypoint defaults it to the value of `--model-path`; an explicit env wins. |

## Build

Build context is the repository root, like the other Dockerfiles under
`docker/`. `BASE_IMAGE` has no default — pin one of our own engine images **by
digest**, so the overlay cannot silently pick up a moved tag. An unset
`BASE_IMAGE` fails the build immediately on a blank base name.

BuildKit (`buildctl`, which is how these images are built):

```
buildctl build \
  --frontend dockerfile.v0 \
  --local context=. \
  --local dockerfile=docker/solar_open2 \
  --opt build-arg:BASE_IMAGE=<engine image pinned by digest> \
  --output type=image,name=<target>,push=true
```

`docker build` works too:

```
docker build -f docker/solar_open2/Dockerfile \
  --build-arg BASE_IMAGE=<engine image pinned by digest> \
  -t <target> .
```

The Dockerfile carries no `# syntax=` directive and no `RUN` heredocs. The
directive would make BuildKit fetch and run an external frontend container
instead of its built-in dockerfile frontend, which does not work in our build
shape; multi-line logic therefore lives in the `COPY`'d scripts and is invoked
as a one-liner.

## The gate

`solar_gate.py` asserts four things. All four are silent-failure modes: the
engine boots, serves and returns 200s either way, and the only symptom is
degraded accuracy, an unenforced reasoning budget, or a dead speculative accept
rate. It exits non-zero on any failure, so a broken image does not serve.

0. **The base image really is a build of this fork** — the Solar-Open2 config,
   model, FSM and detector sources are present and the config is registered.
   This is the check that catches `BASE_IMAGE` pointing at a stock upstream
   image.
1. **`SOLAR_KDA_BETA_SCALE == 2.0` and the beta scale is wired at all three
   sites** — the prefill/extend Python multiply, the decode kernel, and the
   packed-decode kernel. The scale has to be applied wherever the sigmoid is;
   missing only the packed-decode site produces a cell that scores like the
   unscaled-beta regime with every other knob correct.
2. **Draft `num_target_layers` equals the target's `num_hidden_layers`** —
   SGLang reads this field as the target's layer count and validates
   `target_layer_ids` against it. A checkpoint published with vLLM semantics
   carries the tap *count* here instead, and the draft model then refuses to
   build. The fc width is `len(target_layer_ids)`, so correcting the field
   changes no weight shape.
3. **Every aux tap lands on a full-attention (GQA) layer** — a tap on a KDA
   layer boots clean and only kills accept rate.
4. **The FSM is wired into the DSpark verify path when speculative decoding is
   on** — DSpark's verify never goes through `layers/sampler.py`, where the FSM
   hook normally lives, so a build with the FSM but without the verify-path hook
   silently ignores the reasoning budget.

Gates 2–4 need the speculative-decoding launch args, so they report themselves
as skipped during the build-time run and do their real work at container start.
Gate 4 is also skipped (loudly) when `SOLAR_FSM != 1`, since speculative
decoding with the FSM deliberately off is a legitimate A/B.

Callers must pass the engine command as `args:` and must **not** override
`command:` — overriding it bypasses the `ENTRYPOINT`, i.e. tests everything
except the boot contract.

A static gate is necessary but not sufficient: "wired but ineffective" does not
show up in a string check. Pair each new image with one behavioural probe (for
example, confirm that a request with a reasoning budget actually emits
`<|think:end|>` at the budget).
