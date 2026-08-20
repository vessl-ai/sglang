# Solar-Open2-250B-W4AFP8 serving image: boot contract overlay.
#
# This is an OVERLAY, not an engine build. The Solar-Open2 config, model,
# tool-call detector and reasoning-budget FSM are native to this fork, so a
# normal engine image built from this tree already has all of the source. What
# the source alone does not carry is the image's boot contract: the two ENV
# defaults the serving configuration depends on, and a gate that proves the
# wiring is intact before the engine is allowed to start. That is what this
# overlay adds, and nothing else.
#
# Build context is the repository root, matching the other Dockerfiles under
# docker/ -- the COPY paths below are repo-relative.
#
# BASE_IMAGE has no default on purpose. The caller pins one of our own engine
# images BY DIGEST; a default here would be a stale tag waiting to be picked up
# silently. The overlay deliberately pins no upstream image and installs no
# transfer-engine wheel: which engine build and which client libraries go into
# the image are decided one layer up, by the build chain that produces
# BASE_IMAGE. Hard-coding either here would make this overlay disagree with the
# image it is stacked on. If BASE_IMAGE is unset the build fails immediately on
# a blank base name, which is the intended behaviour.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# The gate is a plain script with no imports beyond the stdlib and sglang, so it
# runs identically at build time and at every container start.
COPY docker/solar_open2/solar_gate.py       /opt/solar/solar_gate.py
COPY docker/solar_open2/solar-entrypoint.sh /usr/local/bin/solar-entrypoint.sh

# The build-context file's mode is not trustworthy. When the context is
# delivered as a ConfigMap volume, k8s resets every file to 0644 at mount time
# and the local +x is gone before the builder ever reads it. Without this line
# the image builds and pushes clean and then runc fails container creation with
# "permission denied" at rollout, a long way from the cause. Set the bit here so
# the image is correct however the context arrived.
RUN chmod 0755 /usr/local/bin/solar-entrypoint.sh

# Fail the BUILD if the entrypoint is somehow still not executable, rather than
# discovering it at rollout. The axis exists because a mode defect once passed a
# clean build and push and reached a cluster.
RUN test -x /usr/local/bin/solar-entrypoint.sh

# The KDA beta scale used for serving. The code default is 1.0 (see
# _SOLAR_KDA_BETA_SCALE in python/sglang/srt/models/kimi_linear.py and the two
# fla kernels), which is the unscaled-beta accuracy defect regime -- so this ENV
# is load-bearing, not cosmetic, and the gate refuses to boot without it.
ENV SOLAR_KDA_BETA_SCALE=2.0

# The reasoning-budget FSM is required for serving: without it decode can fall
# into an exact repetition loop. Read by
# python/sglang/srt/sampling/solar_open2_fsm.py.
ENV SOLAR_FSM=1

# Build-time run of the same gate the entrypoint runs. It proves the base image
# really is a build of this fork and that all three beta sites survived into the
# layer. The speculative-decoding args are absent here, so the draft-config and
# FSM verify-path gates report themselves as skipped; those run at boot against
# the actual launch args.
RUN python3 /opt/solar/solar_gate.py

# Every container start goes through the gate. Callers must pass the engine
# command as `args:` and must NOT override `command:` -- overriding it skips the
# ENTRYPOINT, which means testing everything except the boot contract.
ENTRYPOINT ["/usr/local/bin/solar-entrypoint.sh"]
