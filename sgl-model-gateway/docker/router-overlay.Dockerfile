# Overlay build: rebuild ONLY the router artifacts from this repo's
# sgl-model-gateway tree inside the live base image, so glibc/platform match
# is guaranteed and everything else in the image stays untouched. Built with
# a BuildKit git context pointing at a router-* tag of this repo:
#   --opt context=https://github.com/vessl-ai/sglang.git#<tag>
#   --opt filename=sgl-model-gateway/docker/router-overlay.Dockerfile
#
# The live deployment execs the standalone binary /usr/local/bin/sgl-model-gateway,
# so that binary is the primary artifact; the sglang_router wheel is also
# force-reinstalled for python3 -m sglang_router.launch_router parity.
FROM quay.io/vessl-ai/sglang:v0.5.13.post1-cu130-router-26256-27430-prefill-opt26780hc-httpmwfix-leastload-stream-eagleradix

# rust 1.90 + build deps (skip any already present in the base)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.90 \
 && . "$HOME/.cargo/env" \
 && (command -v protoc || (apt-get update && apt-get install -y protobuf-compiler build-essential pkg-config)) \
 && pip install "maturin>=1,<2"

COPY sgl-model-gateway /opt/patch/sgl-model-gateway

# Shared target dir so the wheel build reuses dependency artifacts from the
# binary build.
ENV CARGO_TARGET_DIR=/opt/patch/target

# 1) Standalone binary (what the live Deployment execs).
WORKDIR /opt/patch/sgl-model-gateway
RUN . "$HOME/.cargo/env" && ulimit -n 65536 \
 && cargo build --release --bin sgl-model-gateway --features vendored-openssl \
 && install -m 0755 /opt/patch/target/release/sgl-model-gateway /usr/local/bin/sgl-model-gateway

# 2) Python wheel (launch_router path).
WORKDIR /opt/patch/sgl-model-gateway/bindings/python
RUN . "$HOME/.cargo/env" && ulimit -n 65536 \
 && maturin build --release --features vendored-openssl --out /dist
RUN pip install --no-deps --force-reinstall /dist/*.whl

WORKDIR /
