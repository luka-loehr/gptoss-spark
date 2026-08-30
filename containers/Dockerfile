# syntax=docker/dockerfile:1
# gptoss-spark — gpt-oss-120b serving image for the DGX Spark (linux/arm64, sm_121).
#
# Build (on an sm_121 host; no GPU needed for the build itself):
#   docker build -f containers/Dockerfile -t ghcr.io/OWNER/gptoss-spark:VERSION .
#
# Run:
#   docker run --gpus all --network host --ipc host --shm-size 32g \
#     -v /path/to/gpt-oss-120b:/model:ro \
#     -v /path/to/tiktoken-encodings:/tiktoken:ro \
#     -v gptoss-jit:/root/.cache/flashinfer \
#     -e PROFILE=plain ghcr.io/OWNER/gptoss-spark:VERSION
#
# The checkpoint is NOT in the image. First start JIT-compiles the SM121 MXFP4
# kernels (~10 min); mount a volume at /root/.cache/flashinfer so every later
# start is warm.

ARG VLLM_TAG=nightly
FROM vllm/vllm-openai:${VLLM_TAG}

# The FlashInfer fork carrying the SM121 CUTLASS MXFP4 kernels. Pinned; see
# docs/PATCHES.md for why this is vendored under a separate import name.
ARG FLASHINFER_FORK=https://github.com/christopherowen/flashinfer.git
ARG FLASHINFER_REF=f349e52496a72a00d8c4ac02c7a1e38523ff7194

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# nvrtc headers: the fork's kernels JIT-compile at runtime and need them.
# Do NOT point CPATH at the pip nvidia headers instead - that breaks nvcc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git patch cuda-nvrtc-dev-13-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- vendored SM121 kernels -------------------------------------------------
# Renamed to `flashinfer_spark` so it coexists with the engine's own FlashInfer,
# which is required for sm_121 attention and must not be downgraded.
RUN git clone --depth 1 "${FLASHINFER_FORK}" /tmp/fi \
    && cd /tmp/fi && git fetch --depth 1 origin "${FLASHINFER_REF}" \
    && git checkout FETCH_HEAD && git submodule update --init --recursive --depth 1 \
    && mkdir -p /opt/flashinfer-spark \
    && cp -r /tmp/fi/. /opt/flashinfer-spark/ \
    && cd /opt/flashinfer-spark \
    && mv flashinfer flashinfer_spark \
    && grep -rl --include="*.py" -E "from flashinfer|import flashinfer" flashinfer_spark \
       | xargs -r sed -i -E "s/\bfrom flashinfer\b/from flashinfer_spark/g; \
                             s/\bfrom flashinfer\./from flashinfer_spark./g; \
                             s/\bimport flashinfer\b/import flashinfer_spark/g; \
                             s/\bimport flashinfer\./import flashinfer_spark./g" \
    && cd flashinfer_spark/data \
    && rm -f csrc include cutlass spdlog tvm_ffi \
    && ln -s /opt/flashinfer-spark/csrc csrc \
    && ln -s /opt/flashinfer-spark/include include \
    && ln -s /opt/flashinfer-spark/3rdparty/cutlass cutlass \
    && ln -s /opt/flashinfer-spark/3rdparty/spdlog spdlog \
    && rm -rf /tmp/fi

# ---- engine patches ---------------------------------------------------------
COPY patches /opt/gptoss-spark/patches
RUN cd /usr/local/lib/python3.12/dist-packages \
    && for p in /opt/gptoss-spark/patches/*.patch; do \
         echo "applying $(basename "$p")"; patch -p1 --forward < "$p"; \
       done

COPY containers/entrypoint.sh /usr/local/bin/gptoss-entrypoint
COPY bench /opt/gptoss-spark/bench
RUN chmod +x /usr/local/bin/gptoss-entrypoint

ENV PYTHONPATH=/opt/flashinfer-spark \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    TIKTOKEN_ENCODINGS_BASE=/tiktoken \
    VLLM_SPARK_CUTLASS=1 \
    VLLM_SPARK_DENSE=qkv,o,lm_head,mlp,fc \
    PROFILE=plain \
    PORT=8100

EXPOSE 8100
ENTRYPOINT ["/usr/local/bin/gptoss-entrypoint"]
