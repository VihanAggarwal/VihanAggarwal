# syntax=docker/dockerfile:1
# Policy-server image for the BEHAVIOR-1K 2026 final evaluation.
# The organizers run OmniGibson outside this container and connect to the websocket ports below.
# Must fit one 24 GB GPU (RTX 3090 / A5000 / TITAN RTX); must work under docker and enroot (any UID, no writes
# outside /tmp), with no network access needed at start-up (weights and every file the models fetch at load are baked
# in; checked at build time with RUN --network=none).
#
# Build with docker/build.sh (BuildKit), which stages the build context:
#   context/b1k26/          this repo (behavior-2026), installed as /opt/behavior-2026 (the env scripts need that name)
#   context/ckpt/<name>/    checkpoints referenced by context/serve.yaml (paths /ckpt/<name>/...)
#   context/serve.yaml      the frozen serving config (workers launched as /opt/envs/<env name>/venv/bin/python)
# BACKENDS lists env names for docker/install_envs.sh: openpi_comet openpi_b1k gr00t pibehavior-2025 pibehavior-2026.
# With gr00t, build.sh passes GR00T_ENV_ARGS=--download-backbone and the build secret hf_token (gated backbone).
# Everything the image needs at run time lives outside /tmp: /tmp may be an empty tmpfs under enroot/--tmpfs.
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG BACKENDS="openpi_comet"
ARG GR00T_ENV_ARGS=""
# HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set only after the env install below (the gr00t env script downloads
# the gated backbone into HF_HOME).
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    HF_HOME=/opt/hf \
    OPENPI_DATA_HOME=/opt/openpi_data \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs curl ca-certificates build-essential ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# openpi pi0.5 models (Comet, pi05_b1k, Hoshipu) build a PaligemmaTokenizer at every load, which downloads
# gs://big_vision/paligemma_tokenizer.model into $OPENPI_DATA_HOME unless it is already there: bake it in. The
# cache dir must be mode 0777, otherwise openpi chmods it at load, which fails for a non-root UID (enroot).
RUN mkdir -p /opt/openpi_data/big_vision \
    && curl -fsSL --retry 3 https://storage.googleapis.com/big_vision/paligemma_tokenizer.model \
         -o /opt/openpi_data/big_vision/paligemma_tokenizer.model \
    && test -s /opt/openpi_data/big_vision/paligemma_tokenizer.model \
    && chmod 0777 /opt/openpi_data && chmod 0755 /opt/openpi_data/big_vision \
    && chmod 0644 /opt/openpi_data/big_vision/paligemma_tokenizer.model

# Front server env (light deps only). uv-managed Pythons go to UV_PYTHON_INSTALL_DIR (not under HOME=/tmp).
COPY context/b1k26 /opt/behavior-2026
RUN uv python install 3.11 && uv venv /opt/envs/front --python 3.11 \
    && uv pip install --python /opt/envs/front/bin/python -e "/opt/behavior-2026[control]"

# One isolated env per model family (each pins its own JAX/torch/openpi stack): /opt/envs/<name>/venv.
# The optional build secret hf_token is visible to this step only and is never stored in the image.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" GR00T_ENV_ARGS="${GR00T_ENV_ARGS}" \
    bash /opt/behavior-2026/docker/install_envs.sh --root /opt/envs ${BACKENDS}

# From here on nothing may need the network: the image must start without it.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
RUN --network=none bash /opt/behavior-2026/docker/check_offline.sh /opt/envs

COPY context/ckpt /ckpt
COPY context/serve.yaml /config/serve.yaml
# Fail the build if the config is invalid or names a worker interpreter that does not exist in the image.
RUN /opt/envs/front/bin/b1k26-serve --config /config/serve.yaml --check \
    && chmod -R a+rX /opt /ckpt /config

EXPOSE 8000-8049
# Extra arguments after the image name are passed on, e.g. `--ports 8100-8149` (or set -e B1K26_PORTS=8100-8149).
ENTRYPOINT ["/opt/envs/front/bin/b1k26-serve", "--config", "/config/serve.yaml"]
