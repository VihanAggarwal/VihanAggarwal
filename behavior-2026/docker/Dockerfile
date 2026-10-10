# Policy-server image for the BEHAVIOR-1K 2026 final evaluation.
# The organizers run OmniGibson outside this container and connect to the websocket ports below.
# Must fit one 24 GB GPU (RTX 3090 / A5000 / TITAN RTX); must work under docker and enroot (any UID, no writes
# outside /tmp), with no network access needed at start-up (weights are baked in).
#
# Build with docker/build.sh, which stages the build context:
#   context/b1k26/          this repo (behavior-2026), installed as /opt/behavior-2026 (the env scripts need that name)
#   context/ckpt/<name>/    checkpoints referenced by context/serve.yaml (paths /ckpt/<name>/...)
#   context/serve.yaml      the frozen serving config (workers launched as /opt/envs/<env name>/venv/bin/python)
# BACKENDS lists env names for docker/install_envs.sh: openpi_comet openpi_b1k gr00t pibehavior-2025 pibehavior-2026.
# Everything the image needs at run time lives outside /tmp: /tmp may be an empty tmpfs under enroot/--tmpfs.
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG BACKENDS="openpi_comet"
ARG GR00T_ENV_ARGS=""
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    HF_HOME=/opt/hf \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs curl ca-certificates build-essential ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Front server env (light deps only). uv-managed Pythons go to UV_PYTHON_INSTALL_DIR (not under HOME=/tmp).
COPY context/b1k26 /opt/behavior-2026
RUN uv python install 3.11 && uv venv /opt/envs/front --python 3.11 \
    && uv pip install --python /opt/envs/front/bin/python -e "/opt/behavior-2026[control]"

# One isolated env per model family (each pins its own JAX/torch/openpi stack): /opt/envs/<name>/venv.
RUN GR00T_ENV_ARGS="${GR00T_ENV_ARGS}" bash /opt/behavior-2026/docker/install_envs.sh --root /opt/envs ${BACKENDS}

COPY context/ckpt /ckpt
COPY context/serve.yaml /config/serve.yaml
# Fail the build if the config is invalid or names a worker interpreter that does not exist in the image.
RUN /opt/envs/front/bin/b1k26-serve --config /config/serve.yaml --check \
    && chmod -R a+rX /opt /ckpt /config

EXPOSE 8000-8049
ENTRYPOINT ["/opt/envs/front/bin/b1k26-serve", "--config", "/config/serve.yaml"]
