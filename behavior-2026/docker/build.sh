#!/usr/bin/env bash
# Stage the build context and build the policy image (BuildKit).
#   bash docker/build.sh --config configs/final.yaml --ckpt /ckpt/jackliu_meta100 --ckpt /ckpt/comet_pt50 \
#        --backends "pibehavior-2026 openpi_comet" --tag ghcr.io/<you>/b1k26-policy:final
# --backends takes env names of docker/install_envs.sh (openpi_comet openpi_b1k gr00t pibehavior-2025 pibehavior-2026);
# the config must launch workers as /opt/envs/<name>/venv/bin/python (checked at build time by b1k26-serve --check).
# Checkpoint dirs are hard-linked into the context when possible (same filesystem), so staging is fast.
# In the config, refer to checkpoints as /ckpt/<basename of --ckpt>/... (scripts/download_checkpoints.py --dest /ckpt
# already uses that layout: /ckpt/<candidate name>/...).
#
# gr00t: the backbone nvidia/Cosmos-Reason2-2B is gated. Export HF_TOKEN (an account that accepted its license); it
# is passed as a BuildKit secret (never stored in the image) and the backbone is baked into /opt/hf. The image then
# contains gated weights: check the license before making it public, or give the organizers pull credentials.
# --gr00t-env-args overrides the gr00t env options (default --download-backbone; add --no-flash-attn for a
# Turing-only build host).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"
CONFIG=""; TAG=""; BACKENDS=""; CKPTS=(); CUDA_IMAGE="nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"
GR00T_ENV_ARGS="--download-backbone"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --ckpt) CKPTS+=("$2"); shift 2 ;;
    --backends) BACKENDS="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --cuda-image) CUDA_IMAGE="$2"; shift 2 ;;
    --gr00t-env-args) GR00T_ENV_ARGS="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done
[[ -n "$CONFIG" && -n "$TAG" && -n "$BACKENDS" ]] || { echo "need --config --backends --tag"; exit 2; }
SECRET_ARGS=()
SECRET_FILE=""
cleanup() { [[ -n "$SECRET_FILE" ]] && rm -f "$SECRET_FILE"; return 0; }
trap cleanup EXIT
if [[ " $BACKENDS " == *" gr00t "* ]]; then
  if [[ "$GR00T_ENV_ARGS" == *--download-backbone* ]]; then
    [[ -n "${HF_TOKEN:-}" ]] || { echo "gr00t: export HF_TOKEN (nvidia/Cosmos-Reason2-2B is gated; the image must contain it)"; exit 2; }
    SECRET_FILE="$(mktemp)"; chmod 600 "$SECRET_FILE"; printf '%s' "$HF_TOKEN" > "$SECRET_FILE"
    SECRET_ARGS=(--secret "id=hf_token,src=$SECRET_FILE")
  fi
else
  GR00T_ENV_ARGS=""
fi
CTX="$HERE/context"; rm -rf "$CTX"; mkdir -p "$CTX/ckpt"
rsync -a --exclude '.git' --exclude 'outputs' --exclude 'runs' --exclude '__pycache__' --exclude 'docker/context' \
  "$REPO/" "$CTX/b1k26/"
for c in "${CKPTS[@]}"; do
  cp -al "$c" "$CTX/ckpt/$(basename "$c")" 2>/dev/null || cp -a "$c" "$CTX/ckpt/$(basename "$c")"
done
cp "$CONFIG" "$CTX/serve.yaml"
grep -o '/ckpt/[^ "'"'"']*' "$CTX/serve.yaml" | sort -u | while read -r p; do
  [[ -e "$CTX$p" ]] || { echo "config references $p, which is not in the staged context"; exit 1; }
done
DOCKER_BUILDKIT=1 docker build --build-arg BACKENDS="$BACKENDS" --build-arg CUDA_IMAGE="$CUDA_IMAGE" \
  --build-arg GR00T_ENV_ARGS="$GR00T_ENV_ARGS" "${SECRET_ARGS[@]}" -t "$TAG" -f "$HERE/Dockerfile" "$HERE"
echo "built $TAG"
echo "smoke test:   docker run --rm --gpus '\"device=0\"' -p 8000-8002:8000-8002 $TAG --ports 8000-8002 &"
echo "              bash scripts/smoke_local.sh --no-start --ports 8000-8002"
echo "offline test: docker run --rm --network none --gpus '\"device=0\"' --entrypoint bash $TAG -c \\"
echo "                '/opt/envs/front/bin/b1k26-serve --config /config/serve.yaml --ports 8000 & /opt/envs/front/bin/b1k26-probe --port 8000 --steps 100 --res full --health-timeout 1800'"
echo "push:         docker push $TAG && docker inspect --format '{{index .RepoDigests 0}}' $TAG"
echo "              then make the package public (GHCR packages start private) or give the organizers pull"
echo "              credentials, and check from a logged-out machine: docker logout ghcr.io; docker pull <repo>@sha256:<digest>"
