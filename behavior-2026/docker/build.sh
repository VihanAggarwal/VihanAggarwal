#!/usr/bin/env bash
# Stage the build context and build the policy image.
#   bash docker/build.sh --config configs/final.yaml --ckpt /workspace/ckpt/jackliu_meta100 --ckpt /workspace/ckpt/comet_pt50 \
#        --backends "pibehavior openpi_comet" --tag ghcr.io/<you>/b1k26-policy:final
# Checkpoint dirs are hard-linked into the context when possible (same filesystem), so staging is fast.
# In the config, refer to checkpoints as /ckpt/<basename of --ckpt>/...
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"
CONFIG=""; TAG=""; BACKENDS=""; CKPTS=(); CUDA_IMAGE="nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --ckpt) CKPTS+=("$2"); shift 2 ;;
    --backends) BACKENDS="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --cuda-image) CUDA_IMAGE="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done
[[ -n "$CONFIG" && -n "$TAG" && -n "$BACKENDS" ]] || { echo "need --config --backends --tag"; exit 2; }
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
docker build --build-arg BACKENDS="$BACKENDS" --build-arg CUDA_IMAGE="$CUDA_IMAGE" -t "$TAG" -f "$HERE/Dockerfile" "$HERE"
echo "built $TAG"
echo "smoke test:  docker run --rm --gpus '\"device=0\"' -p 8000:8000 $TAG  &  b1k26-probe --port 8000 --steps 200 --res full --chunk 20"
echo "push:        docker push $TAG && docker inspect --format '{{index .RepoDigests 0}}' $TAG"
