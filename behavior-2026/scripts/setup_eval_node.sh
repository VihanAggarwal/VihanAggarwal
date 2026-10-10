#!/usr/bin/env bash
# Bootstrap a cloud GPU node for BEHAVIOR-1K 2026 self-evaluation.
#
#   bash scripts/setup_eval_node.sh [--work /workspace] [--tag v3.9.3-post2] [--skip-preflight] [--skip-smoke]
#
# Installs BEHAVIOR-1K at the challenge tag into a conda env "behavior" (Isaac Sim 5.1, OmniGibson, BDDL, JoyLo,
# eval extras, robot + scene assets, 2026 task instances), installs b1k26 into the same env (front server, job
# runner), writes $WORK/b1k_env.sh, and runs a 30-step zero-action rollout as a smoke test.
# Run it once, snapshot the machine or volume, and clone the snapshot to the other nodes.
set -euo pipefail

WORK=/workspace
TAG=v3.9.3-post2
SKIP_PREFLIGHT=0
SKIP_SMOKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --work) WORK="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$WORK"
log() { echo -e "\n[setup] $*"; }
die() { echo "[setup] ERROR: $*" >&2; exit 1; }

# ------------------------------------------------------------------------------------------------ preflight
preflight() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found: this node has no NVIDIA driver"
  local name driver vram
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  log "GPU: $name, driver $driver, ${vram} MiB, $(nvidia-smi -L | wc -l) GPU(s)"
  # Isaac Sim renders with RTX ray tracing: GPUs without RT cores render garbage or crash in DLSS init.
  if echo "$name" | grep -Eiq 'A100|A30|A800|H100|H200|H800|H20|B100|B200|B300|GB200|GH200|V100|P100'; then
    die "$name has no RT cores; Isaac Sim cannot render BEHAVIOR scenes on it. Use RTX 4090/5090, L40S, L40, RTX 6000 Ada, A6000, A5000 or 3090."
  fi
  # Isaac Sim 5.1 requires driver >= 580.65.06; 595.x crashes were reported on Blackwell (RTX PRO 6000) hosts.
  python3 - "$driver" <<'EOF' || die "driver too old for Isaac Sim 5.1 (need >= 580.65.06)"
import sys
v = tuple(int(x) for x in sys.argv[1].split("."))
sys.exit(0 if v >= (580, 65, 6) else 1)
EOF
  [[ "$driver" == 595.* ]] && echo "[setup] WARNING: driver 595.x has been reported to crash Isaac Sim 5.1; prefer 580.x or 570.x."
  [[ "$vram" -ge 20000 ]] || die "need >= 20 GB VRAM per GPU (simulator ~8-16 GB + policy ~8 GB)"
  local cpus mem_gb disk_gb
  cpus=$(nproc); mem_gb=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 ))
  disk_gb=$(df -BG --output=avail "$WORK" | tail -1 | tr -dc '0-9')
  log "CPU $cpus, RAM ${mem_gb} GB, free disk ${disk_gb} GB at $WORK"
  [[ "$cpus" -ge 8 ]] || echo "[setup] WARNING: < 8 vCPUs; the simulator is CPU-heavy and will run slowly"
  [[ "$mem_gb" -ge 30 ]] || die "need >= 32 GB RAM"
  [[ "$disk_gb" -ge 150 ]] || die "need >= 150 GB free disk at $WORK (Isaac Sim, assets, texture caches, checkpoints)"
  # Vulkan ICD: containers need NVIDIA_DRIVER_CAPABILITIES=all (graphics) and an ICD file pointing at the driver.
  if ! ls /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json >/dev/null 2>&1; then
    if ldconfig -p | grep -q libGLX_nvidia.so.0; then
      log "creating /etc/vulkan/icd.d/nvidia_icd.json"
      mkdir -p /etc/vulkan/icd.d
      cat > /etc/vulkan/icd.d/nvidia_icd.json <<'EOF'
{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0"}}
EOF
    else
      die "libGLX_nvidia.so.0 is missing: the container was started without graphics capability. Restart it with NVIDIA_DRIVER_CAPABILITIES=all (RunPod/Vast: add the env var to the template)."
    fi
  fi
}
[[ "$SKIP_PREFLIGHT" == 1 ]] || preflight

# ------------------------------------------------------------------------------------------------ system deps
if command -v apt-get >/dev/null; then
  log "installing system packages"
  export DEBIAN_FRONTEND=noninteractive
  SUDO=""; [[ $(id -u) -ne 0 ]] && SUDO=sudo
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq git git-lfs curl wget ca-certificates build-essential libglu1-mesa libxt6 \
    libvulkan1 vulkan-tools ffmpeg tmux htop >/dev/null
fi

# ------------------------------------------------------------------------------------------------ conda
if ! command -v conda >/dev/null; then
  if [[ ! -x "$WORK/miniforge/bin/conda" ]]; then
    log "installing miniforge"
    curl -fsSL -o /tmp/miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    bash /tmp/miniforge.sh -b -p "$WORK/miniforge"
  fi
  export PATH="$WORK/miniforge/bin:$PATH"
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ------------------------------------------------------------------------------------------------ BEHAVIOR-1K
if [[ ! -d "$WORK/BEHAVIOR-1K/.git" ]]; then
  log "cloning BEHAVIOR-1K $TAG (note: there is no plain v3.9.3 tag; post1/post2 are the 2026 eval tags)"
  git clone --depth 1 --branch "$TAG" https://github.com/StanfordVL/BEHAVIOR-1K "$WORK/BEHAVIOR-1K"
fi
cd "$WORK/BEHAVIOR-1K"
if ! conda env list | grep -q '^behavior '; then
  log "running BEHAVIOR-1K setup.sh (Isaac Sim + assets download: 30-60 min)"
  ./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval \
    --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
fi
conda activate behavior

log "installing b1k26 into the behavior env"
python -m pip install -q -e "$REPO_DIR[control]"
python -c "import omnigibson, b1k26; print('omnigibson', omnigibson.__version__, '| b1k26', b1k26.__version__)"

# ------------------------------------------------------------------------------------------------ env file
APPDATA="$WORK/og_appdata"; mkdir -p "$APPDATA"
cat > "$WORK/b1k_env.sh" <<EOF
# source this before running evaluations
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate behavior
export OMNIGIBSON_HEADLESS=1
export OMNIGIBSON_APPDATA_PATH="$APPDATA"      # persistent kit/shader caches (first scene load is ~2x slower)
export B1K_REPO="$WORK/BEHAVIOR-1K"
export B1K26_REPO="$REPO_DIR"
EOF
log "wrote $WORK/b1k_env.sh"

# ------------------------------------------------------------------------------------------------ smoke test
if [[ "$SKIP_SMOKE" != 1 ]]; then
  log "smoke test: 30 zero-action steps on turning_on_radio (index 10 = instance 311) with the official wrapper"
  # shellcheck disable=SC1091
  source "$WORK/b1k_env.sh"
  rm -rf /tmp/b1k_smoke
  start=$(date +%s)
  python -m omnigibson.eval.eval --task-name turning_on_radio --policy local --instance-indices 10 \
    --num-envs 1 --max-steps 30 --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
    --output-dir /tmp/b1k_smoke --write-video 2>&1 | tail -20
  ls /tmp/b1k_smoke/json/turning_on_radio_311_0.json >/dev/null \
    || die "smoke rollout produced no metrics JSON (Isaac exits 0 even on crashes: read the log above)"
  log "smoke test OK in $(( $(date +%s) - start )) s (includes first-time shader compilation)"
fi
log "done. Next: scripts/envs/<backend>.sh for each candidate, then scripts/download_checkpoints.sh"
