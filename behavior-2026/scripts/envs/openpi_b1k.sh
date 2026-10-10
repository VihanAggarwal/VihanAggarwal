#!/usr/bin/env bash
# Create the isolated Python env for the b1k26 "openpi_b1k" worker backend (wensi-ai pi05_b1k and compatible checkpoints).
#
# What it does (uv-based, like the upstream README: `GIT_LFS_SKIP_SMUDGE=1 uv sync`):
#   1. checks out wensi-ai/openpi (branch "behavior") at a pinned commit into PREFIX/src/openpi-b1k
#   2. `uv sync --frozen --no-dev` from the fork's uv.lock into PREFIX/venv (Python 3.11, jax[cuda12]==0.5.3,
#      flax 0.10.2, orbax 0.11.13, torch 2.7.1, transformers 5.5.4, lerobot from wensi-ai/lerobot release/b1k;
#      the fork itself is installed editable by uv sync)
#   3. `uv pip install -e <repo>/behavior-2026` (b1k26) into the same venv, constrained to the versions already
#      installed so nothing from the lock gets up/downgraded
#   4. prints versions and (unless --skip-check) imports the fork on CPU and runs the b1k/R1Pro state-extraction
#      self-check (no omnigibson stub is needed for this fork)
# BEHAVIOR-1K / OmniGibson is NOT installed: the worker only needs openpi (the evaluator runs elsewhere).
#
# Then run the worker with:  PREFIX/venv/bin/python -m b1k26.worker --backend openpi_b1k --checkpoint <ckpt> ...
set -euo pipefail

NAME="openpi_b1k"
FORK_URL="https://github.com/wensi-ai/openpi"
FORK_COMMIT="0cc8e355f7bac0976db1cc3139b1ff0379feea60"  # branch "behavior" head (2026-06-28)
FORK_DIR_NAME="openpi-b1k"
PYTHON_VERSION="3.11"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"           # .../behavior-2026
REPO_DEFAULT="$(dirname "${PKG_DIR_DEFAULT}")"                    # directory that contains behavior-2026/

usage() {
  cat <<EOF
Usage: $(basename "$0") --prefix DIR [options]

Create the ${NAME} worker env.

  --prefix DIR     env root (required): DIR/src/${FORK_DIR_NAME} (fork checkout) and DIR/venv (Python env)
  --repo DIR       directory containing behavior-2026/ (default: ${REPO_DEFAULT})
  --commit SHA     fork commit to install (default: ${FORK_COMMIT})
  --python VER     Python version for uv (default: ${PYTHON_VERSION})
  --install-uv     install uv with the official installer if it is missing
  --skip-check     do not import the fork after installing
  --gpu-check      run the import check on the default JAX backend (GPU) instead of forcing CPU
  -h, --help       show this help
EOF
}

PREFIX=""
REPO="${REPO_DEFAULT}"
COMMIT="${FORK_COMMIT}"
INSTALL_UV=0
SKIP_CHECK=0
GPU_CHECK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="${2:?--prefix needs a value}"; shift 2 ;;
    --repo) REPO="${2:?--repo needs a value}"; shift 2 ;;
    --commit) COMMIT="${2:?--commit needs a value}"; shift 2 ;;
    --python) PYTHON_VERSION="${2:?--python needs a value}"; shift 2 ;;
    --install-uv) INSTALL_UV=1; shift ;;
    --skip-check) SKIP_CHECK=1; shift ;;
    --gpu-check) GPU_CHECK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

die() { echo "[${NAME}] ERROR: $*" >&2; exit 1; }
log() { echo "[${NAME}] $*"; }

[[ -n "${PREFIX}" ]] || { usage >&2; die "--prefix is required"; }
mkdir -p "${PREFIX}"
PREFIX="$(cd "${PREFIX}" && pwd)"
B1K26_DIR="$(cd "${REPO}" && pwd)/behavior-2026"
[[ -f "${B1K26_DIR}/pyproject.toml" ]] || die "b1k26 package not found at ${B1K26_DIR} (use --repo)"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]] || die "--commit must be a full 40-hex SHA, got ${COMMIT}"
command -v git >/dev/null || die "git is required"
if ! command -v uv >/dev/null; then
  if [[ "${INSTALL_UV}" == 1 ]]; then
    command -v curl >/dev/null || die "curl is required to install uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
  fi
  command -v uv >/dev/null || die "uv not found; install it (curl -LsSf https://astral.sh/uv/install.sh | sh) or pass --install-uv"
fi

SRC="${PREFIX}/src/${FORK_DIR_NAME}"
VENV="${PREFIX}/venv"
PY="${VENV}/bin/python"
export GIT_LFS_SKIP_SMUDGE=1   # openpi's README: needed to pull LeRobot as a dependency

# ---- 1. fork checkout at the pinned commit ------------------------------------------------------------------
log "fork ${FORK_URL} @ ${COMMIT} -> ${SRC}"
if [[ -d "${SRC}/.git" ]]; then
  origin="$(git -C "${SRC}" remote get-url origin 2>/dev/null || true)"
  [[ "${origin%.git}" == "${FORK_URL}" ]] || die "${SRC} is a checkout of '${origin}', not ${FORK_URL}"
else
  [[ ! -e "${SRC}" || -z "$(ls -A "${SRC}" 2>/dev/null)" ]] || die "${SRC} exists and is not a git checkout"
  mkdir -p "${SRC}"
  git -C "${SRC}" init -q
  git -C "${SRC}" remote add origin "${FORK_URL}"
fi
if ! git -C "${SRC}" cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
  git -C "${SRC}" fetch -q --depth 1 origin "${COMMIT}" || git -C "${SRC}" fetch -q origin
fi
git -C "${SRC}" checkout -q --detach --force "${COMMIT}"
HEAD_SHA="$(git -C "${SRC}" rev-parse HEAD)"
[[ "${HEAD_SHA}" == "${COMMIT}" ]] || die "checked out ${HEAD_SHA}, expected ${COMMIT}"

# ---- 2. fork environment from its lockfile ------------------------------------------------------------------
log "uv sync (frozen lock, no dev group) -> ${VENV}"
(
  cd "${SRC}"
  UV_PROJECT_ENVIRONMENT="${VENV}" uv sync --frozen --no-dev --python "${PYTHON_VERSION}"
)
[[ -x "${PY}" ]] || die "uv sync did not create ${PY}"

# ---- 3. b1k26 into the same env, without touching the locked versions ------------------------------------------
CONSTRAINTS="${PREFIX}/constraints.txt"
uv pip freeze --python "${PY}" | grep -v -E '^(-e |#)| @ ' > "${CONSTRAINTS}" || true
log "installing b1k26 from ${B1K26_DIR} (constraints: ${CONSTRAINTS})"
uv pip install --python "${PY}" -c "${CONSTRAINTS}" -e "${B1K26_DIR}"

# ---- 4. report --------------------------------------------------------------------------------------------------
cat > "${PREFIX}/ENV_INFO.txt" <<EOF
backend=${NAME}
fork=${FORK_URL}
commit=${HEAD_SHA}
b1k26=${B1K26_DIR}
python=${PY}
created=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
log "versions:"
"${PY}" - <<'PYEOF'
import importlib.metadata as md
import platform
print(f"  python {platform.python_version()}")
for dist in ("jax", "jaxlib", "jax-cuda12-plugin", "flax", "orbax-checkpoint", "numpy", "torch", "transformers",
             "sentencepiece", "openpi", "openpi-client", "websockets", "msgpack", "b1k26"):
    try:
        print(f"  {dist} {md.version(dist)}")
    except md.PackageNotFoundError:
        print(f"  {dist} (not installed)")
PYEOF
"${PY}" -c "import b1k26.backends.openpi_b1k as m; print('  b1k26 backend module OK, pinned', m.OPENPI_B1K_COMMIT)"

if [[ "${SKIP_CHECK}" == 0 ]]; then
  log "import check (fork config + state extraction self-check)"
  if [[ "${GPU_CHECK}" == 0 ]]; then export JAX_PLATFORMS=cpu; fi
  XLA_PYTHON_CLIENT_PREALLOCATE=false "${PY}" - <<'PYEOF'
import numpy as np
import jax
import openpi.training.config as config
from openpi.configs import ROBOT_REGISTRY
from openpi.policies import b1k_policy
from b1k26.backends.openpi_b1k import b1k_state_from_proprio
for name in ("pi05_b1k",):
    assert name in config._CONFIGS_DICT, name
    cfg = config.get_config(name)
    robot = ROBOT_REGISTRY[cfg.data.robot_config_name]
    probe = (np.arange(61, dtype=np.float32) + 1.0) * 0.013
    got = np.asarray(b1k_policy.extract_state_from_proprio(probe, robot), dtype=np.float32)
    assert np.allclose(got, b1k_state_from_proprio(probe, "width"), atol=1e-6), got
    print(f"  config {name}: action_horizon={cfg.model.action_horizon} pi05={cfg.model.pi05} "
          f"repo_id={cfg.data.repo_id} robot={cfg.data.robot_config_name} "
          f"cameras={ {k: o.name for k, o in robot.observations.items()} }")
print("  jax devices:", jax.devices())
print("  import check OK")
PYEOF
fi
log "done: ${PY}"
