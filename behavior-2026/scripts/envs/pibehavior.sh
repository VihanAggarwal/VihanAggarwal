#!/usr/bin/env bash
# Create the isolated Python env for the b1k26 "pibehavior" worker backend (RLC-architecture checkpoints).
#
# Two forks ship a package named `b1k` with different stage tables, so each needs its own env:
#   --fork rlc2025      IliaLarchenko/behavior-1k-solution @ ca556f7 (2025 1st place; 50-task table, 596 stage rows)
#                       for IliaLarchenko/behavior_submission checkpoint_1..4 and behavior_50t_checkpoint.
#                       The repo has no lockfile; its openpi submodule (wensi-ai/openpi @ 01177e0) does. The env is
#                       synced from that lock (jax[cuda12] 0.5.3, flax 0.10.2, orbax 0.11.13, torch 2.7.1,
#                       transformers 4.53.2, lerobot @ 577cd10), then the `b1k` package is added with --no-deps
#                       (every import it makes is covered by the openpi lock).
#   --fork jackliu2026  JackLiu0406/behaviour-1k-2026-meta @ 7146d7b (100-task table, 1120 stage rows; vendored,
#                       patched openpi) for JackLiu0406/meta-SFT-checkpoints. Synced from the repo's own uv.lock
#                       (jax 0.5.3, flax 0.10.2, orbax 0.11.13). That lock points at the Tsinghua PyPI mirror; by
#                       default its URLs are rewritten to pypi.org / files.pythonhosted.org (same paths and hashes;
#                       --keep-mirror keeps them).
# Steps: 1. fork checkout at the pinned commit into PREFIX/src/<fork>
#        2. `uv sync --frozen --no-dev` into PREFIX/venv (Python 3.11)
#        3. `uv pip install -e <repo>/behavior-2026` (b1k26), constrained to the already-installed versions
#        4. prints versions and (unless --skip-check) imports the fork on CPU with b1k26's omnigibson eval_utils stub:
#           config names, TASK_NUM_STAGES length, and the 61-D state-extraction self-check
# BEHAVIOR-1K / OmniGibson is NOT installed (the stub replaces omnigibson.learning.utils.eval_utils).
#
# Then run the worker with:  PREFIX/venv/bin/python -m b1k26.worker --backend pibehavior --checkpoint <ckpt> ...
set -euo pipefail

NAME="pibehavior"
RLC_URL="https://github.com/IliaLarchenko/behavior-1k-solution"
RLC_COMMIT="ca556f74a455cef7987a2be4537b5ac85cc56dd7"          # main head (2026-01-24)
RLC_OPENPI_URL="https://github.com/wensi-ai/openpi"
RLC_OPENPI_COMMIT="01177e0242a1c7e8fad2547caa0e987def614cda"   # the openpi submodule pinned by ca556f7
JACKLIU_URL="https://github.com/JackLiu0406/behaviour-1k-2026-meta"
JACKLIU_COMMIT="7146d7b179d2391db963f564868f6ca313185bce"      # main head (checked 2026-10-10)
PYTHON_VERSION="3.11"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"           # .../behavior-2026
REPO_DEFAULT="$(dirname "${PKG_DIR_DEFAULT}")"                    # directory that contains behavior-2026/

usage() {
  cat <<EOF
Usage: $(basename "$0") --prefix DIR --fork rlc2025|jackliu2026 [options]

Create the ${NAME} worker env for one fork (use one PREFIX per fork).

  --prefix DIR     env root (required): DIR/src/<fork checkout> and DIR/venv (Python env)
  --fork NAME      rlc2025 (2025 RLC checkpoints, tasks 0-49) or jackliu2026 (2026 meta / per-task checkpoints)
  --repo DIR       directory containing behavior-2026/ (default: ${REPO_DEFAULT})
  --commit SHA     fork commit to install (default: rlc2025 ${RLC_COMMIT}, jackliu2026 ${JACKLIU_COMMIT})
  --python VER     Python version for uv (default: ${PYTHON_VERSION})
  --keep-mirror    jackliu2026: keep the lockfile's pypi.tuna.tsinghua.edu.cn URLs
  --install-uv     install uv with the official installer if it is missing
  --skip-check     do not import the fork after installing
  --gpu-check      run the import check on the default JAX backend (GPU) instead of forcing CPU
  -h, --help       show this help
EOF
}

PREFIX=""
FORK=""
REPO="${REPO_DEFAULT}"
COMMIT=""
KEEP_MIRROR=0
INSTALL_UV=0
SKIP_CHECK=0
GPU_CHECK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="${2:?--prefix needs a value}"; shift 2 ;;
    --fork) FORK="${2:?--fork needs a value}"; shift 2 ;;
    --repo) REPO="${2:?--repo needs a value}"; shift 2 ;;
    --commit) COMMIT="${2:?--commit needs a value}"; shift 2 ;;
    --python) PYTHON_VERSION="${2:?--python needs a value}"; shift 2 ;;
    --keep-mirror) KEEP_MIRROR=1; shift ;;
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
case "${FORK}" in
  rlc2025) FORK_URL="${RLC_URL}"; COMMIT="${COMMIT:-${RLC_COMMIT}}"; FORK_DIR_NAME="behavior-1k-solution"; EXPECT_TASKS=50 ;;
  jackliu2026) FORK_URL="${JACKLIU_URL}"; COMMIT="${COMMIT:-${JACKLIU_COMMIT}}"; FORK_DIR_NAME="behaviour-1k-2026-meta"; EXPECT_TASKS=100 ;;
  *) usage >&2; die "--fork must be rlc2025 or jackliu2026" ;;
esac
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
if [[ -f "${PREFIX}/ENV_INFO.txt" ]] && ! grep -q "^fork=${FORK}$" "${PREFIX}/ENV_INFO.txt"; then
  die "${PREFIX} already holds a different fork ($(grep '^fork=' "${PREFIX}/ENV_INFO.txt")); use another --prefix"
fi

SRC="${PREFIX}/src/${FORK_DIR_NAME}"
VENV="${PREFIX}/venv"
PY="${VENV}/bin/python"
export GIT_LFS_SKIP_SMUDGE=1   # openpi's README: needed to pull LeRobot as a dependency

checkout() {  # checkout URL COMMIT DIR
  local url="$1" commit="$2" dir="$3"
  if [[ -d "${dir}/.git" ]]; then
    local origin
    origin="$(git -C "${dir}" remote get-url origin 2>/dev/null || true)"
    [[ "${origin%.git}" == "${url}" ]] || die "${dir} is a checkout of '${origin}', not ${url}"
  else
    [[ ! -e "${dir}" || -z "$(ls -A "${dir}" 2>/dev/null)" ]] || die "${dir} exists and is not a git checkout"
    mkdir -p "${dir}"
    git -C "${dir}" init -q
    git -C "${dir}" remote add origin "${url}"
  fi
  if ! git -C "${dir}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    git -C "${dir}" fetch -q --depth 1 origin "${commit}" || git -C "${dir}" fetch -q origin
  fi
  git -C "${dir}" checkout -q --detach --force "${commit}"
  local head
  head="$(git -C "${dir}" rev-parse HEAD)"
  [[ "${head}" == "${commit}" ]] || die "checked out ${head} in ${dir}, expected ${commit}"
}

# ---- 1. fork checkout at the pinned commit ------------------------------------------------------------------
log "fork ${FORK}: ${FORK_URL} @ ${COMMIT} -> ${SRC}"
checkout "${FORK_URL}" "${COMMIT}" "${SRC}"
OPENPI_SHA=""
if [[ "${FORK}" == "rlc2025" ]]; then
  # The openpi submodule only (the BEHAVIOR-1K submodule is the simulator and is not needed by the worker).
  OPENPI_SHA="$(git -C "${SRC}" ls-tree HEAD openpi | awk '{print $3}')"
  [[ -n "${OPENPI_SHA}" ]] || die "no openpi submodule entry at ${COMMIT}"
  [[ "${COMMIT}" != "${RLC_COMMIT}" || "${OPENPI_SHA}" == "${RLC_OPENPI_COMMIT}" ]] \
    || die "openpi submodule is ${OPENPI_SHA}, expected ${RLC_OPENPI_COMMIT}"
  log "openpi submodule: ${RLC_OPENPI_URL} @ ${OPENPI_SHA} -> ${SRC}/openpi"
  checkout "${RLC_OPENPI_URL}" "${OPENPI_SHA}" "${SRC}/openpi"
fi

# ---- 2. environment from a lockfile --------------------------------------------------------------------------
if [[ "${FORK}" == "rlc2025" ]]; then
  log "uv sync (openpi submodule lock, frozen, no dev group) -> ${VENV}"
  (
    cd "${SRC}/openpi"
    UV_PROJECT_ENVIRONMENT="${VENV}" uv sync --frozen --no-dev --python "${PYTHON_VERSION}"
  )
  [[ -x "${PY}" ]] || die "uv sync did not create ${PY}"
  log "installing the b1k package (--no-deps) from ${SRC}"
  uv pip install --python "${PY}" --no-deps -e "${SRC}"
else
  if [[ "${KEEP_MIRROR}" == 0 ]] && grep -q "pypi.tuna.tsinghua.edu.cn" "${SRC}/uv.lock"; then
    log "rewriting uv.lock mirror URLs to pypi.org / files.pythonhosted.org"
    sed -i \
      -e 's#https://pypi.tuna.tsinghua.edu.cn/simple#https://pypi.org/simple#g' \
      -e 's#https://pypi.tuna.tsinghua.edu.cn/packages/#https://files.pythonhosted.org/packages/#g' \
      "${SRC}/uv.lock"
  fi
  log "uv sync (fork lock, frozen, no dev extra) -> ${VENV}"
  (
    cd "${SRC}"
    UV_PROJECT_ENVIRONMENT="${VENV}" uv sync --frozen --no-dev --python "${PYTHON_VERSION}"
  )
  [[ -x "${PY}" ]] || die "uv sync did not create ${PY}"
fi

# ---- 3. b1k26 into the same env, without touching the locked versions ------------------------------------------
CONSTRAINTS="${PREFIX}/constraints.txt"
uv pip freeze --python "${PY}" | grep -v -E '^(-e |#)| @ ' > "${CONSTRAINTS}" || true
log "installing b1k26 from ${B1K26_DIR} (constraints: ${CONSTRAINTS})"
uv pip install --python "${PY}" -c "${CONSTRAINTS}" -e "${B1K26_DIR}"

# ---- 4. report --------------------------------------------------------------------------------------------------
cat > "${PREFIX}/ENV_INFO.txt" <<EOF
backend=${NAME}
fork=${FORK}
fork_url=${FORK_URL}
commit=$(git -C "${SRC}" rev-parse HEAD)
openpi_submodule=${OPENPI_SHA:-vendored}
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
             "openpi", "openpi-client", "b1k-solution", "lerobot", "websockets", "msgpack", "b1k26"):
    try:
        print(f"  {dist} {md.version(dist)}")
    except md.PackageNotFoundError:
        print(f"  {dist} (not installed)")
PYEOF
"${PY}" -c "import b1k26.backends.pibehavior as m; print('  b1k26 backend module OK, pinned', m.RLC_COMMIT[:7], m.JACKLIU_COMMIT[:7])"

if [[ "${SKIP_CHECK}" == 0 ]]; then
  log "import check (omnigibson stub, configs, stage table, state extraction self-check)"
  if [[ "${GPU_CHECK}" == 0 ]]; then export JAX_PLATFORMS=cpu; fi
  EXPECT_TASKS="${EXPECT_TASKS}" XLA_PYTHON_CLIENT_PREALLOCATE=false "${PY}" - <<'PYEOF'
import os
from b1k26.backends.openpi_comet import install_eval_utils_stub
from b1k26.backends.pibehavior import check_rlc_state_extraction
install_eval_utils_stub()
import jax
from b1k.models import pi_behavior_config as pbc
from b1k.policies import b1k_policy, policy_config  # noqa: F401
from b1k.policies.pi_behavior_policy import PiBehaviorPolicy
import inspect
from b1k.training import config
check_rlc_state_extraction(b1k_policy)
n = len(pbc.TASK_NUM_STAGES)
assert n == int(os.environ["EXPECT_TASKS"]), f"TASK_NUM_STAGES has {n} entries"
assert "initial_actions" in inspect.signature(PiBehaviorPolicy.infer).parameters
names = sorted(config._CONFIGS_DICT)
assert "pi_behavior_b1k_fast" in names, names
cfg = config.get_config("pi_behavior_b1k_fast")
print(f"  configs {names}; pi_behavior_b1k_fast: horizon={cfg.model.action_horizon} num_tasks(default)="
      f"{cfg.model.num_tasks} asset_id={cfg.data.assets.asset_id or cfg.data.repo_id}")
print(f"  TASK_NUM_STAGES: {n} tasks, {sum(pbc.TASK_NUM_STAGES)} stage rows")
print("  jax devices:", jax.devices())
print("  import check OK")
PYEOF
fi
log "done: ${PY}"
