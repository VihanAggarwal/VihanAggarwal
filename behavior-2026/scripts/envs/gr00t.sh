#!/usr/bin/env bash
# Create the isolated Python env for the b1k26 "gr00t" worker backend (GR00T N1.7 checkpoints).
#
# What it does (uv-based, like the fork's README):
#   1. checks out wensi-ai/Isaac-GR00T (branch "behavior") at a pinned commit into PREFIX/src/Isaac-GR00T
#   2. `uv sync --frozen --no-dev` from the fork's uv.lock into PREFIX/venv (Python 3.10, torch 2.7.1 cu128,
#      transformers 4.57.3, flash-attn 2.7.4.post1 prebuilt wheel; the fork itself is installed editable)
#      --no-flash-attn skips the flash-attn wheel (Turing-only nodes: flash-attn 2.x refuses sm < 80 anyway; the
#      backend then uses SDPA)
#   3. `uv pip install -e <repo>/behavior-2026` (b1k26) into the same venv, constrained to the installed versions
#   4. optional --download-backbone: `hf download nvidia/Cosmos-Reason2-2B` into the Hugging Face cache. Gr00tN1d7
#      builds its backbone and processor from that GATED repo at every load (accept its license on huggingface.co
#      and export HF_TOKEN first); afterwards the worker can run with HF_HUB_OFFLINE=1.
#   5. prints versions, GPU capability and the attention/dtype the backend would pick; (unless --skip-check)
#      imports gr00t.policy.gr00t_policy and checks examples/b1k/r1pro.json against b1k26's state/action slices
#
# Then run the worker with:  PREFIX/venv/bin/python -m b1k26.worker --backend gr00t --checkpoint <ckpt> ...
set -euo pipefail

NAME="gr00t"
FORK_URL="https://github.com/wensi-ai/Isaac-GR00T"
FORK_COMMIT="ace36d935b376fbf25cd56371e23877b95407c40"  # branch "behavior" head (2026-07-02)
FORK_DIR_NAME="Isaac-GR00T"
PYTHON_VERSION="3.10"
BACKBONE_REPO="nvidia/Cosmos-Reason2-2B"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"           # .../behavior-2026
REPO_DEFAULT="$(dirname "${PKG_DIR_DEFAULT}")"                    # directory that contains behavior-2026/

usage() {
  cat <<EOF
Usage: $(basename "$0") --prefix DIR [options]

Create the ${NAME} worker env.

  --prefix DIR          env root (required): DIR/src/${FORK_DIR_NAME} (fork checkout) and DIR/venv (Python env)
  --repo DIR            directory containing behavior-2026/ (default: ${REPO_DEFAULT})
  --commit SHA          fork commit to install (default: ${FORK_COMMIT})
  --python VER          Python version for uv (default: ${PYTHON_VERSION}; the fork requires 3.10)
  --no-flash-attn       do not install the flash-attn wheel
  --download-backbone   download ${BACKBONE_REPO} (gated; needs HF_TOKEN) into the HF cache (HF_HOME)
  --install-uv          install uv with the official installer if it is missing
  --skip-check          do not import the fork after installing
  -h, --help            show this help
EOF
}

PREFIX=""
REPO="${REPO_DEFAULT}"
COMMIT="${FORK_COMMIT}"
NO_FLASH=0
DOWNLOAD_BACKBONE=0
INSTALL_UV=0
SKIP_CHECK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="${2:?--prefix needs a value}"; shift 2 ;;
    --repo) REPO="${2:?--repo needs a value}"; shift 2 ;;
    --commit) COMMIT="${2:?--commit needs a value}"; shift 2 ;;
    --python) PYTHON_VERSION="${2:?--python needs a value}"; shift 2 ;;
    --no-flash-attn) NO_FLASH=1; shift ;;
    --download-backbone) DOWNLOAD_BACKBONE=1; shift ;;
    --install-uv) INSTALL_UV=1; shift ;;
    --skip-check) SKIP_CHECK=1; shift ;;
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
if [[ "${DOWNLOAD_BACKBONE}" == 1 && -z "${HF_TOKEN:-}" ]]; then
  log "WARNING: HF_TOKEN is not set; ${BACKBONE_REPO} is gated and the download will fail without a token"
fi

SRC="${PREFIX}/src/${FORK_DIR_NAME}"
VENV="${PREFIX}/venv"
PY="${VENV}/bin/python"
export GIT_LFS_SKIP_SMUDGE=1   # the fork keeps demo data and wheels in LFS; the worker needs none of it

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
SYNC_ARGS=(--frozen --no-dev --python "${PYTHON_VERSION}")
if [[ "${NO_FLASH}" == 1 ]]; then SYNC_ARGS+=(--no-install-package flash-attn); fi
log "uv sync ${SYNC_ARGS[*]} -> ${VENV}"
(
  cd "${SRC}"
  UV_PROJECT_ENVIRONMENT="${VENV}" uv sync "${SYNC_ARGS[@]}"
)
[[ -x "${PY}" ]] || die "uv sync did not create ${PY}"

# ---- 3. b1k26 into the same env, without touching the locked versions ------------------------------------------
CONSTRAINTS="${PREFIX}/constraints.txt"
uv pip freeze --python "${PY}" | grep -v -E '^(-e |#)| @ ' > "${CONSTRAINTS}" || true
log "installing b1k26 from ${B1K26_DIR} (constraints: ${CONSTRAINTS})"
uv pip install --python "${PY}" -c "${CONSTRAINTS}" -e "${B1K26_DIR}"

# ---- 4. gated backbone ------------------------------------------------------------------------------------------
if [[ "${DOWNLOAD_BACKBONE}" == 1 ]]; then
  log "downloading ${BACKBONE_REPO} into the HF cache (${HF_HOME:-~/.cache/huggingface})"
  "${PY}" - "${BACKBONE_REPO}" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
print("  ", snapshot_download(repo_id=sys.argv[1]))
PYEOF
fi

# ---- 5. report --------------------------------------------------------------------------------------------------
cat > "${PREFIX}/ENV_INFO.txt" <<EOF
backend=${NAME}
fork=${FORK_URL}
commit=${HEAD_SHA}
flash_attn=$([[ "${NO_FLASH}" == 1 ]] && echo skipped || echo installed)
b1k26=${B1K26_DIR}
python=${PY}
created=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
log "versions:"
"${PY}" - <<'PYEOF'
import importlib.metadata as md
import platform
print(f"  python {platform.python_version()}")
for dist in ("torch", "transformers", "flash-attn", "numpy", "gr00t", "huggingface-hub", "albumentations",
             "websockets", "msgpack", "b1k26"):
    try:
        print(f"  {dist} {md.version(dist)}")
    except md.PackageNotFoundError:
        print(f"  {dist} (not installed)")
PYEOF
"${PY}" -c "import b1k26.backends.gr00t as m; print('  b1k26 backend module OK, pinned', m.GR00T_COMMIT)"

if [[ "${SKIP_CHECK}" == 0 ]]; then
  log "import check (gr00t policy module, r1pro modality slices, attention/dtype choice)"
  R1PRO_JSON="${SRC}/examples/b1k/r1pro.json" "${PY}" - <<'PYEOF'
import os
import torch
import gr00t.policy.gr00t_policy as gp  # noqa: F401
from b1k26.backends import gr00t as g
g.check_modality_json(os.environ["R1PRO_JSON"])
cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
flash = g.flash_attn_available()
print(f"  cuda available: {torch.cuda.is_available()}, capability {cc}, flash_attn installed: {flash}")
print(f"  backend choice (attn_implementation=auto, dtype=auto): attention "
      f"{g.decide_attention('auto', cc, flash)}, dtype {g.decide_dtype('auto', cc)}")
print("  import check OK")
PYEOF
fi
log "done: ${PY}"
