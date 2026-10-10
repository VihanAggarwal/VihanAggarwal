#!/usr/bin/env bash
# Create the worker Python envs named on the command line under ROOT/<name> (default ROOT=/opt/envs), using the
# per-family scripts in scripts/envs/. A serving config launches each worker as ROOT/<name>/venv/bin/python.
#
#   bash docker/install_envs.sh [--root /opt/envs] [--dry-run] NAME [NAME ...]
#
# NAME                  script                                              config launch interpreter
#   openpi_comet        scripts/envs/openpi_comet.sh                        /opt/envs/openpi_comet/venv/bin/python
#   openpi_b1k          scripts/envs/openpi_b1k.sh                          /opt/envs/openpi_b1k/venv/bin/python
#   gr00t               scripts/envs/gr00t.sh  (+ $GR00T_ENV_ARGS)          /opt/envs/gr00t/venv/bin/python
#   pibehavior-2025     scripts/envs/pibehavior.sh --fork rlc2025           /opt/envs/pibehavior-2025/venv/bin/python
#   pibehavior-2026     scripts/envs/pibehavior.sh --fork jackliu2026       /opt/envs/pibehavior-2026/venv/bin/python
#
# The env scripts install this checkout of b1k26 editable into each env, so it must live in a directory named
# behavior-2026 (the Docker image uses /opt/behavior-2026).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
ROOT="/opt/envs"
DRY=0
NAMES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="${2:?--root needs a value}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "install_envs: unknown option $1" >&2; exit 2 ;;
    *) NAMES+=("$1"); shift ;;
  esac
done
if [[ "$(basename "$PKG")" != "behavior-2026" ]]; then
  echo "install_envs: the package must be in a directory named behavior-2026 (got $PKG)" >&2
  exit 2
fi
for name in "${NAMES[@]}"; do
  case "$name" in
    openpi_comet|openpi_b1k) cmd=(bash "$PKG/scripts/envs/$name.sh" --prefix "$ROOT/$name") ;;
    gr00t)
      # shellcheck disable=SC2206
      extra=(${GR00T_ENV_ARGS:-})
      cmd=(bash "$PKG/scripts/envs/gr00t.sh" --prefix "$ROOT/gr00t" "${extra[@]}") ;;
    pibehavior-2025) cmd=(bash "$PKG/scripts/envs/pibehavior.sh" --prefix "$ROOT/$name" --fork rlc2025) ;;
    pibehavior-2026) cmd=(bash "$PKG/scripts/envs/pibehavior.sh" --prefix "$ROOT/$name" --fork jackliu2026) ;;
    *) echo "install_envs: unknown env name '$name' (openpi_comet openpi_b1k gr00t pibehavior-2025 pibehavior-2026)" >&2
       exit 2 ;;
  esac
  echo "[install_envs] ${cmd[*]}"
  if [[ "$DRY" == 0 ]]; then
    "${cmd[@]}"
    [[ -x "$ROOT/$name/venv/bin/python" ]] || { echo "install_envs: $ROOT/$name/venv/bin/python missing" >&2; exit 1; }
  fi
done
