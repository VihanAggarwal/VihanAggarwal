#!/usr/bin/env bash
# Build-time check that every installed model family can load without network access: what each one fetches at
# load time must already be in the image. The Dockerfile runs it with RUN --network=none.
#
#   bash docker/check_offline.sh [ROOT]          (ROOT default /opt/envs: the envs are ROOT/<name>/venv)
#
# - envs with openpi (openpi_comet, openpi_b1k, pibehavior-*): pi0.5 model transforms build a PaligemmaTokenizer,
#   which reads gs://big_vision/paligemma_tokenizer.model from $OPENPI_DATA_HOME (downloads it if missing).
# - the gr00t env: Gr00tN1d7 loads its backbone and processor from the gated HF repo nvidia/Cosmos-Reason2-2B,
#   which must be in the HF cache ($HF_HOME); docker/build.sh downloads it with HF_TOKEN.
# Exit status 0 if everything loads offline, 1 otherwise.
set -uo pipefail
ROOT="${1:-/opt/envs}"
fail=0
shopt -s nullglob
for py in "$ROOT"/*/venv/bin/python; do
  name="$(basename "$(dirname "$(dirname "$(dirname "$py")")")")"
  if "$py" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('openpi') else 1)" 2>/dev/null; then
    echo "[check_offline] $name: PaliGemma tokenizer (OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-unset})"
    if ! "$py" - <<'EOF'
import importlib
import os
import sys

home = os.environ.get("OPENPI_DATA_HOME")
if not home:
    sys.exit("  OPENPI_DATA_HOME is not set: openpi would use ~/.cache/openpi, which is empty at run time")
tok = importlib.import_module("openpi.models.tokenizer")
if not hasattr(tok, "PaligemmaTokenizer"):
    print("  this fork has no PaligemmaTokenizer: nothing to check")
else:
    tok.PaligemmaTokenizer(48)
    print("  ok")
EOF
    then
      echo "[check_offline] $name: the PaliGemma tokenizer is not available offline"; fail=1
    fi
  fi
  if [[ "$name" == gr00t ]]; then
    echo "[check_offline] gr00t: nvidia/Cosmos-Reason2-2B in the HF cache (HF_HOME=${HF_HOME:-unset})"
    if ! HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$py" - <<'EOF'
from huggingface_hub import snapshot_download
from transformers import AutoConfig

path = snapshot_download("nvidia/Cosmos-Reason2-2B", local_files_only=True)
AutoConfig.from_pretrained("nvidia/Cosmos-Reason2-2B")
print("  ok:", path)
EOF
    then
      echo "[check_offline] gr00t: the backbone is not in the image (build with docker/build.sh and HF_TOKEN set)"; fail=1
    fi
  fi
done
exit $fail
