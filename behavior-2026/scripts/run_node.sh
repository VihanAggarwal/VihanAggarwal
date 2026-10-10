#!/usr/bin/env bash
# Run one worker's evaluation jobs on one GPU: start the policy server, wait for /healthz, run jobs, stop.
#
#   bash scripts/run_node.sh --config configs/final.yaml --jobs jobs/final/worker_03.jsonl --out runs/final_node03 \
#        [--gpu 0] [--port 8000] [--wrapper omnigibson.eval.wrappers.RGBDFullResWrapper] \
#        [--extra "--replay-action-chunk-size 16"] [--no-video]
#
# Resumable: re-running the same command skips every job that already has a metrics JSON in --out. Jobs whose
# evaluator process produced no JSON because of an infrastructure failure (simulator crash, OOM) are retried once
# after checking (and if needed restarting) the server; a rollout that failed because of the policy is not re-run.
# Every attempt goes to status.jsonl with its exact command. If the server cannot be brought back
# (scripts/restart_server.sh gives up), the run stops with exit status 1 instead of running jobs against a dead server.
#
# Keep every evaluator flag identical on all nodes, and pass the same values to b1k26-package: --wrapper becomes its
# --wrapper, a "--replay-action-chunk-size K" in --extra becomes its --replay-chunk-size K. K must divide
# execution.execute_steps of every routed profile (b1k26-package --config checks it); no K (one query per step) is
# always safe.
#
# On a multi-GPU node, start one copy per GPU with different --gpu and --port values (and --out). Launched model
# workers that find their configured port taken move to a free loopback port by themselves.
set -euo pipefail

CONFIG=""; JOBS=""; OUT=""; GPU=0; PORT=8000; EXTRA=""; VIDEO="--write-video"; ENV_FILE="${ENV_FILE:-/workspace/b1k_env.sh}"
WRAPPER="omnigibson.eval.wrappers.RGBDFullResWrapper"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --wrapper) WRAPPER="$2"; shift 2 ;;
    --extra) EXTRA="$2"; shift 2 ;;
    --no-video) VIDEO="--no-write-video"; shift ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done
[[ -n "$CONFIG" && -n "$JOBS" && -n "$OUT" ]] || { echo "need --config --jobs --out"; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$ENV_FILE"
mkdir -p "$OUT"
SERVER_LOG="$OUT/server_gpu${GPU}_port${PORT}.log"
PIDFILE="$OUT/server_gpu${GPU}_port${PORT}.pid"

# The policy workers launched by the front server inherit CUDA_VISIBLE_DEVICES (one GPU per node slot). JAX must not
# grab the whole GPU (the simulator shares it): restart_server.sh sets XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 unless set.
RESTART=(bash "$HERE/restart_server.sh" --config "$CONFIG" --port "$PORT" --gpu "$GPU" --log "$SERVER_LOG"
         --pidfile "$PIDFILE" --health-timeout 1500 --tries 2)

stop_server() {
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill -TERM "$(cat "$PIDFILE")" || true
    for _ in $(seq 1 30); do kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break; sleep 1; done
    kill -KILL "$(cat "$PIDFILE")" 2>/dev/null || true
  fi
}
trap stop_server EXIT

"${RESTART[@]}" || { echo "[run_node] the policy server did not become healthy, see $SERVER_LOG"; exit 1; }
# After a failed attempt: if the server is not healthy, restart it (bounded; b1k26-plan stops the run if this fails).
HEALTH_CMD="curl -sf --noproxy '*' --max-time 5 http://127.0.0.1:$PORT/healthz >/dev/null || $(printf '%q ' "${RESTART[@]}")"

# The simulator selects its GPU through OMNIGIBSON_GPU_ID (Vulkan does not follow CUDA_VISIBLE_DEVICES).
OMNIGIBSON_GPU_ID="$GPU" b1k26-plan run --jobs "$JOBS" --output-dir "$OUT" --host 127.0.0.1 --port "$PORT" \
  --wrapper "$WRAPPER" $([[ "$VIDEO" == "--no-write-video" ]] && echo "--no-write-video") \
  --extra "$EXTRA" --health-cmd "$HEALTH_CMD" --health-timeout-s 3600
