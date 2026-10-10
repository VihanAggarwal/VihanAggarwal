#!/usr/bin/env bash
# Run one worker's evaluation jobs on one GPU: start the policy server, wait for /healthz, run jobs, stop.
#
#   bash scripts/run_node.sh --config configs/final.yaml --jobs jobs/final/worker_03.jsonl --out runs/final \
#        [--gpu 0] [--port 8000] [--extra "--replay-action-chunk-size 20"] [--no-video]
#
# Resumable: re-running the same command skips every job that already has a metrics JSON in --out. Jobs whose
# evaluator process produced no JSON (simulator crash, OOM) are retried once; every attempt goes to status.jsonl.
# On a multi-GPU node, start one copy per GPU with different --gpu and --port values.
set -euo pipefail

CONFIG=""; JOBS=""; OUT=""; GPU=0; PORT=8000; EXTRA=""; VIDEO="--write-video"; ENV_FILE="${ENV_FILE:-/workspace/b1k_env.sh}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --extra) EXTRA="$2"; shift 2 ;;
    --no-video) VIDEO="--no-write-video"; shift ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done
[[ -n "$CONFIG" && -n "$JOBS" && -n "$OUT" ]] || { echo "need --config --jobs --out"; exit 2; }
# shellcheck disable=SC1090
source "$ENV_FILE"
mkdir -p "$OUT"
SERVER_LOG="$OUT/server_gpu${GPU}_port${PORT}.log"
PIDFILE="$OUT/server_gpu${GPU}_port${PORT}.pid"

start_server() {
  # The policy workers launched by the front server inherit CUDA_VISIBLE_DEVICES (one GPU per node slot).
  # JAX must not grab the whole GPU: the simulator shares it.
  CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 \
    nohup b1k26-serve --config "$CONFIG" --ports "$PORT" >> "$SERVER_LOG" 2>&1 &
  echo $! > "$PIDFILE"
  echo "[run_node] server pid $(cat "$PIDFILE"), log $SERVER_LOG"
  local t0; t0=$(date +%s)
  until curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null; do
    if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then echo "[run_node] server died, see $SERVER_LOG"; tail -50 "$SERVER_LOG"; exit 1; fi
    if (( $(date +%s) - t0 > 1800 )); then echo "[run_node] server not healthy after 30 min"; exit 1; fi
    sleep 5
  done
  echo "[run_node] server healthy after $(( $(date +%s) - t0 )) s"
}

stop_server() {
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill -TERM "$(cat "$PIDFILE")" || true
    for _ in $(seq 1 30); do kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break; sleep 1; done
    kill -KILL "$(cat "$PIDFILE")" 2>/dev/null || true
  fi
}
trap stop_server EXIT

start_server
# After a failed attempt, make sure the server is still alive (restart it if not) before the retry.
HEALTH_CMD="curl -sf http://127.0.0.1:$PORT/healthz >/dev/null || (kill \$(cat $PIDFILE) 2>/dev/null; sleep 5; CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 nohup b1k26-serve --config $CONFIG --ports $PORT >> $SERVER_LOG 2>&1 & echo \$! > $PIDFILE; until curl -sf http://127.0.0.1:$PORT/healthz >/dev/null; do sleep 5; done)"

# The simulator selects its GPU through OMNIGIBSON_GPU_ID (Vulkan does not follow CUDA_VISIBLE_DEVICES).
OMNIGIBSON_GPU_ID="$GPU" b1k26-plan run --jobs "$JOBS" --output-dir "$OUT" --host 127.0.0.1 --port "$PORT" \
  $([[ "$VIDEO" == "--no-write-video" ]] && echo "--no-write-video") \
  --extra "$EXTRA" --health-cmd "$HEALTH_CMD"
