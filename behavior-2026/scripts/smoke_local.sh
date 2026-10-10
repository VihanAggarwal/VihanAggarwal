#!/usr/bin/env bash
# End-to-end smoke test of the policy server with the evaluator-faithful probe (b1k26-probe).
#
#   bash scripts/smoke_local.sh [--config configs/fake.yaml] [--ports 18000-18003] [--steps 200]
#   bash scripts/smoke_local.sh --no-start --host 127.0.0.1 --ports 8000-8002     # against a running server/container
#
# 1. starts `b1k26-serve --config CONFIG --ports PORTS` in the background (unless --no-start) and waits for /healthz;
# 2. probes the first port at full resolution (720x720 head + 2x480x480 wrists, RGBA + depth: ~7.8 MB per
#    observation): with chunk replay (__action_chunk_size__ = --chunk, default 20) and with one request per step;
# 3. probes two more ports concurrently (multi-port evaluation: one rollout per port), then a batched (N=3)
#    observation at 224 px on the first port (single-port v3.9.3 evaluation);
# 4. checks /status (workers ready, no config problems, no plan failures) and the median full-res step latency
#    (--max-p50-ms, default 100; 0 disables);
# 5. stops the server with SIGTERM and requires exit status 0 (unless --no-start).
# Exit status 0 when everything passed, 1 otherwise. Logs and probe JSON go to --out (default: a temp dir).
set -uo pipefail

CONFIG="configs/fake.yaml"
PORTS="18000-18003"
HOST="127.0.0.1"
STEPS=200
CHUNK=20
MAX_P50_MS=100
HEALTH_TIMEOUT=600
START=1
OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --ports) PORTS="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --chunk) CHUNK="$2"; shift 2 ;;
    --max-p50-ms) MAX_P50_MS="$2"; shift 2 ;;
    --health-timeout) HEALTH_TIMEOUT="$2"; shift 2 ;;
    --no-start) START=0; shift ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "smoke_local: unknown argument $1" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO" || exit 2
PY="${PYTHON:-python3}"
if command -v b1k26-serve >/dev/null && command -v b1k26-probe >/dev/null; then
  SERVE=(b1k26-serve); PROBE=(b1k26-probe); PY="$(dirname "$(command -v b1k26-serve)")/python"
  [[ -x "$PY" ]] || PY="${PYTHON:-python3}"
else
  SERVE=("$PY" -m b1k26.server); PROBE=("$PY" -m b1k26.client)
fi
[[ -n "$OUT" ]] || OUT="$(mktemp -d "${TMPDIR:-/tmp}/b1k26_smoke.XXXXXX")"
mkdir -p "$OUT"
LOG="$OUT/server.log"

mapfile -t PORT_LIST < <("$PY" -c "import sys; from b1k26.config import parse_ports; print(*parse_ports(sys.argv[1]), sep='\n')" "$PORTS") \
  || { echo "smoke_local: bad --ports $PORTS" >&2; exit 2; }
if (( ${#PORT_LIST[@]} < 3 )); then echo "smoke_local: need at least 3 ports, got ${PORTS}" >&2; exit 2; fi
P0="${PORT_LIST[0]}"; P1="${PORT_LIST[1]}"; P2="${PORT_LIST[2]}"

FAILS=()
fail() { FAILS+=("$1"); echo "[smoke] FAIL: $1"; }
log() { echo "[smoke] $*"; }

SERVER_PID=""
stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.5; done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -KILL "$SERVER_PID" 2>/dev/null
      fail "server did not exit within 30 s of SIGTERM"
    fi
  fi
  wait "$SERVER_PID" 2>/dev/null
  local rc=$?
  SERVER_PID=""
  return $rc
}
trap 'stop_server >/dev/null 2>&1' EXIT

# ---- 1. start + health -------------------------------------------------------------------------------------
if [[ "$START" == 1 ]]; then
  log "starting: ${SERVE[*]} --config $CONFIG --ports $PORTS (log: $LOG)"
  "${SERVE[@]}" --config "$CONFIG" --ports "$PORTS" --host "$HOST" >"$LOG" 2>&1 &
  SERVER_PID=$!
fi
t0=$(date +%s)
until "$PY" -c "import sys, urllib.request as u; o = u.build_opener(u.ProxyHandler({})); sys.exit(0 if o.open(sys.argv[1], timeout=2).status == 200 else 1)" \
      "http://$HOST:$P0/healthz" 2>/dev/null; do
  if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
    wait "$SERVER_PID"; echo "[smoke] server exited with status $? before becoming healthy:"; tail -30 "$LOG"; exit 1
  fi
  if (( $(date +%s) - t0 > HEALTH_TIMEOUT )); then
    echo "[smoke] /healthz on port $P0 not 200 after ${HEALTH_TIMEOUT} s"; [[ -f "$LOG" ]] && tail -30 "$LOG"; exit 1
  fi
  sleep 0.5
done
log "healthy after $(( $(date +%s) - t0 )) s"

# ---- 2./3. probes ----------------------------------------------------------------------------------------------
probe() {  # probe NAME PORT ARGS... ; writes $OUT/NAME.json, returns the probe's exit status
  local name="$1" port="$2"; shift 2
  "${PROBE[@]}" --host "$HOST" --port "$port" --json --health-timeout 30 "$@" >"$OUT/$name.json" 2>"$OUT/$name.err"
}
summarize() {  # summarize NAME -> one line
  "$PY" - "$OUT/$1.json" "$1" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"{sys.argv[2]}: no JSON summary ({e})"); sys.exit(0)
lat = d["latency_ms"]
print(f"{sys.argv[2]}: ok={d['ok']} steps={d['steps']} requests={d['requests']} steps/s={d['steps_per_s']} "
      f"request ms p50={lat['p50']} p90={lat['p90']} p99={lat['p99']} max={lat['max']}"
      + (f" violations={d['violations']}" if d["violations"] else ""))
EOF
}
run_probe() {  # run_probe NAME PORT ARGS...
  local name="$1" rc
  probe "$@"; rc=$?
  if [[ $rc == 0 ]]; then log "$(summarize "$name")"
  else fail "$(summarize "$name") (exit $rc; stderr: $(tail -2 "$OUT/$name.err" | tr '\n' ' '))"; fi
}

run_probe full_chunk "$P0" --steps "$STEPS" --res full --chunk "$CHUNK" --task-id 0
run_probe full_step "$P0" --steps "$STEPS" --res full --chunk 0 --task-id 50

log "two ports concurrently (full res: chunk $CHUNK on $P1, one request per step on $P2)"
probe multi_a "$P1" --steps "$STEPS" --res full --chunk "$CHUNK" --task-id 10 & pa=$!
probe multi_b "$P2" --steps "$STEPS" --res full --chunk 0 --task-id 99 & pb=$!
wait $pa; ra=$?; wait $pb; rb=$?
if [[ $ra == 0 ]]; then log "$(summarize multi_a)"; else fail "$(summarize multi_a) (exit $ra)"; fi
if [[ $rb == 0 ]]; then log "$(summarize multi_b)"; else fail "$(summarize multi_b) (exit $rb)"; fi

run_probe batched_224 "$P0" --steps 100 --res 224 --batch 3 --chunk "$CHUNK" --task-id 0,50,99

# ---- 4. status + latency ---------------------------------------------------------------------------------------
if ! "$PY" - "http://$HOST:$P0/status" "$OUT/full_step.json" "$MAX_P50_MS" <<'EOF'
import json, sys, urllib.request as u
st = json.load(u.build_opener(u.ProxyHandler({})).open(sys.argv[1], timeout=5))
bad = []
eng = st["engine"]
for name, w in eng["workers"].items():
    if w["state"] != "ready":
        bad.append(f"worker {name} is {w['state']} ({w.get('last_error')})")
bad += [f"config problem: {p}" for p in eng.get("config_problems", [])]
print(f"[smoke] status: healthy={st['healthy']} counters={st['counters']} batches={eng['batches']}")
limit = float(sys.argv[3])
try:
    p50 = json.load(open(sys.argv[2]))["latency_ms"]["p50"]
    print(f"[smoke] full-res per-step request p50 = {p50} ms (limit {limit:g} ms)")
    if limit > 0 and p50 > limit:
        bad.append(f"full-res per-step p50 {p50} ms > {limit:g} ms")
except Exception as e:
    bad.append(f"no full_step latency: {e}")
for b in bad:
    print(f"[smoke] FAIL: {b}")
sys.exit(1 if bad else 0)
EOF
then
  FAILS+=("status/latency check")
fi

# ---- 5. shutdown -----------------------------------------------------------------------------------------------
if [[ "$START" == 1 ]]; then
  stop_server; rc=$?
  if [[ $rc != 0 ]]; then fail "server exited with status $rc after SIGTERM"; else log "server stopped cleanly"; fi
  # Every rollout logs one "rollout end" line (at reset or shutdown) with its counters.
  n_end=$(grep -c "rollout end" "$LOG")
  log "server log: $n_end rollout summaries"
  if grep -qE "plan_failures=[1-9]|hold_steps=[1-9]" "$LOG"; then fail "the server log reports plan failures or hold steps (see $LOG)"; fi
  if grep -qE "Traceback|CRITICAL" "$LOG"; then fail "tracebacks or critical errors in $LOG"; fi
fi

if (( ${#FAILS[@]} )); then
  echo "[smoke] FAILED (${#FAILS[@]}): ${FAILS[*]}"
  echo "[smoke] artifacts: $OUT"
  exit 1
fi
echo "[smoke] PASSED (artifacts: $OUT)"
exit 0
