#!/usr/bin/env bash
# (Re)start the policy server for one GPU slot and wait until it is healthy, with bounded waits.
#
#   bash scripts/restart_server.sh --config C --port P --log LOG --pidfile PIDFILE [--gpu G]
#        [--health-timeout 1500] [--tries 2] [--pause 10]
#
# 1. Stops the server recorded in PIDFILE, if it is still running: SIGTERM, up to 30 s, then SIGKILL. Its launched
#    workers die with it (PR_SET_PDEATHSIG + a parent watchdog); a worker port still held meanwhile is no problem, the
#    new server moves its worker to a free port.
# 2. Starts `b1k26-serve --config C --ports P` in the background (CUDA_VISIBLE_DEVICES=G) and records its pid.
# 3. Polls http://127.0.0.1:P/healthz until 200, checking that the server is still alive, for at most
#    --health-timeout seconds.
# 4. On failure, stops it and tries again (--tries attempts in total, --pause seconds apart, doubled each time).
# Exit status 0 once healthy, 1 if every attempt failed (the last lines of LOG are printed). Never waits forever:
# total time <= tries x (health timeout + 40 s) + pauses.
set -uo pipefail

CONFIG=""; PORT=""; LOG=""; PIDFILE=""; GPU="${CUDA_VISIBLE_DEVICES:-0}"; HEALTH_TIMEOUT=1500; TRIES=2; PAUSE=10
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --log) LOG="$2"; shift 2 ;;
    --pidfile) PIDFILE="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --health-timeout) HEALTH_TIMEOUT="$2"; shift 2 ;;
    --tries) TRIES="$2"; shift 2 ;;
    --pause) PAUSE="$2"; shift 2 ;;
    -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "restart_server: unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$CONFIG" && -n "$PORT" && -n "$LOG" && -n "$PIDFILE" ]] || { echo "restart_server: need --config --port --log --pidfile" >&2; exit 2; }

if command -v b1k26-serve >/dev/null; then SERVE=(b1k26-serve); else SERVE=("${PYTHON:-python3}" -m b1k26.server); fi
say() { echo "[restart_server] $*"; }

healthy() {
  curl -sf --noproxy '*' --max-time 3 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1
}

stop_recorded() {
  [[ -f "$PIDFILE" ]] || return 0
  local pid; pid="$(cat "$PIDFILE" 2>/dev/null)"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  say "stopping server pid $pid"
  kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || return 0; sleep 1; done
  say "server pid $pid ignored SIGTERM for 30 s; killing it"
  kill -KILL "$pid" 2>/dev/null
  for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || return 0; sleep 1; done
  return 0
}

pause="$PAUSE"
for attempt in $(seq 1 "$TRIES"); do
  stop_recorded
  mkdir -p "$(dirname "$LOG")"
  say "attempt $attempt/$TRIES: ${SERVE[*]} --config $CONFIG --ports $PORT (GPU $GPU, log $LOG)"
  CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}" \
    B1K26_MEM_FRACTION="${B1K26_MEM_FRACTION:-0.40}" \
    nohup "${SERVE[@]}" --config "$CONFIG" --ports "$PORT" >> "$LOG" 2>&1 &
  pid=$!
  echo "$pid" > "$PIDFILE"
  t0=$(date +%s)
  last_note=$t0
  while true; do
    if healthy; then
      say "server pid $pid healthy after $(( $(date +%s) - t0 )) s"
      exit 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      say "server pid $pid exited before becoming healthy"
      break
    fi
    now=$(date +%s)
    if (( now - t0 > HEALTH_TIMEOUT )); then
      say "server pid $pid not healthy after ${HEALTH_TIMEOUT} s"
      break
    fi
    if (( now - last_note >= 60 )); then
      say "waiting for /healthz ($(( now - t0 )) s): $(tail -1 "$LOG" 2>/dev/null | cut -c1-200)"
      last_note=$now
    fi
    sleep 2
  done
  stop_recorded
  if (( attempt < TRIES )); then sleep "$pause"; pause=$(( pause * 2 )); fi
done
say "giving up after $TRIES attempt(s); last lines of $LOG:"
tail -30 "$LOG" 2>/dev/null
exit 1
