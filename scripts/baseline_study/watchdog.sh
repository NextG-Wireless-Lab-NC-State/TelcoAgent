#!/usr/bin/env bash
# Cron-friendly watchdog. Re-launches any of the 3 supervised sweeps
# (CPU sweep_agents, GPU darts, GPU thuml) if their supervisor is no
# longer alive AND no done-marker exists. Each sweep uses an independent
# pid/done file so OOM in one (which kills launch_*.sh) doesn't block
# the others from being checked.
set -uo pipefail

ROOT="/home/gkim26/Desktop/workplace/telcoagent"
LOG_DIR="$ROOT/output/baseline_study/logs"
mkdir -p "$LOG_DIR"

LOCK="$LOG_DIR/watchdog.lock"
WD_LOG="$LOG_DIR/watchdog.log"

exec 200>"$LOCK"
flock -n 200 || exit 0

restart_if_needed() {
  local label="$1"
  local supervisor="$2"
  local count="$3"
  local launch_pattern="$4"
  local pid_file="$LOG_DIR/${label}_supervised.pid"
  local done_file="$LOG_DIR/${label}_supervised.done"

  # Already finished — nothing to do.
  [ -f "$done_file" ] && return 0

  # Supervisor PID still alive — nothing to do.
  if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    return 0
  fi

  # Belt-and-braces: any leftover launcher in flight.
  if pgrep -f "$launch_pattern" >/dev/null; then
    return 0
  fi

  echo "[$(date '+%F %T')] watchdog restarting $label supervisor (count=$count)" >> "$WD_LOG"
  cd "$ROOT"
  nohup bash "$supervisor" "$count" >/dev/null 2>&1 &
  disown
}

restart_if_needed \
  "sweep" \
  "scripts/baseline_study/run_sweep_supervised.sh" \
  30 \
  "scripts/baseline_study/launch_sweep_agents.sh"

restart_if_needed \
  "darts" \
  "scripts/baseline_study/run_darts_supervised.sh" \
  30 \
  "scripts/baseline_study/launch_darts.sh"

restart_if_needed \
  "thuml" \
  "scripts/baseline_study/run_thuml_supervised.sh" \
  30 \
  "scripts/baseline_study/launch_thuml.sh"
