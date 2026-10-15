#!/usr/bin/env bash
# Wrapper around launch_darts.sh that:
#  - sets cron-safe PYBIN/PATH
#  - writes pid/log/done markers under output/baseline_study/logs
#  - exits 0 only when the full sweep loop completes naturally
set -uo pipefail

COUNT="${1:-30}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$ROOT/output/baseline_study/logs"
mkdir -p "$LOG_DIR"

LOG="$LOG_DIR/darts_supervised.log"
DONE_FILE="$LOG_DIR/darts_supervised.done"
PID_FILE="$LOG_DIR/darts_supervised.pid"

export PYBIN="${PYBIN:-/home/gkim26/miniconda3/envs/telcoagent/bin/python}"
export PATH="/home/gkim26/miniconda3/envs/telcoagent/bin:$PATH"
export BASELINE_RUN_TIMEOUT_SEC="${BASELINE_RUN_TIMEOUT_SEC:-7200}"
export BASELINE_GPU_MEM_FRAC="${BASELINE_GPU_MEM_FRAC:-0.45}"
export WANDB_DIR="${WANDB_DIR:-$ROOT/output/baseline_study/wandb_logs}"
mkdir -p "$WANDB_DIR"

# Avoid double-start
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "[$(date '+%F %T')] [DARTS] supervisor already running pid=$(cat "$PID_FILE"); exiting" >> "$LOG"
  exit 0
fi

echo $$ > "$PID_FILE"
echo "[$(date '+%F %T')] [DARTS] supervised start count=$COUNT pid=$$" >> "$LOG"

cd "$ROOT"
bash scripts/baseline_study/launch_darts.sh "$COUNT" >> "$LOG" 2>&1
RC=$?

echo "[$(date '+%F %T')] [DARTS] supervised exit rc=$RC" >> "$LOG"
if [ "$RC" -eq 0 ]; then
  touch "$DONE_FILE"
fi
rm -f "$PID_FILE"
exit "$RC"
