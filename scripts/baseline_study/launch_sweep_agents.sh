#!/usr/bin/env bash
# Launch W&B sweep agents for the expanded TelcoAgent baseline catalog.
#
# Example:
#   bash scripts/baseline_study/launch_sweep_agents.sh 50
#
# If you have already run `wandb login`, no WANDB_API_KEY env var is needed.
#
# The optional first argument is the per-sweep agent count. Use multiple
# terminals or tmux panes if you want parallel GPU workers.

set -euo pipefail

COUNT="${1:-50}"
PYBIN="${PYBIN:-conda run -n telcoagent python}"
PROJECT="${WANDB_PROJECT:-telcoagent-baselines}"

export BASELINE_RUN_TIMEOUT_SEC="${BASELINE_RUN_TIMEOUT_SEC:-7200}"
export BASELINE_GPU_MEM_FRAC="${BASELINE_GPU_MEM_FRAC:-0.45}"

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

SWEEPS=(
  scripts/baseline_study/configs/sweeps/statistical_grid.yaml
  scripts/baseline_study/configs/sweeps/classical_linear.yaml
  scripts/baseline_study/configs/sweeps/classical_tree.yaml
)
# Note: GPU sweeps (darts_*, thuml_*) moved to dedicated launchers
# launch_darts.sh and launch_thuml.sh for parallel execution.

for sweep in "${SWEEPS[@]}"; do
  echo "[$(date '+%F %T')] Launching ${sweep} count=${COUNT}"
  ${PYBIN} scripts/baseline_study/sweep.py \
    --project "${PROJECT}" \
    --sweep-config "${sweep}" \
    --count "${COUNT}"
done

echo "[$(date '+%F %T')] Baseline sweep launch loop complete."
