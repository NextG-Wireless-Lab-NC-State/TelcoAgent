#!/usr/bin/env bash
# Wait until the running baseline GPU sweep frees the GPU (sole-occupancy is
# required: zeus/codecarbon/util are device-level, so a co-resident job would
# contaminate the Green-AI metrics), then run the Chronos-2 regime sweep alone
# and collapse it to the plot-ready CSV.
set -uo pipefail
cd /home/gkim26/Desktop/workplace/telcoagent

CHRONOS_PY=/home/gkim26/miniconda3/envs/telco-chronos-test/bin/python
TELCO_PY=/home/gkim26/miniconda3/envs/telcoagent/bin/python
# Baseline lifecycle PIDs (driver + launcher). When both exit, the sweep is done.
BASELINE_DRIVER=2798055
BASELINE_LAUNCHER=2798048

echo "[$(date '+%F %T')] watcher up; waiting for baseline (PID $BASELINE_DRIVER/$BASELINE_LAUNCHER) to finish"
while kill -0 "$BASELINE_DRIVER" 2>/dev/null || kill -0 "$BASELINE_LAUNCHER" 2>/dev/null; do
  sleep 300
done
sleep 60  # settle: let GPU memory drain before sole-occupancy run

echo "[$(date '+%F %T')] baseline done; GPU free -> launching chronos_regime (12 points, sole occupancy)"
# Clear any prior regime run dirs (e.g. pre-fix smoke runs) so ALL 12 points are
# measured FRESH under this guaranteed sole-occupancy window. The resume guard in
# run_benchmark.py would otherwise SKIP existing points, splicing their (possibly
# co-resident) Green-AI metrics into the final curve.
rm -rf output/baseline_study_v2/full/chronos/chronos2/chronos2_ftw*_c168
# Generous per-run alarm: 'full' fine-tune is ~20.7k steps; default 7200s could clip it.
export BASELINE_RUN_TIMEOUT_SEC=21600
"$CHRONOS_PY" scripts/baseline_study/run_benchmark.py --group chronos_regime
rc=$?
echo "[$(date '+%F %T')] chronos_regime exited rc=$rc -> aggregating plot-ready CSV"

"$TELCO_PY" scripts/baseline_study/aggregate_regime_sweep.py \
  --root output/baseline_study_v2/full/chronos \
  --out  output/baseline_study_v2/regime_sweep_chronos2.csv
echo "[$(date '+%F %T')] DONE -> output/baseline_study_v2/regime_sweep_chronos2.csv"
