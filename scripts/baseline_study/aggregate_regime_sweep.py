#!/usr/bin/env python3
"""Aggregate the Chronos-2 fine-tune regime sweep into ONE plot-ready CSV.

Each regime point (zero-shot + N trailing weeks 2..11 + full) ran through the
identical leak-free harness and wrote a ``result.json`` with both TEST accuracy
and the runtime resource primitives. This script collapses every Chronos-2 run
under an output root into a single row per regime point so the user can plot:

  * accuracy-vs-weeks  (mean_sMAPE / mean_MASE / per-KPI ... vs finetune_weeks)
  * resource-vs-weeks  (elapsed_sec / energy_kwh / zeus_gpu_energy_j / ... )

The x-axis (``finetune_weeks``) is the amount of fine-tune data in trailing weeks
ending at day 81; the forecast target (day 82-88) is fixed. ``0`` is zero-shot and
``"full"`` is the entire pre-test history. Rows are sorted zero -> 2..11 -> full.

    python scripts/baseline_study/aggregate_regime_sweep.py \
        --root output/baseline_study_v2/full/chronos \
        --out  output/baseline_study_v2/regime_sweep_chronos2.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any, Dict, List

# 7-KPI canonical order (TS 28.554). Per-KPI columns follow this order.
KPIS = ["RRC_Conn", "DL_CQI", "DL_iBler", "DL_rBler", "MAC_DL_Eff", "PRB_Util", "Throughput"]

MEAN_METRICS = ["mean_sMAPE", "mean_MASE", "mean_nRMSE", "mean_MAE", "mean_MSE"]
RESOURCE_KEYS = [
    "elapsed_sec",
    "peak_gpu_mem_gb",
    "gpu_util_mean",
    "energy_kwh",
    "zeus_gpu_energy_j",
    "co2_kg",
    "water_liters",
]

# Stable column order for the plot CSV.
HEADER = (
    ["finetune_weeks", "regime", "n_train_windows", "n_params"]
    + MEAN_METRICS
    + [f"sMAPE_{k}" for k in KPIS]
    + [f"MASE_{k}" for k in KPIS]
    + [f"MAE_{k}" for k in KPIS]
    + [f"nRMSE_{k}" for k in KPIS]
    + RESOURCE_KEYS
    + ["run_dir"]
)


def _weeks_sort_key(value: Any) -> tuple[int, int]:
    """zero(0) first, then ascending weeks, then "full" last."""
    if value == "full":
        return (2, 0)
    weeks = int(value)
    if weeks == 0:
        return (0, 0)
    return (1, weeks)


def _row_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    model = result.get("model", {})
    test = result["test"]
    resources = result.get("resources", {})
    per_kpi = test["per_kpi"]

    row: Dict[str, Any] = {
        "finetune_weeks": model.get("finetune_weeks", resources.get("finetune_weeks")),
        "regime": model.get("regime", resources.get("regime")),
        "n_train_windows": model.get("n_train_windows", resources.get("n_train_windows")),
        "n_params": resources.get("n_params"),
        "run_dir": result.get("run_dir"),
    }
    for metric in MEAN_METRICS:
        row[metric] = test.get(metric)
    for kpi in KPIS:
        row[f"sMAPE_{kpi}"] = per_kpi[kpi]["sMAPE"]
        row[f"MASE_{kpi}"] = per_kpi[kpi]["MASE"]
        row[f"MAE_{kpi}"] = per_kpi[kpi]["MAE"]
        row[f"nRMSE_{kpi}"] = per_kpi[kpi]["nRMSE"]
    for key in RESOURCE_KEYS:
        row[key] = resources.get(key)
    return row


def collect_regime_rows(root: str | Path) -> List[Dict[str, Any]]:
    """Collapse every regime-sweep Chronos-2 ``result.json`` under ``root`` into one row.

    Kept only if ``model_family == "chronos"`` AND the run dir is a regime point
    (``chronos2_ftw*``). This EXCLUDES the zero-shot context-sweep runs (``chronos2_c*``
    from the ``tsfm`` group) that share the ``chronos/chronos2/`` namespace and also
    carry ``finetune_weeks=0`` — without this filter they would collide with
    ``chronos2_ftw0`` as duplicate ``weeks=0`` rows. Rows are sorted zero-shot ->
    ascending weeks -> full. Raises if a kept result misses the test per-KPI block.
    """
    rows: List[Dict[str, Any]] = []
    for rp in sorted(glob.glob(str(Path(root) / "**" / "result.json"), recursive=True)):
        with open(rp, encoding="utf-8") as fh:
            result = json.load(fh)
        if result.get("model", {}).get("model_family") != "chronos":
            continue
        # Regime-sweep runs only (run dir name chronos2_ftw{weeks}_c168). This
        # positive allowlist also excludes the zero-shot context sweep
        # (chronos2_c{ctx}) AND the cumulative-group spatial sweep (chronos2_cum{G})
        # that share the chronos/chronos2/ namespace, so the temporal and spatial
        # CSVs never cross-contaminate (aggregate_spatial_sweep.py owns chronos2_cum*).
        if not Path(rp).parent.name.startswith("chronos2_ftw"):
            continue
        if "per_kpi" not in result.get("test", {}):
            raise ValueError(f"{rp}: chronos result has no test per-KPI block")
        rows.append(_row_from_result(result))
    rows.sort(key=lambda r: _weeks_sort_key(r["finetune_weeks"]))
    return rows


def write_regime_sweep_csv(rows: List[Dict[str, Any]], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in HEADER})


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate the Chronos-2 regime sweep")
    parser.add_argument(
        "--root",
        default="output/baseline_study_v2/full/chronos",
        help="Output root containing the chronos regime-sweep run dirs",
    )
    parser.add_argument(
        "--out",
        default="output/baseline_study_v2/regime_sweep_chronos2.csv",
        help="Destination CSV (one row per regime point)",
    )
    args = parser.parse_args(argv)

    rows = collect_regime_rows(args.root)
    write_regime_sweep_csv(rows, args.out)
    print(f"wrote {args.out}: {len(rows)} regime points")
    for row in rows:
        print(
            f"  weeks={str(row['finetune_weeks']):>4s} regime={row['regime']:<5s} "
            f"n_win={row['n_train_windows']} mean_sMAPE={row['mean_sMAPE']} "
            f"elapsed_sec={row['elapsed_sec']} energy_kwh={row['energy_kwh']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
