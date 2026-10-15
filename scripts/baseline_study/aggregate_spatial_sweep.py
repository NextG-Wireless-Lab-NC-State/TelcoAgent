#!/usr/bin/env python3
"""Aggregate the Chronos-2 CUMULATIVE group sweep into ONE plot-ready CSV.

Each point G fine-tunes on ALL stations of the first G ranked site groups and
evaluates on the remaining (13-G) groups. The test set shrinks as G grows:
G=0 evaluates all 115 stations (pure zero-shot); G=12 evaluates only the last
ranked group. Every point runs through the identical leak-free harness and writes
a ``result.json`` with TEST accuracy + runtime resource primitives. This script
collapses every run under an output root into one row per G so the user can plot:

  * accuracy-vs-n_train_groups  (mean_sMAPE / mean_MASE / per-KPI ...)
  * resource-vs-n_train_groups  (elapsed_sec / energy_kwh / zeus_gpu_energy_j ...)

Rows are sorted ascending by ``n_train_groups`` (0 first).

SELECTOR: only chronos runs whose run-dir name starts ``chronos2_cum`` are
collected, so the temporal regime sweep (``chronos2_ftw*``) and the zero-shot
context sweep (``chronos2_c*``) sharing the ``chronos/chronos2/`` namespace are
never mixed in.

    python scripts/baseline_study/aggregate_spatial_sweep.py \
        --root output/baseline_study_v2/full/chronos \
        --out  output/baseline_study_v2/regime_sweep_chronos2_cumulative.csv
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

# Stable column order for the plot CSV (x-axis = n_train_groups).
HEADER = (
    [
        "n_train_groups",
        "n_train_stations",
        "n_test_groups",
        "n_test_stations",
        "regime",
        "n_train_windows",
        "n_params",
    ]
    + MEAN_METRICS
    + [f"sMAPE_{k}" for k in KPIS]
    + [f"MASE_{k}" for k in KPIS]
    + RESOURCE_KEYS
    + ["run_dir"]
)


def _row_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    model = result.get("model", {})
    test = result["test"]
    resources = result.get("resources", {})
    per_kpi = test["per_kpi"]

    row: Dict[str, Any] = {
        "n_train_groups": model.get("n_train_groups"),
        "n_train_stations": model.get("n_train_stations"),
        "n_test_groups": model.get("n_test_groups"),
        "n_test_stations": model.get("n_test_stations"),
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
    for key in RESOURCE_KEYS:
        row[key] = resources.get(key)
    return row


def collect_spatial_rows(root: str | Path) -> List[Dict[str, Any]]:
    """Collapse every cumulative-sweep Chronos-2 ``result.json`` under ``root``.

    Kept only if ``model_family == "chronos"`` AND the run-dir name starts
    ``chronos2_cum`` (the cumulative points). This EXCLUDES the temporal regime
    runs (``chronos2_ftw*``) and the zero-shot context sweep (``chronos2_c*``)
    that share the ``chronos/chronos2/`` namespace. Rows are sorted ascending by
    ``n_train_groups`` (0 first). Raises if a kept result misses the test per-KPI
    block.
    """
    rows: List[Dict[str, Any]] = []
    for rp in sorted(glob.glob(str(Path(root) / "**" / "result.json"), recursive=True)):
        with open(rp, encoding="utf-8") as fh:
            result = json.load(fh)
        if result.get("model", {}).get("model_family") != "chronos":
            continue
        if not Path(rp).parent.name.startswith("chronos2_cum"):
            continue
        if "per_kpi" not in result.get("test", {}):
            raise ValueError(f"{rp}: chronos result has no test per-KPI block")
        rows.append(_row_from_result(result))
    rows.sort(key=lambda r: r["n_train_groups"])
    return rows


def write_spatial_sweep_csv(rows: List[Dict[str, Any]], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in HEADER})


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate the Chronos-2 cumulative group sweep")
    parser.add_argument(
        "--root",
        default="output/baseline_study_v2/full/chronos",
        help="Output root containing the chronos cumulative-sweep run dirs",
    )
    parser.add_argument(
        "--out",
        default="output/baseline_study_v2/regime_sweep_chronos2_cumulative.csv",
        help="Destination CSV (one row per cumulative point)",
    )
    args = parser.parse_args(argv)

    rows = collect_spatial_rows(args.root)
    write_spatial_sweep_csv(rows, args.out)
    print(f"wrote {args.out}: {len(rows)} cumulative points")
    for row in rows:
        print(
            f"  train_groups={str(row['n_train_groups']):>2} "
            f"train_stations={str(row['n_train_stations']):>3} "
            f"test_groups={str(row['n_test_groups']):>2} "
            f"regime={row['regime']:<5s} mean_sMAPE={row['mean_sMAPE']} "
            f"elapsed_sec={row['elapsed_sec']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
