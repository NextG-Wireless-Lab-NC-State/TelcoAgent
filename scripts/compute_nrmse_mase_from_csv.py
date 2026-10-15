#!/usr/bin/env python3
"""Compute nRMSE and MASE metrics from saved prediction CSVs.

nRMSE = RMSE / (y_max - y_min)        — range-normalized, bounded [0, 1], lower is better
MASE  = MAE / MAE_seasonal_naive(s=24) — scale-free, <1 means better than naive, lower is better

Usage:
    python scripts/compute_nrmse_mase_from_csv.py output/chronos2/csv output/telcoagent_chronos2/csv
    python scripts/compute_nrmse_mase_from_csv.py output/*/csv --names Chronos2 TelcoAgent+Chronos2
    python scripts/compute_nrmse_mase_from_csv.py output/*/csv --save results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from telcoagent.config import CORE_KPI_NAMES

KPI_NAMES = list(CORE_KPI_NAMES)
SEASONAL_PERIOD = 24  # daily seasonality for hourly telecom data


def compute_metrics_from_csv(csv_dir: Path):
    """Compute per-station, per-KPI nRMSE and MASE from prediction CSVs."""
    csv_files = sorted(csv_dir.glob("station_*.csv"))
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return None

    all_per_kpi = {kpi: {"nrmse": [], "mase": []} for kpi in KPI_NAMES}
    station_results = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        station_id = csv_path.stem

        per_kpi = {}
        for kpi in KPI_NAMES:
            pred_col = f"{kpi}_pred"
            true_col = f"{kpi}_true"
            if pred_col not in df.columns or true_col not in df.columns:
                continue

            p = df[pred_col].values.astype(float)
            t = df[true_col].values.astype(float)

            # nRMSE: RMSE / (y_max - y_min)
            t_range = float(t.max() - t.min())
            rmse = float(np.sqrt(np.mean((p - t) ** 2)))
            nrmse = (rmse / t_range) if t_range > 1e-8 else 0.0

            # MASE: MAE / MAE_seasonal_naive(s=24), denominator from test window
            mae = float(np.mean(np.abs(p - t)))
            if len(t) > SEASONAL_PERIOD:
                scale = float(np.mean(np.abs(t[SEASONAL_PERIOD:] - t[:-SEASONAL_PERIOD])))
            else:
                scale = float(np.mean(np.abs(np.diff(t)))) if len(t) > 1 else 1.0
            mase = (mae / scale) if scale > 1e-8 else 0.0

            per_kpi[kpi] = {"nRMSE": round(nrmse, 4), "MASE": round(mase, 4)}
            all_per_kpi[kpi]["nrmse"].append(nrmse)
            all_per_kpi[kpi]["mase"].append(mase)

        station_results.append({"station_id": station_id, "per_kpi": per_kpi})

    # Aggregate: mean across stations per KPI
    agg = {}
    for kpi in KPI_NAMES:
        d = all_per_kpi[kpi]
        if d["nrmse"]:
            agg[kpi] = {
                "avg_nRMSE": round(float(np.mean(d["nrmse"])), 4),
                "avg_MASE": round(float(np.mean(d["mase"])), 4),
                "n_stations": len(d["nrmse"]),
            }

    return {"n_stations": len(csv_files), "per_kpi": agg, "stations": station_results}


def print_comparison(results, names):
    """Print side-by-side comparison of nRMSE and MASE."""
    n = len(results)

    for metric_name, metric_key in [("nRMSE", "avg_nRMSE"), ("MASE", "avg_MASE")]:
        bound_info = "[0, 1]" if metric_name == "nRMSE" else "<1 = better than naive"
        print(f"\n{metric_name} (lower = better, {bound_info}):")
        header = f"  {'KPI':20s}"
        for name in names:
            header += f"  {name:>16s}"
        print(header)
        print("  " + "-" * (22 + 18 * n))

        for kpi in KPI_NAMES:
            row = f"  {kpi:20s}"
            for r in results:
                v = r["per_kpi"].get(kpi, {}).get(metric_key, None)
                if v is not None:
                    row += f"  {v:>16.4f}"
                else:
                    row += f"  {'N/A':>16s}"
            print(row)


def main():
    parser = argparse.ArgumentParser(description="Compute nRMSE and MASE from prediction CSVs")
    parser.add_argument("csv_dirs", nargs="+", help="CSV directories to compare")
    parser.add_argument("--names", nargs="+", help="Display names for each directory")
    parser.add_argument("--save", help="Save results to JSON file")
    args = parser.parse_args()

    names = args.names or [Path(d).parts[-2] for d in args.csv_dirs]
    if len(names) != len(args.csv_dirs):
        names = [f"Exp{i+1}" for i in range(len(args.csv_dirs))]

    results = []
    for csv_dir, name in zip(args.csv_dirs, names):
        print(f"Computing metrics for {name} ({csv_dir})...")
        r = compute_metrics_from_csv(Path(csv_dir))
        if r is None:
            sys.exit(1)
        results.append(r)
        print(f"  {r['n_stations']} stations")

    print(f"\n{'=' * 80}")
    print_comparison(results, names)

    if args.save:
        out = {name: r["per_kpi"] for name, r in zip(names, results)}
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
