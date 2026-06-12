#!/usr/bin/env python3
"""Compute sMAPE and nRMSE metrics from saved prediction CSVs.

No model inference needed — reads pred/true pairs from CSV files.

Usage:
    python scripts/compute_metrics_from_csv.py output/chronos2_true_raw/csv output/telcoagent_chronos2_physics/csv
    python scripts/compute_metrics_from_csv.py dir1/csv dir2/csv --names "No KG" "3GPP KG"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from telcoagent.config import CORE_KPI_NAMES

KPI_NAMES = list(CORE_KPI_NAMES)


def compute_metrics_from_csv(csv_dir: Path):
    """Compute per-station, per-KPI metrics from prediction CSVs."""
    csv_files = sorted(csv_dir.glob("station_*.csv"))
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return None

    SEASONAL_PERIOD = 24
    all_per_kpi = {kpi: {"smapes": [], "mases": []} for kpi in KPI_NAMES}
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

            # sMAPE: symmetric, bounded [0, 200%]
            denom = np.abs(p) + np.abs(t)
            smask = denom > 1e-8
            smape = (
                float(np.mean(2.0 * np.abs(p[smask] - t[smask]) / denom[smask])) * 100
                if smask.sum() > 0
                else 0.0
            )

            # MASE: seasonal naive (s=24) denominator from truth window
            mae = float(np.mean(np.abs(p - t)))
            if len(t) > SEASONAL_PERIOD:
                scale = float(np.mean(np.abs(t[SEASONAL_PERIOD:] - t[:-SEASONAL_PERIOD])))
            else:
                scale = float(np.mean(np.abs(np.diff(t)))) if len(t) > 1 else 1.0
            mase = (mae / scale) if scale > 1e-8 else 0.0

            per_kpi[kpi] = {
                "sMAPE": round(smape, 2),
                "MASE": round(mase, 4),
            }

            all_per_kpi[kpi]["smapes"].append(smape)
            all_per_kpi[kpi]["mases"].append(mase)

        station_results.append({"station_id": station_id, "per_kpi": per_kpi})

    # Aggregate
    agg = {}
    for kpi in KPI_NAMES:
        d = all_per_kpi[kpi]
        if d["smapes"]:
            agg[kpi] = {
                "avg_sMAPE": round(np.mean(d["smapes"]), 2),
                "avg_MASE": round(np.mean(d["mases"]), 4),
            }

    return {"n_stations": len(csv_files), "per_kpi": agg, "stations": station_results}


def print_comparison(results, names):
    """Print side-by-side comparison of multiple experiment results."""
    n = len(results)

    for metric_name, metric_key in [("sMAPE", "avg_sMAPE"), ("MASE", "avg_MASE")]:
        print(f"\n{metric_name} (lower = better):")
        header = f"  {'KPI':20s}"
        for name in names:
            header += f"  {name:>12s}"
        if n == 2:
            header += f"  {'Δ':>9s}"
        print(header)
        print("  " + "-" * (25 + 14 * n + (11 if n == 2 else 0)))

        deltas = []
        for kpi in KPI_NAMES:
            row = f"  {kpi:20s}"
            vals = []
            for r in results:
                v = r["per_kpi"].get(kpi, {}).get(metric_key, 0)
                vals.append(v)
                fmt = f"{v:.2f}%" if metric_key == "avg_sMAPE" else f"{v:.4f}"
                row += f"  {fmt:>11s}"
            if n == 2 and vals[0] > 0:
                d = (vals[1] - vals[0]) / vals[0] * 100
                deltas.append(d)
                marker = " *" if d < -1 else ""
                row += f"  {d:+8.2f}%{marker}"
            print(row)

        if n == 2 and deltas:
            avg_d = np.mean(deltas)
            row = f"  {'Average':20s}"
            for _ in names:
                row += f"  {'':>12s}"
            row += f"  {avg_d:+8.2f}%"
            print(row)

    if n == 2:
        print(f"\nNegative = {names[1]} is better  |  * = >1% improvement")


def main():
    parser = argparse.ArgumentParser(description="Compute metrics from prediction CSVs")
    parser.add_argument("csv_dirs", nargs="+", help="CSV directories to compare")
    parser.add_argument("--names", nargs="+", help="Display names for each directory")
    parser.add_argument("--save", help="Save results to JSON file")
    args = parser.parse_args()

    names = args.names or [Path(d).parent.name for d in args.csv_dirs]
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
