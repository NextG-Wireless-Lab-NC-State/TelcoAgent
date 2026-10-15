#!/usr/bin/env python3
"""AX-1: Plot refinement on/off (baseline vs v1 vs v2) over n=5 stations.

Reads scripts/baselines/run_refinement_ab.py output (rows.csv or summary.json
under <run_dir>/), aggregates per-mode mean sMAPE / MASE, writes:
  - results.csv  (mode x kpi x mean_sMAPE,mean_MASE,n_stations)
  - results.xlsx
  - refinement_onoff_main.png + .pdf  (grouped bars per KPI)

Idempotent.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

DEFAULT_DIR = Path("/home/gkim26/Desktop/workplace/telcoagent/output/ablation/refinement_onoff")
KPI_NAMES = [
    "RRC_Conn",
    "DL_CQI",
    "DL_iBler",
    "DL_rBler",
    "MAC_DL_Eff",
    "PRB_Util",
    "Throughput",
]
MODES = ["baseline", "v1", "v2"]  # baseline=refinement_off, v1/v2=refinement_on variants


def aggregate(run_dir: Path) -> Path:
    """Aggregate from refinement_ab/summary.json (preferred) or rows.csv fallback."""
    # Runner writes ab_summary.json + ab_results.csv (canonical names from run_refinement_ab.py)
    summary_path = run_dir / "ab_summary.json"
    rows_csv = run_dir / "ab_results.csv"
    out_csv = run_dir / "results.csv"

    rows_out = []
    if summary_path.exists():
        summary = json.load(open(summary_path))
        for mode in MODES:
            if mode not in summary:
                continue
            for kpi in KPI_NAMES:
                v = summary[mode]["per_kpi"].get(kpi, {})
                rows_out.append(
                    {
                        "mode": mode,
                        "kpi": kpi,
                        "mean_sMAPE": v.get("avg_sMAPE"),
                        "mean_MASE": v.get("avg_MASE"),
                        "n_stations": summary[mode].get("n_stations"),
                    }
                )
    elif rows_csv.exists():
        per_cell = {(m, k): [] for m in MODES for k in KPI_NAMES}
        per_cell_mase = {(m, k): [] for m in MODES for k in KPI_NAMES}
        n_per_mode = {m: set() for m in MODES}
        with open(rows_csv) as f:
            for r in csv.DictReader(f):
                m, k = r["mode"], r["kpi"]
                if (m, k) not in per_cell:
                    continue
                try:
                    s = float(r["sMAPE"])
                    per_cell[(m, k)].append(s)
                except (TypeError, ValueError):
                    pass
                try:
                    ma = float(r["MASE"])
                    per_cell_mase[(m, k)].append(ma)
                except (TypeError, ValueError):
                    pass
                n_per_mode[m].add(r["station_id"])
        for mode in MODES:
            for kpi in KPI_NAMES:
                vals_s = per_cell[(mode, kpi)]
                vals_m = per_cell_mase[(mode, kpi)]
                rows_out.append(
                    {
                        "mode": mode,
                        "kpi": kpi,
                        "mean_sMAPE": round(statistics.mean(vals_s), 3) if vals_s else None,
                        "mean_MASE": round(statistics.mean(vals_m), 4) if vals_m else None,
                        "n_stations": len(n_per_mode[mode]),
                    }
                )
    else:
        raise FileNotFoundError(f"Neither {summary_path} nor {rows_csv} exist")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    try:
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "refinement_onoff"
        ws.append(list(rows_out[0].keys()))
        for r in rows_out:
            ws.append(list(r.values()))
        wb.save(run_dir / "results.xlsx")
    except ImportError:
        pass
    return out_csv


def plot(csv_path: Path, out_dir: Path) -> None:
    rows = list(csv.DictReader(open(csv_path)))
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )
    colors = {"baseline": "#1f77b4", "v1": "#ff7f0e", "v2": "#2ca02c"}
    import numpy as np

    x = np.arange(len(KPI_NAMES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for i, mode in enumerate(MODES):
        vals = []
        for kpi in KPI_NAMES:
            r = next((r for r in rows if r["mode"] == mode and r["kpi"] == kpi), None)
            vals.append(float(r["mean_sMAPE"]) if r and r["mean_sMAPE"] else 0.0)
        ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=mode,
            color=colors[mode],
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(KPI_NAMES, rotation=25, ha="right", fontsize=7.5)
    ax.set_ylabel("Mean sMAPE (%)")
    ax.set_title("AX-1: Refinement on/off (n=5)")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout(pad=0.5)
    fig.savefig(out_dir / "refinement_onoff_main.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "refinement_onoff_main.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args()
    csv_path = args.input or aggregate(args.dir)
    plot(csv_path, args.dir)
    print(f"wrote {args.dir}/results.csv, results.xlsx, refinement_onoff_main.{{png,pdf}}")


if __name__ == "__main__":
    main()
