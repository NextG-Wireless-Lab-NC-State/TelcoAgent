#!/usr/bin/env python3
"""AX-3: Plot Chronos-2 context-length sweep over n=5 stations.

Aggregates per-station JSONs at contexts {24, 48, 96, 168, 336} h, computes
per-KPI mean nRMSE/MASE, writes:
  - results.csv (context_h x kpi x mean_nRMSE,mean_MASE,n_stations)
  - results.xlsx (same as workbook)
  - context_sweep_main.png + .pdf (mean nRMSE vs context, mean over KPIs)

Idempotent. Re-running overwrites outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

SWEEP_ROOT = Path("/home/gkim26/Desktop/workplace/telcoagent/output/amazon_chronos-2_h7d_ctx_sweep")
DEFAULT_OUT = Path("/home/gkim26/Desktop/workplace/telcoagent/output/ablation/context_sweep")

CONTEXTS_H = [24, 48, 96, 168, 336]
# Deterministic 5-station subset (alpha-sorted first 5 station_A_*)
STATIONS = [
    "station_A_10",
    "station_A_12",
    "station_A_2",
    "station_A_4",
    "station_A_5",
]
KPI_NAMES = [
    "RRC_Conn",
    "DL_CQI",
    "DL_iBler",
    "DL_rBler",
    "MAC_DL_Eff",
    "PRB_Util",
    "Throughput",
]


def _load_metric(ctx_h: int, station: str, kpi: str, metric: str):
    p = SWEEP_ROOT / f"ctx_{ctx_h}h" / f"{station}.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return d.get("metrics", {}).get("per_kpi", {}).get(kpi, {}).get(metric)


def aggregate(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ctx_h in CONTEXTS_H:
        for kpi in KPI_NAMES:
            nrmse_vals, mase_vals = [], []
            for sid in STATIONS:
                n = _load_metric(ctx_h, sid, kpi, "nRMSE")
                m = _load_metric(ctx_h, sid, kpi, "MASE")
                if n is not None:
                    nrmse_vals.append(n)
                if m is not None:
                    mase_vals.append(m)
            rows.append(
                {
                    "context_h": ctx_h,
                    "kpi": kpi,
                    "mean_nRMSE": round(statistics.mean(nrmse_vals), 4) if nrmse_vals else None,
                    "std_nRMSE": (
                        round(statistics.stdev(nrmse_vals), 4) if len(nrmse_vals) > 1 else 0.0
                    ),
                    "mean_MASE": round(statistics.mean(mase_vals), 4) if mase_vals else None,
                    "std_MASE": (
                        round(statistics.stdev(mase_vals), 4) if len(mase_vals) > 1 else 0.0
                    ),
                    "n_stations": len(nrmse_vals),
                }
            )
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    try:
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "context_sweep"
        ws.append(list(rows[0].keys()))
        for r in rows:
            ws.append(list(r.values()))
        wb.save(out_dir / "results.xlsx")
    except ImportError:
        pass

    return csv_path


def plot(csv_path: Path, out_dir: Path) -> None:
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
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
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    by_kpi = {kpi: [] for kpi in KPI_NAMES}
    ctxs = sorted({int(r["context_h"]) for r in rows})
    for kpi in KPI_NAMES:
        for ctx in ctxs:
            v = next(
                (r for r in rows if int(r["context_h"]) == ctx and r["kpi"] == kpi),
                None,
            )
            by_kpi[kpi].append(float(v["mean_nRMSE"]) if v and v["mean_nRMSE"] else None)
    for kpi, ys in by_kpi.items():
        ax.plot(ctxs, ys, marker="o", linewidth=1.2, markersize=4, label=kpi)
    ax.set_xlabel("Context length (hours)")
    ax.set_ylabel("Mean nRMSE (n=5)")
    ax.set_title("AX-3: Chronos-2 context-length sweep")
    ax.set_xscale("log")
    ax.set_xticks(ctxs)
    ax.set_xticklabels([str(c) for c in ctxs])
    ax.legend(loc="upper right", fontsize=7, ncol=2, frameon=True)
    fig.tight_layout(pad=0.5)
    fig.savefig(out_dir / "context_sweep_main.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "context_sweep_main.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None, help="CSV path (default: derived)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out_dir = args.out
    csv_path = args.input or aggregate(out_dir)
    if args.input is None:
        # Just aggregated; csv_path already valid.
        pass
    plot(csv_path, out_dir)
    print(f"wrote {out_dir}/results.csv, results.xlsx, context_sweep_main.{{png,pdf}}")


if __name__ == "__main__":
    main()
