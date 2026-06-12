#!/usr/bin/env python3
"""Plot Chronos-2 context-length sweep — MAE / RMSE / NMAE / NRMSE.

Aggregates per-station JSONs across all ctx_*h directories under
``output/chronos2_ctx_sweep/`` and produces error plots in absolute
KPI units (MAE/RMSE) plus normalized variants (NMAE/NRMSE = MAE / mean(truth))
so KPIs with very different scales become comparable.

Outputs (under output/chronos2_ctx_sweep/plots/):
    per_kpi_mae.png         — small-multiples grid of avg MAE vs context (per KPI)
    per_kpi_rmse.png        — same for RMSE
    per_kpi_nmae.png        — small-multiples of normalized MAE
    overall_nmae.png        — mean NMAE across 7 KPIs vs context, with best
    mae_long_table.csv      — long-format CSV (context_d, kpi, avg_MAE, avg_RMSE,
                              avg_NMAE, avg_NRMSE)

Usage:
    conda run -n telcoagent python scripts/plots/plot_chronos2_context_sweep_mae.py \\
        [--sweep-root output/chronos2_ctx_sweep] \\
        [--out output/chronos2_ctx_sweep/plots]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KPI_ORDER = [
    "RRC_Conn",
    "DL_CQI",
    "DL_iBler",
    "DL_rBler",
    "MAC_DL_Eff",
    "PRB_Util",
    "Throughput",
]
KPI_COLORS = {
    "RRC_Conn": "#1f77b4",
    "DL_CQI": "#ff7f0e",
    "DL_iBler": "#2ca02c",
    "DL_rBler": "#d62728",
    "MAC_DL_Eff": "#9467bd",
    "PRB_Util": "#8c564b",
    "Throughput": "#e377c2",
}
KPI_UNITS = {
    "RRC_Conn": "count",
    "DL_CQI": "index",
    "DL_iBler": "%",
    "DL_rBler": "%",
    "MAC_DL_Eff": "kbps",
    "PRB_Util": "%",
    "Throughput": "kbps",
}

CTX_DIR_RE = re.compile(r"^ctx_(\d+)h$")


def collect_station_metrics(sweep_root: Path) -> pd.DataFrame:
    """Load every ctx_*/station_*.json and return a tidy long DataFrame.

    Columns: context_h, context_d, station_id, kpi, MAE, RMSE, sMAPE, MASE
    """
    rows = []
    ctx_dirs = sorted(
        [d for d in sweep_root.iterdir() if d.is_dir() and CTX_DIR_RE.match(d.name)],
        key=lambda d: int(CTX_DIR_RE.match(d.name).group(1)),
    )
    print(f"Found {len(ctx_dirs)} ctx directories under {sweep_root}")

    for ctx_dir in ctx_dirs:
        ctx_h = int(CTX_DIR_RE.match(ctx_dir.name).group(1))
        for jp in ctx_dir.glob("station_*.json"):
            with open(jp) as f:
                d = json.load(f)
            sid = d["station_id"]
            for kpi, m in d["metrics"]["per_kpi"].items():
                rows.append(
                    {
                        "context_h": ctx_h,
                        "context_d": ctx_h / 24.0,
                        "station_id": sid,
                        "kpi": kpi,
                        "MAE": m["MAE"],
                        "RMSE": m["RMSE"],
                        "sMAPE": m["sMAPE"],
                        "MASE": m["MASE"],
                    }
                )
    df = pd.DataFrame(rows)
    print(
        f"Loaded {len(df):,} rows  "
        f"({df['context_h'].nunique()} contexts × {df['station_id'].nunique()} stations × "
        f"{df['kpi'].nunique()} KPIs)"
    )
    return df


def compute_kpi_truth_means(sweep_root: Path) -> dict[str, float]:
    """Compute per-KPI mean(|truth|) from one ctx's CSVs to use as NMAE denominator.

    Truth is identical across contexts (same target window), so any ctx works.
    """
    any_ctx = next(
        (d for d in sweep_root.iterdir() if d.is_dir() and CTX_DIR_RE.match(d.name)),
        None,
    )
    if any_ctx is None:
        raise RuntimeError(f"No ctx_*h directory found under {sweep_root}")
    csv_dir = any_ctx / "csv"
    sums = {k: [] for k in KPI_ORDER}
    for cp in csv_dir.glob("station_*.csv"):
        df = pd.read_csv(cp)
        for k in KPI_ORDER:
            col = f"{k}_true"
            if col in df.columns:
                sums[k].append(float(np.mean(np.abs(df[col].values))))
    means = {k: float(np.mean(v)) if v else 1.0 for k, v in sums.items()}
    print("Per-KPI mean(|truth|) used for normalization:")
    for k, v in means.items():
        print(f"  {k:14s}  {v:>12.4f}  ({KPI_UNITS[k]})")
    return means


def aggregate(df: pd.DataFrame, truth_means: dict) -> pd.DataFrame:
    """Aggregate to (context_d × kpi) averages and add NMAE / NRMSE."""
    agg = (
        df.groupby(["context_h", "context_d", "kpi"])
        .agg(
            avg_MAE=("MAE", "mean"),
            avg_RMSE=("RMSE", "mean"),
            avg_sMAPE=("sMAPE", "mean"),
            avg_MASE=("MASE", "mean"),
            n_stations=("station_id", "nunique"),
        )
        .reset_index()
    )
    agg["truth_mean"] = agg["kpi"].map(truth_means)
    agg["avg_NMAE"] = agg["avg_MAE"] / agg["truth_mean"]
    agg["avg_NRMSE"] = agg["avg_RMSE"] / agg["truth_mean"]
    return agg.sort_values(["kpi", "context_d"])


def small_multiples(
    agg: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    unit_per_kpi: bool = False,
):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    axes = axes.flatten()

    for i, kpi in enumerate(KPI_ORDER):
        ax = axes[i]
        sub = agg[agg["kpi"] == kpi].sort_values("context_d")
        ax.plot(sub["context_d"], sub[metric], "-o", color=KPI_COLORS[kpi], markersize=3)
        best = sub.loc[sub[metric].idxmin()]
        ax.scatter([best["context_d"]], [best[metric]], color="red", s=80, zorder=5)
        ax.axvline(best["context_d"], color="red", linestyle="--", alpha=0.4)
        suffix = f" {KPI_UNITS[kpi]}" if unit_per_kpi else ""
        ax.set_title(f"{kpi}\nbest = {int(best['context_d'])}d  " f"({best[metric]:.3f}{suffix})")
        ax.set_xlabel("Context (days)")
        ax.set_ylabel(ylabel + suffix)
        ax.grid(alpha=0.3)

    axes[-1].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overall_nmae(agg: pd.DataFrame, out_path: Path):
    overall = (
        agg.groupby("context_d")
        .agg(avg_NMAE=("avg_NMAE", "mean"), avg_NRMSE=("avg_NRMSE", "mean"))
        .reset_index()
        .sort_values("context_d")
    )
    best = overall.loc[overall["avg_NMAE"].idxmin()]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(
        overall["context_d"],
        overall["avg_NMAE"],
        "-o",
        color="#1f77b4",
        markersize=4,
        label="avg NMAE",
    )
    ax1.scatter(
        [best["context_d"]],
        [best["avg_NMAE"]],
        color="red",
        s=120,
        zorder=5,
        label=f"best = {int(best['context_d'])}d ({best['avg_NMAE']:.3f})",
    )
    ax1.axvline(best["context_d"], color="red", linestyle="--", alpha=0.4)
    ax1.set_xlabel("Context length (days)")
    ax1.set_ylabel("Avg NMAE  — mean over 7 KPIs", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        overall["context_d"],
        overall["avg_NRMSE"],
        "-s",
        color="#2ca02c",
        markersize=3,
        alpha=0.7,
        label="avg NRMSE",
    )
    ax2.set_ylabel("Avg NRMSE", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    ax1.set_title(
        "Chronos-2 normalized error vs input context length\n"
        "NMAE = MAE / mean(|truth|), averaged over 7 KPIs (115 stations)"
    )
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", default="output/chronos2_ctx_sweep")
    parser.add_argument("--out", default="output/chronos2_ctx_sweep/plots")
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_station_metrics(sweep_root)
    truth_means = compute_kpi_truth_means(sweep_root)
    agg = aggregate(df, truth_means)

    long_csv = out_dir / "mae_long_table.csv"
    agg.to_csv(long_csv, index=False)
    print(f"\nSaved long table: {long_csv}")

    small_multiples(
        agg,
        "avg_MAE",
        "Avg MAE",
        "Per-KPI: Chronos-2 MAE vs context length (absolute units)",
        out_dir / "per_kpi_mae.png",
        unit_per_kpi=True,
    )
    small_multiples(
        agg,
        "avg_RMSE",
        "Avg RMSE",
        "Per-KPI: Chronos-2 RMSE vs context length (absolute units)",
        out_dir / "per_kpi_rmse.png",
        unit_per_kpi=True,
    )
    small_multiples(
        agg,
        "avg_NMAE",
        "Avg NMAE",
        "Per-KPI: Chronos-2 normalized MAE (NMAE = MAE / mean|truth|)",
        out_dir / "per_kpi_nmae.png",
    )
    best = plot_overall_nmae(agg, out_dir / "overall_nmae.png")

    print("\n=== Best context per KPI (lowest avg MAE) ===")
    for kpi in KPI_ORDER:
        sub = agg[agg["kpi"] == kpi]
        b = sub.loc[sub["avg_MAE"].idxmin()]
        print(
            f"  {kpi:14s}  {int(b['context_d']):>3d}d   "
            f"MAE={b['avg_MAE']:>10.4f} {KPI_UNITS[kpi]:6s}  "
            f"NMAE={b['avg_NMAE']:.3f}"
        )
    print(
        f"\nBest overall context (avg NMAE): {int(best['context_d'])}d   "
        f"NMAE={best['avg_NMAE']:.3f}  NRMSE={best['avg_NRMSE']:.3f}"
    )

    print(f"\nPlots saved to: {out_dir}/")


if __name__ == "__main__":
    main()
