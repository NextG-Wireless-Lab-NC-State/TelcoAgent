#!/usr/bin/env python3
"""INFOCOM-style 4-panel figure: NMAE / sMAPE / MASE / MDA vs context length.

Metrics:
    NMAE  - normalized MAE = MAE / mean(|truth|)              (lower better)
    sMAPE - symmetric MAPE (%)                                (lower better)
    MASE  - vs 24h seasonal naive                             (lower better)
    MDA   - Mean Directional Accuracy, step-to-step [0, 1]   (higher better)

MDA is computed fresh from the per-station CSVs (pred / truth hourly series)
because it is not stored in the existing sweep summary CSVs.

Usage:
    conda run -n telcoagent python scripts/plots/plot_context_sweep_infocom.py \\
        [--sweep-root output/chronos2_ctx_sweep] \\
        [--out output/chronos2_ctx_sweep/plots]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# INFOCOM / IEEE style
# --------------------------------------------------------------------------- #
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Palatino"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 150,
    }
)

DOUBLE_COL_W = 7.16  # IEEE double-column width in inches
SINGLE_COL_W = 3.5  # IEEE single-column width in inches

KPI_ORDER = [
    "RRC_Conn",
    "DL_CQI",
    "DL_iBler",
    "DL_rBler",
    "MAC_DL_Eff",
    "PRB_Util",
    "Throughput",
]

CTX_DIR_RE = re.compile(r"^ctx_(\d+)h$")

# One line style per KPI for the overlay plots
LINE_STYLES = [
    {"color": "#1f77b4", "linestyle": "-", "marker": ""},
    {"color": "#ff7f0e", "linestyle": "--", "marker": ""},
    {"color": "#2ca02c", "linestyle": "-.", "marker": ""},
    {"color": "#d62728", "linestyle": ":", "marker": ""},
    {"color": "#9467bd", "linestyle": "-", "marker": ""},
    {"color": "#8c564b", "linestyle": "--", "marker": ""},
    {"color": "#e377c2", "linestyle": "-.", "marker": ""},
]


# --------------------------------------------------------------------------- #
# Data loading helpers
# --------------------------------------------------------------------------- #
def load_station_json_metrics(sweep_root: Path) -> pd.DataFrame:
    """Return per-(ctx, station, kpi) MAE / sMAPE / MASE from JSON files."""
    rows = []
    ctx_dirs = sorted(
        [d for d in sweep_root.iterdir() if d.is_dir() and CTX_DIR_RE.match(d.name)],
        key=lambda d: int(CTX_DIR_RE.match(d.name).group(1)),
    )
    for ctx_dir in ctx_dirs:
        ctx_h = int(CTX_DIR_RE.match(ctx_dir.name).group(1))
        for jp in ctx_dir.glob("station_*.json"):
            d = json.loads(jp.read_text())
            for kpi, m in d["metrics"]["per_kpi"].items():
                rows.append(
                    {
                        "context_h": ctx_h,
                        "context_d": ctx_h / 24.0,
                        "station_id": d["station_id"],
                        "kpi": kpi,
                        "MAE": m["MAE"],
                        "sMAPE": m["sMAPE"],
                        "MASE": m["MASE"],
                    }
                )
    return pd.DataFrame(rows)


def compute_truth_means(sweep_root: Path) -> dict[str, float]:
    """Mean(|truth|) per KPI from any one ctx's CSV files (truth is identical)."""
    any_ctx = sorted(
        [d for d in sweep_root.iterdir() if d.is_dir() and CTX_DIR_RE.match(d.name)],
        key=lambda d: int(CTX_DIR_RE.match(d.name).group(1)),
    )[0]
    sums: dict[str, list] = {k: [] for k in KPI_ORDER}
    for cp in (any_ctx / "csv").glob("station_*.csv"):
        df = pd.read_csv(cp)
        for k in KPI_ORDER:
            col = f"{k}_true"
            if col in df.columns:
                sums[k].append(float(np.mean(np.abs(df[col].values))))
    return {k: float(np.mean(v)) if v else 1.0 for k, v in sums.items()}


def compute_mda_from_csvs(sweep_root: Path) -> pd.DataFrame:
    """MDA = step-to-step directional accuracy within the 168h forecast window.

    For each (ctx, station, kpi):
        MDA = mean(sign(pred[t] - pred[t-1]) == sign(true[t] - true[t-1]))
              over t = 1..167   (168 steps → 167 differences)
    """
    rows = []
    ctx_dirs = sorted(
        [d for d in sweep_root.iterdir() if d.is_dir() and CTX_DIR_RE.match(d.name)],
        key=lambda d: int(CTX_DIR_RE.match(d.name).group(1)),
    )
    for ctx_dir in ctx_dirs:
        ctx_h = int(CTX_DIR_RE.match(ctx_dir.name).group(1))
        for cp in (ctx_dir / "csv").glob("station_*.csv"):
            df = pd.read_csv(cp)
            sid = df["station_id"].iloc[0]
            for kpi in KPI_ORDER:
                pcol, tcol = f"{kpi}_pred", f"{kpi}_true"
                if pcol not in df.columns:
                    continue
                p = df[pcol].values.astype(float)
                t = df[tcol].values.astype(float)
                # step-to-step differences
                dp = np.sign(np.diff(p))
                dt = np.sign(np.diff(t))
                mda = float(np.mean(dp == dt))
                rows.append(
                    {
                        "context_h": ctx_h,
                        "context_d": ctx_h / 24.0,
                        "station_id": sid,
                        "kpi": kpi,
                        "MDA": mda,
                    }
                )
    return pd.DataFrame(rows)


def aggregate_all(df_json: pd.DataFrame, df_mda: pd.DataFrame, truth_means: dict) -> pd.DataFrame:
    """Join and aggregate all metrics to (context_d × kpi)."""
    agg_j = (
        df_json.groupby(["context_h", "context_d", "kpi"])
        .agg(avg_MAE=("MAE", "mean"), avg_sMAPE=("sMAPE", "mean"), avg_MASE=("MASE", "mean"))
        .reset_index()
    )
    agg_j["truth_mean"] = agg_j["kpi"].map(truth_means)
    agg_j["avg_NMAE"] = agg_j["avg_MAE"] / agg_j["truth_mean"]

    agg_m = (
        df_mda.groupby(["context_h", "context_d", "kpi"]).agg(avg_MDA=("MDA", "mean")).reset_index()
    )

    agg = agg_j.merge(agg_m, on=["context_h", "context_d", "kpi"])
    return agg.sort_values(["kpi", "context_d"])


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _mark_best(ax, x, y, better="lower"):
    idx = np.argmin(y) if better == "lower" else np.argmax(y)
    ax.axvline(x[idx], color="red", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.scatter([x[idx]], [y[idx]], color="red", s=50, zorder=6, label=f"best = {x[idx]:.0f}d")


def make_4panel_figure(agg: pd.DataFrame, out_path: Path):
    """Single-column 4-panel figure suitable for an INFOCOM-style paper."""
    # Overall averages (mean over 7 KPIs)
    ov = (
        agg.groupby("context_d")
        .agg(
            NMAE=("avg_NMAE", "mean"),
            sMAPE=("avg_sMAPE", "mean"),
            MASE=("avg_MASE", "mean"),
            MDA=("avg_MDA", "mean"),
        )
        .reset_index()
        .sort_values("context_d")
    )

    x = ov["context_d"].values

    panels = [
        ("NMAE", r"NMAE (norm. MAE)", "lower", "#1f77b4"),
        ("sMAPE", r"sMAPE (%)", "lower", "#ff7f0e"),
        ("MASE", r"MASE", "lower", "#2ca02c"),
        ("MDA", r"MDA", "higher", "#d62728"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(SINGLE_COL_W, 7.5), sharex=True)

    for ax, (col, ylabel, better, color) in zip(axes, panels):
        y = ov[col].values
        ax.plot(x, y, color=color, linewidth=1.4)
        _mark_best(ax, x, y, better)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, loc="upper right", fontsize=7)
        # keep x-axis label only on bottom panel
    axes[-1].set_xlabel("Context length (days)")

    fig.suptitle(
        "Chronos-2: prediction metrics vs context length\n" "(mean over 7 KPIs, 115 stations)",
        fontsize=9,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_4panel_per_kpi_overlay(agg: pd.DataFrame, out_path: Path):
    """Double-column 2×2 figure with one KPI-overlay per metric."""
    panels = [
        ("avg_NMAE", r"NMAE", "lower"),
        ("avg_sMAPE", r"sMAPE (%)", "lower"),
        ("avg_MASE", r"MASE", "lower"),
        ("avg_MDA", r"MDA", "higher"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_W, 5.5), sharex=True)
    axes = axes.flatten()

    for ax, (col, ylabel, better) in zip(axes, panels):
        for i, kpi in enumerate(KPI_ORDER):
            sub = agg[agg["kpi"] == kpi].sort_values("context_d")
            x, y = sub["context_d"].values, sub[col].values
            ls = LINE_STYLES[i]
            # markers only every 10 points to avoid clutter
            step = max(1, len(x) // 10)
            markevery = list(range(0, len(x), step))
            ax.plot(
                x,
                y,
                label=kpi,
                color=ls["color"],
                linestyle=ls["linestyle"],
                marker="o",
                markersize=3,
                markevery=markevery,
            )
        ax.set_ylabel(ylabel)
        ax.set_title(f"({chr(97 + panels.index((col, ylabel, better)))}) {ylabel}")

    axes[2].set_xlabel("Context length (days)")
    axes[3].set_xlabel("Context length (days)")

    # single shared legend below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle(
        "Chronos-2 context-length sweep — per-KPI metrics",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_best_context_table(agg: pd.DataFrame):
    ov = (
        agg.groupby("context_d")
        .agg(
            NMAE=("avg_NMAE", "mean"),
            sMAPE=("avg_sMAPE", "mean"),
            MASE=("avg_MASE", "mean"),
            MDA=("avg_MDA", "mean"),
        )
        .reset_index()
    )
    metrics = [("NMAE", "lower"), ("sMAPE", "lower"), ("MASE", "lower"), ("MDA", "higher")]
    print(f"\n{'Metric':8s}  {'Best ctx':>10s}  {'Value':>10s}")
    print("-" * 34)
    for col, better in metrics:
        best = ov.loc[ov[col].idxmin() if better == "lower" else ov[col].idxmax()]
        print(f"{col:8s}  {int(best['context_d']):>7d}d  {best[col]:>10.4f}")

    print("\nPer-KPI best context (overall NMAE basis):")
    print(f"  {'KPI':14s}  {'NMAE ctx':>9s}  {'MDA ctx':>9s}")
    print("  " + "-" * 36)
    for kpi in KPI_ORDER:
        sub = agg[agg["kpi"] == kpi]
        bn = sub.loc[sub["avg_NMAE"].idxmin()]
        bm = sub.loc[sub["avg_MDA"].idxmax()]
        print(f"  {kpi:14s}  {int(bn['context_d']):>6d}d  {int(bm['context_d']):>6d}d")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", default="output/chronos2_ctx_sweep")
    parser.add_argument("--out", default="output/chronos2_ctx_sweep/plots")
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading JSON metrics …")
    df_json = load_station_json_metrics(sweep_root)
    print(
        f"  {len(df_json):,} rows  ({df_json['context_h'].nunique()} contexts × "
        f"{df_json['station_id'].nunique()} stations × {df_json['kpi'].nunique()} KPIs)"
    )

    print("Computing MDA from CSVs …")
    df_mda = compute_mda_from_csvs(sweep_root)
    print(f"  {len(df_mda):,} rows")

    print("Computing truth means for NMAE normalization …")
    truth_means = compute_truth_means(sweep_root)

    agg = aggregate_all(df_json, df_mda, truth_means)

    # Save combined long table
    long_csv = out_dir / "all_metrics_long.csv"
    agg.to_csv(long_csv, index=False)
    print(f"Saved: {long_csv}")

    print("Plotting …")
    make_4panel_figure(agg, out_dir / "infocom_4panel_overall.png")
    make_4panel_per_kpi_overlay(agg, out_dir / "infocom_4panel_per_kpi.png")

    print_best_context_table(agg)


if __name__ == "__main__":
    main()
