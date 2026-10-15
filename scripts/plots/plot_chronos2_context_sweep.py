#!/usr/bin/env python3
"""Plot Chronos-2 context-length sweep results.

Reads ``output/chronos2_ctx_sweep/sweep_summary.csv`` and produces:
    1. overall_smape.png   — mean sMAPE across 7 KPIs vs context length, with
                              the best context annotated.
    2. per_kpi_smape.png   — small-multiples grid: each KPI's sMAPE curve with
                              its individual best context marked.
    3. all_kpis_overlay.png — single axes with all 7 KPI curves overlaid.
    4. best_context_bar.png — bar chart of best context per KPI.

Usage:
    conda run -n telcoagent python scripts/plots/plot_chronos2_context_sweep.py \\
        [--csv output/chronos2_ctx_sweep/sweep_summary.csv] \\
        [--out output/chronos2_ctx_sweep/plots]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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


def plot_overall(df: pd.DataFrame, out_path: Path):
    overall = (
        df.groupby("context_d")
        .agg(avg_sMAPE=("avg_sMAPE", "mean"), avg_MASE=("avg_MASE", "mean"))
        .reset_index()
        .sort_values("context_d")
    )
    best = overall.loc[overall["avg_sMAPE"].idxmin()]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(
        overall["context_d"],
        overall["avg_sMAPE"],
        "-o",
        color="#1f77b4",
        markersize=4,
        label="avg sMAPE",
    )
    ax1.scatter(
        [best["context_d"]],
        [best["avg_sMAPE"]],
        color="red",
        s=120,
        zorder=5,
        label=f"best = {int(best['context_d'])}d ({best['avg_sMAPE']:.2f}%)",
    )
    ax1.axvline(best["context_d"], color="red", linestyle="--", alpha=0.4)

    ax1.set_xlabel("Context length (days)")
    ax1.set_ylabel("Avg sMAPE (%)  — mean over 7 KPIs", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        overall["context_d"],
        overall["avg_MASE"],
        "-s",
        color="#2ca02c",
        markersize=3,
        alpha=0.7,
        label="avg MASE",
    )
    ax2.set_ylabel("Avg MASE", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    ax1.set_title(
        "Chronos-2 prediction quality vs input context length\n"
        "(115 stations, target = last 168h)"
    )

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return best


def plot_per_kpi(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    axes = axes.flatten()

    for i, kpi in enumerate(KPI_ORDER):
        ax = axes[i]
        sub = df[df["kpi"] == kpi].sort_values("context_d")
        ax.plot(sub["context_d"], sub["avg_sMAPE"], "-o", color=KPI_COLORS[kpi], markersize=3)

        best = sub.loc[sub["avg_sMAPE"].idxmin()]
        ax.scatter([best["context_d"]], [best["avg_sMAPE"]], color="red", s=80, zorder=5)
        ax.axvline(best["context_d"], color="red", linestyle="--", alpha=0.4)
        ax.set_title(f"{kpi}\nbest = {int(best['context_d'])}d  " f"({best['avg_sMAPE']:.2f}%)")
        ax.set_xlabel("Context (days)")
        ax.set_ylabel("Avg sMAPE (%)")
        ax.grid(alpha=0.3)

    axes[-1].axis("off")
    fig.suptitle("Per-KPI: Chronos-2 sMAPE vs context length", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for kpi in KPI_ORDER:
        sub = df[df["kpi"] == kpi].sort_values("context_d")
        ax.plot(
            sub["context_d"], sub["avg_sMAPE"], "-o", color=KPI_COLORS[kpi], markersize=3, label=kpi
        )
    ax.set_xlabel("Context length (days)")
    ax.set_ylabel("Avg sMAPE (%)")
    ax.set_title("Chronos-2: per-KPI sMAPE across context sweep (7d-81d)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_best_bar(df: pd.DataFrame, out_path: Path):
    rows = []
    for kpi in KPI_ORDER:
        sub = df[df["kpi"] == kpi]
        best = sub.loc[sub["avg_sMAPE"].idxmin()]
        rows.append({"kpi": kpi, "best_d": int(best["context_d"]), "smape": best["avg_sMAPE"]})
    bdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(bdf["kpi"], bdf["best_d"], color=[KPI_COLORS[k] for k in bdf["kpi"]])
    for bar, sm in zip(bars, bdf["smape"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{int(bar.get_height())}d\n({sm:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Best context length (days)")
    ax.set_title("Per-KPI best context length\n(annotation: best context, sMAPE)")
    ax.set_ylim(0, max(bdf["best_d"]) + 8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return bdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="output/chronos2_ctx_sweep/sweep_summary.csv")
    parser.add_argument("--out", default="output/chronos2_ctx_sweep/plots")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(
        f"Loaded {csv_path}  shape={df.shape}  "
        f"contexts={df['context_d'].nunique()}  kpis={df['kpi'].nunique()}"
    )

    best = plot_overall(df, out_dir / "overall_smape.png")
    plot_per_kpi(df, out_dir / "per_kpi_smape.png")
    plot_overlay(df, out_dir / "all_kpis_overlay.png")
    bdf = plot_best_bar(df, out_dir / "best_context_bar.png")

    print(
        f"\nBest overall context: {int(best['context_d'])}d  "
        f"avg sMAPE = {best['avg_sMAPE']:.2f}%  avg MASE = {best['avg_MASE']:.4f}"
    )
    print("\nBest per-KPI:")
    for _, r in bdf.iterrows():
        print(f"  {r['kpi']:14s}  {r['best_d']:>3d}d  sMAPE={r['smape']:.2f}%")
    print(f"\nPlots saved to: {out_dir}/")


if __name__ == "__main__":
    main()
