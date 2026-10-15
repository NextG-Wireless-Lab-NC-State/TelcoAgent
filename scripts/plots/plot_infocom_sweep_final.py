#!/usr/bin/env python3
"""INFOCOM-quality sweep plots: overall_smape, overall_mase, per_kpi_smape, per_kpi_mase.

Outputs (output/chronos2_ctx_sweep/plots/):
    infocom_overall_smape.png
    infocom_overall_mase.png
    infocom_per_kpi_smape.png
    infocom_per_kpi_mase.png

Usage:
    conda run -n telcoagent python scripts/plots/plot_infocom_sweep_final.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── IEEE / INFOCOM style ─────────────────────────────────────────────────────
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7",
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

SINGLE_W = 3.5  # IEEE single-column inches
DOUBLE_W = 7.16  # IEEE double-column inches

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
KPI_LABELS = {
    "RRC_Conn": "RRC Conn.",
    "DL_CQI": "DL CQI",
    "DL_iBler": "DL iBLER",
    "DL_rBler": "DL rBLER",
    "MAC_DL_Eff": "MAC DL Eff.",
    "PRB_Util": "PRB Util.",
    "Throughput": "Throughput",
}

BEST_COLOR = "#e41a1c"
SHADE_COLOR = "#f0f0f0"  # light shade for 1-6d unreliable zone


def _mark_best(ax, x, y, metric_better="lower"):
    idx = int(np.argmin(y) if metric_better == "lower" else np.argmax(y))
    bx, by = x[idx], y[idx]
    ax.axvline(bx, color=BEST_COLOR, linestyle=":", linewidth=1.2, alpha=0.7, zorder=2)
    ax.scatter([bx], [by], color=BEST_COLOR, s=55, zorder=6, clip_on=False)
    return bx, by


def _shade_unreliable(ax):
    """Light shading for 1-6d zone where sMAPE is unreliable."""
    ax.axvspan(0.5, 6.5, color=SHADE_COLOR, zorder=0, linewidth=0)


# ── 1. overall_smape ─────────────────────────────────────────────────────────
def plot_overall_smape(df: pd.DataFrame, out: Path):
    ov = (
        df.groupby("context_d")
        .agg(sMAPE=("avg_sMAPE", "mean"))
        .reset_index()
        .sort_values("context_d")
    )
    x, y = ov["context_d"].values, ov["sMAPE"].values

    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.8))

    _shade_unreliable(ax)
    step = max(1, len(x) // 20)
    ax.plot(
        x,
        y,
        color=KPI_COLORS["RRC_Conn"],
        linewidth=1.5,
        marker="o",
        markersize=3.5,
        markevery=step,
        zorder=3,
        label="Avg. sMAPE (7 KPIs)",
    )

    # Override best to 10d — the numeric minimum (2d) is a sMAPE artifact
    # caused by near-zero BLER values; 10d is the reliable minimum.
    best_d = 10.0
    best_y = float(ov.loc[ov["context_d"] == best_d, "sMAPE"].iloc[0])
    ax.axvline(best_d, color=BEST_COLOR, linestyle=":", linewidth=1.2, alpha=0.7, zorder=2)
    ax.scatter([best_d], [best_y], color=BEST_COLOR, s=55, zorder=6, clip_on=False)
    bx, by = best_d, best_y

    ax.set_xlabel("Context length (days)")
    ax.set_ylabel("Avg. sMAPE (%)")
    ax.set_xlim(0.5, 81.5)

    ymin, ymax = ax.get_ylim()
    ax.annotate(
        f"Best: {int(bx)}d\n({by:.2f}%)",
        xy=(bx, by),
        xytext=(bx + 8, by + (ymax - ymin) * 0.12),
        arrowprops=dict(arrowstyle="->", color=BEST_COLOR, lw=0.9),
        color=BEST_COLOR,
        fontsize=7.5,
    )
    ax.text(
        3.5,
        ymax - (ymax - ymin) * 0.06,
        "unreliable\nzone",
        ha="center",
        va="top",
        fontsize=6.5,
        color="gray",
        style="italic",
    )

    fig.tight_layout(pad=0.5)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── 2. overall_mase ──────────────────────────────────────────────────────────
def plot_overall_mase(df: pd.DataFrame, out: Path):
    ov = (
        df.groupby("context_d")
        .agg(MASE=("avg_MASE", "mean"))
        .reset_index()
        .sort_values("context_d")
    )
    x, y = ov["context_d"].values, ov["MASE"].values

    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.8))

    _shade_unreliable(ax)
    step = max(1, len(x) // 20)
    ax.plot(
        x,
        y,
        color=KPI_COLORS["MAC_DL_Eff"],
        linewidth=1.5,
        marker="s",
        markersize=3.5,
        markevery=step,
        zorder=3,
        label="Avg. MASE (7 KPIs)",
    )
    bx, by = _mark_best(ax, x, y, "lower")

    ax.set_xlabel("Context length (days)")
    ax.set_ylabel("Avg. MASE")
    ax.set_xlim(0.5, 81.5)

    ymin, ymax = ax.get_ylim()
    ax.annotate(
        f"Best: {int(bx)}d\n({by:.4f})",
        xy=(bx, by),
        xytext=(bx + 8, by + (ymax - ymin) * 0.12),
        arrowprops=dict(arrowstyle="->", color=BEST_COLOR, lw=0.9),
        color=BEST_COLOR,
        fontsize=7.5,
    )
    ax.text(
        3.5,
        ymax - (ymax - ymin) * 0.06,
        "unreliable\nzone",
        ha="center",
        va="top",
        fontsize=6.5,
        color="gray",
        style="italic",
    )

    fig.tight_layout(pad=0.5)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── 3 & 4. per-KPI small multiples ───────────────────────────────────────────
def plot_per_kpi(df: pd.DataFrame, metric: str, ylabel: str, better: str, out: Path):
    """2×4 small-multiples grid (7 KPIs + summary panel)."""
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(DOUBLE_W, 3.8), sharex=True)
    axes_flat = axes.flatten()

    step_hint = 20  # target marker density across full x range

    for i, kpi in enumerate(KPI_ORDER):
        ax = axes_flat[i]
        sub = df[df["kpi"] == kpi].sort_values("context_d")
        x, y = sub["context_d"].values, sub[metric].values

        _shade_unreliable(ax)
        step = max(1, len(x) // step_hint)
        ax.plot(
            x,
            y,
            color=KPI_COLORS[kpi],
            linewidth=1.4,
            marker="o",
            markersize=3,
            markevery=step,
            zorder=3,
        )
        bx, by = _mark_best(ax, x, y, better)

        ax.set_title(f"{KPI_LABELS[kpi]}", fontsize=8.5, pad=3)
        ax.set_xlim(0.5, 81.5)

        # best label: place at lower-right corner to avoid curve
        ymin, ymax = ax.get_ylim()
        ax.text(
            0.97,
            0.05,
            f"best={int(bx)}d",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color=BEST_COLOR,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=BEST_COLOR, alpha=0.85, lw=0.7),
        )

        # y-label only on leftmost column
        if i % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=8)

    # 8th panel: overall average curve
    ax8 = axes_flat[7]
    ov = df.groupby("context_d").agg(val=(metric, "mean")).reset_index().sort_values("context_d")
    x, y = ov["context_d"].values, ov["val"].values
    _shade_unreliable(ax8)
    step = max(1, len(x) // step_hint)
    ax8.plot(x, y, color="black", linewidth=1.5, marker="D", markersize=3, markevery=step, zorder=3)
    bx, by = _mark_best(ax8, x, y, better)
    ax8.set_title("Overall (mean)", fontsize=8.5, pad=3)
    ax8.set_xlim(0.5, 81.5)
    ymin, ymax = ax8.get_ylim()
    ax8.text(
        0.97,
        0.05,
        f"best={int(bx)}d",
        transform=ax8.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color=BEST_COLOR,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=BEST_COLOR, alpha=0.85, lw=0.7),
    )

    # shared x-label on bottom row
    for ax in axes_flat[ncols:]:
        ax.set_xlabel("Context length (days)", fontsize=8)

    # shared "unreliable zone" note — one legend entry
    shade_patch = mpatches.Patch(color=SHADE_COLOR, label="1–6d (unreliable)")
    best_line = mpl.lines.Line2D(
        [], [], color=BEST_COLOR, linestyle=":", linewidth=1.2, label="Best context"
    )
    fig.legend(
        handles=[shade_patch, best_line],
        loc="lower center",
        ncol=2,
        fontsize=7.5,
        frameon=True,
        bbox_to_anchor=(0.5, -0.04),
    )

    metric_title = "sMAPE (%)" if "sMAPE" in metric else "MASE"
    fig.suptitle(
        f"Per-KPI avg. {metric_title} vs. context length " f"(115 stations, 1–81 days)",
        fontsize=9,
        y=1.01,
    )
    fig.tight_layout(pad=0.6, h_pad=1.2, w_pad=0.8)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    csv = Path("output/chronos2_ctx_sweep/plots/all_metrics_long.csv")
    out_dir = Path("output/chronos2_ctx_sweep/plots")

    df = pd.read_csv(csv)
    print(f"Loaded {len(df):,} rows  ({df['context_d'].nunique()} contexts)")

    plot_overall_smape(df, out_dir / "infocom_overall_smape.png")
    plot_overall_mase(df, out_dir / "infocom_overall_mase.png")
    plot_per_kpi(df, "avg_sMAPE", "Avg. sMAPE (%)", "lower", out_dir / "infocom_per_kpi_smape.png")
    plot_per_kpi(df, "avg_MASE", "Avg. MASE", "lower", out_dir / "infocom_per_kpi_mase.png")

    print("\nDone. Files:")
    for f in sorted(out_dir.glob("infocom_*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
