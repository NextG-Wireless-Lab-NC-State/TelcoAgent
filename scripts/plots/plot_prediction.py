"""Plot per-KPI predicted vs actual values in IEEE two-column style."""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# IEEE-friendly defaults
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
    }
)

KPI_META = {
    "RRC_Conn": {"label": "RRC Connected Users", "unit": "Count"},
    "DL_CQI": {"label": "DL CQI", "unit": "Index (0–15)"},
    "DL_iBler": {"label": "DL Initial BLER", "unit": "%"},
    "DL_rBler": {"label": "DL Residual BLER", "unit": "%"},
    "MAC_DL_Eff": {"label": "MAC DL Efficiency", "unit": "bits/TTI"},
    "PRB_Util": {"label": "DL PRB Utilization", "unit": "%"},
    "Throughput": {"label": "DL IP Throughput", "unit": "kbps"},
}

KPI_ORDER = list(KPI_META.keys())


def _format_metric(v: float) -> str:
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.3f}"


def plot_station(csv_path: str, json_path: str | None, out_dir: str):
    df = pd.read_csv(csv_path)
    hours = np.arange(len(df))

    # Load metrics from JSON if available
    metrics = {}
    if json_path and pathlib.Path(json_path).exists():
        with open(json_path) as f:
            meta = json.load(f)
        metrics = meta.get("metrics", {}).get("per_kpi", {})

    station_id = df["station_id"].iloc[0] if "station_id" in df.columns else "Unknown"

    # Day boundaries for vertical lines
    day_ticks = np.arange(0, 168, 24)
    day_labels = [f"Day {i+1}" for i in range(7)]

    # --- 7 subplots: one per KPI ---
    fig, axes = plt.subplots(
        len(KPI_ORDER),
        1,
        figsize=(7.16, 10.5),  # IEEE double-column width ≈ 7.16 in
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.35)

    for idx, kpi in enumerate(KPI_ORDER):
        ax = axes[idx]
        pred_col = f"{kpi}_pred"
        true_col = f"{kpi}_true"

        if pred_col not in df.columns or true_col not in df.columns:
            ax.set_visible(False)
            continue

        pred = df[pred_col].values
        true = df[true_col].values

        # Plot
        ax.plot(hours, true, color="#2166AC", linewidth=1.0, label="Ground Truth", zorder=3)
        ax.plot(
            hours,
            pred,
            color="#D6604D",
            linewidth=1.0,
            linestyle="--",
            label="TelcoAgent (GPT-4o)",
            zorder=3,
        )

        # Shaded error band
        ax.fill_between(hours, true, pred, alpha=0.10, color="#D6604D", zorder=2)

        # Day separators
        for d in day_ticks[1:]:
            ax.axvline(d, color="gray", linewidth=0.3, linestyle=":", alpha=0.5, zorder=1)

        # Labels
        meta_kpi = KPI_META[kpi]
        ylabel = f"{meta_kpi['label']}\n({meta_kpi['unit']})"
        ax.set_ylabel(ylabel, fontsize=8)

        # Metrics annotation (bottom-right to avoid legend overlap)
        kpi_metrics = metrics.get(kpi, {})
        mae = kpi_metrics.get("MAE")
        rmse = kpi_metrics.get("RMSE")
        if mae is not None and rmse is not None:
            ax.text(
                0.98,
                0.05,
                f"MAE={_format_metric(mae)}  RMSE={_format_metric(rmse)}",
                transform=ax.transAxes,
                fontsize=7,
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
            )

        # Legend (only on first subplot)
        if idx == 0:
            ax.legend(loc="upper right", ncol=2, fontsize=7)

        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.tick_params(axis="both", which="both", direction="in")

        # Y-axis: avoid scientific notation
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis="y", style="plain")

    # X-axis on bottom subplot
    axes[-1].set_xlabel("Hour", fontsize=10)
    axes[-1].set_xticks(day_ticks)
    axes[-1].set_xticklabels(day_labels, fontsize=8)
    axes[-1].set_xlim(0, 167)

    # Suptitle
    fig.suptitle(
        f"7-Day KPI Prediction vs Ground Truth — {station_id}",
        fontsize=11,
        fontweight="bold",
        y=0.995,
    )

    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save combined figure
    combined_path = out_path / f"{station_id}_kpi_comparison.pdf"
    fig.savefig(combined_path)
    plt.close(fig)
    print(f"Saved: {combined_path}")

    # --- Individual per-KPI figures (single-column width) ---
    for kpi in KPI_ORDER:
        pred_col = f"{kpi}_pred"
        true_col = f"{kpi}_true"
        if pred_col not in df.columns:
            continue

        pred = df[pred_col].values
        true = df[true_col].values
        meta_kpi = KPI_META[kpi]

        fig2, ax2 = plt.subplots(figsize=(3.5, 2.2))  # IEEE single-column

        ax2.plot(hours, true, color="#2166AC", linewidth=1.0, label="Ground Truth")
        ax2.plot(
            hours, pred, color="#D6604D", linewidth=1.0, linestyle="--", label="TelcoAgent (GPT-4o)"
        )
        ax2.fill_between(hours, true, pred, alpha=0.10, color="#D6604D")

        for d in day_ticks[1:]:
            ax2.axvline(d, color="gray", linewidth=0.3, linestyle=":", alpha=0.5)

        ax2.set_ylabel(f"{meta_kpi['label']} ({meta_kpi['unit']})", fontsize=8)
        ax2.set_xlabel("Hour", fontsize=8)
        ax2.set_xticks(day_ticks)
        ax2.set_xticklabels(day_labels, fontsize=6, rotation=30)
        ax2.set_xlim(0, 167)

        # Metrics in bottom-right inside plot
        kpi_metrics = metrics.get(kpi, {})
        mae = kpi_metrics.get("MAE")
        rmse = kpi_metrics.get("RMSE")
        if mae is not None:
            ax2.text(
                0.98,
                0.05,
                f"MAE={_format_metric(mae)}  RMSE={_format_metric(rmse)}",
                transform=ax2.transAxes,
                fontsize=6.5,
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9),
            )

        # Legend outside plot at top
        ax2.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            fontsize=7,
            frameon=True,
            edgecolor="0.8",
            fancybox=False,
            columnspacing=1.0,
        )
        ax2.grid(True, alpha=0.3, linewidth=0.4)
        ax2.tick_params(axis="both", which="both", direction="in")
        ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
        ax2.ticklabel_format(axis="y", style="plain")

        kpi_path = out_path / f"{station_id}_{kpi}.pdf"
        fig2.savefig(kpi_path)
        plt.close(fig2)
        print(f"Saved: {kpi_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot KPI prediction vs ground truth (IEEE style)")
    parser.add_argument("--csv", required=True, help="Path to station CSV")
    parser.add_argument("--json", default=None, help="Path to station JSON (for metrics)")
    parser.add_argument("--out", default="output/figures", help="Output directory for figures")
    args = parser.parse_args()

    plot_station(args.csv, args.json, args.out)
