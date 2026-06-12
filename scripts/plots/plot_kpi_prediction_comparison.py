"""Plot predicted vs actual KPIs (3-panel) for GLOBECOM paper.

Generates a 3-panel figure showing Moirai predictions vs ground truth
for RRC Conn, MAC DL Eff, and PRB Util over 7 days.

Replaces: paper/figures/dataset_overview.pdf
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# IEEE GLOBECOM style
plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "lines.linewidth": 0.9,
        "text.usetex": False,
    }
)

ROOT = Path(__file__).resolve().parent.parent.parent
STATION = "station_A_10"

csv_path = ROOT / "output" / "moirai" / "csv" / f"{STATION}.csv"
json_path = ROOT / "output" / "moirai" / f"{STATION}.json"

df = pd.read_csv(csv_path)
hours = np.arange(len(df))
n_days = 7

KPIS = [
    ("RRC_Conn", "RRC Connected Users", "count"),
    ("MAC_DL_Eff", "MAC Down Link Throughput", "kbps"),
    ("PRB_Util", "PRB Utilization", "%"),
]

COLOR_TRUE = "#2166AC"
COLOR_PRED = "#B2182B"

fig, axes = plt.subplots(
    len(KPIS),
    1,
    figsize=(3.5, 2.6),
    sharex=True,
    gridspec_kw={"hspace": 0.35},
)

for i, (kpi, label, unit) in enumerate(KPIS):
    ax = axes[i]
    true_vals = df[f"{kpi}_true"].values
    pred_vals = df[f"{kpi}_pred"].values

    ax.plot(hours, true_vals, color=COLOR_TRUE, linewidth=0.9, label="Actual", zorder=3)
    ax.plot(
        hours,
        pred_vals,
        color=COLOR_PRED,
        linewidth=0.9,
        linestyle="--",
        label="Prediction",
        zorder=3,
    )

    ax.fill_between(hours, true_vals, pred_vals, alpha=0.08, color=COLOR_PRED, zorder=2)

    for d in range(1, n_days):
        ax.axvline(d * 24, color="#AAAAAA", linewidth=0.4, linestyle="--", alpha=0.6)

    ax.set_ylabel("")
    ax.set_title(f"{label} ({unit})", fontsize=6, loc="left", pad=2)

    ax.tick_params(axis="both", which="both", length=2, pad=2)
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.3)
    ax.set_xlim(0, len(df) - 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if kpi == "RRC_Conn":
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, integer=True))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis="y", style="plain")
    elif kpi == "MAC_DL_Eff":
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis="y", style="plain")

# Legend closer to graph
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    fontsize=6,
    frameon=True,
    framealpha=0.9,
    edgecolor="0.8",
    handlelength=1.5,
    columnspacing=1.0,
    bbox_to_anchor=(0.55, 0.99),
    borderpad=0.3,
)

axes[-1].set_xlabel("Time (hours)", fontsize=8)
axes[-1].set_xticks([d * 24 for d in range(n_days + 1)])
axes[-1].set_xticklabels([str(d * 24) for d in range(n_days + 1)])
axes[-1].xaxis.set_minor_locator(ticker.MultipleLocator(12))

out_dir = ROOT / "paper" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
out_pdf = out_dir / "kpi_prediction_comparison.pdf"
fig.savefig(out_pdf)
print(f"Saved: {out_pdf}")
