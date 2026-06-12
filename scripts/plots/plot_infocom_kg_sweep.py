#!/usr/bin/env python3
"""INFOCOM-quality KG quality-threshold (Θ) sweep plots.

Reads ``output/kg_sweep/metrics.csv`` produced by ``scripts/run_kg_sweep.py``
and visualises how the spec-extracted KG changes with the evaluator
approval threshold Θ (applied to the integrate_score = (C+Cl+R)/3 emitted
by the 3-signal Evaluator agent).  The story we want to make:

    * low Θ  → many edges, but lower mean quality score
    * mid Θ  → stable plateau (canonical KPI relationship count is flat,
                quality score creeps up)
    * Θ ≳ 0.95 → cliff: KG collapses to ~1 edge (unusable)

The chosen production threshold (Θ* = 0.7, set in
``telcoagent.kg_construction.pipeline``) sits inside the plateau, and is annotated
on every panel.

Outputs (under output/kg_sweep/plots/):
    infocom_kg_sweep_overview.png   — single-column dual-axis: canonical
                                       edges (left) + avg quality (right)
    infocom_kg_sweep_breakdown.png  — double-column 1×3 panel:
                                       (a) edges-by-type vs Θ
                                       (b) avg quality score vs Θ
                                       (c) retention ratio vs Θ
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
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
        "legend.fontsize": 7.5,
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

SINGLE_W = 3.5
DOUBLE_W = 7.16

# Same palette as the chronos-2 INFOCOM plots so the paper feels coherent.
C_TOTAL = "#8c564b"  # total triples
C_REL = "#1f77b4"  # ontology relationship edges
C_CANON = "#2ca02c"  # canonical {INCREASES, DECREASES, …}
C_QUALITY = "#d62728"  # avg quality score
CHOSEN = "#e41a1c"
CLIFF_SHADE = "#f0f0f0"

# Production default — see telcoagent/kg_construction/pipeline.py.
# Quality-first choice: Θ*=0.9 maximises avg quality score (0.9205) within the
# usable region.  We trade 5 canonical edges (76 → 71, ≈6.6 %) for the highest
# mean integrate score before the Θ≥0.95 collapse zone (1 edge → unusable).
THETA_STAR = 0.9


def _load(csv: Path) -> pd.DataFrame:
    """Collapse min_confidence dimension (KG-level metrics are constant in it)."""
    df = pd.read_csv(csv)
    keep = ["total_edges", "relationship_edges", "canonical_edges", "avg_quality_score"]
    g = df.groupby("theta")[keep].first().reset_index().sort_values("theta")
    return g


def _shade_cliff(ax, theta_min=0.925, theta_max=1.005):
    ax.axvspan(theta_min, theta_max, color=CLIFF_SHADE, zorder=0, linewidth=0)


def _mark_chosen(ax, x_val: float, y_val: float, color=CHOSEN, label=True):
    ax.axvline(x_val, color=color, linestyle=":", linewidth=1.2, alpha=0.75, zorder=2)
    ax.scatter(
        [x_val],
        [y_val],
        color=color,
        s=55,
        zorder=6,
        clip_on=False,
        edgecolor="white",
        linewidth=0.6,
    )


# ── 1. single-column overview ───────────────────────────────────────────────
def plot_overview(g: pd.DataFrame, out: Path):
    fig, ax1 = plt.subplots(figsize=(SINGLE_W, 2.9))
    _shade_cliff(ax1)

    x = g["theta"].values
    y_canon = g["canonical_edges"].values

    line1 = ax1.plot(
        x,
        y_canon,
        color=C_CANON,
        marker="o",
        markersize=3.6,
        linewidth=1.5,
        zorder=3,
        label="Canonical KPI edges",
    )[0]

    ax1.set_xlabel(r"Approval threshold $\Theta$")
    ax1.set_ylabel("Canonical KPI edges", color=C_CANON)
    ax1.tick_params(axis="y", labelcolor=C_CANON)
    ax1.set_xlim(0.475, 1.025)
    ax1.set_xticks(np.arange(0.5, 1.01, 0.1))

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    y_q = g["avg_quality_score"].values
    # Mask the Θ=1.0 row (no edges → q=0 is meaningless, would mislead the eye)
    mask = g["total_edges"].values > 0
    line2 = ax2.plot(
        x[mask],
        y_q[mask],
        color=C_QUALITY,
        marker="s",
        markersize=3.4,
        linewidth=1.4,
        linestyle="--",
        zorder=3,
        label="Avg. quality score",
    )[0]
    ax2.set_ylabel("Avg. quality score", color=C_QUALITY)
    ax2.tick_params(axis="y", labelcolor=C_QUALITY)
    ax2.grid(False)

    # Annotate chosen Θ on the canonical-edges curve
    chosen_idx = int(np.argmin(np.abs(x - THETA_STAR)))
    _mark_chosen(ax1, x[chosen_idx], y_canon[chosen_idx])
    ymin, ymax = ax1.get_ylim()
    ax1.annotate(
        rf"chosen $\Theta^*$={THETA_STAR}" "\n" f"({int(y_canon[chosen_idx])} edges)",
        xy=(THETA_STAR, y_canon[chosen_idx]),
        xytext=(THETA_STAR - 0.30, ymin + (ymax - ymin) * 0.22),
        arrowprops=dict(arrowstyle="->", color=CHOSEN, lw=0.9),
        color=CHOSEN,
        fontsize=7.2,
    )

    # Cliff label
    ax1.text(
        0.965,
        ymax - (ymax - ymin) * 0.06,
        "collapse\nzone",
        ha="center",
        va="top",
        fontsize=6.8,
        color="gray",
        style="italic",
    )

    ax1.legend(
        [line1, line2],
        ["Canonical KPI edges", "Avg. quality score"],
        loc="lower left",
        frameon=True,
        ncol=1,
    )

    fig.tight_layout(pad=0.5)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── 2. double-column breakdown (3 panels) ───────────────────────────────────
def plot_breakdown(g: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_W, 2.55))
    x = g["theta"].values

    # ----- (a) edges by type -----
    ax = axes[0]
    _shade_cliff(ax)
    ax.plot(
        x,
        g["total_edges"].values,
        color=C_TOTAL,
        marker="^",
        markersize=3.4,
        linewidth=1.4,
        label="All triples",
    )
    ax.plot(
        x,
        g["relationship_edges"].values,
        color=C_REL,
        marker="o",
        markersize=3.4,
        linewidth=1.4,
        label="Relationship edges",
    )
    ax.plot(
        x,
        g["canonical_edges"].values,
        color=C_CANON,
        marker="s",
        markersize=3.4,
        linewidth=1.4,
        label="Canonical KPI edges",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Theta$")
    ax.set_ylabel("Edge count (log scale)")
    ax.set_title("(a) KG size vs. $\\Theta$", pad=3)
    ax.set_xlim(0.475, 1.025)
    ax.set_xticks(np.arange(0.5, 1.01, 0.1))
    ax.legend(loc="lower left", frameon=True, fontsize=7)

    chosen_idx = int(np.argmin(np.abs(x - THETA_STAR)))
    _mark_chosen(ax, THETA_STAR, g["canonical_edges"].values[chosen_idx])

    # ----- (b) avg quality score -----
    ax = axes[1]
    _shade_cliff(ax)
    mask = g["total_edges"].values > 0
    ax.plot(
        x[mask],
        g["avg_quality_score"].values[mask],
        color=C_QUALITY,
        marker="D",
        markersize=3.4,
        linewidth=1.4,
    )
    ax.set_xlabel(r"$\Theta$")
    ax.set_ylabel("Avg. quality score")
    ax.set_title("(b) Mean integrate score of admitted edges", pad=3)
    ax.set_xlim(0.475, 1.025)
    ax.set_xticks(np.arange(0.5, 1.01, 0.1))
    _mark_chosen(ax, THETA_STAR, g["avg_quality_score"].values[chosen_idx])

    ymin, ymax = ax.get_ylim()
    ax.annotate(
        rf"$\Theta^*$={THETA_STAR}",
        xy=(THETA_STAR, g["avg_quality_score"].values[chosen_idx]),
        xytext=(THETA_STAR - 0.32, ymin + (ymax - ymin) * 0.78),
        arrowprops=dict(arrowstyle="->", color=CHOSEN, lw=0.9),
        color=CHOSEN,
        fontsize=7.2,
    )

    # ----- (c) retention ratio -----
    ax = axes[2]
    _shade_cliff(ax)
    base_canon = g["canonical_edges"].values[0]
    base_rel = g["relationship_edges"].values[0]
    ax.plot(
        x,
        g["canonical_edges"].values / base_canon,
        color=C_CANON,
        marker="s",
        markersize=3.4,
        linewidth=1.4,
        label="Canonical",
    )
    ax.plot(
        x,
        g["relationship_edges"].values / base_rel,
        color=C_REL,
        marker="o",
        markersize=3.4,
        linewidth=1.4,
        label="Relationship",
    )
    ax.set_xlabel(r"$\Theta$")
    ax.set_ylabel(r"Retention vs. $\Theta=0.5$")
    ax.set_title("(c) Edge retention ratio", pad=3)
    ax.set_xlim(0.475, 1.025)
    ax.set_xticks(np.arange(0.5, 1.01, 0.1))
    ax.set_ylim(-0.05, 1.08)
    ax.legend(loc="lower left", frameon=True, fontsize=7)
    _mark_chosen(ax, THETA_STAR, g["canonical_edges"].values[chosen_idx] / base_canon)

    fig.suptitle(
        r"3GPP spec-extracted KG vs. approval threshold $\Theta$"
        "  (13 specs, 8 935 raw triples at $\\Theta=0.5$)",
        fontsize=9,
        y=1.04,
    )
    fig.tight_layout(pad=0.5, w_pad=1.2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    csv = Path("output/kg_sweep/metrics.csv")
    out_dir = Path("output/kg_sweep/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    g = _load(csv)
    print(f"Loaded {csv}  Θ values: {g['theta'].tolist()}")
    print(g.to_string(index=False))

    plot_overview(g, out_dir / "infocom_kg_sweep_overview.png")
    plot_breakdown(g, out_dir / "infocom_kg_sweep_breakdown.png")

    chosen = g[np.isclose(g["theta"], THETA_STAR)].iloc[0]
    print(
        f"\nChosen Θ*={THETA_STAR}: "
        f"{int(chosen['canonical_edges'])} canonical / "
        f"{int(chosen['relationship_edges'])} rel / "
        f"{int(chosen['total_edges'])} total edges, "
        f"avg quality = {chosen['avg_quality_score']:.4f}"
    )
    print(
        f"Cliff at Θ=0.95: {int(g.loc[np.isclose(g['theta'], 0.95), 'relationship_edges'].iloc[0])} rel edges left"
    )


if __name__ == "__main__":
    main()
