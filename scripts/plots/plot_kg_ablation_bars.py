"""Paper-grade IEEE INFOCOM bar chart for the KG-on / KG-off ablation.

Renders a side-by-side comparison of Faithfulness and Answer Relevancy
across the two ablation arms (with and without the 3GPP knowledge
graph), using the actual ORANSight-judged values from
``output/kg_ablation_full_v2/judge_grounded/judge_summary.csv``.

Output: ``output/plots/kg_ablation_bars.pdf`` — IEEE single-column
3.5", serif Times, no in-figure title (LaTeX caption replaces it).

The numbers are reported verbatim — Faithfulness comes out marginally
higher in kg_off (0.643 vs 0.615), while Answer Relevancy is clearly
higher in kg_on (0.807 vs 0.748). The accompanying paper paragraph
should acknowledge the Faithfulness inversion honestly; see the
script's docstring footer for a suggested discussion.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)


# ─── Data (frozen from judge_grounded run on 89 / 97 stations) ────────────


_FAITHFULNESS_KG_ON: float = 0.615
_FAITHFULNESS_KG_ON_SD: float = 0.046
_FAITHFULNESS_KG_OFF: float = 0.643
_FAITHFULNESS_KG_OFF_SD: float = 0.055
_FAITHFULNESS_N: int = 97

_ANSWER_RELEVANCY_KG_ON: float = 0.807
_ANSWER_RELEVANCY_KG_ON_SD: float = 0.066
_ANSWER_RELEVANCY_KG_OFF: float = 0.748
_ANSWER_RELEVANCY_KG_OFF_SD: float = 0.076
_ANSWER_RELEVANCY_N: int = 21


# ─── Style ────────────────────────────────────────────────────────────────


_KG_ON_COLOUR: str = "#2c5fa6"  # deep blue — w/ KG
_KG_OFF_COLOUR: str = "#bdbdbd"  # light grey — w/o KG
_KG_ON_HATCH: str = ""
_KG_OFF_HATCH: str = "////"


def _apply_paper_style(ieee_column: str = "single") -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8 if ieee_column == "single" else 9,
            "axes.labelsize": 8 if ieee_column == "single" else 10,
            "axes.titlesize": 9 if ieee_column == "single" else 11,
            "xtick.labelsize": 7 if ieee_column == "single" else 9,
            "ytick.labelsize": 7 if ieee_column == "single" else 9,
            "legend.fontsize": 7 if ieee_column == "single" else 9,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# ─── Renderer ─────────────────────────────────────────────────────────────


def _render(
    output_path: Path,
    ieee_column: str = "single",
) -> None:
    width = 3.5 if ieee_column == "single" else 7.16
    height = 2.4 if ieee_column == "single" else 3.5
    _apply_paper_style(ieee_column)

    fig, ax = plt.subplots(figsize=(width, height))

    metric_labels: List[str] = ["Faithfulness", "Answer Relevancy"]
    kg_on_vals = [_FAITHFULNESS_KG_ON, _ANSWER_RELEVANCY_KG_ON]
    kg_off_vals = [_FAITHFULNESS_KG_OFF, _ANSWER_RELEVANCY_KG_OFF]
    kg_on_sd = [_FAITHFULNESS_KG_ON_SD, _ANSWER_RELEVANCY_KG_ON_SD]
    kg_off_sd = [_FAITHFULNESS_KG_OFF_SD, _ANSWER_RELEVANCY_KG_OFF_SD]

    x = np.arange(len(metric_labels), dtype=float)
    bar_width = 0.34
    err_kw = dict(elinewidth=0.7, capsize=2.5, ecolor="black")

    bars_on = ax.bar(
        x - bar_width / 2,
        kg_on_vals,
        width=bar_width,
        color=_KG_ON_COLOUR,
        edgecolor="black",
        linewidth=0.55,
        hatch=_KG_ON_HATCH,
        label="w/ KG",
        yerr=kg_on_sd,
        error_kw=err_kw,
    )
    bars_off = ax.bar(
        x + bar_width / 2,
        kg_off_vals,
        width=bar_width,
        color=_KG_OFF_COLOUR,
        edgecolor="black",
        linewidth=0.55,
        hatch=_KG_OFF_HATCH,
        label="w/o KG",
        yerr=kg_off_sd,
        error_kw=err_kw,
    )

    # Numerical labels above each error bar (so they don't collide).
    for bars, vals, sds in (
        (bars_on, kg_on_vals, kg_on_sd),
        (bars_off, kg_off_vals, kg_off_sd),
    ):
        for bar, val, sd in zip(bars, vals, sds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + sd + 0.018,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.minorticks_on()
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(False, which="minor")
    ax.tick_params(axis="x", which="minor", bottom=False)

    leg = ax.legend(
        frameon=True,
        loc="upper left",
        framealpha=0.9,
        edgecolor="black",
        borderpad=0.35,
        labelspacing=0.3,
        handlelength=1.8,
        handletextpad=0.5,
    )
    if leg is not None:
        leg.get_frame().set_linewidth(0.5)

    fig.tight_layout(pad=0.3)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


# ─── Main ─────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render the IEEE INFOCOM bar chart comparing Faithfulness "
            "and Answer Relevancy across kg_on / kg_off arms."
        ),
    )
    parser.add_argument(
        "--output",
        default="output/plots/kg_ablation_bars.pdf",
        help="Output PDF path.",
    )
    parser.add_argument(
        "--ieee-column",
        default="single",
        choices=("single", "double"),
        help="IEEE column width preset.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    _render(out, ieee_column=args.ieee_column)
    return 0


if __name__ == "__main__":
    sys.exit(main())
