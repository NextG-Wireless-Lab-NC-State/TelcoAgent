"""Plot PRB_Util forecast accuracy vs input-context length.

Two figures emitted, one for each metric (so each is publication-grade
on its own without a paired y-axis):

* ``prb_util_ctx_sweep_nRMSE.png``
* ``prb_util_ctx_sweep_MASE.png``

X-axis = input context length in hours (auto-discovered from the
``ctx_<N>h/`` subfolders inside each model's sweep dir).

Y-axis = the named metric, averaged across all stations available at
that ``(model, ctx)`` cell. Each model is one line; markers indicate
exact ctx points that were measured. The legend uses
publication-friendly model names.

Inputs are the per-station JSON files written by
``scripts/baselines/run_<model>_h7d_ctx_sweep.py``:

::

    output/<model_sweep_dir>/ctx_<N>h/station_<id>.json
        → metrics → per_kpi → PRB_Util → {nRMSE, MASE, ...}
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────


_DEFAULT_MODEL_DIRS: List[Tuple[str, str, str]] = [
    # (sweep_dir_name, display_label, line_colour)
    ("amazon_chronos-2_h7d_ctx_sweep", "Chronos-2", "#1f77b4"),
    ("AutonLab_MOMENT-1-large_h7d_ctx_sweep", "MOMENT-1-large", "#d62728"),
    ("moirai-1.1-R-large_h7d_ctx_sweep", "Moirai-1.1-R-large", "#2ca02c"),
]

_TARGET_KPI: str = "PRB_Util"
_METRICS: List[str] = ["nRMSE", "MASE"]

_CTX_DIR_RE = re.compile(r"^ctx_(\d+)h$")


# ─── Discovery + aggregation ──────────────────────────────────────────────


def _list_ctx_lengths(model_dir: Path) -> List[int]:
    """Return sorted list of context-length integers under ``model_dir``."""
    out: List[int] = []
    for child in model_dir.iterdir():
        if not child.is_dir():
            continue
        match = _CTX_DIR_RE.match(child.name)
        if match:
            out.append(int(match.group(1)))
    return sorted(out)


def _collect_metric(
    model_dir: Path,
    ctx_h: int,
    kpi: str,
    metric: str,
) -> List[float]:
    """Read every per-station JSON under ``ctx_<ctx_h>h/`` and pull
    ``metrics → per_kpi → kpi → metric`` into a list (skipping missing)."""
    ctx_dir = model_dir / f"ctx_{ctx_h}h"
    values: List[float] = []
    if not ctx_dir.is_dir():
        return values
    for json_path in sorted(ctx_dir.glob("station_*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics = payload.get("metrics") or {}
        per_kpi = metrics.get("per_kpi") or {}
        kpi_metrics = per_kpi.get(kpi) or {}
        value = kpi_metrics.get(metric)
        if isinstance(value, (int, float)) and not (
            isinstance(value, float) and (value != value)  # NaN guard
        ):
            values.append(float(value))
    return values


def _aggregate_one_model(
    sweep_root: Path,
    sweep_name: str,
    kpi: str,
    metric: str,
) -> Tuple[List[int], List[float], List[int], List[float]]:
    """Return ``(ctx_list, mean_list, n_list, sem_list)`` for one model.

    ``sem_list[i]`` is the standard error of the mean
    (``std / sqrt(n)``) at ``ctx_list[i]`` — used for the
    publication-grade error envelope.
    """
    model_dir = sweep_root / sweep_name
    if not model_dir.is_dir():
        logger.warning("Missing sweep dir: %s — skipping", model_dir)
        return [], [], [], []

    ctx_lengths = _list_ctx_lengths(model_dir)
    means: List[float] = []
    counts: List[int] = []
    sems: List[float] = []
    kept_ctx: List[int] = []
    for ctx_h in ctx_lengths:
        values = _collect_metric(model_dir, ctx_h, kpi, metric)
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        kept_ctx.append(ctx_h)
        means.append(float(arr.mean()))
        counts.append(len(values))
        sems.append(float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0)
    return kept_ctx, means, counts, sems


# ─── Renderer ─────────────────────────────────────────────────────────────


_PAPER_LINESTYLES: List[str] = ["-", "--", "-."]
_PAPER_MARKERS: List[str] = ["o", "s", "^"]


def _render_figure(
    metric: str,
    series: List[Tuple[str, str, List[int], List[float], List[int], List[float]]],
    output_path: Path,
    kpi: str,
    ieee_column: str = "single",
    paper_grade: bool = False,
    metric_label_override: Optional[str] = None,
) -> None:
    """Render one (label-coloured) line per model.

    ``series`` items:
    ``(display_label, colour, ctx_list, mean_list, n_list, sem_list)``.

    Parameters
    ----------
    ieee_column
        Figure width preset: ``"single"`` (3.5 in) or ``"double"`` (7.16 in).
    paper_grade
        When True, switches to a publication-grade rendering optimised
        for IEEE INFOCOM: distinct line styles + marker shapes for
        B&W readability, ±1 SEM envelope shading, finer minor ticks,
        no figure title (the LaTeX caption replaces it), tighter axis
        margins. Suitable as a drop-in
        ``\\includegraphics{...}`` source.
    metric_label_override
        Optional pretty label for the y-axis (e.g. ``"nRMSE"`` →
        ``"nRMSE [unitless]"``). Defaults to ``metric``.
    """
    width = 3.5 if ieee_column == "single" else 7.16
    height = 2.4 if ieee_column == "single" else 4.0

    # INFOCOM / IEEE conference style — mirrors SciencePlots 'ieee+grid'.
    # Conventions: serif Times, 8 pt label / 7 pt tick + legend, thin
    # axes (0.5), major-only dashed grid, B&W-readable line styles.
    rc_overrides = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8 if ieee_column == "single" else 9,
        "axes.labelsize": 8 if ieee_column == "single" else 10,
        "axes.titlesize": 9 if ieee_column == "single" else 11,
        "xtick.labelsize": 7 if ieee_column == "single" else 9,
        "ytick.labelsize": 7 if ieee_column == "single" else 9,
        "legend.fontsize": 7 if ieee_column == "single" else 9,
        "pdf.fonttype": 42,  # TrueType for editable PDF text
        "ps.fonttype": 42,
    }
    if paper_grade:
        rc_overrides.update(
            {
                "axes.linewidth": 0.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.minor.width": 0.4,
                "ytick.minor.width": 0.4,
                "xtick.major.size": 3.0,
                "ytick.major.size": 3.0,
                "xtick.minor.size": 1.6,
                "ytick.minor.size": 1.6,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "legend.handlelength": 2.6,
                "legend.borderpad": 0.35,
                "legend.columnspacing": 1.0,
            }
        )
    plt.rcParams.update(rc_overrides)

    fig, ax = plt.subplots(figsize=(width, height))

    for idx, item in enumerate(series):
        label, colour, ctx_list, mean_list, _n_list, sem_list = item
        if not ctx_list:
            continue
        x = np.array(ctx_list, dtype=float)
        y = np.array(mean_list, dtype=float)
        # Convert hours → days for the x-axis (paper preference).
        x_days = x / 24.0
        if paper_grade:
            linestyle = _PAPER_LINESTYLES[idx % len(_PAPER_LINESTYLES)]
            marker = _PAPER_MARKERS[idx % len(_PAPER_MARKERS)]
            # Mark a point every 5 days — pick the index of the ctx
            # value closest to each multiple of 5 days.
            target_days = np.arange(5.0, float(x_days[-1]) + 0.5, 5.0)
            marker_indices = [int(np.argmin(np.abs(x_days - td))) for td in target_days]
            ax.plot(
                x_days,
                y,
                color=colour,
                label=label,
                linestyle=linestyle,
                marker=marker,
                markersize=4.0,
                markevery=marker_indices,
                linewidth=1.0,
                markerfacecolor=colour,
                markeredgecolor="white",
                markeredgewidth=0.5,
                alpha=0.95,
            )
        else:
            ax.plot(
                x_days,
                y,
                marker="o",
                markersize=3.0 if ieee_column == "single" else 4.5,
                linewidth=1.2 if ieee_column == "single" else 1.6,
                color=colour,
                label=label,
                alpha=0.9,
            )

    ax.set_xlabel("Input context length (days)")
    ax.set_ylabel(metric_label_override or metric)
    # No in-figure title — the LaTeX caption carries the description.
    if paper_grade:
        ax.minorticks_on()
        # INFOCOM convention: major-only dashed grid, no minor grid.
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.grid(False, which="minor")
        ax.legend(
            frameon=True,
            loc="best",
            framealpha=0.9,
            edgecolor="black",
            borderpad=0.35,
            labelspacing=0.3,
        )
        # Tight legend frame line (matches INFOCOM thin-axis aesthetic).
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_linewidth(0.5)
        ax.margins(x=0.02)
    else:
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.legend(frameon=False, loc="best")

    fig.tight_layout(pad=0.3 if paper_grade else 0.4)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


# ─── Main ─────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            f"Plot {_TARGET_KPI} forecast accuracy (nRMSE / MASE) vs input "
            "context length across foundation baselines."
        ),
    )
    parser.add_argument(
        "--sweep-root",
        default="output",
        help="Root dir containing the per-model ``*_ctx_sweep`` folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/plots",
        help="Where to write the two PNG figures.",
    )
    parser.add_argument(
        "--kpi",
        default=_TARGET_KPI,
        help=f"Target KPI (default: {_TARGET_KPI}).",
    )
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated 'sweep_dir:label[:colour]' triples to override "
            "the default 3 foundation baselines. "
            "Example: 'amazon_chronos-2_h7d_ctx_sweep:Chronos-2:#1f77b4'."
        ),
    )
    parser.add_argument(
        "--ieee-column",
        default="single",
        choices=("single", "double"),
        help="IEEE column width preset (single = 3.5 in, double = 7.16 in).",
    )
    parser.add_argument(
        "--format",
        default="pdf",
        choices=("pdf", "png"),
        help="Output file format (pdf for IEEE Overleaf, png for previews).",
    )
    parser.add_argument(
        "--paper-grade",
        action="store_true",
        help=(
            "Render publication-grade figures: distinct line styles + "
            "marker shapes for B&W readability, ±1 SEM envelope shading, "
            "minor ticks. No in-figure title (LaTeX caption replaces it)."
        ),
    )
    parser.add_argument(
        "--paper-grade-metrics",
        default="nRMSE",
        help=(
            "Comma-separated metric names to render in paper-grade style "
            "(default: nRMSE only). Other metrics use the standard style. "
            "Ignored unless --paper-grade is set. Use 'all' for every metric."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (DEBUG / INFO / WARNING).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sweep_root = Path(args.sweep_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.models:
        triples: List[Tuple[str, str, str]] = []
        for raw in args.models.split(","):
            parts = [p.strip() for p in raw.split(":") if p.strip()]
            if len(parts) < 2:
                logger.error(
                    "--models entry %r is malformed; need at least sweep_dir:label",
                    raw,
                )
                return 2
            sweep_dir, label = parts[0], parts[1]
            colour = parts[2] if len(parts) >= 3 else None
            triples.append((sweep_dir, label, colour or "#444444"))
    else:
        triples = list(_DEFAULT_MODEL_DIRS)

    paper_grade_metrics = (
        {m.strip() for m in args.paper_grade_metrics.split(",") if m.strip()}
        if args.paper_grade_metrics
        else set()
    )
    apply_paper_grade_to_all = args.paper_grade and "all" in paper_grade_metrics

    summary_rows: List[Dict[str, object]] = []
    for metric in _METRICS:
        series: List[Tuple[str, str, List[int], List[float], List[int], List[float]]] = []
        for sweep_name, label, colour in triples:
            ctx_list, mean_list, n_list, sem_list = _aggregate_one_model(
                sweep_root,
                sweep_name,
                args.kpi,
                metric,
            )
            if not ctx_list:
                logger.warning(
                    "%s: no data for %s/%s — line skipped",
                    label,
                    args.kpi,
                    metric,
                )
                continue
            series.append((label, colour, ctx_list, mean_list, n_list, sem_list))
            for ctx_h, mean, n, sem in zip(ctx_list, mean_list, n_list, sem_list):
                summary_rows.append(
                    {
                        "model": label,
                        "metric": metric,
                        "kpi": args.kpi,
                        "ctx_h": ctx_h,
                        "mean": round(mean, 6),
                        "sem": round(sem, 6),
                        "n_stations": n,
                    }
                )

        out_path = output_dir / f"{args.kpi}_ctx_sweep_{metric}.{args.format}"
        use_paper_grade = args.paper_grade and (
            apply_paper_grade_to_all or metric in paper_grade_metrics
        )
        _render_figure(
            metric,
            series,
            out_path,
            args.kpi,
            ieee_column=args.ieee_column,
            paper_grade=use_paper_grade,
        )

    csv_path = output_dir / f"{args.kpi}_ctx_sweep_summary.csv"
    if summary_rows:
        import csv

        fieldnames = ["model", "metric", "kpi", "ctx_h", "mean", "sem", "n_stations"]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        logger.info("Saved %s (%d rows)", csv_path, len(summary_rows))

    return 0


if __name__ == "__main__":
    sys.exit(main())
