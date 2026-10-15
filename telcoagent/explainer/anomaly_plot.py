"""Forecast + anomaly visualisation for the Explainer Agent report.

Generates per-KPI line plots that overlay:

* The 168-hour forecast trajectory.
* The input baseline window (last 168 hours of observed history).
* Shaded vertical bands for every detected anomaly event on that KPI.
* Optional dashed reference lines for the KPI's physical valid range.

The PNG paths are returned to the caller so the Explainer Agent can
embed them in the markdown report and reference them in the §4
scenario narratives. A separate overview plot stacks every KPI on a
shared time axis for at-a-glance reading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Headless rendering — no display required.

import matplotlib.pyplot as plt  # noqa: E402  (must follow `use("Agg")`)
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)


# ─── Visual constants ────────────────────────────────────────────────────

#: Hex colour codes per anomaly event type. Values chosen for high
#: contrast on a white background and partial transparency in fills.
_EVENT_COLOURS: Dict[str, str] = {
    "level_shift": "#d62728",
    "spike": "#ff7f0e",
    "trend_break": "#9467bd",
    "multi_kpi": "#17becf",
}

_FORECAST_COLOUR: str = "#1f77b4"
_BASELINE_COLOUR: str = "#7f7f7f"
_VALID_RANGE_COLOUR: str = "#2ca02c"

_FILL_ALPHA: float = 0.18
_BASELINE_ALPHA: float = 0.55
_VALID_RANGE_LINESTYLE: str = "--"

_FIGURE_DPI: int = 120
_PER_KPI_FIGSIZE: Tuple[float, float] = (9.0, 3.0)
_OVERVIEW_FIGSIZE: Tuple[float, float] = (10.0, 11.0)


# ─── Helpers ─────────────────────────────────────────────────────────────


def _events_for_kpi(events: List[Dict[str, Any]], kpi: str) -> List[Dict[str, Any]]:
    """Return the events whose ``kpi`` field matches ``kpi`` exactly.

    Multi-KPI events list participating channels under
    ``co_occurring_kpis`` and are included whenever ``kpi`` appears
    there, so the per-KPI plot still surfaces systemic incidents.
    """
    matched = []
    for event in events:
        if event.get("kpi") == kpi:
            matched.append(event)
        elif event.get("type") == "multi_kpi" and kpi in event.get("co_occurring_kpis", []):
            matched.append(event)
    return matched


def _format_event_label(event: Dict[str, Any]) -> str:
    direction = event.get("direction", "")
    arrow = "↑" if direction == "up" else "↓" if direction == "down" else "↕"
    return f"{event['type']} {arrow} " f"|z|={abs(event.get('z_score', 0.0)):.2f}"


def _draw_baseline(
    ax: plt.Axes,
    baseline: np.ndarray,
    forecast_length: int,
) -> None:
    """Plot the baseline window to the left of t = 0 with a divider."""
    n_baseline = baseline.shape[0]
    x_baseline = np.arange(-n_baseline, 0)
    ax.plot(
        x_baseline,
        baseline,
        color=_BASELINE_COLOUR,
        alpha=_BASELINE_ALPHA,
        linewidth=1.0,
        label="Input baseline (observed)",
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlim(-n_baseline, forecast_length - 1)


def _draw_anomaly_bands(
    ax: plt.Axes,
    events: List[Dict[str, Any]],
) -> None:
    """Shade the time interval of every anomaly event on the KPI."""
    seen_types: set = set()
    for event in events:
        colour = _EVENT_COLOURS.get(event.get("type", ""), "#888888")
        t_start = event.get("t_start", 0)
        t_end = event.get("t_end", t_start) + 1
        ax.axvspan(t_start, t_end, color=colour, alpha=_FILL_ALPHA, linewidth=0)
        # Only label the first occurrence of each event type so the
        # legend stays compact.
        event_type = event.get("type", "anomaly")
        if event_type not in seen_types:
            ax.axvspan(
                t_start,
                t_end,
                color=colour,
                alpha=_FILL_ALPHA,
                linewidth=0,
                label=event_type,
            )
            seen_types.add(event_type)
        ax.annotate(
            _format_event_label(event),
            xy=((t_start + t_end) / 2.0, ax.get_ylim()[1]),
            xytext=(0, -10),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7,
            color=colour,
            alpha=0.85,
        )


def _draw_valid_range(
    ax: plt.Axes,
    valid_range: Tuple[float, float],
) -> None:
    """Mark the physical valid range as horizontal dashed reference lines."""
    low, high = valid_range
    ax.axhline(
        low,
        color=_VALID_RANGE_COLOUR,
        linestyle=_VALID_RANGE_LINESTYLE,
        linewidth=0.8,
        alpha=0.6,
    )
    ax.axhline(
        high,
        color=_VALID_RANGE_COLOUR,
        linestyle=_VALID_RANGE_LINESTYLE,
        linewidth=0.8,
        alpha=0.6,
        label=f"Valid range [{low:g}, {high:g}]",
    )


# ─── Public API ──────────────────────────────────────────────────────────


_PER_KPI_PLOT_TARGETS: Tuple[str, ...] = ("PRB_Util",)


def plot_forecast_with_anomalies(
    forecast: np.ndarray,
    baseline: np.ndarray,
    kpi_names: List[str],
    anomaly_events: List[Dict[str, Any]],
    output_dir: str,
    station_id: str = "",
    valid_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, str]:
    """Render per-KPI and overview anomaly plots; return their file paths.

    Parameters
    ----------
    forecast
        Predicted KPI series of shape ``(T, C)``.
    baseline
        Observed input baseline of shape ``(T_b, C)``; typically the
        last 168 hours of the input window.
    kpi_names
        Ordered KPI names; only the first ``C`` are plotted.
    anomaly_events
        Output of :func:`telcoagent.anomaly_detector.events_to_dicts`.
    output_dir
        Directory where PNG files are written. Created if missing.
    station_id
        Optional identifier embedded in the figure title.
    valid_ranges
        Optional ``{kpi_name: (low, high)}`` from the 3GPP ontology.
        When provided, dashed reference lines are drawn on each plot.

    Returns
    -------
    dict
        ``{kpi_name → png_path}`` plus an ``"overview"`` entry with a
        stacked figure spanning every KPI.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    valid_ranges = valid_ranges or {}

    horizon, n_channels = forecast.shape
    n_kpis = min(n_channels, len(kpi_names))
    paths: Dict[str, str] = {}

    for c in range(n_kpis):
        kpi = kpi_names[c]
        if kpi not in _PER_KPI_PLOT_TARGETS:
            continue
        events = _events_for_kpi(anomaly_events, kpi)
        path = _render_per_kpi(
            forecast=forecast[:, c],
            baseline=baseline[:, c],
            kpi=kpi,
            events=events,
            station_id=station_id,
            valid_range=valid_ranges.get(kpi),
            output_path=out / f"anomaly_{kpi}.png",
        )
        paths[kpi] = str(path)

    overview_path = _render_overview(
        forecast=forecast,
        baseline=baseline,
        kpi_names=kpi_names[:n_kpis],
        anomaly_events=anomaly_events,
        station_id=station_id,
        valid_ranges=valid_ranges,
        output_path=out / "anomaly_overview.png",
    )
    paths["overview"] = str(overview_path)

    logger.info(
        "Anomaly plots: %d per-KPI + 1 overview written to %s",
        n_kpis,
        out,
    )
    return paths


# ─── Renderers ───────────────────────────────────────────────────────────


def _render_per_kpi(
    forecast: np.ndarray,
    baseline: np.ndarray,
    kpi: str,
    events: List[Dict[str, Any]],
    station_id: str,
    valid_range: Optional[Tuple[float, float]],
    output_path: Path,
) -> Path:
    horizon = forecast.shape[0]
    fig, ax = plt.subplots(figsize=_PER_KPI_FIGSIZE, dpi=_FIGURE_DPI)

    _draw_baseline(ax, baseline, horizon)
    ax.plot(
        np.arange(horizon),
        forecast,
        color=_FORECAST_COLOUR,
        linewidth=1.6,
        label="Forecast (168 h)",
    )
    if valid_range is not None:
        _draw_valid_range(ax, valid_range)
    _draw_anomaly_bands(ax, events)

    title = f"{kpi} — Forecast & detected anomalies" + (f"  ({station_id})" if station_id else "")
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel("Hours (negative = baseline, 0 = forecast start)", fontsize=8)
    ax.set_ylabel(kpi, fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=7, loc="best", framealpha=0.85)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _render_overview(
    forecast: np.ndarray,
    baseline: np.ndarray,
    kpi_names: List[str],
    anomaly_events: List[Dict[str, Any]],
    station_id: str,
    valid_ranges: Dict[str, Tuple[float, float]],
    output_path: Path,
) -> Path:
    horizon, n_channels = forecast.shape
    n_kpis = len(kpi_names)
    fig, axes = plt.subplots(
        n_kpis,
        1,
        figsize=_OVERVIEW_FIGSIZE,
        dpi=_FIGURE_DPI,
        sharex=True,
    )
    if n_kpis == 1:
        axes = [axes]

    for c, (ax, kpi) in enumerate(zip(axes, kpi_names)):
        events = _events_for_kpi(anomaly_events, kpi)
        _draw_baseline(ax, baseline[:, c], horizon)
        ax.plot(
            np.arange(horizon),
            forecast[:, c],
            color=_FORECAST_COLOUR,
            linewidth=1.4,
        )
        if kpi in valid_ranges:
            _draw_valid_range(ax, valid_ranges[kpi])
        _draw_anomaly_bands(ax, events)
        ax.set_ylabel(kpi, fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    axes[-1].set_xlabel("Hours (negative = baseline, 0 = forecast start)", fontsize=8)
    fig.suptitle(
        "Forecast overview & detected anomalies" + (f"  ({station_id})" if station_id else ""),
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
