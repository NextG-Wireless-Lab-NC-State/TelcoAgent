"""Report post-processing helpers for the Explanation pipeline.

Deterministic text/markdown utilities applied to the LLM-authored
report by :class:`telcoagent.explainer.orchestrator.TelcoAgentExplainer`:
figure-embed backstops, preamble stripping, plot catalogues, and the
RAGAS retrieved-context injection. Pure functions — no LLM calls, no
I/O beyond reading the supplied strings/paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def ensure_figures_embedded(
    report: str,
    plot_paths: Optional[Dict[str, str]],
) -> str:
    """Guarantee that every anomaly figure appears at least once.

    The ReAct prompt asks the LLM to embed each per-KPI figure
    inside the matching §4 narrative, but small models occasionally
    drop figures. This pass:

    * Inserts the overview figure once, immediately above §4 if it
      is not already referenced.
    * Appends an "Anomaly figure gallery" subsection at the end of
      the report containing markdown image links for every per-KPI
      plot the LLM did not embed itself.

    Existing figure references the LLM produced are left untouched.
    """
    if not plot_paths:
        return report

    def _is_embedded(filename: str) -> bool:
        return filename in report

    # Insert the overview figure above §4 (or append to end if no §4).
    overview = plot_paths.get("overview")
    if overview and not _is_embedded(Path(overview).name):
        overview_block = "\n\n![Forecast & anomaly overview]" f"(plots/{Path(overview).name})\n"
        section4_marker = "## §4"
        idx = report.find(section4_marker)
        if idx >= 0:
            report = report[:idx] + overview_block + "\n" + report[idx:]
        else:
            report = report.rstrip() + overview_block

    # Per-KPI figures that the LLM forgot to embed.
    missing: List[tuple] = []
    for key, path in plot_paths.items():
        if key == "overview":
            continue
        if not _is_embedded(Path(path).name):
            missing.append((key, path))

    if missing:
        gallery = ["", "", "## Anomaly Figure Gallery", ""]
        for kpi, path in missing:
            gallery.append(f"**{kpi}**")
            gallery.append("")
            gallery.append(f"![{kpi} anomaly](plots/{Path(path).name})")
            gallery.append("")
        report = report.rstrip() + "\n" + "\n".join(gallery)

    return report


def format_plot_catalogue(plot_paths: Dict[str, str]) -> str:
    """Format the anomaly-plot catalogue for the ReAct system prompt.

    The Explainer Agent receives this catalogue so it can embed
    figure references in §4 narratives. Each per-KPI plot already
    contains shaded anomaly bands and a baseline overlay, so the
    narrative only needs to point to the figure rather than
    re-describe the visual evidence.
    """
    lines: List[str] = ["[ANOMALY FIGURES — embed in §4 narratives]"]
    overview = plot_paths.get("overview")
    if overview:
        lines.append(f"  - overview: {Path(overview).name}")
    for key, path in plot_paths.items():
        if key == "overview":
            continue
        lines.append(f"  - {key}: {Path(path).name}")
    lines.append(
        "  Reference each figure with markdown image syntax:" "  ![{KPI} anomaly](plots/{filename})"
    )
    lines.append("")
    return "\n".join(lines)


def strip_preamble(report: str) -> str:
    """Trim any LLM chatter that precedes the first report section."""
    markers = (
        "### Phase 2: Report Writing",
        "## Phase 2: Report Writing",
        "### Phase 3: Report Writing",
        "## Phase 3: Report Writing",
        "## §0 Executive Summary",
        "#### §0 Executive Summary",
        "### §0 Executive Summary",
        "## §0",
        "## §1 Forecast Summary",
        "#### §1 Forecast Summary",
        "## Forecast Summary",
        "#### Forecast Summary",
        "## §1",
    )
    for marker in markers:
        idx = report.find(marker)
        if idx < 0:
            continue
        trimmed = report[idx:]
        for prefix in ("### Phase 3: Report Writing", "## Phase 3: Report Writing"):
            if trimmed.startswith(prefix):
                return trimmed[len(prefix) :].lstrip("\n")
        return trimmed
    return report


def daily_means(arr: np.ndarray) -> List[float]:
    """Collapse an hourly series into up to seven daily means."""
    if len(arr) < 24:
        return [float(value) for value in arr]
    n_days = min(7, len(arr) // 24)
    return [float(np.mean(arr[d * 24 : (d + 1) * 24])) for d in range(n_days)]


def linear_slope(daily: List[float]) -> float:
    """Least-squares slope of the daily-mean series (units / day)."""
    if len(daily) < 2:
        return 0.0
    x = np.arange(len(daily), dtype=float)
    design = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(design, daily, rcond=None)[0]
    return float(slope)


def inject_forecast_contexts(contexts: List[str], summary: str) -> None:
    """Prepend per-KPI forecast summary lines to the RAGAS context list."""
    if not summary:
        return
    for line in summary.strip().splitlines():
        line = line.strip()
        if line.startswith("- "):
            contexts.insert(0, f"[Forecast] {line[2:]}")
        elif line:
            contexts.insert(0, f"[Forecast] {line}")


def dedupe_contexts(contexts: List[str]) -> List[str]:
    """Drop duplicate context entries while preserving first-seen order."""
    seen: set = set()
    unique: List[str] = []
    for entry in contexts:
        if entry not in seen:
            seen.add(entry)
            unique.append(entry)
    return unique
