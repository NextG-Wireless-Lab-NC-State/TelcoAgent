"""Formatting and persistence helpers for PAX-TS SensitivityResult artefacts.

Converts the raw numpy tensors in a :class:`~telcoagent.explainer.pax_ts.SensitivityResult`
into human-readable markdown, flat CSV rows, and on-disk artefact files. No
domain-specific causal interpretation is embedded here — all labels come from
the magnitude classifier in :func:`telcoagent.agents.registry._classify_sensitivity_magnitude`.

Engineering policy
------------------
All public functions validate their inputs at the boundary and raise
:class:`ValueError` loudly on shape mismatches. There are no silent
``except`` clauses and no fallback default returns.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from telcoagent.agents.registry import _classify_sensitivity_magnitude

from .pax_ts import PERTURBATION_TYPES, SensitivityResult

# ─── Constants ────────────────────────────────────────────────────────────

#: Default top-K cap on the global sensitivity table.
_DEFAULT_TOP_K_GLOBAL: int = 5

#: Default top-K cap on local per-event source lines.
_DEFAULT_TOP_K_LOCAL: int = 3

#: CSV header columns for global sensitivity rows.
_GLOBAL_CSV_COLUMNS: Tuple[str, ...] = (
    "scope",
    "source",
    "target",
    "perturbation_type",
    "S_raw",
    "S_norm",
    "magnitude",
)

#: CSV header columns for local sensitivity rows.
_LOCAL_CSV_COLUMNS: Tuple[str, ...] = (
    "scope",
    "event_idx",
    "t_center",
    "source",
    "target",
    "S_raw",
    "S_norm",
    "magnitude",
)


# ─── Private helpers ──────────────────────────────────────────────────────


def _validate_kpi_names(kpi_names: List[str], expected_channels: int) -> None:
    if len(kpi_names) != expected_channels:
        raise ValueError(
            f"kpi_names has {len(kpi_names)} entries but sensitivity tensor "
            f"has {expected_channels} channels"
        )


def _top_k_global_pairs(
    S_norm: np.ndarray,
    kpi_names: List[str],
    top_k: int,
) -> List[Tuple[int, int, int, float]]:
    """Return top-K off-diagonal (src, tgt, pert_idx) triples by |S_norm|.

    Returns list of (src_idx, tgt_idx, pert_idx, s_norm_value) sorted
    descending by absolute value.
    """
    C = len(kpi_names)
    entries: List[Tuple[float, int, int, int]] = []
    for src in range(C):
        for tgt in range(C):
            if src == tgt:
                continue
            for p_idx in range(S_norm.shape[2]):
                val = float(S_norm[src, tgt, p_idx])
                entries.append((abs(val), src, tgt, p_idx))
    entries.sort(key=lambda x: x[0], reverse=True)
    return [
        (src, tgt, p_idx, float(S_norm[src, tgt, p_idx])) for _, src, tgt, p_idx in entries[:top_k]
    ]


def _format_csv_row(
    scope: str,
    source: str,
    target: str,
    perturbation_type: str,
    s_raw: float,
    s_norm: float,
    magnitude: str,
    **extra: Any,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"scope": scope}
    row.update(extra)
    row["source"] = source
    row["target"] = target
    row["perturbation_type"] = perturbation_type
    row["S_raw"] = s_raw
    row["S_norm"] = s_norm
    row["magnitude"] = magnitude
    return row


# ─── Public API ───────────────────────────────────────────────────────────


__all__ = [
    "format_global_sensitivity_table",
    "format_local_sensitivity_lines",
    "to_csv_rows",
    "write_sensitivity_artefacts",
]


def format_global_sensitivity_table(
    result: SensitivityResult,
    kpi_names: List[str],
    top_k: int = _DEFAULT_TOP_K_GLOBAL,
) -> str:
    """Return a markdown table of the top-K global sensitivity pairs.

    Rows are ranked by ``|S_global_norm|``, off-diagonal only.  Each row
    contains rank, source KPI, target KPI, perturbation type, S_norm,
    S_raw, and magnitude classification.

    Parameters
    ----------
    result:
        PAX-TS output containing ``S_global_raw`` and ``S_global_norm``
        of shape ``(C, C, 3)``.
    kpi_names:
        Ordered KPI name list of length ``C``.
    top_k:
        Maximum number of rows to include.

    Returns
    -------
    str
        Markdown table string including header, separator, and data rows.

    Raises
    ------
    ValueError
        If ``kpi_names`` length does not match the tensor channel count.
    """
    C = result.S_global_norm.shape[0]
    _validate_kpi_names(kpi_names, C)

    header = "| rank | source | target | perturbation_type | S_norm | S_raw | magnitude |\n"
    sep = "|------|--------|--------|-------------------|--------|-------|----------|\n"
    lines = [header, sep]

    pairs = _top_k_global_pairs(result.S_global_norm, kpi_names, top_k)
    for rank, (src, tgt, p_idx, s_norm_val) in enumerate(pairs, start=1):
        s_raw_val = float(result.S_global_raw[src, tgt, p_idx])
        mag = _classify_sensitivity_magnitude(s_norm_val)["magnitude"]
        pert_type = PERTURBATION_TYPES[p_idx]
        lines.append(
            f"| {rank} | {kpi_names[src]} | {kpi_names[tgt]} | "
            f"{pert_type} | {s_norm_val:.6f} | {s_raw_val:.6f} | {mag} |\n"
        )

    return "".join(lines)


def format_global_matrix_grid(
    result: SensitivityResult,
    kpi_names: List[str],
    use_normalized: bool = True,
) -> str:
    """Render the full ``C × C`` cross-channel sensitivity matrix as a
    markdown grid (one matrix collapsing the three perturbation types).

    Each cell ``[i, j]`` reports the maximum absolute sensitivity of
    target ``j``'s forecast under any of the three perturbation types
    applied to source ``i``:

        cell[i, j] = max(|S_norm[i, j, p]|) over p in {mean, var, trend}

    The diagonal is left at 0 (self-sensitivity is uninformative). The
    grid mirrors the seven-row, seven-column "Cross-Channel Sensitivity
    Matrix (7×7)" panel in Figure 4 of the paper draft so the LLM can
    copy this block verbatim into §2 instead of hand-rendering the
    matrix from raw scores.
    """
    S = result.S_global_norm if use_normalized else result.S_global_raw
    if S.ndim != 3:
        raise ValueError(f"S_global tensor must be 3-D (C, C, P); got shape {S.shape}")
    n_channels, _, _ = S.shape
    _validate_kpi_names(kpi_names, n_channels)

    # Collapse perturbation types to per-cell max |sensitivity|.
    grid = np.max(np.abs(S), axis=-1)
    np.fill_diagonal(grid, 0.0)

    header = "| src \\ tgt | " + " | ".join(kpi_names) + " |\n"
    sep = "|" + "---|" * (n_channels + 1) + "\n"
    rows = [header, sep]
    for i in range(n_channels):
        cells = " | ".join(f"{grid[i, j]:.4f}" for j in range(n_channels))
        rows.append(f"| {kpi_names[i]} | {cells} |\n")
    return "".join(rows)


def format_local_sensitivity_lines(
    result: SensitivityResult,
    anomaly_events: List[Dict[str, Any]],
    kpi_names: List[str],
    top_k: int = _DEFAULT_TOP_K_LOCAL,
) -> str:
    """Return per-event sensitivity summary lines.

    For each event in ``result.event_centers``, emits one line of the form::

        Event {e} (h{t_start}-h{t_end}): {target_KPI} <- {src1} ({s1:.4f}), {src2} ({s2:.4f}), ...

    Events whose ``S_local_norm`` row is all-zero are skipped.

    Parameters
    ----------
    result:
        PAX-TS output.  ``S_local_norm`` shape is ``(n_events, C)``.
    anomaly_events:
        Original anomaly event dicts (must carry ``kpi``, ``t_start``,
        ``t_end`` keys).
    kpi_names:
        Ordered KPI name list of length ``C``.
    top_k:
        Maximum number of source KPIs listed per event line.

    Returns
    -------
    str
        Concatenated newline-terminated lines; empty string if no events.

    Raises
    ------
    ValueError
        If ``kpi_names`` length does not match ``S_local_norm`` channel count,
        or if ``result.event_centers`` references an out-of-bounds anomaly index.
    """
    if result.S_local_norm.shape[0] == 0:
        return ""
    C = result.S_local_norm.shape[1]
    _validate_kpi_names(kpi_names, C)

    lines: List[str] = []
    for local_idx, (event_orig_idx, t_center) in enumerate(result.event_centers):
        if event_orig_idx >= len(anomaly_events):
            raise ValueError(
                f"event_centers references event_idx={event_orig_idx} but "
                f"anomaly_events has only {len(anomaly_events)} entries"
            )
        row = result.S_local_norm[local_idx, :]
        if not np.any(row != 0.0):
            continue

        ev = anomaly_events[event_orig_idx]
        target_kpi = ev.get("kpi", "?")
        t_start = ev.get("t_start", t_center)
        t_end = ev.get("t_end", t_center)

        sorted_src = sorted(range(C), key=lambda i: abs(row[i]), reverse=True)
        top_sources = sorted_src[:top_k]
        src_parts = ", ".join(f"{kpi_names[i]} ({row[i]:.4f})" for i in top_sources)
        lines.append(
            f"Event {event_orig_idx} (h{t_start}-h{t_end}): " f"{target_kpi} <- {src_parts}\n"
        )

    return "".join(lines)


def to_csv_rows(
    result: SensitivityResult,
    kpi_names: List[str],
) -> List[Dict[str, Any]]:
    """Flatten global and local sensitivity into a list of CSV-ready dicts.

    Global rows have ``scope="global"`` with columns: source, target,
    perturbation_type, S_raw, S_norm, magnitude.  Diagonal entries
    (source == target) are excluded.

    Local rows have ``scope="local"`` with additional columns:
    event_idx, t_center.

    Parameters
    ----------
    result:
        PAX-TS output.
    kpi_names:
        Ordered KPI name list.

    Returns
    -------
    List[Dict[str, Any]]
        Plain dicts; no pandas dependency.

    Raises
    ------
    ValueError
        If ``kpi_names`` length does not match tensor channel count.
    """
    C = result.S_global_norm.shape[0]
    _validate_kpi_names(kpi_names, C)

    rows: List[Dict[str, Any]] = []

    # Global rows — all off-diagonal (C*(C-1)*3 total)
    for src in range(C):
        for tgt in range(C):
            if src == tgt:
                continue
            for p_idx, pert_type in enumerate(PERTURBATION_TYPES):
                s_raw = float(result.S_global_raw[src, tgt, p_idx])
                s_norm = float(result.S_global_norm[src, tgt, p_idx])
                mag = _classify_sensitivity_magnitude(s_norm)["magnitude"]
                rows.append(
                    _format_csv_row(
                        scope="global",
                        source=kpi_names[src],
                        target=kpi_names[tgt],
                        perturbation_type=pert_type,
                        s_raw=s_raw,
                        s_norm=s_norm,
                        magnitude=mag,
                    )
                )

    # Local rows — one per (event, source_channel)
    for local_idx, (event_orig_idx, t_center) in enumerate(result.event_centers):
        for src in range(C):
            s_raw = float(result.S_local_raw[local_idx, src])
            s_norm = float(result.S_local_norm[local_idx, src])
            mag = _classify_sensitivity_magnitude(s_norm)["magnitude"]
            row = _format_csv_row(
                scope="local",
                source=kpi_names[src],
                target="",
                perturbation_type="",
                s_raw=s_raw,
                s_norm=s_norm,
                magnitude=mag,
                event_idx=event_orig_idx,
                t_center=t_center,
            )
            rows.append(row)

    return rows


def write_sensitivity_artefacts(
    out_dir: str,
    result: SensitivityResult,
    kpi_names: List[str],
    anomaly_events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    """Write sensitivity CSVs and markdown summary to ``out_dir``.

    Creates three files:

    * ``sensitivity_global.csv``  — global off-diagonal rows from
      :func:`to_csv_rows`.
    * ``sensitivity_local.csv``   — local per-event rows from
      :func:`to_csv_rows`.
    * ``sensitivity_summary.md``  — concatenation of
      :func:`format_global_sensitivity_table` and (when
      ``anomaly_events`` is supplied) :func:`format_local_sensitivity_lines`.

    Parameters
    ----------
    out_dir:
        Directory path.  Must already exist.
    result:
        PAX-TS output.
    kpi_names:
        Ordered KPI name list.
    anomaly_events:
        Optional list of anomaly event dicts (one per event in
        ``result.event_centers``).  Required to render the per-event
        local sensitivity section in the markdown summary; when
        omitted, the summary contains only the global section.  The
        local CSV is always written regardless.

    Returns
    -------
    Dict[str, str]
        Mapping ``{"sensitivity_global": <path>, "sensitivity_local": <path>,
        "sensitivity_summary": <path>}``.

    Raises
    ------
    ValueError
        If ``kpi_names`` length does not match tensor channel count, or
        if ``anomaly_events`` is supplied but its length does not cover
        all indices referenced by ``result.event_centers``.
    FileNotFoundError
        If ``out_dir`` does not exist.
    """
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"out_dir does not exist: {out_dir!r}")

    all_rows = to_csv_rows(result, kpi_names)
    global_rows = [r for r in all_rows if r["scope"] == "global"]
    local_rows = [r for r in all_rows if r["scope"] == "local"]

    global_path = os.path.join(out_dir, "sensitivity_global.csv")
    local_path = os.path.join(out_dir, "sensitivity_local.csv")
    summary_path = os.path.join(out_dir, "sensitivity_summary.md")

    global_cols = list(_GLOBAL_CSV_COLUMNS)
    local_cols = list(_LOCAL_CSV_COLUMNS)

    with open(global_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=global_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(global_rows)

    with open(local_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=local_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(local_rows)

    global_table = format_global_sensitivity_table(result, kpi_names)
    summary_content = "## Global Sensitivity\n\n" + global_table
    if anomaly_events is not None:
        local_lines = format_local_sensitivity_lines(
            result,
            anomaly_events,
            kpi_names,
        )
        if local_lines:
            summary_content += "\n## Local Event Sensitivity\n\n" + local_lines

    with open(summary_path, "w") as f:
        f.write(summary_content)

    return {
        "sensitivity_global": global_path,
        "sensitivity_local": local_path,
        "sensitivity_summary": summary_path,
    }
