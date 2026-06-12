"""PAX-TS sensitivity tools — global matrix and per-event localization.

``query_sensitivity_matrix`` ranks (source → target, perturbation_type)
pairs from the cap-normalized global sensitivity tensor;
``query_localized_sensitivity`` ranks source KPIs for a single anomaly
event from the localized (Type-1) perturbation vector. When no
:class:`~telcoagent.explainer.pax_ts.SensitivityResult` is attached,
both return an explicit error payload (no synthetic fallback), per
CLAUDE.md policy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from telcoagent.agents.registry import ToolRegistry, _capture_context, _schema

if TYPE_CHECKING:
    from .factory import ExplainerToolsConfig

logger = logging.getLogger(__name__)


def _register_query_sensitivity_matrix(
    registry: ToolRegistry,
    cfg: "ExplainerToolsConfig",
    ctx: List[str],
    n_channels: int,
) -> None:
    """Register PAX-TS global sensitivity tool.

    The tool returns the top-K (source → target, perturbation_type)
    pairs from the cap-normalized sensitivity tensor. When no
    sensitivity result is attached, the tool returns an explicit error
    payload (no synthetic fallback), per CLAUDE.md policy.
    """

    kpi_names = list(cfg.kpi_names)

    def query_sensitivity_matrix(
        target_kpi: str = "",
        perturbation_type: str = "all",
        top_k: int = 5,
        use_normalized: bool = True,
    ) -> Dict[str, Any]:
        """Return ranked KPI sensitivity pairs from the PAX-TS tensor."""
        result = cfg.sensitivity_result
        if result is None:
            return {"error": "no sensitivity result attached"}

        from telcoagent.agents.registry import _classify_sensitivity_magnitude
        from telcoagent.explainer.pax_ts import PERTURBATION_TYPES

        S = result.S_global_norm if use_normalized else result.S_global_raw

        ptype_indices: List[int]
        if perturbation_type == "all":
            ptype_indices = list(range(len(PERTURBATION_TYPES)))
        elif perturbation_type in PERTURBATION_TYPES:
            ptype_indices = [PERTURBATION_TYPES.index(perturbation_type)]
        else:
            return {
                "error": (
                    f"unknown perturbation_type {perturbation_type!r}; "
                    f"expected one of {list(PERTURBATION_TYPES) + ['all']}"
                )
            }

        target_indices: List[int]
        if target_kpi == "":
            target_indices = list(range(n_channels))
        elif target_kpi in kpi_names:
            target_indices = [kpi_names.index(target_kpi)]
        else:
            return {
                "error": f"unknown target_kpi {target_kpi!r}; "
                f"expected one of {kpi_names + ['']}"
            }

        scored: List[Dict[str, Any]] = []
        for i in range(n_channels):
            for j in target_indices:
                if i == j:
                    continue
                for p in ptype_indices:
                    s_norm = float(result.S_global_norm[i, j, p])
                    s_raw = float(result.S_global_raw[i, j, p])
                    rank_value = abs(float(S[i, j, p]))
                    scored.append(
                        {
                            "source_kpi": kpi_names[i],
                            "target_kpi": kpi_names[j],
                            "perturbation_type": PERTURBATION_TYPES[p],
                            "sensitivity_norm": round(s_norm, 6),
                            "sensitivity_raw": round(s_raw, 6),
                            "magnitude_class": _classify_sensitivity_magnitude(
                                rank_value,
                            )["magnitude"],
                            "rank_value": round(rank_value, 6),
                        }
                    )

        scored.sort(key=lambda row: row["rank_value"], reverse=True)
        top = scored[:top_k]

        for row in top:
            _capture_context(
                ctx,
                f"[Sensitivity] {row['source_kpi']}→{row['target_kpi']} "
                f"({row['perturbation_type']}): S_norm={row['sensitivity_norm']:+.4f} "
                f"({row['magnitude_class']})",
            )

        # Render the full C×C grid (per-cell max |S_norm| across the
        # three perturbation types) as a markdown table so the LLM can
        # copy it verbatim into the §2 "Cross-Channel Sensitivity
        # Matrix" panel without hand-rendering values from per-row
        # data. The grid mirrors the 7×7 panel in the paper figure.
        try:
            from telcoagent.explainer.sensitivity_format import (
                format_global_matrix_grid,
            )

            matrix_markdown = format_global_matrix_grid(
                result,
                kpi_names,
                use_normalized=use_normalized,
            )
        except Exception as exc:
            logger.warning(
                "query_sensitivity_matrix: grid render failed: %s",
                exc,
            )
            matrix_markdown = ""

        return {
            "available": True,
            "top_pairs": top,
            "matrix_markdown": matrix_markdown,
            "kpi_names": kpi_names,
            "perturbation_types": list(PERTURBATION_TYPES),
            "use_normalized": use_normalized,
        }

    registry.register(
        "query_sensitivity_matrix",
        query_sensitivity_matrix,
        _schema(
            "query_sensitivity_matrix",
            "Query the PAX-TS-derived global cross-channel sensitivity "
            "matrix. Returns the top-K (source → target, perturbation_type) "
            "pairs ranked by |S_norm|. Use perturbation_type to filter to a "
            "single decomposition (mean / variance / trend) or pass 'all' "
            "to rank across types. target_kpi='' aggregates across targets.",
            {
                "type": "object",
                "properties": {
                    "target_kpi": {
                        "type": "string",
                        "description": "Target KPI name; '' for global top-K.",
                        "enum": kpi_names + [""],
                    },
                    "perturbation_type": {
                        "type": "string",
                        "description": "Perturbation decomposition to query.",
                        "enum": ["mean", "variance", "trend", "all"],
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 49,
                        "description": "Number of top pairs to return.",
                    },
                    "use_normalized": {
                        "type": "boolean",
                        "description": (
                            "True (default) ranks by cap-normalized "
                            "sensitivity; False ranks by raw KPI-units "
                            "sensitivity."
                        ),
                    },
                },
                "required": [],
            },
        ),
    )


def _register_query_localized_sensitivity(
    registry: ToolRegistry,
    cfg: "ExplainerToolsConfig",
    ctx: List[str],
    n_channels: int,
) -> None:
    """Register PAX-TS per-event localized sensitivity tool.

    For a given anomaly event index, returns the top-K source KPIs
    whose recent-week localized perturbation most affects the event's
    target KPI within its hour window.
    """

    kpi_names = list(cfg.kpi_names)

    def query_localized_sensitivity(
        event_index: int,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Return per-event source KPI ranking from S_local_norm."""
        result = cfg.sensitivity_result
        if result is None:
            return {"error": "no sensitivity result attached"}

        from telcoagent.agents.registry import _classify_sensitivity_magnitude

        n_events = result.S_local_norm.shape[0]
        if not (0 <= event_index < n_events):
            return {"error": (f"event_index {event_index} out of range " f"[0, {n_events})")}

        # event_centers[event_index] = (original_event_idx, t_center).
        if event_index >= len(result.event_centers):
            return {"error": (f"event_centers truncated; index {event_index} " f"unavailable")}
        original_idx, t_center = result.event_centers[event_index]

        target_kpi = ""
        target_window: Tuple[int, int] = (-1, -1)
        if 0 <= original_idx < len(cfg.anomaly_events):
            ev = cfg.anomaly_events[original_idx]
            target_kpi = str(ev.get("kpi", ""))
            target_window = (
                int(ev.get("t_start", -1)),
                int(ev.get("t_end", -1)),
            )

        norm_row = result.S_local_norm[event_index]
        raw_row = result.S_local_raw[event_index]
        ranked: List[Dict[str, Any]] = []
        for src in range(n_channels):
            s_norm = float(norm_row[src])
            s_raw = float(raw_row[src])
            ranked.append(
                {
                    "source_kpi": kpi_names[src],
                    "sensitivity_norm": round(s_norm, 6),
                    "sensitivity_raw": round(s_raw, 6),
                    "magnitude_class": _classify_sensitivity_magnitude(
                        abs(s_norm),
                    )["magnitude"],
                    "rank_value": round(abs(s_norm), 6),
                }
            )
        ranked.sort(key=lambda row: row["rank_value"], reverse=True)
        top = ranked[:top_k]

        for row in top:
            _capture_context(
                ctx,
                f"[Sensitivity-Local] event_idx={event_index} "
                f"({target_kpi}) ← {row['source_kpi']}: "
                f"S_norm={row['sensitivity_norm']:+.4f} "
                f"({row['magnitude_class']})",
            )

        return {
            "available": True,
            "event_index": event_index,
            "original_event_idx": original_idx,
            "t_center": t_center,
            "target_kpi": target_kpi,
            "target_window": {"t_start": target_window[0], "t_end": target_window[1]},
            "top_sources": top,
        }

    registry.register(
        "query_localized_sensitivity",
        query_localized_sensitivity,
        _schema(
            "query_localized_sensitivity",
            "Query the PAX-TS-derived per-event localized sensitivity "
            "vector. Given an anomaly event index (matching the order in "
            "cfg.sensitivity_result.event_centers), returns the top-K "
            "source KPIs whose Type-1 localized perturbation at the event's "
            "center timestep most affects the event's target KPI inside "
            "its hour window.",
            {
                "type": "object",
                "properties": {
                    "event_index": {
                        "type": "integer",
                        "minimum": 0,
                        "description": (
                            "Index into " "sensitivity_result.event_centers (0-based)."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 7,
                        "description": "Number of top source KPIs to return.",
                    },
                },
                "required": ["event_index"],
            },
        ),
    )
