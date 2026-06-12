"""Evidence assembly for the Explainer Agent prompt (paper Sec. III-C).

Two responsibilities, both deterministic (no LLM calls):

* ``_format_structured_context`` -- render the detected anomaly events
  (with their KG-grounded candidate causes and OSM environmental
  context) into the compact orientation snapshot placed at the top of
  the ReAct user prompt.
* ``_prestuff_evidence`` -- for the ReAct-off ablation arm, execute the
  full explainer tool suite once and serialise every tool output into a
  single evidence block, so the single-shot LLM sees the same evidence
  the ReAct loop would have fetched on demand.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_ANOMALY_PREVIEW: int = 10


def _format_candidate_causes_inline(
    candidate_causes: List[Dict[str, Any]],
    max_shown: int = 2,
) -> str:
    """Render the top-N candidate causes as a single trailing string.

    Returns an empty string if no causes are present. The format is
    chosen so it slots after the existing one-line event preview, e.g.
    ``: driven by DL_iBler spike @ h41-h46 (KG: DECREASES, strong)``.
    """
    if not candidate_causes:
        return ""
    fragments: List[str] = []
    for cause in candidate_causes[:max_shown]:
        if cause.get("shared_mechanism"):
            fragments.append(
                f"shared-mechanism with {cause['source_kpi']} "
                f"(KG: {cause['relation_type']}, {cause['strength']})"
            )
            continue
        evidence = cause.get("concurrent_evidence")
        if evidence:
            fragments.append(
                f"{cause['source_kpi']} {evidence['type']} "
                f"@ h{evidence['t_start']}-h{evidence['t_end']} "
                f"(KG: {cause['relation_type']}, {cause['strength']})"
            )
            continue
        background = cause.get("background_value")
        if background:
            fragments.append(
                f"{cause['source_kpi']} background z={background['z_deviation']:+.2f} "
                f"(KG: {cause['relation_type']}, {cause['strength']})"
            )
    if not fragments:
        return ""
    return ": driven by " + "; ".join(fragments)


def _format_environmental_context_inline(
    environmental_context: Optional[Dict[str, Any]],
) -> str:
    """Render the per-event environmental_context as a trailing string.

    Returns an empty string when no environmental_context is attached.
    The OSM summary is taken verbatim from
    ``environmental_context['osm_summary']``; when the harness LLM has
    attached an ``llm_assessment``, its primary_layer + confidence are
    appended in ``[...]`` so the orientation snapshot conveys the LLM's
    own judgement instead of any pre-named rule flag.
    """
    if not environmental_context:
        return ""
    summary = environmental_context.get("osm_summary") or ""
    assessment = environmental_context.get("llm_assessment")
    if not summary and not assessment:
        return ""
    suffix = "; env: " + summary if summary else "; env"
    if isinstance(assessment, dict):
        layer = assessment.get("primary_layer") or "?"
        confidence = assessment.get("confidence") or "?"
        cited = assessment.get("cited_factors") or []
        cited_str = (", " + ", ".join(cited)) if cited else ""
        suffix += f" [llm:{layer}/{confidence}{cited_str}]"
    return suffix


_WEEKDAY_ABBR_FOR_PREVIEW: List[str] = [
    "Mon",
    "Tue",
    "Wed",
    "Thu",
    "Fri",
    "Sat",
    "Sun",
]


def _hour_window_label(
    t_start: int,
    t_end: int,
    forecast_start_weekday: int,
) -> str:
    """Convert forecast-hour indices into an operator-friendly clock label.

    The forecast horizon is a 168-hour stretch starting at ``t = 0``.
    Each hour index is mapped to a ``Day N (Wkd) HH:00`` clock label so
    explanations no longer need to expose the raw zero-based index
    (e.g. ``h39``).  ``t_end`` is the inclusive right edge of the event
    hour and is rendered as the *start* of the next hour
    (``HH:00 → (HH+1):00``) so a single-hour event still yields a
    non-empty window.

    When the event spans midnight (``t_start`` and ``t_end + 1`` fall
    on different forecast days), the label switches to the longer
    cross-day form
    ``Day N (Wkd) HH:00 → Day M (Wkd) HH:00`` instead of the more
    compact same-day ``Day N (Wkd) HH:00–HH:00`` form.
    """
    start_day = t_start // 24 + 1
    end_total = t_end + 1
    end_day = end_total // 24 + 1
    if end_total % 24 == 0:
        # Right edge falls on midnight — render as 24:00 of the same
        # day rather than 00:00 of the next, keeping the label
        # contained when the event ends exactly on the day boundary.
        end_day_display = end_day - 1
        end_hour = 24
    else:
        end_day_display = end_day
        end_hour = end_total % 24
    start_hour = t_start % 24
    start_wkd = _WEEKDAY_ABBR_FOR_PREVIEW[(forecast_start_weekday + start_day - 1) % 7]
    end_wkd = _WEEKDAY_ABBR_FOR_PREVIEW[(forecast_start_weekday + end_day_display - 1) % 7]
    if end_day_display == start_day:
        return f"Day {start_day} ({start_wkd}) " f"{start_hour:02d}:00–{end_hour:02d}:00"
    return (
        f"Day {start_day} ({start_wkd}) {start_hour:02d}:00 "
        f"→ Day {end_day_display} ({end_wkd}) {end_hour:02d}:00"
    )


def _format_structured_context(
    anomaly_events: List[Dict[str, Any]],
    kpi_names: List[str],
    forecast_start_weekday: int = 0,
) -> str:
    """Compact text snapshot of anomaly state for the ReAct prompt.

    The Explainer Agent reads this to orient itself before issuing tool
    calls; detailed data is then fetched on demand. When the events
    carry ``candidate_causes`` (KG-grounded upstream KPIs), the top two
    are appended inline to the existing one-line preview, followed by
    an ``env:`` suffix when ``environmental_context`` is non-empty.

    Each event line begins with a ``Day N (Wkd), hH1–hH2`` anchor so §4
    narratives and §5 actions can quote the same hour window verbatim
    without doing the ``t_start // 24`` arithmetic themselves.
    """
    lines: List[str] = []

    if anomaly_events:
        n_shown = min(_ANOMALY_PREVIEW, len(anomaly_events))
        lines.append(f"[TOP {n_shown} ANOMALY EVENTS]")
        for rank, event in enumerate(anomaly_events[:n_shown], start=1):
            cause_suffix = _format_candidate_causes_inline(
                event.get("candidate_causes", []),
            )
            env_suffix = _format_environmental_context_inline(
                event.get("environmental_context"),
            )
            window_label = _hour_window_label(
                int(event["t_start"]),
                int(event["t_end"]),
                forecast_start_weekday,
            )
            lines.append(
                f"  {rank}. [{window_label}] {event['type']} on {event['kpi']}: "
                f"|z|={abs(event['z_score']):.2f}, dir={event['direction']}"
                f"{cause_suffix}{env_suffix}"
            )
        lines.append("")

    return "\n".join(lines)


def _prestuff_evidence(
    tool_registry,
    anomaly_events: List[Dict[str, Any]],
    kpi_names: List[str],
    sensitivity_result: Optional[Any] = None,
) -> str:
    """Execute the explainer tool suite once and serialise the outputs.

    The ReAct-off ablation arm has no agent loop, so every tool the
    ReAct-on agent would normally call must be invoked deterministically
    and folded into the user prompt. The tool registry mutates its
    ``retrieved_contexts`` state list as a side-effect, so the
    downstream Faithfulness audit sees the same context blob as in
    the ReAct-on case.

    Block emission order is fixed:
        [Spatial] → [Temporal] → [Sensitivity] (global) →
        [Sensitivity-Local] (per-event) → [GraphRAG] (per pair).

    The two ``[Sensitivity*]`` blocks emit unconditionally; when the
    explainer is constructed without an attached
    ``sensitivity_result``, the sensitivity tools return their
    `{"error": "no sensitivity result attached"}` payload and the
    error itself is preserved verbatim in the prestuff so the LLM
    follows the system prompt's "do not fabricate / do not cite"
    rule. The ``[GraphRAG]`` block is similarly preserved across the
    KG-on / KG-off arms so the user prompt evidence VOLUME is
    symmetric — the only between-arm difference is the payload of
    the ``[GraphRAG]`` block itself.
    """
    import json

    blocks: List[str] = []

    def _emit(category: str, payload: Any) -> None:
        try:
            text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        except Exception:
            text = str(payload)
        blocks.append(f"### {category}\n```json\n{text}\n```")

    try:
        spatial = tool_registry.execute("query_spatial_context", {})
        _emit("[Spatial] query_spatial_context()", spatial)
    except Exception:
        logger.exception("Pre-stuff: unexpected error in query_spatial_context")
        raise

    try:
        temporal = tool_registry.execute(
            "query_forecast_temporal",
            {"notable_only": True},
        )
        _emit("[Temporal] query_forecast_temporal(notable_only=True)", temporal)
    except Exception:
        logger.exception("Pre-stuff: unexpected error in query_forecast_temporal")
        raise

    # ── Sensitivity prestuff (symmetric across kg_on / kg_off arms) ──
    # Always invoke the global sensitivity tool; the tool itself
    # returns an explicit error payload when no SensitivityResult is
    # attached, which the prompt is written to recognise. This keeps
    # the kg_on and kg_off user prompts the same length so the
    # ablation isolates the [GraphRAG] block payload, not its presence.
    try:
        sens_global = tool_registry.execute(
            "query_sensitivity_matrix",
            {"top_k": 10, "use_normalized": True},
        )
        _emit("[Sensitivity] query_sensitivity_matrix(top_k=10)", sens_global)
    except Exception:
        logger.exception("Pre-stuff: unexpected error in query_sensitivity_matrix")
        raise

    # Full C×C cross-channel sensitivity matrix grid (collapsed across
    # the three perturbation types to per-cell max |S_norm|). When a
    # SensitivityResult is attached we render the grid here as raw
    # markdown so the LLM copies it verbatim into §2 instead of trying
    # to hand-render the matrix from the top-K per-row table.
    if sensitivity_result is not None:
        try:
            from telcoagent.explainer.sensitivity_format import (
                format_global_matrix_grid,
            )

            grid_md = format_global_matrix_grid(
                sensitivity_result,
                kpi_names,
                use_normalized=True,
            )
            blocks.append(
                "### [Sensitivity-Matrix] cross-channel sensitivity grid "
                "(per-cell max |S_norm| across mean/variance/trend)\n" + grid_md
            )
        except Exception:
            logger.exception("Pre-stuff: unexpected error in matrix grid render")
            raise

    if anomaly_events:
        for ev_idx in range(len(anomaly_events)):
            try:
                sens_local = tool_registry.execute(
                    "query_localized_sensitivity",
                    {"event_index": ev_idx, "top_k": 3},
                )
                _emit(
                    f"[Sensitivity-Local] query_localized_sensitivity" f"(event_index={ev_idx})",
                    sens_local,
                )
            except Exception:
                logger.exception(
                    "Pre-stuff: unexpected error in query_localized_sensitivity(event_index=%d)",
                    ev_idx,
                )
                raise

    # Walk every candidate_cause across all events so the GraphRAG
    # block carries the mechanism description for every (src, tgt) pair
    # the §4 narrative might reference. Without this, the ReAct-off
    # arm would have anomaly events with relation_type but no mechanism
    # text — an unfair handicap. In KG-off mode this loop still runs
    # but query_graphrag_mechanism returns error payloads, which is
    # exactly what we want to measure.
    pairs: List[tuple] = []
    seen: set = set()
    for event in anomaly_events:
        target_kpi = event.get("kpi", "")
        for cause in event.get("candidate_causes", []) or []:
            source_kpi = cause.get("source_kpi", "")
            if not source_kpi or not target_kpi:
                continue
            key = (source_kpi, target_kpi)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    for source_kpi, target_kpi in pairs:
        try:
            mech = tool_registry.execute(
                "query_graphrag_mechanism",
                {"source_kpi": source_kpi, "target_kpi": target_kpi},
            )
            _emit(
                f"[GraphRAG] query_graphrag_mechanism({source_kpi}→{target_kpi})",
                mech,
            )
        except Exception:
            logger.exception(
                "Pre-stuff: unexpected error in query_graphrag_mechanism(%s→%s)",
                source_kpi,
                target_kpi,
            )
            raise

    if not pairs:
        blocks.append(
            "### [GraphRAG]\n_(no candidate causes available — "
            "causal-mechanism evidence channel is empty for this run)_"
        )

    return "\n\n".join(blocks)
