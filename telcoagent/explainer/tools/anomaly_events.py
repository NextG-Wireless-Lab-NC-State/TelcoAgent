"""``query_anomaly_events`` — deterministic anomaly events ranked by |z|.

Surfaces the detector's event list (with KG-grounded candidate causes
and OSM environmental context already attached upstream) to the ReAct
agent, annotated with operator-friendly clock labels.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from telcoagent.agents.registry import ToolRegistry, _capture_context, _schema

from .shared import _annotate_event_time

if TYPE_CHECKING:
    from .factory import ExplainerToolsConfig

logger = logging.getLogger(__name__)


def _register_query_anomaly_events(
    registry: ToolRegistry,
    cfg: "ExplainerToolsConfig",
    ctx: List[str],
    n_channels: int,
) -> None:
    events = cfg.anomaly_events

    def query_anomaly_events(event_type: str = "") -> Dict:
        """Return the deterministic anomaly event list ranked by |z-score|.

        Each event reports magnitude, severity, and baseline-relative
        deviation. ``event_type`` filters by ``level_shift`` / ``spike``
        / ``trend_break`` / ``multi_kpi``.
        """
        if not events:
            return {"available": False, "events": [], "message": "No anomaly events detected."}

        filtered = [event for event in events if not event_type or event.get("type") == event_type]

        annotated: List[Dict[str, Any]] = [
            _annotate_event_time(event, cfg.forecast_start_weekday) for event in filtered
        ]
        for event in annotated:
            label = (
                f"Anomaly {event['type']} on {event['kpi']}: "
                f"{event['hour_window_label']} (t={event['t_start']}–{event['t_end']}), "
                f"z={event['z_score']:+.2f}, direction={event['direction']}"
            )
            _capture_context(ctx, label)

        return {
            "available": True,
            "n_total": len(events),
            "n_returned": len(annotated),
            "filter": event_type or "all",
            "events": annotated,
        }

    registry.register(
        "query_anomaly_events",
        query_anomaly_events,
        _schema(
            "query_anomaly_events",
            "Return deterministic forecast anomaly events (level shifts, spikes, "
            "trend breaks, multi-KPI co-occurrence) ranked by |z-score|. Each "
            "event reports timing (t_start, t_end as forecast hour indices "
            "0–167; plus forecast_day, weekday_label, and hour_window_label "
            "such as 'Day 3 (Wed), h64–h71' that §5 actions must quote "
            "verbatim), magnitude, direction, baseline-relative "
            "severity, KG-grounded candidate_causes (ranked upstream KPIs from "
            "the 3GPP KG with concurrent-event evidence, background-value "
            "deviation, and direction-consistency flags), plus an OSM-derived "
            "environmental_context per event (osm_summary, parsed factor dict, "
            "and an optional llm_assessment produced by the relevance harness "
            "model that judges whether the environment plausibly contributes "
            "to each anomaly). No telecom-domain rule list is hardcoded -- "
            "the harness LLM reasons over the raw factors. Filter by "
            "event_type if needed.",
            {
                "type": "object",
                "properties": {
                    "event_type": {
                        "type": "string",
                        "description": "Filter by event type (omit for all).",
                        "enum": ["level_shift", "spike", "trend_break", "multi_kpi"],
                    },
                },
                "required": [],
            },
        ),
    )
