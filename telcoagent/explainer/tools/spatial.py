"""``query_spatial_context`` — OSM context + 3GPP environment label.

Hand-curated ``_AREA_TYPE_CHARACTERISTICS`` and ``_REPORT_GUIDANCE``
tables were removed per the project's "no domain knowledge in
code/prompts" policy (see CLAUDE.md). Spatial context is now passed
through to the downstream LLM as the raw OSM blob plus the
deterministic ``_infer_environment_type`` mapping; the LLM derives the
operationally relevant guidance from that context rather than from a
canned mobility/traffic table.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from telcoagent.agents.registry import ToolRegistry, _capture_context, _schema

from .shared import _WEEKDAY_NAMES

if TYPE_CHECKING:
    from .factory import ExplainerToolsConfig

logger = logging.getLogger(__name__)


def _infer_environment_type(spatial_context: Any, mobility_info: Optional[str]) -> str:
    """Heuristic mapping from OSM context strings to a 3GPP environment label."""
    text = ""
    if isinstance(spatial_context, str):
        text = spatial_context.lower()
    elif isinstance(spatial_context, dict):
        text = " ".join(str(value).lower() for value in spatial_context.values())

    if "rural" in text or "rma" in text:
        return "RMa"
    if "indoor" in text or "inh" in text:
        return "InH"
    if any(token in text for token in ("micro", "umi", "dense urban")):
        return "UMi"
    if any(token in text for token in ("urban", "uma", "commercial", "residential")):
        return "UMa"
    if mobility_info and any(
        token in mobility_info.lower() for token in ("rural", "sparse", "farmland")
    ):
        return "RMa"
    return "UMa"


def _register_query_spatial_context(
    registry: ToolRegistry,
    cfg: "ExplainerToolsConfig",
    ctx: List[str],
) -> None:
    osm = cfg.osm
    station_id = cfg.station_id

    def query_spatial_context() -> Dict:
        """Retrieve raw OSM context plus a deterministic environment label."""
        try:
            spatial_context = osm.get_spatial_context(station_id)
        except Exception as exc:
            spatial_context = {"error": str(exc)}

        weekday_counts = {"weekday": 0, "weekend": 0}
        forecast_days: List[str] = []
        for day_offset in range(7):
            weekday_idx = (cfg.forecast_start_weekday + day_offset) % 7
            weekday_name = _WEEKDAY_NAMES[weekday_idx]
            is_weekend = weekday_idx >= 5
            weekday_counts["weekend" if is_weekend else "weekday"] += 1
            label = f"Day {day_offset + 1}: {weekday_name}"
            if is_weekend:
                label += " (weekend)"
            forecast_days.append(label)

        environment_type = _infer_environment_type(spatial_context, None)

        result: Dict[str, Any] = {
            "station_id": station_id,
            "spatial_context": spatial_context,
            "forecast_window": {
                "days": forecast_days,
                "weekday_count": weekday_counts["weekday"],
                "weekend_count": weekday_counts["weekend"],
            },
            "environment_type": environment_type,
        }

        if isinstance(spatial_context, str):
            for line in spatial_context.splitlines():
                line = line.strip()
                if line:
                    _capture_context(ctx, f"Station {station_id}: {line}")
                    _capture_context(ctx, line)
                    if ": " in line:
                        _capture_context(ctx, line.split(": ", 1)[1])
        elif isinstance(spatial_context, dict) and "error" not in spatial_context:
            for key, value in spatial_context.items():
                _capture_context(ctx, f"Station {station_id} {key}: {value}")
                _capture_context(ctx, f"{key}: {value}")
                _capture_context(ctx, str(value))

        return result

    registry.register(
        "query_spatial_context",
        query_spatial_context,
        _schema(
            "query_spatial_context",
            "Retrieve OpenStreetMap spatial context for the station: raw OSM "
            "blob, the deterministic 3GPP environment classification "
            "(RMa/UMa/UMi/InH derived from OSM keywords), and the forecast "
            "weekday window. Operational guidance is left to the explainer "
            "LLM rather than a hardcoded table.",
            {"type": "object", "properties": {}, "required": []},
        ),
    )
