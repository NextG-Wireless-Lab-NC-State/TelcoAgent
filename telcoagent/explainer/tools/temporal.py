"""``query_forecast_temporal`` — per-day forecast breakdown.

Daily means, peak/trough hours, day-over-day percentages, cross-KPI
co-occurrences, weekday/weekend patterns, and (in overview mode) the
``multi_kpi_coverage`` list of KPIs that §4 narratives must cover.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from telcoagent.agents.registry import ToolRegistry, _capture_context, _schema

from .shared import _WEEKDAY_NAMES

if TYPE_CHECKING:
    from .factory import ExplainerToolsConfig

logger = logging.getLogger(__name__)


#: Day-over-day percent threshold for marking a forecast day as
#: "notable" inside :func:`query_forecast_temporal`.
_NOTABLE_DOD_PCT: float = 10.0

#: Minimum |day-over-day %| considered to indicate a co-occurring
#: change across multiple KPIs.
_CO_OCCURRENCE_DOD_PCT: float = 5.0

#: Weekday/weekend mean-difference threshold (percent) for inclusion
#: in the weekly pattern summary.
_WEEKLY_PATTERN_MIN_DIFF_PCT: float = 3.0


_HOUR_LABELS: Dict[tuple, str] = {
    (0, 6): "night",
    (6, 10): "morning-rush",
    (10, 16): "midday",
    (16, 20): "evening-rush",
    (20, 24): "night",
}


def _hour_period(hour: int) -> str:
    for (start, end), label in _HOUR_LABELS.items():
        if start <= hour < end:
            return label
    return "night"


def _summarise_day(
    chunk: np.ndarray,
    day_index: int,
    weekday_idx: int,
) -> Dict[str, Any]:
    peak_hour = int(chunk.argmax())
    trough_hour = int(chunk.argmin())
    return {
        "day": day_index + 1,
        "weekday": _WEEKDAY_NAMES[weekday_idx],
        "is_weekend": weekday_idx >= 5,
        "mean": round(float(chunk.mean()), 2),
        "min": round(float(chunk.min()), 2),
        "max": round(float(chunk.max()), 2),
        "std": round(float(chunk.std()), 2),
        "peak_hour": peak_hour,
        "peak_period": _hour_period(peak_hour),
        "trough_hour": trough_hour,
        "trough_period": _hour_period(trough_hour),
    }


def _attach_dod_and_baseline(
    days: List[Dict[str, Any]],
    baseline_arr: Optional[np.ndarray],
) -> None:
    """Decorate per-day summaries with day-over-day and baseline deltas."""
    for d, day_info in enumerate(days):
        if d > 0:
            prev_mean = days[d - 1]["mean"]
            if abs(prev_mean) > 1e-8:
                pct = 100.0 * (day_info["mean"] - prev_mean) / abs(prev_mean)
                day_info["day_over_day_change_pct"] = round(pct, 2)
                if abs(pct) >= _NOTABLE_DOD_PCT:
                    direction = "spike" if pct > 0 else "drop"
                    day_info["notable_change"] = f"{direction} of {abs(pct):.1f}%"

        if baseline_arr is not None and len(baseline_arr) >= (d + 1) * 24:
            base_chunk = baseline_arr[d * 24 : (d + 1) * 24]
            base_mean = float(base_chunk.mean())
            if abs(base_mean) > 1e-8:
                vs_base = 100.0 * (day_info["mean"] - base_mean) / abs(base_mean)
                day_info["vs_baseline_pct"] = round(vs_base, 2)
                day_info["baseline_day_mean"] = round(base_mean, 2)


def _register_query_forecast_temporal(
    registry: ToolRegistry,
    cfg: "ExplainerToolsConfig",
    ctx: List[str],
    n_channels: int,
) -> None:
    kpi_names = cfg.kpi_names

    def query_forecast_temporal(kpi_name: str = "", notable_only: bool = False) -> Dict:
        """Per-day forecast breakdown with co-occurrence and weekly patterns."""
        prediction = cfg.prediction
        if prediction is None:
            return {"available": False, "message": "No prediction data available"}

        baseline = cfg.input_baseline or {}
        target_kpis = [kpi_name] if kpi_name and kpi_name in kpi_names else list(kpi_names)

        # Per-KPI per-day summary (computed for all KPIs to enable co-occurrence).
        per_kpi_days: Dict[str, List[Dict[str, Any]]] = {}
        for kpi in kpi_names:
            values = prediction.get(kpi)
            if not values:
                continue
            arr = np.asarray(values, dtype=float)
            n_days = min(7, len(arr) // 24)
            days: List[Dict[str, Any]] = []
            for d in range(n_days):
                chunk = arr[d * 24 : (d + 1) * 24]
                weekday_idx = (cfg.forecast_start_weekday + d) % 7
                days.append(_summarise_day(chunk, d, weekday_idx))
            baseline_arr = np.asarray(baseline[kpi], dtype=float) if kpi in baseline else None
            _attach_dod_and_baseline(days, baseline_arr)
            per_kpi_days[kpi] = days

        n_days = min(len(days) for days in per_kpi_days.values()) if per_kpi_days else 0

        co_occurrences: List[Dict[str, Any]] = []
        for d in range(1, n_days):
            changed = []
            for kpi, days in per_kpi_days.items():
                pct = days[d].get("day_over_day_change_pct", 0.0)
                if abs(pct) >= _CO_OCCURRENCE_DOD_PCT:
                    changed.append(
                        {
                            "kpi": kpi,
                            "day_over_day_pct": pct,
                            "direction": "up" if pct > 0 else "down",
                        }
                    )
            if len(changed) < 2:
                continue
            weekday = _WEEKDAY_NAMES[(cfg.forecast_start_weekday + d) % 7]
            co_occurrences.append(
                {
                    "day": d + 1,
                    "weekday": weekday,
                    "kpis_changed": changed,
                }
            )

        weekday_weekend: Dict[str, Dict[str, float]] = {}
        for kpi, days in per_kpi_days.items():
            wd_means = [day["mean"] for day in days if not day["is_weekend"]]
            we_means = [day["mean"] for day in days if day["is_weekend"]]
            if not (wd_means and we_means):
                continue
            wd_avg = float(np.mean(wd_means))
            we_avg = float(np.mean(we_means))
            if abs(wd_avg) <= 1e-8:
                continue
            diff_pct = 100.0 * (we_avg - wd_avg) / abs(wd_avg)
            if abs(diff_pct) < _WEEKLY_PATTERN_MIN_DIFF_PCT:
                continue
            weekday_weekend[kpi] = {
                "weekday_avg": round(wd_avg, 2),
                "weekend_avg": round(we_avg, 2),
                "weekend_vs_weekday_pct": round(diff_pct, 2),
            }

        result: Dict[str, Any] = {"available": True, "kpis": {}}
        for kpi in target_kpis:
            if kpi not in per_kpi_days:
                continue
            days = per_kpi_days[kpi]
            arr = np.asarray(prediction[kpi], dtype=float)
            forecast_mean = float(arr[: len(days) * 24].mean())
            visible_days = (
                [day for day in days if "notable_change" in day] if notable_only else days
            )
            kpi_result: Dict[str, Any] = {
                "forecast_mean": round(forecast_mean, 2),
                "per_day": visible_days,
                "notable_days_count": sum(1 for day in days if "notable_change" in day),
            }
            dod_changes = [
                (day["day"], day.get("day_over_day_change_pct", 0.0))
                for day in days
                if "day_over_day_change_pct" in day
            ]
            if dod_changes:
                most_volatile = max(dod_changes, key=lambda entry: abs(entry[1]))
                kpi_result["most_volatile_day"] = most_volatile[0]
                kpi_result["most_volatile_change_pct"] = most_volatile[1]
            if kpi in weekday_weekend:
                kpi_result["weekday_weekend_pattern"] = weekday_weekend[kpi]
            result["kpis"][kpi] = kpi_result

        if not kpi_name:
            required_kpis: set = set()
            reasons: Dict[str, List[str]] = {}

            def _require(k: str, reason: str) -> None:
                required_kpis.add(k)
                reasons.setdefault(k, []).append(reason)

            for kpi_key, kpi_data in result["kpis"].items():
                if kpi_data.get("notable_days_count", 0) > 0:
                    _require(kpi_key, "notable day-over-day change ≥ 10%")
                volatility = kpi_data.get("most_volatile_change_pct", 0)
                if abs(volatility) >= 5.0:
                    _require(kpi_key, f"most volatile change {volatility:+.1f}%")
            for occurrence in co_occurrences:
                for change in occurrence["kpis_changed"]:
                    _require(change["kpi"], f"co-occurrence on Day {occurrence['day']}")

            result["multi_kpi_coverage"] = {
                "required_kpis": sorted(required_kpis),
                "reasons": dict(sorted(reasons.items())),
                "note": "ALL listed KPIs MUST appear in §4 temporal narratives.",
            }

        if co_occurrences:
            result["cross_kpi_co_occurrences"] = co_occurrences
        if weekday_weekend:
            result["weekday_weekend_summary"] = weekday_weekend

        for kpi in target_kpis:
            for day in per_kpi_days.get(kpi, []):
                line = (
                    f"Forecast {kpi} Day {day['day']} ({day['weekday']}"
                    f"{', weekend' if day['is_weekend'] else ', weekday'}): "
                    f"mean={day['mean']}, min={day['min']}, max={day['max']}, "
                    f"peak_hour={day['peak_hour']} ({day['peak_period']}), "
                    f"trough_hour={day['trough_hour']} ({day['trough_period']})"
                )
                if "day_over_day_change_pct" in day:
                    line += f", day-over-day={day['day_over_day_change_pct']:+.2f}%"
                if "vs_baseline_pct" in day:
                    line += (
                        f", vs_baseline={day['vs_baseline_pct']:+.2f}% "
                        f"(baseline_day_mean={day['baseline_day_mean']})"
                    )
                if "notable_change" in day:
                    line += f" [NOTABLE: {day['notable_change']}]"
                _capture_context(ctx, line)

        for occurrence in co_occurrences:
            kpi_summary = ", ".join(
                f"{change['kpi']}({change['day_over_day_pct']:+.2f}%)"
                for change in occurrence["kpis_changed"]
            )
            _capture_context(
                ctx,
                f"Cross-KPI co-occurrence Day {occurrence['day']} "
                f"({occurrence['weekday']}): {kpi_summary} changed simultaneously",
            )

        for kpi, ww in weekday_weekend.items():
            if kpi in target_kpis:
                _capture_context(
                    ctx,
                    f"Weekday/weekend pattern {kpi}: weekday_avg={ww['weekday_avg']}, "
                    f"weekend_avg={ww['weekend_avg']}, "
                    f"weekend_vs_weekday={ww['weekend_vs_weekday_pct']:+.2f}%",
                )

        return result

    registry.register(
        "query_forecast_temporal",
        query_forecast_temporal,
        _schema(
            "query_forecast_temporal",
            "Per-day forecast breakdown for one KPI or all KPIs. Returns daily "
            "means, peak/trough hours, day-over-day percentages, cross-KPI "
            "co-occurrences, weekday/weekend pattern, and (in overview mode) a "
            "multi_kpi_coverage list of KPIs that must appear in §4. Set "
            "notable_only=true to keep the per-day list compact.",
            {
                "type": "object",
                "properties": {
                    "kpi_name": {
                        "type": "string",
                        "description": "KPI name; omit for all-KPI overview.",
                    },
                    "notable_only": {
                        "type": "boolean",
                        "description": "Keep only days with day-over-day ≥ 10%.",
                    },
                },
                "required": [],
            },
        ),
    )
