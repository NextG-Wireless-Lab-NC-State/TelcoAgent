"""Helpers shared by more than one explainer tool module.

Weekday label tables plus the event time annotator used by both the
anomaly-events tool and the spatial tool's forecast-window rendering.
"""

from __future__ import annotations

from typing import Any, Dict, List

_WEEKDAY_NAMES: List[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

_WEEKDAY_ABBR: List[str] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _annotate_event_time(
    event: Dict[str, Any],
    forecast_start_weekday: int,
) -> Dict[str, Any]:
    """Attach forecast_day, weekday_label, and hour_window_label to a copy.

    The hour fields ``t_start`` / ``t_end`` are forecast-hour indices
    (0-167). This converts them into operator-friendly clock anchors,
    e.g. ``"Day 4 (Thu) 08:00-12:00"`` (or, when crossing midnight,
    ``"Day 4 (Thu) 22:00 -> Day 5 (Fri) 02:00"``). §5 actions quote
    these labels verbatim so reports never expose the raw zero-based
    index.
    """
    annotated = dict(event)
    t_start = int(event["t_start"])
    t_end = int(event["t_end"])
    start_day = t_start // 24 + 1
    end_total = t_end + 1
    end_day = end_total // 24 + 1
    if end_total % 24 == 0:
        end_day_display = end_day - 1
        end_hour = 24
    else:
        end_day_display = end_day
        end_hour = end_total % 24
    start_hour = t_start % 24
    start_wkd = _WEEKDAY_ABBR[(forecast_start_weekday + start_day - 1) % 7]
    end_wkd = _WEEKDAY_ABBR[(forecast_start_weekday + end_day_display - 1) % 7]
    if end_day_display == start_day:
        label = f"Day {start_day} ({start_wkd}) " f"{start_hour:02d}:00–{end_hour:02d}:00"
    else:
        label = (
            f"Day {start_day} ({start_wkd}) {start_hour:02d}:00 "
            f"→ Day {end_day_display} ({end_wkd}) {end_hour:02d}:00"
        )
    annotated["forecast_day"] = start_day
    annotated["weekday_label"] = start_wkd
    annotated["hour_window_label"] = label
    return annotated
