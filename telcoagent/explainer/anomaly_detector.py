"""Forecast anomaly detection for scenario-level explanation.

Deterministic detector that surfaces a small, ranked set of
"interesting" moments in a multivariate KPI forecast. The
:class:`ExplainerAgent` consumes this list to write the §4 (Temporal &
Scenario) section of its report without having to scan the entire
forecast horizon hour by hour.

Event taxonomy
--------------
* ``level_shift`` — the forecast deviates from the input baseline by at
  least ``z_threshold`` σ for ``min_duration`` consecutive hours.
* ``spike`` — instantaneous one-step change exceeding
  ``delta_threshold`` × Δσ_baseline.
* ``trend_break`` — the rolling 12-hour slope reverses sign and the
  reversal magnitude exceeds twice the baseline slope volatility.
* ``multi_kpi`` — at least ``multi_kpi_min`` channels are simultaneously
  in any of the three states above. These tend to correspond to
  congestion or handover events that span several KPIs.

Events are ranked by absolute z-score with a small priority boost for
``multi_kpi`` events so that systemic incidents surface even when each
individual channel is only moderately extreme.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────────────────

#: Default z-score threshold for level-shift detection.
_DEFAULT_LEVEL_Z_THRESHOLD: float = 2.5

#: Default Δσ multiple for spike detection.
_DEFAULT_SPIKE_DELTA_THRESHOLD: float = 3.0

#: Minimum consecutive hours for a level shift to be reported.
_DEFAULT_MIN_DURATION_H: int = 2

#: Minimum number of co-occurring channel anomalies to emit a
#: ``multi_kpi`` event.
_DEFAULT_MULTI_KPI_MIN: int = 3

#: Maximum events returned to the ExplainerAgent. Limits the LLM
#: context size and forces the detector to rank.
_DEFAULT_TOP_N: int = 10

#: Rolling window (hours) used for trend-break slope estimation.
_TREND_WINDOW_H: int = 12

#: Trend-break sensitivity: a slope reversal must exceed this multiple
#: of the baseline slope volatility to be reported.
_TREND_BREAK_SIGMA_MULTIPLE: float = 2.0

#: Score boost applied to multi-KPI events during ranking.
_MULTI_KPI_RANK_BOOST: float = 1.5

#: Floor used to avoid divide-by-zero on degenerate baselines.
_BASELINE_VARIANCE_FLOOR: float = 1e-8


# ─── Result container ─────────────────────────────────────────────────────


@dataclass
class AnomalyEvent:
    """One detected event in the forecast horizon.

    Attributes
    ----------
    kpi
        Affected KPI name. Set to ``"MULTI"`` for ``multi_kpi`` events.
    type
        One of ``level_shift``, ``spike``, ``trend_break``, ``multi_kpi``.
    t_start, t_end
        Inclusive forecast hour indices bounding the event.
    magnitude
        Maximum |deviation| in original KPI units.
    z_score
        Signed z-score (or co-occurrence count for ``multi_kpi``).
    direction
        ``"up"``, ``"down"``, or ``"mixed"`` (for ``multi_kpi``).
    co_occurring_kpis
        Names of all KPIs participating in a ``multi_kpi`` event;
        empty for single-channel events.
    """

    kpi: str
    type: str
    t_start: int
    t_end: int
    magnitude: float
    z_score: float
    direction: str
    co_occurring_kpis: List[str] = field(default_factory=list)


# ─── Mask segmentation ────────────────────────────────────────────────────


def _mask_to_segments(
    mask: np.ndarray,
    min_duration: int,
) -> List[Tuple[int, int]]:
    """Convert a 1D boolean mask into ``[(start, end_inclusive), …]``.

    Only segments of length ``≥ min_duration`` are returned. The mask
    is treated as non-circular: a True run reaching the end of the
    array still counts as a closed segment.
    """
    segments: List[Tuple[int, int]] = []
    in_segment = False
    start = 0
    for t, flag in enumerate(mask):
        if flag and not in_segment:
            in_segment = True
            start = t
        elif not flag and in_segment:
            in_segment = False
            if t - start >= min_duration:
                segments.append((start, t - 1))
    if in_segment and len(mask) - start >= min_duration:
        segments.append((start, len(mask) - 1))
    return segments


# ─── Per-channel detectors ────────────────────────────────────────────────


def _detect_level_shifts(
    kpi: str,
    forecast: np.ndarray,
    baseline_mean: float,
    baseline_sigma: float,
    z_threshold: float,
    min_duration: int,
) -> Tuple[List[AnomalyEvent], np.ndarray]:
    """Detect sustained level shifts on one channel."""
    z_scores = (forecast - baseline_mean) / baseline_sigma
    level_mask = np.abs(z_scores) > z_threshold

    events: List[AnomalyEvent] = []
    for start, end in _mask_to_segments(level_mask, min_duration):
        window = forecast[start : end + 1]
        z_window = z_scores[start : end + 1]
        peak = int(np.argmax(np.abs(z_window)))
        events.append(
            AnomalyEvent(
                kpi=kpi,
                type="level_shift",
                t_start=int(start),
                t_end=int(end),
                magnitude=float(np.max(np.abs(window - baseline_mean))),
                z_score=float(z_window[peak]),
                direction="up" if z_window[peak] > 0 else "down",
            )
        )
    return events, level_mask


def _detect_spikes(
    kpi: str,
    forecast: np.ndarray,
    last_baseline_value: float,
    baseline_delta_sigma: float,
    delta_threshold: float,
) -> Tuple[List[AnomalyEvent], np.ndarray]:
    """Detect single-step spikes on one channel.

    The first forecast step is differenced against the last baseline
    value to avoid an artificial zero at the boundary.
    """
    deltas = np.diff(forecast, prepend=last_baseline_value)
    spike_mask = np.abs(deltas) > delta_threshold * baseline_delta_sigma

    events = [
        AnomalyEvent(
            kpi=kpi,
            type="spike",
            t_start=int(t),
            t_end=int(t),
            magnitude=float(abs(deltas[t])),
            z_score=float(deltas[t] / baseline_delta_sigma),
            direction="up" if deltas[t] > 0 else "down",
        )
        for t in np.where(spike_mask)[0]
    ]
    return events, spike_mask


def _detect_trend_breaks(
    kpi: str,
    forecast: np.ndarray,
    baseline_slope_sigma: float,
    horizon: int,
) -> List[AnomalyEvent]:
    """Detect rolling-slope sign reversals on one channel."""
    if horizon < 2 * _TREND_WINDOW_H:
        return []

    rolling_mean = np.convolve(
        forecast,
        np.ones(_TREND_WINDOW_H) / _TREND_WINDOW_H,
        mode="valid",
    )
    slopes = np.diff(rolling_mean)

    events: List[AnomalyEvent] = []
    threshold = _TREND_BREAK_SIGMA_MULTIPLE * baseline_slope_sigma
    for t in range(1, len(slopes)):
        sign_changed = np.sign(slopes[t]) != np.sign(slopes[t - 1])
        large_swing = abs(slopes[t] - slopes[t - 1]) > threshold
        if sign_changed and large_swing:
            center = t + _TREND_WINDOW_H // 2
            events.append(
                AnomalyEvent(
                    kpi=kpi,
                    type="trend_break",
                    t_start=int(max(0, center - 2)),
                    t_end=int(min(horizon - 1, center + 2)),
                    magnitude=float(abs(slopes[t] - slopes[t - 1])),
                    z_score=float((slopes[t] - slopes[t - 1]) / baseline_slope_sigma),
                    direction="up" if slopes[t] > 0 else "down",
                )
            )
    return events


def _detect_multi_kpi_events(
    kpi_names: List[str],
    per_channel_anomaly_mask: np.ndarray,
    multi_kpi_min: int,
) -> List[AnomalyEvent]:
    """Aggregate per-channel masks into multi-KPI co-occurrence events."""
    coverage = per_channel_anomaly_mask.sum(axis=0)
    multi_mask = coverage >= multi_kpi_min

    events: List[AnomalyEvent] = []
    for start, end in _mask_to_segments(multi_mask, min_duration=1):
        active_kpis = [
            kpi_names[c]
            for c in range(per_channel_anomaly_mask.shape[0])
            if per_channel_anomaly_mask[c, start : end + 1].any()
        ]
        events.append(
            AnomalyEvent(
                kpi="MULTI",
                type="multi_kpi",
                t_start=int(start),
                t_end=int(end),
                magnitude=float(coverage[start : end + 1].max()),
                z_score=float(coverage[start : end + 1].max()),
                direction="mixed",
                co_occurring_kpis=active_kpis,
            )
        )
    return events


# ─── Public API ───────────────────────────────────────────────────────────


def detect_anomaly_events(
    forecast: np.ndarray,
    baseline: np.ndarray,
    kpi_names: List[str],
    z_threshold: float = _DEFAULT_LEVEL_Z_THRESHOLD,
    delta_threshold: float = _DEFAULT_SPIKE_DELTA_THRESHOLD,
    min_duration: int = _DEFAULT_MIN_DURATION_H,
    multi_kpi_min: int = _DEFAULT_MULTI_KPI_MIN,
    top_n: int = _DEFAULT_TOP_N,
) -> List[AnomalyEvent]:
    """Detect notable anomalies in a multivariate forecast.

    Parameters
    ----------
    forecast
        Predicted KPI series of shape ``(T, C)``.
    baseline
        Recent input window of shape ``(T_baseline, C)`` used to derive
        per-KPI mean, σ, and Δ-volatility. ``T_baseline`` need not
        equal ``T``.
    kpi_names
        Ordered KPI names; only the first ``C`` are used.
    z_threshold, delta_threshold, min_duration, multi_kpi_min, top_n
        Detector hyperparameters; defaults are tuned for hourly 7-KPI
        telco forecasts of length 168.

    Returns
    -------
    list of :class:`AnomalyEvent`
        Up to ``top_n`` events ranked by absolute z-score, with a 1.5×
        boost on ``multi_kpi`` events.
    """
    if forecast.ndim != 2 or baseline.ndim != 2:
        raise ValueError(
            f"forecast and baseline must be 2D (T, C); "
            f"got forecast={forecast.shape}, baseline={baseline.shape}"
        )
    horizon, n_channels = forecast.shape
    if baseline.shape[1] != n_channels:
        raise ValueError(
            f"channel count mismatch: forecast has {n_channels}, "
            f"baseline has {baseline.shape[1]}"
        )

    n_kpis = min(n_channels, len(kpi_names))
    events: List[AnomalyEvent] = []
    per_channel_mask = np.zeros((n_kpis, horizon), dtype=bool)

    for c in range(n_kpis):
        kpi = kpi_names[c]
        baseline_col = baseline[:, c]
        forecast_col = forecast[:, c]

        baseline_mean = float(np.mean(baseline_col))
        baseline_sigma = float(np.std(baseline_col)) or _BASELINE_VARIANCE_FLOOR
        baseline_delta_sigma = float(np.std(np.diff(baseline_col))) or _BASELINE_VARIANCE_FLOOR

        level_events, level_mask = _detect_level_shifts(
            kpi,
            forecast_col,
            baseline_mean,
            baseline_sigma,
            z_threshold,
            min_duration,
        )
        spike_events, spike_mask = _detect_spikes(
            kpi,
            forecast_col,
            float(baseline_col[-1]),
            baseline_delta_sigma,
            delta_threshold,
        )
        trend_events = _detect_trend_breaks(
            kpi,
            forecast_col,
            baseline_delta_sigma,
            horizon,
        )

        events.extend(level_events)
        events.extend(spike_events)
        events.extend(trend_events)
        per_channel_mask[c] = level_mask | spike_mask

    events.extend(
        _detect_multi_kpi_events(
            kpi_names[:n_kpis],
            per_channel_mask,
            multi_kpi_min,
        )
    )

    def _rank(event: AnomalyEvent) -> float:
        boost = _MULTI_KPI_RANK_BOOST if event.type == "multi_kpi" else 1.0
        return abs(event.z_score) * boost

    events.sort(key=_rank, reverse=True)
    if top_n is not None:
        events = events[:top_n]

    logger.info(
        "Anomaly detector: %d events (z≥%.1f, δ≥%.1fσ, multi_kpi≥%d, top_n=%d)",
        len(events),
        z_threshold,
        delta_threshold,
        multi_kpi_min,
        top_n,
    )
    return events


def events_to_dicts(events: List[AnomalyEvent]) -> List[dict]:
    """Serialize events for ReAct tool output (ExplainerAgent input)."""
    return [
        {
            "kpi": e.kpi,
            "type": e.type,
            "t_start": e.t_start,
            "t_end": e.t_end,
            "duration_h": e.t_end - e.t_start + 1,
            "magnitude": round(e.magnitude, 4),
            "z_score": round(e.z_score, 3),
            "direction": e.direction,
            "co_occurring_kpis": e.co_occurring_kpis,
        }
        for e in events
    ]
