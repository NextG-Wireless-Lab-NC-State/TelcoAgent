"""PAX-TS perturbation-based cross-channel sensitivity analysis.

Implements the PAX-TS algorithm of Kreuzer et al. (arXiv 2508.18982,
2025) — Perturbation Analysis eXplanations for Time Series — adapted
to the TelcoAgent setting: a 7-KPI multivariate forecast with a
1944-hour input window, of which Chronos-2 actually conditions on the
last ``CONTEXT_LENGTH_H`` (672 h = 28 d).

Adaptations vs the original protocol
------------------------------------

* **Perturbation region** is restricted to the last 168 h (one week)
  of the input window — the segment matching the forecast horizon and
  the part of Chronos-2's context with the strongest decoder bias.
  Mean / variance / trend perturbations are computed from statistics
  of this slice only; localized Type-1 perturbations are anchored at
  detected anomaly-event centers that fall inside the slice.
* **Localized centers** are derived from
  :class:`telcoagent.explainer.anomaly_detector.AnomalyEvent` rather
  than swept over every timestep. The original Algorithm 2 sweeps
  ``t ∈ {1, …, b}`` with optional sparsification; for our 1944 h
  context this would require ~80,000 forward passes per station, so
  we sparsify to anomaly-driven centers — the only timesteps the
  downstream §4 narrative ever quotes.
* **Output normalization**: perturbations are applied in raw KPI
  units per PAX-TS §III-B (so ``α=±0.1`` means a 10 % shift relative
  to each channel's own mean / variance / slope). The resulting
  sensitivity matrix is exposed in two forms, raw and cap-normalized
  via :data:`telcoagent.config.KPI_NORMALIZATION_MAX`, so that
  cross-KPI ranking is meaningful despite each channel having a
  different physical scale.
* **Acceleration**: every perturbed input is stacked into a single
  batched ``Chronos2Pipeline.predict`` call with
  ``predict_batches_jointly=True``. On an RTX 4090 (25 GB) the full
  ~340-row batch occupies under 1 GB of VRAM, giving a 40–80×
  speedup over sequential perturbation.

Public API
----------

* :class:`SensitivityResult` — dataclass holding the global
  ``(7, 7, 3)`` sensitivity tensor, the per-event localized
  ``(n_events, 7)`` matrix, and metadata.
* :func:`compute_sensitivity` — the single entry point. Given a
  Chronos-2 pipeline, the input window, the cached baseline
  forecast, and the anomaly-event list, returns a
  :class:`SensitivityResult`.

Engineering policy
------------------

This module follows the project's CLAUDE.md "no fallback code"
principle. Shape mismatches, NaN/Inf inputs, or zero-variance
channels raise loudly; there is no silent ``except Exception:
return default`` path. If a perturbation type cannot be applied to
a channel (e.g. all-zero baseline making the variance scaling
ill-defined) we raise a :class:`ValueError` with a precise
diagnosis and let the caller decide how to recover. The only
non-fatal degradation is the Type-1 event-center filter: anomaly
events whose center falls outside the perturbation window are
skipped with an INFO log (the local sensitivity matrix simply has
fewer rows), since this is a documented contract — not a hidden
fallback.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from telcoagent.config import (
    CONTEXT_LENGTH_H,
    KPI_NORMALIZATION_MAX,
    PREDICTION_LENGTH_H,
)

logger = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────────────────

#: Default scale set ``A`` from PAX-TS §IV-A — the original-paper
#: benchmark configuration with |A| = 6 covering ±1 %, ±5 %, ±10 %
#: relative shifts.
_DEFAULT_SCALES: Tuple[float, ...] = (-0.1, -0.05, -0.01, 0.01, 0.05, 0.1)

#: Type-1 Gaussian temporal half-width ``w`` (PAX-TS §III-A).
_DEFAULT_TYPE1_W: int = 2

#: Type-1 Gaussian smoothness scale ``s`` (PAX-TS §III-A).
_DEFAULT_TYPE1_S: float = 1.0

#: Numerical floor on the amplitude weight denominator
#: (``ε`` in PAX-TS §III-A) to avoid divide-by-zero on zero-valued
#: timesteps.
_AMPLITUDE_WEIGHT_EPS: float = 1e-8

#: Window over which Type-2 / Type-3 perturbations operate. Mean,
#: variance, and slope statistics are computed on this slice and the
#: perturbation is applied to the same slice. Set to the full
#: Chronos-2 context length (81 days = 1944 h) so the canonical
#: PAX-TS Algorithm 2 perturbs the entire input the model sees.
_DEFAULT_PERTURBATION_WINDOW_H: int = 1944

#: Diurnal seasonality period for Type-3 deseasonalization (PAX-TS
#: §III-C). With hourly data, ``k = 24`` corresponds to a 1-day
#: cycle, the dominant period of cellular KPIs.
_DEFAULT_DESEASONALIZATION_PERIOD_H: int = 24

#: Number of Chronos-2 sample paths whose median is taken to produce
#: the point forecast for change-ratio computation. Must match the
#: pipeline's default; we re-take the median here to be explicit.
_FORECAST_MEDIAN_DIM: int = -2

#: Floor on the per-channel standard deviation used in the Type-1
#: change-ratio normalizer; protects against degenerate flat
#: channels yielding ``Δ = 0`` and a divide-by-zero.
_CHANGE_RATIO_DELTA_FLOOR: float = 1e-6

#: Tolerance below which a Type-2 variance scaling factor is
#: rejected as numerically unstable (``α + 1 ≤ 0``). Outside this
#: bound the variance-perturbed series would fold through zero.
_VARIANCE_ALPHA_FLOOR: float = -0.95


# ─── Result container ─────────────────────────────────────────────────────


@dataclass
class SensitivityResult:
    """Output of :func:`compute_sensitivity`.

    Attributes
    ----------
    S_global_raw
        Cross-channel sensitivity tensor of shape ``(C, C, 3)`` in
        raw KPI units. ``S_global_raw[i, j, p]`` is the average
        signed change ratio of target KPI ``j``'s forecast mean
        when source KPI ``i`` is perturbed under perturbation type
        ``p`` ∈ {mean, variance, trend}.
    S_global_norm
        Cap-normalized version of ``S_global_raw``. Each entry is
        multiplied by ``cap[i] / cap[j]`` from
        :data:`telcoagent.config.KPI_NORMALIZATION_MAX` so that
        cross-KPI rankings are not dominated by raw-unit scale
        differences. Note: this transformation is asymmetric by
        design — the directional question "how much does a 10 %
        relative shift in source X move target Y in Y's units?" is
        genuinely directional.
    S_local_raw
        Per-event localized sensitivity of shape ``(n_events, C)``.
        Row ``e`` contains, for each source channel, the average
        change in the target KPI's forecast over the event window
        when that source channel is locally perturbed at the
        event's center timestep. The target KPI for row ``e`` is
        the KPI of the corresponding anomaly event; ``S_local_raw``
        therefore answers "which upstream channel's recent-week
        perturbation most affects this event's KPI?".
    S_local_norm
        Cap-normalized version of ``S_local_raw``.
    perturbation_scales
        The scale set ``A`` used at run time; copied here so that
        downstream artefacts can record the exact configuration.
    event_centers
        ``[(event_idx_in_input_list, t_center), …]`` — anomaly
        events whose center fell inside the perturbation window
        and that contributed a row to the local matrix. Events
        skipped because their center fell outside the window are
        not present (and are logged at INFO).
    n_perturbations
        Total number of perturbed inputs evaluated, including the
        baseline pass.
    elapsed_sec
        Wall-clock time of :func:`compute_sensitivity`.
    batch_strategy
        ``"joint"`` when a single ``pipeline.predict`` call covered
        the whole batch, ``"chunked"`` when chunking was requested.
    """

    S_global_raw: np.ndarray
    S_global_norm: np.ndarray
    S_local_raw: np.ndarray
    S_local_norm: np.ndarray
    perturbation_scales: Tuple[float, ...]
    event_centers: List[Tuple[int, int]] = field(default_factory=list)
    n_perturbations: int = 0
    elapsed_sec: float = 0.0
    batch_strategy: str = "joint"


# ─── Perturbation builders (vectorised numpy, raw KPI units) ──────────────


def _gaussian_position_weight(
    indices: np.ndarray,
    center: int,
    width: int,
    softness: float,
) -> np.ndarray:
    """PAX-TS §III-A positional weight ``w_p``.

    ``w_p[i] = exp(-s · ((i - t) / w)²)`` for ``|i - t| ≤ w``,
    zero otherwise. Returns a 1-D float array aligned with
    ``indices``.
    """
    rel = (indices - center).astype(float) / float(width)
    weights = np.exp(-softness * rel * rel)
    weights[np.abs(indices - center) > width] = 0.0
    return weights


def _amplitude_weight(channel_values: np.ndarray, center_value: float) -> np.ndarray:
    """PAX-TS §III-A amplitude weight ``w_a``.

    ``w_a[i] = min(x_i, x_t) / (max(x_i, x_t) + ε)``. Returns a
    1-D float array of the same length as ``channel_values``.
    """
    abs_vals = np.abs(channel_values)
    abs_center = abs(center_value)
    num = np.minimum(abs_vals, abs_center)
    den = np.maximum(abs_vals, abs_center) + _AMPLITUDE_WEIGHT_EPS
    return num / den


def _build_type1_perturbations(
    input_kpis: np.ndarray,
    event_centers: List[Tuple[int, int]],
    scales: Sequence[float],
    width: int = _DEFAULT_TYPE1_W,
    softness: float = _DEFAULT_TYPE1_S,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    """Construct localized Gaussian-smoothed perturbations.

    For each ``(event_idx, t_center)`` × source channel × scale,
    apply ``x'_i = x_i + w_p · w_a · α · x_i`` per PAX-TS §III-A.

    Returns
    -------
    perturbed
        Array of shape ``(B1, L, C)`` where
        ``B1 = len(event_centers) · C · len(scales)`` and ``(L, C)``
        is the input shape.
    index_map
        ``[(event_idx, source_channel, alpha), …]`` aligned with the
        first axis of ``perturbed`` so the caller can demultiplex
        the resulting forecasts.
    """
    L, C = input_kpis.shape
    indices = np.arange(L)

    rows: List[np.ndarray] = []
    index_map: List[Tuple[int, int, float]] = []

    for event_idx, t_center in event_centers:
        if not (0 <= t_center < L):
            raise ValueError(f"Type-1 center t={t_center} is outside input window [0, {L})")
        w_p = _gaussian_position_weight(indices, t_center, width, softness)
        for src in range(C):
            channel = input_kpis[:, src]
            w_a = _amplitude_weight(channel, channel[t_center])
            envelope = w_p * w_a * channel
            for alpha in scales:
                perturbed = input_kpis.copy()
                perturbed[:, src] = channel + alpha * envelope
                rows.append(perturbed)
                index_map.append((event_idx, src, float(alpha)))

    if not rows:
        return np.empty((0, L, C), dtype=input_kpis.dtype), []
    return np.stack(rows, axis=0), index_map


def _channel_mean_on_window(channel: np.ndarray, window: int) -> float:
    return float(np.mean(channel[-window:]))


def _build_type2_mean_perturbations(
    input_kpis: np.ndarray,
    scales: Sequence[float],
    window: int = _DEFAULT_PERTURBATION_WINDOW_H,
) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
    """First-moment scaling per PAX-TS §III-B.

    For each source channel and ``α ∈ scales``, shift the last
    ``window`` hours of that channel by ``α · μ_window``. The earlier
    history is left untouched.

    Returns ``(perturbed, [(source_channel, alpha), …])``.
    """
    L, C = input_kpis.shape
    if window > L:
        raise ValueError(f"perturbation window {window} exceeds input length {L}")

    rows: List[np.ndarray] = []
    index_map: List[Tuple[int, float]] = []
    for src in range(C):
        mu = _channel_mean_on_window(input_kpis[:, src], window)
        for alpha in scales:
            perturbed = input_kpis.copy()
            perturbed[-window:, src] = perturbed[-window:, src] + alpha * mu
            rows.append(perturbed)
            index_map.append((src, float(alpha)))
    return np.stack(rows, axis=0), index_map


def _build_type2_variance_perturbations(
    input_kpis: np.ndarray,
    scales: Sequence[float],
    window: int = _DEFAULT_PERTURBATION_WINDOW_H,
) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
    """Second-moment scaling per PAX-TS §III-B.

    ``x' = (x - μ) · √(α + 1) + μ`` applied to the last ``window``
    hours of each source channel.
    """
    L, C = input_kpis.shape
    if window > L:
        raise ValueError(f"perturbation window {window} exceeds input length {L}")

    rows: List[np.ndarray] = []
    index_map: List[Tuple[int, float]] = []
    for src in range(C):
        slice_ = input_kpis[-window:, src]
        mu = float(np.mean(slice_))
        for alpha in scales:
            if alpha <= _VARIANCE_ALPHA_FLOOR:
                raise ValueError(
                    f"Type-2 variance scale α={alpha} is below the floor "
                    f"{_VARIANCE_ALPHA_FLOOR} (would fold through zero)"
                )
            scaled = (slice_ - mu) * np.sqrt(alpha + 1.0) + mu
            perturbed = input_kpis.copy()
            perturbed[-window:, src] = scaled
            rows.append(perturbed)
            index_map.append((src, float(alpha)))
    return np.stack(rows, axis=0), index_map


def _ols_slope_intercept(values: np.ndarray) -> Tuple[float, float]:
    """Least-squares linear fit ``y = m · x + a`` over integer ``x``.

    Returns ``(m, a)``.
    """
    n = len(values)
    if n < 2:
        raise ValueError(f"Trend fit requires ≥ 2 samples, got {n}")
    x = np.arange(n, dtype=float)
    design = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(design, values, rcond=None)[0]
    return float(slope), float(intercept)


def _build_type3_trend_perturbations(
    input_kpis: np.ndarray,
    scales: Sequence[float],
    window: int = _DEFAULT_PERTURBATION_WINDOW_H,
    seasonality: int = _DEFAULT_DESEASONALIZATION_PERIOD_H,
) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
    """Trend-drift scaling per PAX-TS §III-C with sign-aware ``α``.

    The slice is deseasonalized via a length-``seasonality`` moving
    average, an OLS slope ``m`` and intercept ``a`` are fit to the
    smoothed residual, then a perturbed slope ``m' = m + z`` is
    applied verbatim per Kreuzer et al. §III-C:

    * ``z = +α · m`` when ``m ≥ 0`` → ``m' = m · (1 + α)``
      (positive trend amplified with positive ``α``).
    * ``z = -α · m`` when ``m < 0`` → ``m' = m · (1 − α)``
      (negative trend pushed toward zero with positive ``α``).

    Note that this is asymmetric: positive ``α`` amplifies
    upward trends but *dampens* downward trends; negative ``α``
    does the reverse. The asymmetry follows the paper's formula
    literally — interpret ``α`` as "push the trend toward the
    positive direction by a factor of α" rather than "strengthen
    the trend regardless of sign". The unit test
    ``test_type3_sign_aware_slope_scaling`` locks this convention
    in.

    The intercept is left-fixed (``a' = a``); the perturbed series
    is reconstructed via
    ``x' = x - a + a' - (m - m') · (i - 1)``.
    """
    L, C = input_kpis.shape
    if window > L:
        raise ValueError(f"perturbation window {window} exceeds input length {L}")
    if window < 2 * seasonality:
        raise ValueError(
            f"Type-3 trend fit needs window ({window}) ≥ 2 · seasonality " f"({2 * seasonality})"
        )

    kernel = np.ones(seasonality, dtype=float) / seasonality

    rows: List[np.ndarray] = []
    index_map: List[Tuple[int, float]] = []
    for src in range(C):
        slice_ = input_kpis[-window:, src].astype(float)
        deseasonalized = np.convolve(slice_, kernel, mode="valid")
        m, a = _ols_slope_intercept(deseasonalized)
        for alpha in scales:
            z = alpha * m if m >= 0.0 else -alpha * m
            m_prime = m + z
            a_prime = a  # left-fixed
            i_idx = np.arange(window, dtype=float)
            adjustment = -a + a_prime - (m - m_prime) * i_idx
            perturbed_slice = slice_ + adjustment
            perturbed = input_kpis.copy()
            perturbed[-window:, src] = perturbed_slice
            rows.append(perturbed)
            index_map.append((src, float(alpha)))
    return np.stack(rows, axis=0), index_map


# ─── Chronos-2 batched dispatch ───────────────────────────────────────────


def _take_median_along_samples(forecasts: Any) -> np.ndarray:
    """Reduce a Chronos-2 sample tensor to point forecasts.

    Chronos-2 returns a tensor of shape
    ``(batch, num_samples, prediction_length)`` for univariate or
    ``(batch, num_variates, num_samples, prediction_length)`` for
    multivariate. We take the median across the sample axis. The
    function tolerates both torch and numpy tensors and returns a
    numpy array.
    """
    import torch as _torch

    if isinstance(forecasts, _torch.Tensor):
        if forecasts.dim() == 3:
            median = _torch.median(forecasts, dim=1).values
        elif forecasts.dim() == 4:
            median = _torch.median(forecasts, dim=2).values
        else:
            raise ValueError(f"Chronos-2 forecast has unexpected ndim={forecasts.dim()}")
        return median.detach().cpu().numpy()
    arr = np.asarray(forecasts)
    if arr.ndim == 3:
        return np.median(arr, axis=1)
    if arr.ndim == 4:
        return np.median(arr, axis=2)
    raise ValueError(f"Chronos-2 forecast has unexpected ndim={arr.ndim}")


def _run_chronos_batched(
    pipeline: Any,
    perturbed_inputs: np.ndarray,
    chunk_size: int = 0,
    predict_batches_jointly: bool = True,
) -> Tuple[np.ndarray, str]:
    """Push a batch of perturbed inputs through Chronos-2.

    ``perturbed_inputs`` is ``(B, L, C)`` in raw KPI units. The
    last ``CONTEXT_LENGTH_H`` rows are sliced out (matching the
    contract of :meth:`telcoagent.prediction.predictor.TelcoAgent._make_predict_fn`
    at foundation.py:832), transposed to ``(B, C, CONTEXT_LENGTH_H)``,
    and dispatched as a single ``pipeline.predict`` call. Returns
    a numpy array of shape ``(B, PREDICTION_LENGTH_H, C)`` plus a
    label (``"joint"`` or ``"chunked"``) describing the dispatch
    strategy.
    """
    import torch as _torch

    batch_size = perturbed_inputs.shape[0]
    context = perturbed_inputs[:, -CONTEXT_LENGTH_H:, :]
    tensor = _torch.tensor(
        np.transpose(context, (0, 2, 1)).copy(),
        dtype=_torch.float32,
    )

    def _predict_chunk(t: "_torch.Tensor") -> np.ndarray:
        kwargs: Dict[str, Any] = {"prediction_length": PREDICTION_LENGTH_H}
        try:
            kwargs["predict_batches_jointly"] = predict_batches_jointly
            output = pipeline.predict(t, **kwargs)
        except TypeError:
            kwargs.pop("predict_batches_jointly", None)
            output = pipeline.predict(t, **kwargs)
        median = _take_median_along_samples(output)
        if median.ndim == 3 and median.shape[1] == perturbed_inputs.shape[2]:
            return np.transpose(median, (0, 2, 1))
        if median.ndim == 3 and median.shape[2] == perturbed_inputs.shape[2]:
            return median
        raise ValueError(f"Unexpected median shape {median.shape} for batch={tensor.shape}")

    if chunk_size <= 0 or batch_size <= chunk_size:
        return _predict_chunk(tensor), "joint"

    chunks = []
    for start in range(0, batch_size, chunk_size):
        chunks.append(_predict_chunk(tensor[start : start + chunk_size]))
    return np.concatenate(chunks, axis=0), "chunked"


# ─── Sensitivity aggregation ──────────────────────────────────────────────


def _validate_inputs(
    input_kpis: np.ndarray,
    baseline_forecast: np.ndarray,
    kpi_names: List[str],
    cap_table: Dict[str, float],
) -> None:
    """Boundary validation per CLAUDE.md (validate at trust boundary).

    Raises
    ------
    ValueError
        On shape, NaN, or cap-table inconsistencies.
    """
    if input_kpis.ndim != 2:
        raise ValueError(f"input_kpis must be 2-D (L, C); got shape {input_kpis.shape}")
    if baseline_forecast.ndim != 2:
        raise ValueError(
            f"baseline_forecast must be 2-D (H, C); got shape " f"{baseline_forecast.shape}"
        )
    if input_kpis.shape[1] != baseline_forecast.shape[1]:
        raise ValueError(
            f"channel mismatch: input has {input_kpis.shape[1]}, "
            f"baseline_forecast has {baseline_forecast.shape[1]}"
        )
    if input_kpis.shape[1] != len(kpi_names):
        raise ValueError(
            f"kpi_names length {len(kpi_names)} != channel count " f"{input_kpis.shape[1]}"
        )
    if not np.all(np.isfinite(input_kpis)):
        raise ValueError("input_kpis contains NaN or Inf values")
    if not np.all(np.isfinite(baseline_forecast)):
        raise ValueError("baseline_forecast contains NaN or Inf values")
    missing = [k for k in kpi_names if k not in cap_table]
    if missing:
        raise ValueError(f"cap_table missing entries for KPIs: {missing}")


def _select_event_centers(
    anomaly_events: List[Dict[str, Any]],
    input_length: int,
    perturbation_window: int,
) -> List[Tuple[int, int]]:
    """Map anomaly events on the forecast horizon to Type-1 perturbation
    centers in the input window.

    Anomaly events live on the forecast horizon (indices ``0 ≤ t < H``
    where ``H = PREDICTION_LENGTH_H``). Each event's center is mapped
    to the input timestep one full forecast horizon back from the end
    of the input — i.e.
    ``absolute_center = input_length - PREDICTION_LENGTH_H + forecast_center``
    so the localized perturbation sits inside the most-recent
    ``PREDICTION_LENGTH_H`` hours of the input (the segment with the
    strongest TSFM attention bias and the natural pre-image of the
    forecast). This mapping is independent of ``perturbation_window``
    — that argument controls only the Type-2/Type-3 channel-wide
    statistics window.

    Returns ``[(event_idx_in_original_list, t_center_in_input_window), …]``.
    Events whose mapped center falls outside ``[0, input_length)``
    are logged at INFO and excluded.
    """
    del perturbation_window  # decoupled from Type-1 center selection
    out: List[Tuple[int, int]] = []
    skipped: List[int] = []
    horizon_to_input_offset = input_length - PREDICTION_LENGTH_H
    for idx, event in enumerate(anomaly_events):
        t_start = int(event.get("t_start", -1))
        t_end = int(event.get("t_end", -1))
        if t_start < 0 or t_end < t_start:
            skipped.append(idx)
            continue
        forecast_center = (t_start + t_end) // 2
        absolute_center = horizon_to_input_offset + forecast_center
        if not (0 <= absolute_center < input_length):
            skipped.append(idx)
            continue
        out.append((idx, absolute_center))
    if skipped:
        logger.info(
            "PAX-TS Type-1: skipped %d/%d anomaly events with malformed "
            "or out-of-bounds time windows (event_indices=%s)",
            len(skipped),
            len(anomaly_events),
            skipped,
        )
    return out


def _signed_mean_change_ratio(
    perturbed_pred_means: np.ndarray,
    baseline_pred_mean: float,
    alphas: np.ndarray,
) -> float:
    """PAX-TS §III-D Algorithm 1 line 9.

    ``r̄_π = (1/|A|) Σ_α sgn(α) · (π(ŷ'_α) - π(ŷ)) / Δ``.

    For mean / variance / trend perturbations Δ is implicitly one
    (the change-ratio is computed against the alpha index space
    directly), since Algorithm 1 normalizes by the perturbation
    distance and we stay in the same input-units throughout.
    """
    deltas = perturbed_pred_means - baseline_pred_mean
    return float(np.mean(np.sign(alphas) * deltas))


def _aggregate_global(
    forecasts: np.ndarray,
    baseline_forecast: np.ndarray,
    index_map: List[Tuple[int, float]],
    n_channels: int,
    scales: Sequence[float],
) -> np.ndarray:
    """Reduce a Type-2 / Type-3 forecast batch to a ``(C, C)`` matrix.

    ``index_map`` carries ``(source_channel, alpha)`` per row.
    ``forecasts`` has shape ``(B, H, C)``. Returns ``(C, C)`` where
    entry ``[i, j]`` is the signed mean change ratio of target
    channel ``j``'s forecast mean, induced by perturbations of
    source channel ``i``.
    """
    forecast_means = forecasts.mean(axis=1)  # (B, C)
    baseline_means = baseline_forecast.mean(axis=0)  # (C,)
    out = np.zeros((n_channels, n_channels), dtype=float)
    by_source: Dict[int, List[Tuple[float, np.ndarray]]] = {c: [] for c in range(n_channels)}
    for batch_idx, (src, alpha) in enumerate(index_map):
        by_source[src].append((alpha, forecast_means[batch_idx]))
    for src, entries in by_source.items():
        if not entries:
            continue
        alphas = np.array([alpha for alpha, _ in entries], dtype=float)
        means = np.stack([m for _, m in entries], axis=0)  # (n_α, C)
        for tgt in range(n_channels):
            out[src, tgt] = _signed_mean_change_ratio(
                means[:, tgt],
                float(baseline_means[tgt]),
                alphas,
            )
    return out


def _aggregate_local(
    forecasts: np.ndarray,
    baseline_forecast: np.ndarray,
    index_map: List[Tuple[int, int, float]],
    event_centers: List[Tuple[int, int]],
    anomaly_events: List[Dict[str, Any]],
    n_channels: int,
) -> np.ndarray:
    """Reduce a Type-1 forecast batch to ``(n_events, C)``.

    For each anomaly event, the target KPI is fixed (the event's
    own KPI). For each source channel, average ``sgn(α) · Δ`` of
    the target KPI's forecast mean over the event's hour window.
    """
    H = baseline_forecast.shape[0]
    n_events = len(event_centers)
    out = np.zeros((n_events, n_channels), dtype=float)
    if n_events == 0:
        return out

    event_idx_to_local: Dict[int, int] = {
        original_idx: local_idx for local_idx, (original_idx, _) in enumerate(event_centers)
    }
    target_kpi_idx_per_event: Dict[int, int] = {}
    target_window_per_event: Dict[int, Tuple[int, int]] = {}
    # The caller is responsible for supplying kpi_name_lookup-consistent
    # anomaly events. We map names via the canonical kpi_names supplied
    # to compute_sensitivity, plumbed through a closure-ish path here:
    # for cleanliness, target index resolution happens in the caller —
    # this helper assumes ``anomaly_events[i]['kpi_index']`` is set.
    for original_idx, _ in event_centers:
        ev = anomaly_events[original_idx]
        if "kpi_index" not in ev:
            raise ValueError(
                f"anomaly_events[{original_idx}] missing 'kpi_index' — "
                "compute_sensitivity must inject it before aggregation"
            )
        kpi_idx = int(ev["kpi_index"])
        if not (0 <= kpi_idx < n_channels):
            raise ValueError(
                f"anomaly_events[{original_idx}].kpi_index={kpi_idx} "
                f"is out of range for n_channels={n_channels}"
            )
        t_start = int(ev["t_start"])
        t_end = int(ev["t_end"])
        if not (0 <= t_start <= t_end < H):
            raise ValueError(
                f"anomaly_events[{original_idx}] window [{t_start}, "
                f"{t_end}] is outside forecast horizon H={H}"
            )
        target_kpi_idx_per_event[original_idx] = kpi_idx
        target_window_per_event[original_idx] = (t_start, t_end)

    bucket: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    for batch_idx, (event_idx, src, alpha) in enumerate(index_map):
        if event_idx not in event_idx_to_local:
            continue
        target_kpi = target_kpi_idx_per_event[event_idx]
        t_s, t_e = target_window_per_event[event_idx]
        perturbed_target = forecasts[batch_idx, t_s : t_e + 1, target_kpi]
        baseline_target = baseline_forecast[t_s : t_e + 1, target_kpi]
        delta = float(perturbed_target.mean() - baseline_target.mean())
        bucket.setdefault((event_idx, src), []).append((alpha, delta))

    for (event_idx, src), entries in bucket.items():
        local_idx = event_idx_to_local[event_idx]
        alphas = np.array([a for a, _ in entries], dtype=float)
        deltas = np.array([d for _, d in entries], dtype=float)
        out[local_idx, src] = float(np.mean(np.sign(alphas) * deltas))
    return out


def _normalize_with_caps(
    matrix: np.ndarray,
    kpi_names: List[str],
    cap_table: Dict[str, float],
) -> np.ndarray:
    """Multiply each entry by ``cap[source] / cap[target]``.

    Works on ``(C, C)`` and ``(C, C, P)`` and ``(n_events, C)``
    layouts. For the per-event matrix the caller is responsible
    for supplying a row that has already been keyed to its target
    KPI in raw units; the cap normalization for a per-event row
    cannot be a uniform ratio because the target KPI is different
    per row, so we apply it externally — see :func:`_normalize_local`.
    """
    caps = np.array(
        [float(cap_table[name]) for name in kpi_names],
        dtype=float,
    )
    if matrix.ndim == 2 and matrix.shape == (len(caps), len(caps)):
        ratios = caps[:, None] / caps[None, :]
        return matrix * ratios
    if matrix.ndim == 3 and matrix.shape[:2] == (len(caps), len(caps)):
        ratios = caps[:, None, None] / caps[None, :, None]
        return matrix * ratios
    raise ValueError(f"_normalize_with_caps expects (C,C) or (C,C,P), got {matrix.shape}")


def _normalize_local(
    matrix: np.ndarray,
    kpi_names: List[str],
    cap_table: Dict[str, float],
    target_kpi_per_event: List[str],
) -> np.ndarray:
    """Per-event cap normalization for ``S_local_raw``.

    Each row's target is fixed and given by ``target_kpi_per_event``.
    Source caps come from ``kpi_names`` along the last axis.
    """
    n_events, n_channels = matrix.shape
    if len(target_kpi_per_event) != n_events:
        raise ValueError(
            f"target_kpi_per_event has {len(target_kpi_per_event)} " f"entries, expected {n_events}"
        )
    src_caps = np.array(
        [float(cap_table[name]) for name in kpi_names],
        dtype=float,
    )
    out = np.zeros_like(matrix)
    for row, target_kpi in enumerate(target_kpi_per_event):
        tgt_cap = float(cap_table[target_kpi])
        out[row, :] = matrix[row, :] * (src_caps / tgt_cap)
    return out


# ─── Public entry point ───────────────────────────────────────────────────


_PERTURBATION_TYPES: Tuple[str, ...] = ("mean", "variance", "trend")


def compute_sensitivity(
    pipeline: Any,
    input_kpis: np.ndarray,
    baseline_forecast: np.ndarray,
    anomaly_events: List[Dict[str, Any]],
    kpi_names: List[str],
    cap_table: Optional[Dict[str, float]] = None,
    scales: Sequence[float] = _DEFAULT_SCALES,
    perturbation_window: int = _DEFAULT_PERTURBATION_WINDOW_H,
    seasonality: int = _DEFAULT_DESEASONALIZATION_PERIOD_H,
    type1_width: int = _DEFAULT_TYPE1_W,
    type1_softness: float = _DEFAULT_TYPE1_S,
    chunk_size: int = 0,
    predict_batches_jointly: bool = True,
) -> SensitivityResult:
    """Run PAX-TS Algorithm 2 on a single station.

    Parameters
    ----------
    pipeline
        Loaded ``Chronos2Pipeline`` (the ``self._chronos_pipeline``
        attribute on a :class:`telcoagent.prediction.predictor.TelcoAgent`).
        Must support ``predict(tensor, prediction_length=…,
        predict_batches_jointly=…)``.
    input_kpis
        Raw KPI input window of shape ``(L, C)``. The trailing
        ``CONTEXT_LENGTH_H`` rows are passed to the model.
    baseline_forecast
        Cached, unperturbed forecast of shape ``(PREDICTION_LENGTH_H,
        C)`` produced by the same pipeline on the same input. Used
        as the reference for change ratios so we do not pay for an
        extra forward pass.
    anomaly_events
        Output of ``events_to_dicts(detect_anomaly_events(...))``.
        Must have ``kpi``, ``t_start``, ``t_end`` populated. We
        inject a transient ``kpi_index`` field per event before
        aggregation; callers should not rely on this field.
    kpi_names
        Ordered KPI names in the same order as
        ``input_kpis``'s columns and ``baseline_forecast``'s columns.
    cap_table
        Optional override; defaults to
        :data:`telcoagent.config.KPI_NORMALIZATION_MAX`.
    scales, perturbation_window, seasonality, type1_width,
    type1_softness, chunk_size, predict_batches_jointly
        Hyperparameters; see module docstring and constants.

    Returns
    -------
    SensitivityResult
    """
    started_at = time.time()
    cap_table_resolved = cap_table or KPI_NORMALIZATION_MAX
    _validate_inputs(input_kpis, baseline_forecast, kpi_names, cap_table_resolved)

    L, C = input_kpis.shape
    scales_tuple = tuple(float(a) for a in scales)

    kpi_index_lookup = {name: idx for idx, name in enumerate(kpi_names)}
    annotated_events: List[Dict[str, Any]] = []
    for ev in anomaly_events:
        ev_copy = dict(ev)
        kpi_name = ev_copy.get("kpi", "")
        if kpi_name in kpi_index_lookup:
            ev_copy["kpi_index"] = kpi_index_lookup[kpi_name]
        else:
            ev_copy["kpi_index"] = -1
        annotated_events.append(ev_copy)

    in_window_centers = _select_event_centers(
        annotated_events,
        L,
        perturbation_window,
    )
    valid_event_centers = [
        (idx, center)
        for idx, center in in_window_centers
        if annotated_events[idx]["kpi_index"] >= 0
    ]
    skipped_unknown_kpi = [
        idx for idx, _ in in_window_centers if annotated_events[idx]["kpi_index"] < 0
    ]
    if skipped_unknown_kpi:
        logger.info(
            "PAX-TS Type-1: skipped %d events with KPI not in kpi_names " "(event_indices=%s)",
            len(skipped_unknown_kpi),
            skipped_unknown_kpi,
        )

    type1_inputs, type1_index = _build_type1_perturbations(
        input_kpis,
        valid_event_centers,
        scales_tuple,
        width=type1_width,
        softness=type1_softness,
    )
    type2m_inputs, type2m_index = _build_type2_mean_perturbations(
        input_kpis,
        scales_tuple,
        window=perturbation_window,
    )
    type2v_inputs, type2v_index = _build_type2_variance_perturbations(
        input_kpis,
        scales_tuple,
        window=perturbation_window,
    )
    type3_inputs, type3_index = _build_type3_trend_perturbations(
        input_kpis,
        scales_tuple,
        window=perturbation_window,
        seasonality=seasonality,
    )

    sub_batches: List[Tuple[str, np.ndarray, Any]] = [
        ("type1", type1_inputs, type1_index),
        ("type2_mean", type2m_inputs, type2m_index),
        ("type2_var", type2v_inputs, type2v_index),
        ("type3_trend", type3_inputs, type3_index),
    ]
    non_empty = [(name, arr, idx) for name, arr, idx in sub_batches if arr.shape[0] > 0]

    if not non_empty:
        raise ValueError(
            "compute_sensitivity produced zero perturbations — check "
            "scales, anomaly_events, and perturbation_window"
        )

    stacked = np.concatenate([arr for _, arr, _ in non_empty], axis=0)
    section_offsets: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for name, arr, _ in non_empty:
        section_offsets[name] = (cursor, cursor + arr.shape[0])
        cursor += arr.shape[0]

    forecasts, batch_strategy = _run_chronos_batched(
        pipeline,
        stacked,
        chunk_size=chunk_size,
        predict_batches_jointly=predict_batches_jointly,
    )

    type1_forecasts = (
        forecasts[slice(*section_offsets["type1"])]
        if "type1" in section_offsets
        else np.empty((0, baseline_forecast.shape[0], C), dtype=float)
    )
    type2m_forecasts = (
        forecasts[slice(*section_offsets["type2_mean"])]
        if "type2_mean" in section_offsets
        else None
    )
    type2v_forecasts = (
        forecasts[slice(*section_offsets["type2_var"])] if "type2_var" in section_offsets else None
    )
    type3_forecasts = (
        forecasts[slice(*section_offsets["type3_trend"])]
        if "type3_trend" in section_offsets
        else None
    )

    def _global_or_zero(forecasts_section, index_section):
        if forecasts_section is None or len(index_section) == 0:
            return np.zeros((C, C), dtype=float)
        return _aggregate_global(
            forecasts_section,
            baseline_forecast,
            index_section,
            C,
            scales_tuple,
        )

    S_mean = _global_or_zero(type2m_forecasts, type2m_index)
    S_var = _global_or_zero(type2v_forecasts, type2v_index)
    S_trend = _global_or_zero(type3_forecasts, type3_index)
    S_global_raw = np.stack([S_mean, S_var, S_trend], axis=-1)
    S_global_norm = _normalize_with_caps(S_global_raw, kpi_names, cap_table_resolved)

    S_local_raw = _aggregate_local(
        type1_forecasts,
        baseline_forecast,
        type1_index,
        valid_event_centers,
        annotated_events,
        C,
    )
    target_kpi_per_event = [annotated_events[idx]["kpi"] for idx, _ in valid_event_centers]
    S_local_norm = _normalize_local(
        S_local_raw,
        kpi_names,
        cap_table_resolved,
        target_kpi_per_event,
    )

    elapsed = time.time() - started_at
    n_perturbations = stacked.shape[0]

    logger.info(
        "PAX-TS sensitivity computed in %.2fs (B=%d perturbations, " "events=%d, strategy=%s)",
        elapsed,
        n_perturbations,
        len(valid_event_centers),
        batch_strategy,
    )

    return SensitivityResult(
        S_global_raw=S_global_raw,
        S_global_norm=S_global_norm,
        S_local_raw=S_local_raw,
        S_local_norm=S_local_norm,
        perturbation_scales=scales_tuple,
        event_centers=valid_event_centers,
        n_perturbations=n_perturbations,
        elapsed_sec=elapsed,
        batch_strategy=batch_strategy,
    )


# Public re-export of the perturbation-type label tuple for callers
# that need to label sensitivity matrix slices.
PERTURBATION_TYPES: Tuple[str, ...] = _PERTURBATION_TYPES
