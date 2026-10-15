"""Pure-NumPy input-window utilities for the prediction pipeline.

Deliberately torch/chronos-free so lightweight consumers (CI slices,
baseline scripts, the explainer's KPI-order lookups) can import the
canonical KPI order and input statistics without dragging the TSFM
dependency closure into their import graph.

Contents:
    KPI_NAMES                  -- canonical 7-KPI order (3GPP TS 28.554).
    kpis_to_text               -- render an input window as an LLM-readable
                                  table (used by prompt-building harnesses).
    compute_input_statistics   -- per-KPI mean/std/trend/diurnal summary of
                                  the input window (validation aid; reads
                                  input only, so no leakage).
"""

from typing import Dict, List

import numpy as np

from telcoagent.config import CORE_KPI_NAMES

# Canonical 7-KPI order (3GPP TS 28.554). Public re-export of the
# tuple-shaped constant from telcoagent.config; kept as a list for
# backward compatibility with call sites that slice or .index() it.
KPI_NAMES: List[str] = list(CORE_KPI_NAMES)


def kpis_to_text(
    kpi_data: np.ndarray,
    kpi_names: List[str] = KPI_NAMES,
) -> str:
    """Convert KPI time series to natural language description with ACTUAL DATA VALUES."""
    L, C = kpi_data.shape
    days = L // 24

    text_parts = [
        f"=== Network Measurements ({days} days, {L} hours) ===",
        "NOTE: All KPI values are STANDARDIZED (zero mean, unit variance).",
        "Your prediction output must also be in this standardized scale.\n",
    ]

    # Daily averages
    text_parts.append("## DAILY AVERAGES (Use these for prediction baseline)")
    text_parts.append(f"{'Day':<6} " + " ".join([f"{name:<12}" for name in kpi_names[:C]]))
    text_parts.append("-" * 100)
    for day in range(days):
        day_data = kpi_data[day * 24 : (day + 1) * 24]
        day_means = day_data.mean(axis=0)
        values_str = " ".join([f"{val:<12.2f}" for val in day_means[:C]])
        text_parts.append(f"Day {day+1:<2} {values_str}")
    text_parts.append("")

    # Full hourly data (168h × 7 KPI)
    text_parts.append("## FULL HOURLY DATA (168 hours × 7 KPIs)")
    text_parts.append("Format: Day D H0 H1 ... H23 (per KPI)")
    for i, name in enumerate(kpi_names[:C]):
        text_parts.append(f"### {name}")
        for day in range(days):
            day_vals = kpi_data[day * 24 : (day + 1) * 24, i]
            vals_str = " ".join(f"{v:.2f}" for v in day_vals)
            text_parts.append(f"D{day+1}: {vals_str}")
    text_parts.append("")

    # Summary statistics
    text_parts.append("## KPI STATISTICS SUMMARY")
    for i, name in enumerate(kpi_names[:C]):
        series = kpi_data[:, i]
        mean_val = float(np.mean(series))
        std_val = float(np.std(series))
        min_val = float(np.min(series))
        max_val = float(np.max(series))
        amplitude = max_val - min_val
        first_half = series[: L // 2].mean()
        second_half = series[L // 2 :].mean()
        trend_pct = ((second_half - first_half) / (first_half + 1e-8)) * 100
        trend_str = "UP" if trend_pct > 5 else "DOWN" if trend_pct < -5 else "STABLE"
        text_parts.append(
            f"- {name}: Mean={mean_val:.2f}, Std={std_val:.2f}, "
            f"Range=[{min_val:.2f}, {max_val:.2f}], Amplitude={amplitude:.2f}, "
            f"Trend={trend_str}({trend_pct:+.1f}%)"
        )

    # Diurnal fingerprint — average 24h pattern per KPI (shape template for prediction)
    text_parts.append(
        "\n## INPUT DIURNAL FINGERPRINT (average 24h shape — use as prediction template)"
    )
    text_parts.append("Your predicted diurnal shape should closely follow this pattern.")
    text_parts.append("Format: H00 H01 ... H23 (standardized values)")
    if L >= 24:
        n_full_days = L // 24
        for i, name in enumerate(kpi_names[:C]):
            hourly_matrix = kpi_data[: n_full_days * 24, i].reshape(n_full_days, 24)
            avg_diurnal = hourly_matrix.mean(axis=0)
            peak_h = int(np.argmax(avg_diurnal))
            trough_h = int(np.argmin(avg_diurnal))
            vals_str = " ".join(f"{v:+.2f}" for v in avg_diurnal)
            text_parts.append(
                f"- {name}: [{vals_str}]  (peak=H{peak_h:02d}, trough=H{trough_h:02d})"
            )

    return "\n".join(text_parts)


def compute_input_statistics(input_kpis: np.ndarray) -> Dict[str, Dict]:
    """Compute statistics from input data for validation (NO DATA LEAKAGE)."""
    stats = {}
    L, C = input_kpis.shape
    for i, name in enumerate(KPI_NAMES[:C]):
        series = input_kpis[:, i]
        mean = float(np.mean(series))
        std = float(np.std(series))
        trend = (
            float(series[-168:].mean() - series[:168].mean())
            if L >= 168
            else float(series[L // 2 :].mean() - series[: L // 2].mean())
        )
        daily = series.reshape(-1, 24).mean(axis=0) if L >= 24 else series
        stats[name] = {
            "mean": mean,
            "std": std,
            "min": float(np.min(series)),
            "max": float(np.max(series)),
            "trend": trend,
            "daily_pattern": daily.tolist(),
            "pred_min": max(0, mean - 3 * std + min(0, trend * 2)),
            "pred_max": mean + 3 * std + max(0, trend * 2),
            "pred_mean_expected": mean + trend,
        }
    return stats
