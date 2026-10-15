"""Paired bootstrap statistics for component-ablation comparisons.

Public API
----------
paired_comparison(arr_a, arr_b, n_boot=10000, seed=42) -> dict
    Compute mean difference, 95% bootstrap CI, p-value, and Cohen's d
    for station-level paired arrays.

Notes
-----
- Bootstrap unit: individual stations (row-wise paired resample).
- p-value: Wilcoxon signed-rank test (scipy.stats.wilcoxon).
- Cohen's d: paired variant — mean(diff) / std(diff, ddof=1).
- No LLM calls; numpy + scipy only.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _scipy_stats


def paired_comparison(
    arr_a,
    arr_b,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    """Return paired statistics for two station-level score arrays.

    Parameters
    ----------
    arr_a, arr_b:
        1-D array-like of equal length.  Each position corresponds to the
        same station — comparisons are always paired.
    n_boot:
        Number of bootstrap resamples for the 95% CI.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        mean_diff   float  mean(arr_a - arr_b)
        ci_low      float  2.5th percentile of bootstrap distribution
        ci_high     float  97.5th percentile of bootstrap distribution
        p_value     float  two-sided p-value (Wilcoxon signed-rank)
        cohens_d    float  paired Cohen's d = mean(diff) / std(diff, ddof=1)
        n           int    number of paired observations
    """
    a = np.asarray(arr_a, dtype=float)
    b = np.asarray(arr_b, dtype=float)

    if a.size == 0 or b.size == 0:
        raise ValueError("arr_a and arr_b must be non-empty.")
    if a.shape != b.shape:
        raise ValueError(f"arr_a and arr_b must have the same shape; got {a.shape} vs {b.shape}.")

    diff = a - b
    n = len(diff)
    mean_diff = float(np.mean(diff))

    # --- paired bootstrap 95% CI -------------------------------------------
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[indices].mean(axis=1)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    # --- p-value: Wilcoxon signed-rank (handles n >= 2) ---------------------
    if n < 2:
        # single pair: convention is p=1.0 (can't test)
        p_value = 1.0
    else:
        # zero_method="wilcox" drops zero differences (standard behaviour)
        stat_result = _scipy_stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        p_value = float(stat_result.pvalue)

    # --- paired Cohen's d ---------------------------------------------------
    if n < 2:
        # std(ddof=1) is undefined for a single observation; convention: NaN.
        cohens_d = float("nan")
    else:
        std_diff = float(np.std(diff, ddof=1))
        if std_diff == 0.0:
            # All differences are identical: zero effect if mean_diff==0, otherwise
            # perfectly consistent effect — treat as infinite (sign preserved).
            cohens_d = 0.0 if mean_diff == 0.0 else float(np.sign(mean_diff)) * np.inf
        else:
            cohens_d = mean_diff / std_diff

    return {
        "mean_diff": mean_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "n": n,
    }
