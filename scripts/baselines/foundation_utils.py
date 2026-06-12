"""Common utilities for foundation model evaluation scripts.

Shared functions for data loading, metrics computation, and result saving.
"""

import importlib.metadata
import json
import logging
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Determinism / seeding
# =============================================================================

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed all RNGs for paper-grade reproducibility.

    Sets Python's ``random``, NumPy, PyTorch CPU/CUDA, and the relevant env
    vars (``PYTHONHASHSEED``, ``CUBLAS_WORKSPACE_CONFIG``). Also flips
    ``torch.use_deterministic_algorithms(True, warn_only=True)`` —
    ``warn_only`` because some uni2ts / chronos kernels still trigger a
    non-deterministic CUDA path; we log a warning rather than aborting.

    Returns the seed for logging convenience.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required for deterministic cuBLAS GEMMs on CUDA >= 10.2.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except AttributeError:
                pass
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError) as exc:  # very old torch / unsupported op
            logger.warning("set_seed: could not enable deterministic algorithms: %s", exc)
    except ImportError:  # torch missing — pure-numpy callers (analysis scripts)
        pass
    return seed


# Family label -> distribution name on PyPI / conda. Used by
# ``_resolve_lib_version`` so each per-station JSON records the actual
# installed version of the inference library, not a docstring claim.
_FAMILY_TO_DIST = {
    "chronos2": "chronos-forecasting",
    "moment": "momentfm",
    "moirai": "uni2ts",
    "moirai_moe": "uni2ts",
    "toto": "toto-ts",
    "mamba4cast": "mamba-ssm",
}


def _resolve_lib_version(family: str) -> str:
    """Return the installed version of the model-family's inference lib.

    Falls back to ``"unknown"`` if the package is missing (e.g. an env that
    only has analysis tools). For the Mamba4Cast family we additionally
    suffix the vendored repo's git short SHA when available — Mamba4Cast
    is a vendored sub-tree, not a pip package.
    """
    dist = _FAMILY_TO_DIST.get(family)
    base = "unknown"
    if dist:
        try:
            base = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            base = "unknown"

    if family == "mamba4cast":
        repo_root = Path(__file__).resolve().parent.parent.parent / "Mamba4Cast"
        if (repo_root / ".git").exists():
            try:
                sha = (
                    subprocess.check_output(
                        ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
                        stderr=subprocess.DEVNULL,
                        timeout=2,
                    )
                    .decode()
                    .strip()
                )
                base = f"{base}+repo:{sha}"
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
    return base


# KPI column names in CSV files
KPI_COLS_CSV = [
    "RRC_Conn_Count_Avg",
    "DL_CQI",
    "DL_iBler",
    "DL_rBler",
    "MAC DL Eff TP",
    "DL PRB Utilization",
    "DL_IP_Throughput",
]

# Standardized KPI names — single source of truth in telcoagent.config.
# Use a fallback so this module also imports cleanly in lightweight envs that
# don't have the full telcoagent package dependency chain installed (used for
# isolated baseline-model envs, e.g. a chronos-2 sweep env that only needs
# numpy/pandas/torch/chronos).
try:
    from telcoagent.config import CORE_KPI_NAMES  # type: ignore
except ImportError:
    CORE_KPI_NAMES = (
        "RRC_Conn",
        "DL_CQI",
        "DL_iBler",
        "DL_rBler",
        "MAC_DL_Eff",
        "PRB_Util",
        "Throughput",
    )
KPI_NAMES = list(CORE_KPI_NAMES)
N_CHANNELS = len(CORE_KPI_NAMES)

try:
    # Re-exported for baseline runners; importing it here also asserts the
    # telcoagent.config dependency is present (fail loud below if absent).
    from telcoagent.config import PREDICTION_LENGTH_H as PREDICTION_LENGTH_H  # noqa: F401
except ImportError as exc:  # fail loud — no silent divergent default (principle #1)
    raise ImportError(
        "foundation_utils requires telcoagent.config for PREDICTION_LENGTH_H; "
        "isolated baseline envs must import it from a standalone constant or "
        "pass it as a parameter, not rely on a silent default."
    ) from exc

# Cross-runner constants used by the shared sweep harness below.
TOTAL_DAYS = 88  # full station series length
MIN_CONTEXT_DAYS = 1
DEFAULT_HORIZON_DAYS = 7


def load_station_data(csv_path: Path) -> np.ndarray:
    """Load a station CSV and return KPI array (hours, 7)."""
    df = pd.read_csv(csv_path)
    return df[KPI_COLS_CSV].values.astype(np.float64)


def split_data(
    data: np.ndarray,
    target_hours: int = 168,
) -> Dict[str, np.ndarray]:
    """Split data into input window and test target.

    Args:
        data: Full time series (hours, 7)
        target_hours: Prediction horizon (default 168 = 7 days)

    Returns:
        Dict with 'input_window', 'test_target'
        input_window: all data except last target_hours (e.g., 81 days = 1944h)
        test_target:  last target_hours (e.g., 7 days = 168h)
    """
    return {
        "input_window": data[:-target_hours],
        "test_target": data[-target_hours:],
    }


def apply_revin(
    data: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Reversible Instance Normalization.

    Args:
        data: Input array (hours, channels)
        mean: Pre-computed mean (if None, computed from data)
        std: Pre-computed std (if None, computed from data)

    Returns:
        (normalized_data, mean, std)
    """
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
        std[std < 1e-5] = 1.0

    normalized = (data - mean) / std
    return normalized, mean, std


def revert_revin(
    data: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Revert Reversible Instance Normalization."""
    return data * std + mean


# Re-export the shared strict clamp (ADR 0001 / B-MED-3).
# All baseline scripts must call apply_physical_clamp from this module;
# internally it is the same object as telcoagent.prediction.physical_clamp.apply_physical_clamp_strict.
from telcoagent.prediction.physical_clamp import (  # noqa: E402
    apply_physical_clamp_strict as apply_physical_clamp,
)


def compute_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    input_window: Optional[np.ndarray] = None,
) -> Dict:
    """Compute per-KPI and overall evaluation metrics.

    Metrics:
        sMAPE  - symmetric MAPE (0-200% scale, lower is better)
        MASE   - Mean Absolute Scaled Error vs. daily seasonal naive (s=24).
                 Denominator is computed from input_window if provided,
                 otherwise from the truth window itself.
    """
    SEASONAL_PERIOD = 24  # daily seasonality for hourly telecom data

    n_hours = min(pred.shape[0], truth.shape[0])
    pred, truth = pred[:n_hours], truth[:n_hours]

    per_kpi = {}
    for i, kpi in enumerate(KPI_NAMES):
        if i >= pred.shape[1]:
            break
        p, t = pred[:, i], truth[:, i]

        # sMAPE: symmetric, bounded [0, 200%]
        denom = np.abs(p) + np.abs(t)
        smask = denom > 1e-8
        smape = (
            float(np.mean(2.0 * np.abs(p[smask] - t[smask]) / denom[smask])) * 100
            if smask.sum() > 0
            else 0.0
        )

        # MASE: scale by seasonal naive MAE (s=24)
        # Use input_window for denominator if available, else fall back to truth
        baseline = input_window[:, i] if input_window is not None else t
        if len(baseline) > SEASONAL_PERIOD:
            scale = float(np.mean(np.abs(baseline[SEASONAL_PERIOD:] - baseline[:-SEASONAL_PERIOD])))
        else:
            scale = float(np.mean(np.abs(np.diff(baseline)))) if len(baseline) > 1 else 1.0
        mae = float(np.mean(np.abs(p - t)))
        mase = (mae / scale) if scale > 1e-8 else 0.0

        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        # nRMSE: range-normalised so it's scale-free across the 7 KPIs and
        # safe to average. Range is computed on the test target (paper-
        # standard for forecast evaluation; truth window is fixed at 168h
        # across the whole sweep, so this denominator is constant per KPI
        # / station).
        rng = float(t.max() - t.min())
        nrmse = (rmse / rng) if rng > 1e-8 else 0.0

        per_kpi[kpi] = {
            "sMAPE": round(smape, 2),
            "MASE": round(mase, 4),
            # Scale-free RMSE companion (paper primary alongside MASE).
            "nRMSE": round(nrmse, 4),
            # Raw absolute metrics — kept per-KPI for debugging /
            # supplementary tables. Not aggregated across KPIs because the
            # 7 KPIs have different units (kbps, %, count, index).
            "MAE": round(float(np.mean(np.abs(p - t))), 4),
            "RMSE": round(rmse, 4),
        }

    # Overall MAE/RMSE retain their previous semantics (cross-KPI mean of
    # raw errors — useful only as a debug aggregate). Overall nRMSE is the
    # unweighted mean of per-KPI nRMSE — the natural multi-KPI summary.
    all_mae = float(np.mean(np.abs(pred - truth)))
    all_rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    nrmse_values = [v["nRMSE"] for v in per_kpi.values()]
    overall_nrmse = float(np.mean(nrmse_values)) if nrmse_values else 0.0

    return {
        "per_kpi": per_kpi,
        "overall_MAE": round(all_mae, 4),
        "overall_RMSE": round(all_rmse, 4),
        "overall_nRMSE": round(overall_nrmse, 4),
    }


def compute_naive_seasonal_metrics(
    input_window: np.ndarray,
    test_target: np.ndarray,
    season: int = 24,
) -> Dict:
    """Reference metrics for the seasonal-naive(s=24) forecast.

    The naive seasonal forecast tiles the last ``season`` hours of the
    input window across the prediction horizon. By construction this
    predictor's per-KPI MASE equals 1.0 (since MASE's denominator is the
    same seasonal-24h MAE on the input window — Hyndman 2006).

    Useful as a paper-table reference row: any model with MASE < 1 beats
    the naive seasonal baseline on that KPI.
    """
    H = test_target.shape[0]
    last_season = input_window[-season:]
    n_repeats = (H + season - 1) // season  # ceil division
    naive_pred = np.tile(last_season, (n_repeats, 1))[:H]
    return compute_metrics(naive_pred, test_target, input_window=input_window)


def save_prediction_csv(
    pred: np.ndarray,
    truth: np.ndarray,
    station_id: str,
    csv_path: Path,
):
    """Save hourly prediction vs ground truth as CSV."""
    n_hours = min(pred.shape[0], truth.shape[0])
    rows = []
    for h in range(n_hours):
        day = h // 24 + 1
        hour = h % 24
        row = {"station_id": station_id, "day": day, "hour": hour}
        for i, kpi in enumerate(KPI_NAMES):
            if i < pred.shape[1]:
                row[f"{kpi}_pred"] = round(float(pred[h, i]), 4)
                row[f"{kpi}_true"] = round(float(truth[h, i]), 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def save_station_result(
    station_id: str,
    model_name: str,
    prediction: np.ndarray,
    truth: np.ndarray,
    metrics: Dict,
    elapsed_sec: float,
    output_dir: Path,
    extra_info: Optional[Dict] = None,
) -> Dict:
    """Save per-station prediction results (JSON + CSV).

    Returns summary dict for batch report.
    """
    # Save CSV
    csv_path = output_dir / "csv" / f"{station_id}.csv"
    save_prediction_csv(prediction, truth, station_id, csv_path)

    # Build JSON output
    output = {
        "station_id": station_id,
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed_sec, 2),
        "prediction_shape": list(prediction.shape),
        "ground_truth_shape": list(truth.shape),
        "metrics": metrics,
        "prediction_daily": {},
        "ground_truth_daily": {},
    }

    # Add daily averages
    n_days = min(7, prediction.shape[0] // 24)
    for i, kpi in enumerate(KPI_NAMES):
        if i < prediction.shape[1]:
            output["prediction_daily"][kpi] = [
                round(float(np.mean(prediction[d * 24 : (d + 1) * 24, i])), 4)
                for d in range(n_days)
            ]
            output["ground_truth_daily"][kpi] = [
                round(float(np.mean(truth[d * 24 : (d + 1) * 24, i])), 4) for d in range(n_days)
            ]

    if extra_info:
        output["extra_info"] = extra_info

    # Save JSON
    json_path = output_dir / f"{station_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    return {
        "station_id": station_id,
        "status": "success",
        "elapsed_sec": round(elapsed_sec, 2),
        "metrics": metrics,
        "csv_file": str(csv_path),
        "json_file": str(json_path),
    }


def save_batch_summary(
    results: List[Dict],
    model_name: str,
    output_dir: Path,
    total_elapsed: float,
    extra_config: Optional[Dict] = None,
):
    """Save batch summary JSON with aggregate metrics."""
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "error"]

    summary = {
        "run_info": {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "total_stations": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "total_elapsed_sec": round(total_elapsed, 1),
        },
        "aggregate": {},
        "stations": results,
    }

    if extra_config:
        summary["run_info"].update(extra_config)

    if successful:
        times = [r["elapsed_sec"] for r in successful]

        # Per-KPI aggregate (paper primary: sMAPE + MASE + nRMSE).
        per_kpi_avg = {}
        for kpi in KPI_NAMES:
            entries = [
                r["metrics"]["per_kpi"][kpi] for r in successful if kpi in r["metrics"]["per_kpi"]
            ]
            if not entries:
                continue
            per_kpi_avg[kpi] = {
                "avg_sMAPE": round(sum(e["sMAPE"] for e in entries) / len(entries), 2),
                "avg_MASE": round(sum(e["MASE"] for e in entries) / len(entries), 4),
                "avg_nRMSE": round(sum(e.get("nRMSE", 0.0) for e in entries) / len(entries), 4),
            }

        summary["aggregate"] = {
            "per_kpi": per_kpi_avg,
            "avg_time_per_station_sec": round(sum(times) / len(times), 1),
        }

    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    return summary


def print_batch_report(summary: Dict, output_dir: Path):
    """Print final batch report to console."""
    info = summary["run_info"]
    agg = summary.get("aggregate", {})

    print(f"\n{'=' * 60}")
    print(f"BATCH COMPLETE: {info['successful']}/{info['total_stations']} stations")
    print(f"{'=' * 60}")
    print(f"  Model:          {info['model']}")
    print(
        f"  Total time:     {info['total_elapsed_sec']:.0f}s ({info['total_elapsed_sec'] / 60:.1f}min)"
    )

    if agg:
        print(f"  Avg time/stn:   {agg['avg_time_per_station_sec']:.1f}s")
        print(f"\n  Per-KPI Metrics (averaged over {info['successful']} stations):")
        print(f"    {'KPI':20s}  {'sMAPE(%)':>10s}  {'MASE':>10s}  {'nRMSE':>10s}")
        print(f"    {'-'*56}")
        for kpi, vals in agg.get("per_kpi", {}).items():
            print(
                f"    {kpi:20s}  {vals['avg_sMAPE']:>10.2f}  "
                f"{vals['avg_MASE']:>10.4f}  {vals.get('avg_nRMSE', 0):>10.4f}"
            )

    failed = [s for s in summary["stations"] if s.get("status") == "error"]
    if failed:
        print("\n  Failed stations:")
        for r in failed:
            print(f"    - {r['station_id']}: {r.get('error', 'Unknown error')}")

    print(f"\n  Results: {output_dir}/")
    print(f"  Summary: {output_dir}/summary.json")


# =============================================================================
# Shared TSFM sweep harness
# =============================================================================
#
# All baseline runners share the same sweep loop, argparse, and output schema.
# The classes below (BaseTSFMPredictor + run_sweep + make_sweep_argparser +
# assert_nonflat_prediction) factor that out so each runner reduces to:
#
#     1. a Predictor subclass that implements load() and predict()
#     2. a main() that builds the argparser, instantiates the subclass,
#        and calls run_sweep().
#
# Design rules:
#   - Subclasses must call .eval() and .to(device) inside their load().
#   - Subclasses must NOT wrap predict() in torch.no_grad themselves —
#     BaseTSFMPredictor.__call__ wraps it in torch.inference_mode() once.
#   - The synthetic sanity check (`assert_nonflat_prediction`) runs once
#     before the first context sweep starts, catching broken libraries
#     (e.g. chronos-forecasting v2.2.2 returning input mean) immediately
#     instead of after a 9000-station sweep.


class BaseTSFMPredictor:
    """Abstract base for time-series foundation model sweep predictors.

    Subclasses implement ``load(device)`` and ``predict(input_window,
    context_h, prediction_length)``. ``__call__`` wraps every prediction
    in ``torch.inference_mode()`` and validates the output shape +
    finiteness, so subclasses cannot forget eval-mode safety.

    Attributes set by subclasses (typically in ``__init__``):
        model_id   : HuggingFace / local id used for logging.
        family     : short label, e.g. "chronos2", "moirai", "moment".
        n_channels : number of KPIs (default 7 from CORE_KPI_NAMES).
        extra_info : dict merged into save_station_result(extra_info=...).
    """

    model_id: str = "<unset>"
    family: str = "base"
    n_channels: int = N_CHANNELS

    def __init__(self):
        self.extra_info: Dict = {}

    # ---- Subclass interface ------------------------------------------------
    def load(self, device: str) -> None:
        """Load weights + move to device + put model in eval mode.

        Subclasses MUST call .eval() on the underlying nn.Module(s) here.
        """
        raise NotImplementedError

    def predict(
        self,
        input_window: np.ndarray,
        context_h: int,
        prediction_length: int,
    ) -> np.ndarray:
        """Subclass forecast.

        Args:
            input_window: full available input window (L, n_channels).
            context_h:   trailing window length to feed the model (hours).
            prediction_length: forecast horizon (hours).

        Returns:
            prediction array of shape (prediction_length, n_channels).
        """
        raise NotImplementedError

    # ---- Public entry point ------------------------------------------------
    def __call__(
        self,
        input_window: np.ndarray,
        context_h: int,
        prediction_length: int,
    ) -> np.ndarray:
        import torch  # deferred to keep slim envs from importing torch when only utilities are used

        with torch.inference_mode():
            pred = self.predict(input_window, context_h, prediction_length)

        if not isinstance(pred, np.ndarray):
            raise TypeError(f"{self.model_id}: predict() must return np.ndarray, got {type(pred)}")
        if pred.shape != (prediction_length, self.n_channels):
            raise ValueError(
                f"{self.model_id}: predict() returned shape {pred.shape}, "
                f"expected ({prediction_length}, {self.n_channels})"
            )
        if not np.isfinite(pred).all():
            n_bad = int((~np.isfinite(pred)).sum())
            raise RuntimeError(
                f"{self.model_id}: prediction contains {n_bad} non-finite "
                f"value(s) (NaN/inf); refusing to score."
            )
        return pred


def assert_nonflat_prediction(
    predictor: BaseTSFMPredictor,
    *,
    prediction_length: int = 168,
    context_h: int = 24 * 28,  # 28d default
    threshold: float = 1e-3,
) -> None:
    """Run a one-shot synthetic-input sanity check on a loaded predictor.

    Builds a clean multivariate seasonal signal (per-channel sin wave with
    different amplitudes / phases), runs the predictor once, and asserts
    that the prediction is not constant across time. Catches situations
    where the underlying library or checkpoint is broken and silently
    returns the input mean for every step (we hit this with
    chronos-forecasting v2.2.2 + amazon/chronos-2 in March 2026).

    Raises ``RuntimeError`` with a descriptive message if the per-channel
    std of the prediction across time is below ``threshold`` for *any*
    channel. ``threshold=1e-3`` is conservative — even mean-only predictors
    show std around 1e-2 from numerical noise, so 1e-3 reliably catches
    truly degenerate output without false positives on stable signals.
    """
    n_input_h = context_h + prediction_length
    t = np.arange(n_input_h)
    channels = []
    for k in range(predictor.n_channels):
        amp = 50.0 + 20.0 * k
        phase = 0.3 * k
        channels.append(100.0 + amp * np.sin(2.0 * np.pi * t / 24.0 + phase))
    synthetic = np.stack(channels, axis=1).astype(np.float64)  # (T, C)

    # Mimic split_data: the "input window" is everything except the last
    # prediction_length hours (those would be the held-out target in a real run).
    input_window = synthetic[:-prediction_length]

    pred = predictor(input_window, context_h, prediction_length)
    per_channel_std = pred.std(axis=0)
    min_std = float(per_channel_std.min())
    if min_std < threshold:
        bad_idx = int(np.argmin(per_channel_std))
        bad_kpi = KPI_NAMES[bad_idx] if bad_idx < len(KPI_NAMES) else f"ch{bad_idx}"
        raise RuntimeError(
            f"{predictor.model_id}: sanity check failed — prediction is "
            f"essentially constant on channel {bad_idx} ({bad_kpi}), "
            f"std={min_std:.6g} < {threshold:.6g}. The model is likely "
            f"broken (forgot .eval(), wrong library version, or "
            f"checkpoint mismatch). Refusing to start the sweep."
        )
    logger.info(
        "%s: sanity check OK (per-channel std min=%.4f, max=%.4f)",
        predictor.model_id,
        min_std,
        float(per_channel_std.max()),
    )


def safe_output_name(model_id: str) -> str:
    """Sanitise a HuggingFace-style model id for use as a directory name."""
    return model_id.replace("Salesforce/", "").replace("/", "_")


def make_sweep_argparser(
    description: str,
    *,
    default_horizon: int = DEFAULT_HORIZON_DAYS,
    total_days: int = TOTAL_DAYS,
    default_model_id: Optional[str] = None,
    require_model_id: bool = False,
):
    """Build the argparser shared by every sweep runner.

    Each runner can ``add_argument`` extras on the returned parser before
    calling parser.parse_args().
    """
    import argparse

    parser = argparse.ArgumentParser(description=description)
    if require_model_id:
        parser.add_argument(
            "--model",
            required=True,
            help="HuggingFace model id (required).",
        )
    elif default_model_id is not None:
        parser.add_argument(
            "--model",
            default=default_model_id,
            help=f"HuggingFace model id (default: {default_model_id}).",
        )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=default_horizon,
        help=f"Forecast horizon in days (default {default_horizon}). "
        f"Max input becomes TOTAL_DAYS - horizon = "
        f"{total_days} - horizon days.",
    )
    parser.add_argument(
        "--context-days",
        nargs="+",
        type=int,
        default=None,
        help=f"Context lengths in days. If omitted, sweeps the full grid "
        f"1..(TOTAL_DAYS - horizon_days). Max valid value depends on "
        f"horizon (must be <= {total_days} - horizon_days).",
    )
    parser.add_argument("--data-dir", default="data/station")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output dir (default: output/<safe_id>_h<H>d_ctx_sweep).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stations", nargs="*", help="Subset of station stems")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip per-station runs whose JSON already exists in the ctx dir",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED}). "
        "Sets Python/NumPy/torch RNGs and enables deterministic torch ops.",
    )
    parser.add_argument(
        "--no-naive",
        action="store_true",
        help="Skip the seasonal-24h naive baseline alongside the model. "
        "By default the naive baseline is computed once per (ctx, station) "
        "for paper-grade reference rows.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def resolve_sweep_grid(args, parser, total_days: int = TOTAL_DAYS):
    """Validate horizon/context flags and return (context_days_sorted,
    context_hours_list, max_context_days, prediction_length_hours).
    """
    max_context_days = total_days - args.horizon_days
    if max_context_days < MIN_CONTEXT_DAYS:
        parser.error(
            f"horizon-days={args.horizon_days} leaves no room for input "
            f"(TOTAL_DAYS={total_days})."
        )
    if args.context_days is None:
        context_days = list(range(MIN_CONTEXT_DAYS, max_context_days + 1))
    else:
        bad = [d for d in args.context_days if d < MIN_CONTEXT_DAYS or d > max_context_days]
        if bad:
            parser.error(
                f"context-days must be in [{MIN_CONTEXT_DAYS}, "
                f"{max_context_days}] for horizon={args.horizon_days}d; "
                f"got invalid values: {bad}"
            )
        context_days = sorted(set(args.context_days))
    context_hours = [d * 24 for d in context_days]
    prediction_length = args.horizon_days * 24
    return context_days, context_hours, max_context_days, prediction_length


def run_sweep(
    predictor: BaseTSFMPredictor,
    *,
    csv_files: List[Path],
    context_hours: List[int],
    output_root: Path,
    prediction_length: int,
    skip_existing: bool = False,
    verbose: bool = False,
    sanity_check: bool = True,
    seed: int = DEFAULT_SEED,
    compute_naive: bool = True,
):
    """Common sweep loop.

    For each context_h in context_hours, runs every station in csv_files
    through ``predictor(input_window, context_h, prediction_length)`` and
    saves the result with the existing ``save_station_result`` /
    ``save_batch_summary`` schema.

    Reproducibility:
        - ``set_seed(seed)`` is called once at the start so torch / numpy /
          random RNGs are all aligned. The seed is recorded into
          ``predictor.extra_info["seed"]`` and ends up in every per-station
          JSON.
        - ``predictor.extra_info["lib_version"]`` is auto-filled from the
          pinned distribution name for ``predictor.family`` (overrides
          whatever the runner already wrote, so the value reflects the
          actually-installed wheel rather than a docstring claim).

    Sanity:
        - ``assert_nonflat_prediction`` runs exactly once before the first
          context starts (unless ``sanity_check=False``) to catch broken
          models early.

    Naive baseline:
        - When ``compute_naive=True`` (default), the seasonal-naive(s=24)
          metrics are computed once per (ctx, station) and merged into
          each station's ``extra_info["naive_seasonal_metrics"]``. The
          aggregate naive metrics for the ctx are saved as
          ``ctx_dir/naive_summary.json`` so the paper can show a baseline
          row alongside every model.
    """
    output_root.mkdir(parents=True, exist_ok=True)

    # Reproducibility: seed BEFORE any forward pass.
    set_seed(seed)
    predictor.extra_info.setdefault("seed", seed)
    predictor.extra_info["lib_version"] = _resolve_lib_version(predictor.family)

    if sanity_check:
        # Use a context that fits inside the smallest scheduled context,
        # so the sanity probe doesn't ask the model to consume more than
        # the user's grid would.
        probe_ctx = max(MIN_CONTEXT_DAYS * 24, min(context_hours) if context_hours else 24 * 28)
        try:
            assert_nonflat_prediction(
                predictor,
                prediction_length=prediction_length,
                context_h=probe_ctx,
            )
        except RuntimeError as e:
            logger.error("Aborting sweep: %s", e)
            raise

    per_context_summaries: Dict[int, Dict] = {}
    total_t0 = time.time()
    for ctx_h in context_hours:
        per_context_summaries[ctx_h] = _run_one_context(
            predictor=predictor,
            csv_files=csv_files,
            context_h=ctx_h,
            output_root=output_root,
            prediction_length=prediction_length,
            skip_existing=skip_existing,
            verbose=verbose,
            compute_naive=compute_naive,
        )

    write_sweep_summary(per_context_summaries, output_root)
    total_min = (time.time() - total_t0) / 60
    print(
        f"\nTotal sweep time: {total_min:.1f} min "
        f"({len(context_hours)} contexts × {len(csv_files)} stations)"
    )
    return per_context_summaries


def _run_one_context(
    *,
    predictor: BaseTSFMPredictor,
    csv_files: List[Path],
    context_h: int,
    output_root: Path,
    prediction_length: int,
    skip_existing: bool,
    verbose: bool,
    compute_naive: bool = True,
) -> Dict:
    ctx_dir = output_root / f"ctx_{context_h}h"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    pending = csv_files
    if skip_existing:
        pending = [f for f in pending if not (ctx_dir / f"{f.stem}.json").exists()]

    print(f"\n{'=' * 60}")
    print(f"Context = {context_h}h ({context_h / 24:.1f}d)")
    print(f"  Stations to run: {len(pending)} / {len(csv_files)}")
    print(f"  Output:          {ctx_dir}")
    print(f"{'=' * 60}")

    # Per-station naive seasonal-24h baseline. Same for every model run on
    # the same dataset, so we compute it inline here and stash both per-
    # station (in extra_info) and as an aggregate (naive_summary.json).
    naive_results: List[Dict] = []

    results: List[Dict] = []
    t0 = time.time()
    for idx, csv_path in enumerate(pending, 1):
        sid = csv_path.stem
        print(f"  [{idx:3d}/{len(pending)}] {sid} ... ", end="", flush=True)
        try:
            data = load_station_data(csv_path)
            splits = split_data(data, target_hours=prediction_length)
            input_window = splits["input_window"]
            test_target = splits["test_target"]

            naive_metrics = (
                compute_naive_seasonal_metrics(input_window, test_target, season=24)
                if compute_naive
                else None
            )

            t_start = time.time()
            prediction = predictor(input_window, context_h, prediction_length)
            elapsed = time.time() - t_start

            prediction = apply_physical_clamp(prediction, KPI_NAMES)
            metrics = compute_metrics(prediction, test_target, input_window=input_window)

            extra = {
                "context_h": context_h,
                "context_d": context_h / 24.0,
                "model_family": predictor.family,
            }
            extra.update(predictor.extra_info)
            if naive_metrics is not None:
                extra["naive_seasonal_metrics"] = naive_metrics
                naive_results.append(
                    {
                        "station_id": sid,
                        "status": "success",
                        "metrics": naive_metrics,
                        "elapsed_sec": 0.0,
                    }
                )

            result = save_station_result(
                station_id=sid,
                model_name=predictor.model_id,
                prediction=prediction,
                truth=test_target,
                metrics=metrics,
                elapsed_sec=elapsed,
                output_dir=ctx_dir,
                extra_info=extra,
            )
            results.append(result)
            tp_smape = result["metrics"]["per_kpi"].get("Throughput", {}).get("sMAPE", 0)
            print(f"OK  sMAPE(TP)={tp_smape:.2f}  {result['elapsed_sec']:.1f}s")
        except Exception as exc:
            logger.error("Failed %s @ ctx=%dh: %s", sid, context_h, exc, exc_info=verbose)
            results.append({"station_id": sid, "status": "error", "error": str(exc)})
            print(f"FAIL ({exc})")

    summary = save_batch_summary(
        results,
        predictor.model_id,
        ctx_dir,
        time.time() - t0,
        extra_config={
            "context_h": context_h,
            "context_d": context_h / 24.0,
            "model_family": predictor.family,
            **predictor.extra_info,
        },
    )
    print_batch_report(summary, ctx_dir)

    # Persist the naive baseline aggregate alongside the model summary so
    # the paper's "model vs naive" reference row is reproducible from any
    # ctx_dir alone. We write directly to ``naive_summary.json`` here
    # (cannot reuse ``save_batch_summary`` — that helper is hardcoded to
    # ``summary.json`` and would clobber the model aggregate just written).
    if compute_naive and naive_results:
        per_kpi_avg = {}
        for kpi in KPI_NAMES:
            kpi_smapes = [r["metrics"]["per_kpi"][kpi]["sMAPE"] for r in naive_results]
            kpi_mases = [r["metrics"]["per_kpi"][kpi]["MASE"] for r in naive_results]
            kpi_nrmses = [r["metrics"]["per_kpi"][kpi]["nRMSE"] for r in naive_results]
            per_kpi_avg[kpi] = {
                "avg_sMAPE": round(sum(kpi_smapes) / len(kpi_smapes), 2),
                "avg_MASE": round(sum(kpi_mases) / len(kpi_mases), 4),
                "avg_nRMSE": round(sum(kpi_nrmses) / len(kpi_nrmses), 4),
            }
        naive_summary = {
            "run_info": {
                "timestamp": datetime.now().isoformat(),
                "model": "seasonal_naive_24h",
                "model_family": "naive",
                "season": 24,
                "context_h": context_h,
                "context_d": context_h / 24.0,
                "total_stations": len(naive_results),
                "successful": len(naive_results),
            },
            "aggregate": {"per_kpi": per_kpi_avg},
            "stations": [
                {
                    "station_id": r["station_id"],
                    "metrics": r["metrics"],
                }
                for r in naive_results
            ],
        }
        with open(ctx_dir / "naive_summary.json", "w") as f:
            json.dump(naive_summary, f, indent=2, ensure_ascii=False, default=str)

    return summary


def write_sweep_summary(per_context_summaries: Dict[int, Dict], output_root: Path):
    """Write the cross-context aggregate sweep_summary.{csv,json}."""
    import csv as _csv

    csv_path = output_root / "sweep_summary.csv"
    json_path = output_root / "sweep_summary.json"

    rows = []
    for ctx_h in sorted(per_context_summaries):
        summ = per_context_summaries[ctx_h]
        agg = summ.get("aggregate", {})
        per_kpi = agg.get("per_kpi", {})
        for kpi in KPI_NAMES:
            vals = per_kpi.get(kpi, {})
            rows.append(
                {
                    "context_h": ctx_h,
                    "context_d": round(ctx_h / 24.0, 2),
                    "kpi": kpi,
                    "avg_sMAPE": vals.get("avg_sMAPE"),
                    "avg_MASE": vals.get("avg_MASE"),
                    "avg_nRMSE": vals.get("avg_nRMSE"),
                    "n_stations": summ["run_info"].get("successful", 0),
                    "avg_time_per_station_sec": agg.get("avg_time_per_station_sec"),
                }
            )

    with open(csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w") as f:
        json.dump(
            {str(k): v for k, v in per_context_summaries.items()},
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print(f"\n{'=' * 60}")
    print("SWEEP SUMMARY — best context per KPI (lowest avg sMAPE)")
    print(f"{'=' * 60}")
    print(f"  {'KPI':20s}  {'best ctx':>10s}  {'sMAPE(%)':>10s}  {'MASE':>10s}")
    print(f"  {'-' * 56}")
    for kpi in KPI_NAMES:
        kpi_rows = [r for r in rows if r["kpi"] == kpi and r["avg_sMAPE"] is not None]
        if not kpi_rows:
            continue
        best = min(kpi_rows, key=lambda r: r["avg_sMAPE"])
        print(
            f"  {kpi:20s}  {best['context_h']:>7d}h  "
            f"{best['avg_sMAPE']:>10.2f}  {best['avg_MASE']:>10.4f}"
        )

    print("\n  Overall avg sMAPE (mean across 7 KPIs) by context:")
    print(f"  {'context':>10s}  {'avg sMAPE(%)':>14s}  {'avg MASE':>12s}")
    print(f"  {'-' * 40}")
    for ctx_h in sorted(per_context_summaries):
        kpi_rows = [r for r in rows if r["context_h"] == ctx_h and r["avg_sMAPE"] is not None]
        if not kpi_rows:
            continue
        mean_smape = float(np.mean([r["avg_sMAPE"] for r in kpi_rows]))
        mean_mase = float(np.mean([r["avg_MASE"] for r in kpi_rows]))
        print(f"  {ctx_h:>7d}h    {mean_smape:>14.2f}  {mean_mase:>12.4f}")

    print(f"\n  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
