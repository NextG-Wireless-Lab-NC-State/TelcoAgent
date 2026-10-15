"""Component-ablation prediction metrics — Chronos-2 vs seasonal-naive.

Quantifies the prediction-side impact of the −Predictor ablation arm:
the Full / −KG / −Explainer arms all use the Chronos-2 (ctx_672h)
forecast, while −Predictor swaps in a seasonal-naive(s=24) forecast.

For each of the n=30 selected stations this script computes per-KPI
and 7-KPI-mean nRMSE + MASE for both forecasters, reusing the metric
logic in ``scripts/baselines/foundation_utils.py``:

* Chronos-2 metrics — ``compute_metrics`` on the ``<kpi>_pred`` columns
  of the ctx_672h forecast CSV.
* seasonal-naive metrics — ``compute_naive_seasonal_metrics`` on the
  168-hour input baseline (days 75-81), tiling its last 24 h.

MASE denominator (seasonal-24h MAE) is taken from the **same** 168-hour
input window for both forecasters, so the two are directly comparable.
No LLM calls — cost $0.

Outputs (under ``--out-dir``):
* ``prediction_per_station.csv`` — one row per (station, forecaster)
  with overall + per-KPI nRMSE / MASE.
* ``prediction_per_station.json`` — same rows, JSON.
* ``prediction_summary.csv`` — forecaster-level mean ± stdev.

Example
-------
::

    PYTHONPATH=. python scripts/ablation/compute_prediction_metrics.py \\
        --stations-file output/ablation/component/stations_n30.txt \\
        --forecast-dir output/amazon_chronos-2_h7d_ctx_sweep/ctx_672h/csv \\
        --station-dir data/station \\
        --out-dir output/ablation/component
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.baselines.foundation_utils import (  # noqa: E402
    KPI_NAMES,
    compute_metrics,
    compute_naive_seasonal_metrics,
)

#: 81-day input window — matches ``run_explainer_kg_ablation._INPUT_WINDOW_HOURS``.
_INPUT_WINDOW_HOURS: int = 1944
#: Baseline window the live Explainer uses (last 168 h = days 75-81).
_BASELINE_HOURS: int = 168
#: 7-day forecast horizon.
_FORECAST_HORIZON_H: int = 168

#: Operator EMS column names mapped 1:1 to the canonical 7-KPI order.
_STATION_KPI_COLS_CSV: List[str] = [
    "RRC_Conn_Count_Avg",
    "DL_CQI",
    "DL_iBler",
    "DL_rBler",
    "MAC DL Eff TP",
    "DL PRB Utilization",
    "DL_IP_Throughput",
]


def _norm_station(name: str) -> str:
    """Normalise to bare station id (e.g. ``station_L_3`` -> ``L_3``)."""
    return name[len("station_") :] if name.startswith("station_") else name


def _load_input_baseline(station_csv: Path) -> np.ndarray:
    """Return the last 168 h of the 1944-h input window (days 75-81)."""
    df = pd.read_csv(station_csv)
    arr = df[_STATION_KPI_COLS_CSV].values.astype(np.float64)
    input_window = arr[:_INPUT_WINDOW_HOURS]
    return input_window[-_BASELINE_HOURS:]


def _load_chronos_forecast(pred_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (forecast, target) arrays of shape (168, 7) from the ctx CSV."""
    df = pd.read_csv(pred_csv)
    if len(df) != _FORECAST_HORIZON_H:
        raise ValueError(
            f"{pred_csv.name}: expected {_FORECAST_HORIZON_H} forecast rows, got {len(df)}"
        )
    forecast = np.zeros((_FORECAST_HORIZON_H, len(KPI_NAMES)))
    target = np.zeros_like(forecast)
    for i, kpi in enumerate(KPI_NAMES):
        forecast[:, i] = df[f"{kpi}_pred"].values.astype(float)
        target[:, i] = df[f"{kpi}_true"].values.astype(float)
    return forecast, target


def _flat_row(station: str, forecaster: str, metrics: Dict) -> Dict:
    """Flatten a ``compute_metrics`` result into one CSV row."""
    row: Dict[str, object] = {
        "station_id": station,
        "forecaster": forecaster,
        "overall_nRMSE": metrics["overall_nRMSE"],
    }
    nrmse_vals = [metrics["per_kpi"][k]["nRMSE"] for k in KPI_NAMES]
    mase_vals = [metrics["per_kpi"][k]["MASE"] for k in KPI_NAMES]
    row["mean_nRMSE"] = round(float(np.mean(nrmse_vals)), 6)
    row["mean_MASE"] = round(float(np.mean(mase_vals)), 6)
    for k in KPI_NAMES:
        row[f"nRMSE_{k}"] = metrics["per_kpi"][k]["nRMSE"]
        row[f"MASE_{k}"] = metrics["per_kpi"][k]["MASE"]
    return row


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Chronos-2 vs seasonal-naive prediction metrics for component ablation."
    )
    parser.add_argument("--stations-file", required=True)
    parser.add_argument("--forecast-dir", required=True)
    parser.add_argument("--station-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    stations = [
        _norm_station(line.strip())
        for line in Path(args.stations_file).read_text().splitlines()
        if line.strip()
    ]
    forecast_dir = Path(args.forecast_dir)
    station_dir = Path(args.station_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for sid in stations:
        pred_csv = forecast_dir / f"station_{sid}.csv"
        station_csv = station_dir / f"station_{sid}.csv"
        if not pred_csv.exists():
            raise FileNotFoundError(f"missing Chronos-2 forecast: {pred_csv}")
        if not station_csv.exists():
            raise FileNotFoundError(f"missing station data: {station_csv}")

        forecast, target = _load_chronos_forecast(pred_csv)
        baseline = _load_input_baseline(station_csv)

        chronos_m = compute_metrics(forecast, target, input_window=baseline)
        naive_m = compute_naive_seasonal_metrics(baseline, target, season=24)

        rows.append(_flat_row(sid, "chronos2", chronos_m))
        rows.append(_flat_row(sid, "seasonal_naive", naive_m))
        print(
            f"{sid:6s}  chronos2 nRMSE={chronos_m['overall_nRMSE']:.4f}  "
            f"naive nRMSE={naive_m['overall_nRMSE']:.4f}"
        )

    # ── per-station CSV / JSON ───────────────────────────────────────────
    fieldnames = list(rows[0].keys())
    csv_path = out_dir / "prediction_per_station.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "prediction_per_station.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    # ── forecaster-level summary ─────────────────────────────────────────
    summary_rows: List[Dict] = []
    for forecaster in ("chronos2", "seasonal_naive"):
        sub = [r for r in rows if r["forecaster"] == forecaster]
        for metric in ("mean_nRMSE", "mean_MASE", "overall_nRMSE"):
            vals = np.array([r[metric] for r in sub], dtype=float)
            summary_rows.append(
                {
                    "forecaster": forecaster,
                    "metric": metric,
                    "n": len(vals),
                    "mean": round(float(vals.mean()), 6),
                    "stdev": round(float(vals.std(ddof=1)), 6),
                    "min": round(float(vals.min()), 6),
                    "max": round(float(vals.max()), 6),
                }
            )
    summary_path = out_dir / "prediction_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["forecaster", "metric", "n", "mean", "stdev", "min", "max"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
