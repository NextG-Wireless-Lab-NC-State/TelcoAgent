"""Metrics and artifact writers shared by all baseline-study models."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from scripts.baseline_study.data import KPI_NAMES
from telcoagent.prediction.physical_clamp import apply_physical_clamp_strict


def apply_clamp(pred: np.ndarray) -> np.ndarray:
    return apply_physical_clamp_strict(pred, KPI_NAMES)


def compute_metrics(
    pred: np.ndarray, truth: np.ndarray, input_window: np.ndarray | None = None
) -> Dict:
    n_hours = min(pred.shape[0], truth.shape[0])
    pred = pred[:n_hours]
    truth = truth[:n_hours]
    season = 24
    per_kpi = {}
    for idx, kpi in enumerate(KPI_NAMES):
        p = pred[:, idx]
        t = truth[:, idx]
        denom = np.abs(p) + np.abs(t)
        mask = denom > 1e-8
        smape = (
            float(np.mean(2.0 * np.abs(p[mask] - t[mask]) / denom[mask]) * 100)
            if mask.any()
            else 0.0
        )
        baseline = input_window[:, idx] if input_window is not None else t
        if len(baseline) > season:
            scale = float(np.mean(np.abs(baseline[season:] - baseline[:-season])))
        else:
            scale = float(np.mean(np.abs(np.diff(baseline)))) if len(baseline) > 1 else 1.0
        mae = float(np.mean(np.abs(p - t)))
        mse = float(np.mean((p - t) ** 2))
        rmse = float(np.sqrt(mse))
        rng = float(t.max() - t.min())
        per_kpi[kpi] = {
            "sMAPE": round(smape, 2),
            "MASE": round(mae / scale, 4) if scale > 1e-8 else 0.0,
            "nRMSE": round(rmse / rng, 4) if rng > 1e-8 else 0.0,
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
        }
    return {
        "per_kpi": per_kpi,
        "mean_sMAPE": round(float(np.mean([v["sMAPE"] for v in per_kpi.values()])), 4),
        "mean_MASE": round(float(np.mean([v["MASE"] for v in per_kpi.values()])), 6),
        "mean_nRMSE": round(float(np.mean([v["nRMSE"] for v in per_kpi.values()])), 6),
        "mean_MAE": round(float(np.mean([v["MAE"] for v in per_kpi.values()])), 6),
        "mean_MSE": round(float(np.mean([v["MSE"] for v in per_kpi.values()])), 6),
    }


def aggregate_station_metrics(rows: List[Dict]) -> Dict:
    successful = [r for r in rows if r.get("status") == "success"]
    if not successful:
        return {"successful": 0}
    per_kpi = {}
    for kpi in KPI_NAMES:
        vals = [r["metrics"]["per_kpi"][kpi] for r in successful]
        per_kpi[kpi] = {
            "sMAPE": round(float(np.mean([v["sMAPE"] for v in vals])), 4),
            "MASE": round(float(np.mean([v["MASE"] for v in vals])), 6),
            "nRMSE": round(float(np.mean([v["nRMSE"] for v in vals])), 6),
            "MAE": round(float(np.mean([v["MAE"] for v in vals])), 6),
            "MSE": round(float(np.mean([v["MSE"] for v in vals])), 6),
            "RMSE": round(float(np.mean([v["RMSE"] for v in vals])), 6),
        }
    return {
        "successful": len(successful),
        "mean_sMAPE": round(float(np.mean([r["metrics"]["mean_sMAPE"] for r in successful])), 4),
        "mean_MASE": round(float(np.mean([r["metrics"]["mean_MASE"] for r in successful])), 6),
        "mean_nRMSE": round(float(np.mean([r["metrics"]["mean_nRMSE"] for r in successful])), 6),
        "mean_MAE": round(float(np.mean([r["metrics"]["mean_MAE"] for r in successful])), 6),
        "mean_MSE": round(float(np.mean([r["metrics"]["mean_MSE"] for r in successful])), 6),
        "per_kpi": per_kpi,
    }


def prediction_frame(station_id: str, pred: np.ndarray, truth: np.ndarray) -> pd.DataFrame:
    rows = []
    for h in range(min(len(pred), len(truth))):
        row = {"station_id": station_id, "horizon_h": h, "day": h // 24 + 1, "hour": h % 24}
        for idx, kpi in enumerate(KPI_NAMES):
            row[f"{kpi}_pred"] = round(float(pred[h, idx]), 4)
            row[f"{kpi}_true"] = round(float(truth[h, idx]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def write_station_artifacts(
    *,
    output_dir: Path,
    station_id: str,
    split_name: str,
    pred: np.ndarray,
    truth: np.ndarray,
    metrics: Dict,
) -> Dict:
    split_dir = output_dir / split_name
    csv_dir = split_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame(station_id, pred, truth).to_csv(csv_dir / f"{station_id}.csv", index=False)
    payload = {
        "station_id": station_id,
        "split": split_name,
        "timestamp": datetime.now().isoformat(),
        "prediction_shape": list(pred.shape),
        "truth_shape": list(truth.shape),
        "metrics": metrics,
    }
    (split_dir / f"{station_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"station_id": station_id, "status": "success", "metrics": metrics}


def write_summary(output_dir: Path, split_name: str, rows: List[Dict], extra: Dict) -> Dict:
    summary = {
        "run_info": {"timestamp": datetime.now().isoformat(), **extra},
        "aggregate": aggregate_station_metrics(rows),
        "stations": rows,
    }
    path = output_dir / split_name / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "station_id": r["station_id"],
                "status": r["status"],
                "mean_sMAPE": r.get("metrics", {}).get("mean_sMAPE"),
                "mean_MASE": r.get("metrics", {}).get("mean_MASE"),
                "mean_nRMSE": r.get("metrics", {}).get("mean_nRMSE"),
                "mean_MAE": r.get("metrics", {}).get("mean_MAE"),
                "mean_MSE": r.get("metrics", {}).get("mean_MSE"),
            }
            for r in rows
        ]
    ).to_csv(output_dir / split_name / "metrics_per_station.csv", index=False)
    return summary
