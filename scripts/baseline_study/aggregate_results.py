#!/usr/bin/env python3
"""Aggregate baseline-study local artifacts into paper-ready tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.baseline_study.data import KPI_NAMES


def _load_run(run_dir: Path) -> Dict | None:
    result_path = run_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    skipped_path = run_dir / "skipped.json"
    if skipped_path.exists():
        skipped = json.loads(skipped_path.read_text(encoding="utf-8"))
        return {"run_dir": str(run_dir), "skipped": skipped}
    return None


def aggregate(root: Path, out_dir: Path) -> None:
    runs: List[Dict] = []
    for path in sorted(root.glob("*/*/*")):
        if path.is_dir():
            loaded = _load_run(path)
            if loaded is not None:
                runs.append(loaded)
    out_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = []
    per_kpi_rows = []
    for run in runs:
        if "skipped" in run:
            skipped = run["skipped"]
            leaderboard.append(
                {
                    "model_family": skipped.get("model_family"),
                    "model_name": skipped.get("model_name"),
                    "forecast_mode": skipped.get("forecast_mode"),
                    "status": "skipped",
                    "reason": skipped.get("reason"),
                    "run_dir": run["run_dir"],
                }
            )
            continue
        model = run["model"]
        test = run["test"]
        leaderboard.append(
            {
                "model_family": model["model_family"],
                "model_name": model["model_name"],
                "forecast_mode": model.get("forecast_mode"),
                "status": "completed",
                "params": model.get("model_params", 0),
                "val_mean_sMAPE": run["val"].get("mean_sMAPE"),
                "val_mean_MAE": run["val"].get("mean_MAE"),
                "val_mean_MSE": run["val"].get("mean_MSE"),
                "test_mean_sMAPE": test.get("mean_sMAPE"),
                "test_mean_MASE": test.get("mean_MASE"),
                "test_mean_nRMSE": test.get("mean_nRMSE"),
                "test_mean_MAE": test.get("mean_MAE"),
                "test_mean_MSE": test.get("mean_MSE"),
                "sec_per_station": run["runtime"].get("sec_per_station"),
                "run_dir": run["run_dir"],
            }
        )
        for kpi in KPI_NAMES:
            vals = test.get("per_kpi", {}).get(kpi, {})
            per_kpi_rows.append(
                {
                    "model_family": model["model_family"],
                    "model_name": model["model_name"],
                    "forecast_mode": model.get("forecast_mode"),
                    "kpi": kpi,
                    "sMAPE": vals.get("sMAPE"),
                    "MASE": vals.get("MASE"),
                    "nRMSE": vals.get("nRMSE"),
                    "MAE": vals.get("MAE"),
                    "MSE": vals.get("MSE"),
                    "RMSE": vals.get("RMSE"),
                    "run_dir": run["run_dir"],
                }
            )
    df = pd.DataFrame(leaderboard)
    if not df.empty and "test_mean_MAE" in df:
        df = df.sort_values(["status", "test_mean_MAE"], na_position="last")
    df.to_csv(out_dir / "leaderboard.csv", index=False)
    pd.DataFrame(per_kpi_rows).to_csv(out_dir / "metrics_per_kpi.csv", index=False)
    (out_dir / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'leaderboard.csv'}")
    print(f"Wrote {out_dir / 'metrics_per_kpi.csv'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="output/baseline_study/full")
    parser.add_argument("--out-dir", default="output/baseline_study/full/paper_tables")
    args = parser.parse_args()
    aggregate(Path(args.root), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
