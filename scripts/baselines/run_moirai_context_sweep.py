#!/usr/bin/env python3
"""Moirai context-length sweep (cross-channel / Any-Variate Attention).

Sweeps the input context length fed to Moirai while keeping the
prediction horizon fixed at 168h (last 7 days of the 88-day series).
All 7 KPIs are forecast jointly via Moirai's Any-Variate Attention
(target_dim=7).

Data layout (per station, 88 days hourly):
    - Total:        2112h (88d)
    - Test target:  last 168h (day 82-88, fixed)
    - Available:    1944h (81d) for the input window — the sweep picks a
                    suffix of this window of length ``context_h``.

Usage:
    conda run -n telcoagent python scripts/baselines/run_moirai_context_sweep.py \\
        --context-days 1 3 7 14 21 28 42 56 70 81

    # full range (1..81d), subset of stations for smoke test
    conda run -n telcoagent python scripts/baselines/run_moirai_context_sweep.py \\
        --stations station_A_10 station_C_2

Outputs:
    output/moirai_ctx_sweep/
        ctx_<H>h/                # per-context Moirai results (same schema
                                  # as scripts/baselines/run_moirai.py)
            <station>.json
            csv/<station>.csv
            summary.json
        sweep_summary.csv         # one row per (context_h, kpi) with avg metrics
        sweep_summary.json        # full nested summary across all contexts
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

from scripts.baselines.foundation_utils import (
    KPI_NAMES,
    apply_physical_clamp,
    compute_metrics,
    load_station_data,
    print_batch_report,
    save_batch_summary,
    save_station_result,
    split_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from telcoagent.config import PREDICTION_LENGTH_H as PREDICTION_LENGTH

DEFAULT_MODEL_ID = "Salesforce/moirai-1.1-R-large"
N_CHANNELS = 7

# Default sweep grid (days). 81d = full available input window.
MIN_CONTEXT_DAYS = 1
MAX_CONTEXT_DAYS = 81  # 88d total - 7d target
DEFAULT_CONTEXT_DAYS = list(range(MIN_CONTEXT_DAYS, MAX_CONTEXT_DAYS + 1))  # 1..81


def load_module(model_id: str, device: str):
    """Load the Moirai weights once. Wrapper is rebuilt per context length."""
    from uni2ts.model.moirai import MoiraiModule

    logger.info("Loading Moirai module: %s", model_id)
    module = MoiraiModule.from_pretrained(model_id)
    if device == "cuda" and torch.cuda.is_available():
        module = module.to(device)
    return module


def build_forecast(module, context_h: int):
    """Build a MoiraiForecast wrapper for a specific context length."""
    from uni2ts.model.moirai import MoiraiForecast

    return MoiraiForecast(
        module=module,
        prediction_length=PREDICTION_LENGTH,
        context_length=context_h,
        patch_size="auto",
        num_samples=20,
        target_dim=N_CHANNELS,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )


def predict(model, input_data: np.ndarray, context_h: int) -> np.ndarray:
    """Moirai cross-channel prediction with custom context length.

    Args:
        model:      MoiraiForecast wrapper built for this ``context_h``.
        input_data: Full available input window, shape (L, C). The last
                    ``context_h`` hours are used as the model context.
        context_h:  Number of hours fed to the model (same for all KPIs).

    Returns:
        Forecast of shape (PREDICTION_LENGTH, C).
    """
    import pandas as pd
    from gluonts.dataset.common import ListDataset

    if input_data.shape[0] < context_h:
        raise ValueError(f"input_data has only {input_data.shape[0]}h, need {context_h}h")
    context = input_data[-context_h:]  # (context_h, C)

    dataset = ListDataset(
        [
            {
                "start": pd.Timestamp("2024-01-01"),
                "target": context.T,  # (C, context_h)
            }
        ],
        freq="h",
        one_dim_target=False,
    )

    predictor = model.create_predictor(batch_size=1)
    forecasts = list(predictor.predict(dataset))
    forecast = forecasts[0]
    pred = np.median(forecast.samples, axis=0)  # (PREDICTION_LENGTH, C)

    if pred.shape[0] < PREDICTION_LENGTH:
        pad = PREDICTION_LENGTH - pred.shape[0]
        pred = np.pad(pred, ((0, pad), (0, 0)), mode="edge")
    return pred[:PREDICTION_LENGTH]


def run_station(
    model,
    station_id: str,
    data: np.ndarray,
    context_h: int,
    output_dir: Path,
    model_id: str,
) -> dict:
    splits = split_data(data)
    input_window = splits["input_window"]
    test_target = splits["test_target"]

    t0 = time.time()
    prediction = predict(model, input_window, context_h)
    elapsed = time.time() - t0

    prediction = apply_physical_clamp(prediction, KPI_NAMES)
    metrics = compute_metrics(prediction, test_target, input_window=input_window)

    return save_station_result(
        station_id=station_id,
        model_name=model_id,
        prediction=prediction,
        truth=test_target,
        metrics=metrics,
        elapsed_sec=elapsed,
        output_dir=output_dir,
        extra_info={
            "context_h": context_h,
            "context_d": context_h / 24.0,
            "multivariate_mode": "any_variate_attention",
        },
    )


def run_one_context(
    module,
    model_id: str,
    csv_files: List[Path],
    context_h: int,
    output_root: Path,
    skip_existing: bool,
    verbose: bool,
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

    # Build wrapper once per context length.
    model = build_forecast(module, context_h)

    results: List[Dict] = []
    t0 = time.time()
    for idx, csv_path in enumerate(pending, 1):
        sid = csv_path.stem
        print(f"  [{idx:3d}/{len(pending)}] {sid} ... ", end="", flush=True)
        try:
            data = load_station_data(csv_path)
            result = run_station(model, sid, data, context_h, ctx_dir, model_id)
            results.append(result)
            tp_smape = result["metrics"]["per_kpi"].get("Throughput", {}).get("sMAPE", 0)
            print(f"OK  sMAPE(TP)={tp_smape:.2f}  {result['elapsed_sec']:.1f}s")
        except Exception as exc:
            logger.error("Failed %s @ ctx=%dh: %s", sid, context_h, exc, exc_info=verbose)
            results.append({"station_id": sid, "status": "error", "error": str(exc)})
            print(f"FAIL ({exc})")

    summary = save_batch_summary(
        results,
        model_id,
        ctx_dir,
        time.time() - t0,
        extra_config={
            "context_h": context_h,
            "context_d": context_h / 24.0,
            "multivariate_mode": "any_variate_attention",
        },
    )
    print_batch_report(summary, ctx_dir)
    return summary


def write_sweep_summary(per_context_summaries: Dict[int, Dict], output_root: Path):
    """Write CSV + JSON aggregating per-KPI metrics across context lengths."""
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
                    "n_stations": summ["run_info"].get("successful", 0),
                    "avg_time_per_station_sec": agg.get("avg_time_per_station_sec"),
                }
            )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
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


def main():
    parser = argparse.ArgumentParser(description="Moirai context-length sweep")
    parser.add_argument(
        "--context-days",
        nargs="+",
        type=int,
        default=DEFAULT_CONTEXT_DAYS,
        help=f"Context lengths in days (default: 1..{MAX_CONTEXT_DAYS})",
    )
    parser.add_argument("--data-dir", default="data/station")
    parser.add_argument("--output-dir", default="output/moirai_ctx_sweep")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stations", nargs="*", help="Subset of station stems")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip per-station runs whose JSON already exists in the ctx dir",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    bad = [d for d in args.context_days if d < MIN_CONTEXT_DAYS or d > MAX_CONTEXT_DAYS]
    if bad:
        parser.error(
            f"context-days must be in [{MIN_CONTEXT_DAYS}, {MAX_CONTEXT_DAYS}]; "
            f"got invalid values: {bad}"
        )
    context_days = sorted(set(args.context_days))
    context_hours = [d * 24 for d in context_days]

    device = args.device if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("station_*.csv"))
    if args.stations:
        csv_files = [f for f in csv_files if f.stem in args.stations]
    if not csv_files:
        parser.error(f"No station CSVs found under {data_dir}")

    module = load_module(args.model, device)

    print("\nMoirai Context-Length Sweep (cross-channel, Any-Variate Attention)")
    print(f"  Model:        {args.model}")
    print(f"  Horizon:      {PREDICTION_LENGTH}h (7d, fixed)")
    print(f"  Context grid: {context_days} days  ({len(context_hours)} values)")
    print(f"  Stations:     {len(csv_files)}")
    print(f"  Device:       {device}")
    print(f"  Output:       {output_root}")

    per_context_summaries: Dict[int, Dict] = {}
    total_t0 = time.time()
    for ctx_h in context_hours:
        per_context_summaries[ctx_h] = run_one_context(
            module,
            args.model,
            csv_files,
            ctx_h,
            output_root,
            skip_existing=args.skip_existing,
            verbose=args.verbose,
        )

    write_sweep_summary(per_context_summaries, output_root)
    total_min = (time.time() - total_t0) / 60
    print(
        f"\nTotal sweep time: {total_min:.1f} min "
        f"({len(context_hours)} contexts × {len(csv_files)} stations)"
    )


if __name__ == "__main__":
    main()
