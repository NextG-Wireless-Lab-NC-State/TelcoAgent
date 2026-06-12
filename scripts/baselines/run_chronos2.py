#!/usr/bin/env python3
"""Chronos-2 foundation model baseline (raw, no TelcoAgent).

Amazon Chronos-2: multivariate cross-channel forecasting.
Context: 672h (28 days), Prediction: 168h (7 days).

Usage:
    conda run -n telco-chronos python scripts/baselines/run_chronos2.py
    conda run -n telco-chronos python scripts/baselines/run_chronos2.py --skip-existing
"""

import argparse
import logging
import sys
import time
from pathlib import Path

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

from telcoagent.config import CONTEXT_LENGTH_H as CONTEXT_LENGTH
from telcoagent.config import PREDICTION_LENGTH_H as PREDICTION_LENGTH

MODEL_ID = "amazon/chronos-2"


def load_model(device: str):
    from chronos import Chronos2Pipeline

    logger.info("Loading Chronos-2: %s", MODEL_ID)
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_ID,
        device_map=device,
        dtype=torch.float32,
    )
    return pipeline


def predict(pipeline, input_data: np.ndarray) -> np.ndarray:
    """Chronos-2 cross-channel prediction: (L, C) -> (168, C)."""
    context = input_data[-CONTEXT_LENGTH:]  # (672, 7)
    input_tensor = torch.tensor(
        context.T[np.newaxis, :, :],
        dtype=torch.float32,
    )  # (1, C, 672)
    output = pipeline.predict(input_tensor, prediction_length=PREDICTION_LENGTH)
    pred_samples = output[0]  # (C, n_samples, H)
    pred = torch.median(pred_samples, dim=1).values.numpy()  # (C, H)
    return pred.T  # (H, C) = (168, 7)


def run_station(pipeline, station_id: str, data: np.ndarray, output_dir: Path) -> dict:
    splits = split_data(data)
    input_window = splits["input_window"]
    test_target = splits["test_target"]

    t0 = time.time()
    prediction = predict(pipeline, input_window)
    elapsed = time.time() - t0

    prediction = apply_physical_clamp(prediction, KPI_NAMES)
    metrics = compute_metrics(prediction, test_target, input_window=input_window)

    return save_station_result(
        station_id=station_id,
        model_name=MODEL_ID,
        prediction=prediction,
        truth=test_target,
        metrics=metrics,
        elapsed_sec=elapsed,
        output_dir=output_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Chronos-2 Foundation Model Baseline")
    parser.add_argument("--data-dir", default="data/station")
    parser.add_argument("--output-dir", default="output/chronos2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stations", nargs="*")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    device = args.device if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("station_*.csv"))
    if args.stations:
        csv_files = [f for f in csv_files if f.stem in args.stations]
    if args.skip_existing:
        csv_files = [f for f in csv_files if not (output_dir / f"{f.stem}.json").exists()]

    pipeline = load_model(device)

    print("\nChronos-2 Baseline")
    print(f"  Model:    {MODEL_ID}")
    print(f"  Context:  {CONTEXT_LENGTH}h ({CONTEXT_LENGTH // 24}d)")
    print(f"  Device:   {device}")
    print(f"  Stations: {len(csv_files)}")
    print(f"  Output:   {output_dir}\n")

    results = []
    total_t0 = time.time()

    for idx, csv_path in enumerate(csv_files, 1):
        sid = csv_path.stem
        print(f"[{idx:3d}/{len(csv_files)}] {sid} ... ", end="", flush=True)
        try:
            data = load_station_data(csv_path)
            result = run_station(pipeline, sid, data, output_dir)
            results.append(result)
            tp_smape = result["metrics"]["per_kpi"].get("Throughput", {}).get("sMAPE", 0)
            print(f"OK  sMAPE(TP)={tp_smape:.2f}  {result['elapsed_sec']:.1f}s")
        except Exception as exc:
            logger.error("Failed %s: %s", sid, exc, exc_info=args.verbose)
            results.append({"station_id": sid, "status": "error", "error": str(exc)})
            print(f"FAIL ({exc})")

    summary = save_batch_summary(results, MODEL_ID, output_dir, time.time() - total_t0)
    print_batch_report(summary, output_dir)


if __name__ == "__main__":
    main()
