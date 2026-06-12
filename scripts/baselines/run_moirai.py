#!/usr/bin/env python3
"""Moirai Foundation Model Evaluation Script.

Moirai uses Any-Variate Attention for native multivariate forecasting,
producing a joint forecast across all KPI channels — a prerequisite for
the directed Transfer Entropy analysis run by the Explainer Agent.

Usage:
    python scripts/run_moirai.py --data-dir data/station --output-dir output/moirai
    python scripts/run_moirai.py --data-dir data/station --model Salesforce/moirai-1.1-R-large
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

logger = logging.getLogger(__name__)

MODEL_NAME = "Moirai"
DEFAULT_MODEL_ID = "Salesforce/moirai-1.1-R-large"


N_CHANNELS = 7  # number of KPI channels (RRC_Conn, DL_CQI, DL_iBler, DL_rBler, MAC_DL_Eff, PRB_Util, Throughput)


def load_model(model_id: str, device: str):
    """Load Moirai model in Any-Variate (cross-channel) mode with target_dim=7."""
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    logger.info(f"Loading Moirai: {model_id} (cross-channel, target_dim={N_CHANNELS})")

    module = MoiraiModule.from_pretrained(model_id)

    # target_dim=N_CHANNELS enables Any-Variate Attention across all 7 KPIs jointly
    model = MoiraiForecast(
        module=module,
        prediction_length=168,
        context_length=1944,
        patch_size="auto",
        num_samples=20,
        target_dim=N_CHANNELS,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )

    if device == "cuda" and torch.cuda.is_available():
        module = module.to(device)

    return model, device


def predict(
    model,
    device: str,
    input_data: np.ndarray,
    prediction_length: int = 168,
) -> np.ndarray:
    """Run Moirai prediction (cross-channel, Any-Variate Attention).

    All 7 KPI channels are fed jointly so Moirai can model inter-KPI
    dependencies via its Any-Variate Attention mechanism.

    Args:
        model: Moirai model (target_dim=7)
        device: Device string
        input_data: Input array (hours, channels) - normalized
        prediction_length: Forecast horizon

    Returns:
        Prediction array (prediction_length, channels) - normalized
    """
    import pandas as pd
    from gluonts.dataset.common import ListDataset

    n_hours, n_channels = input_data.shape

    # Feed all channels jointly: target shape (n_channels, n_hours)
    dataset = ListDataset(
        [
            {
                "start": pd.Timestamp("2024-01-01"),
                "target": input_data.T,  # (n_channels, n_hours)
            }
        ],
        freq="h",
        one_dim_target=False,
    )

    predictor = model.create_predictor(batch_size=1)
    forecasts = list(predictor.predict(dataset))

    # samples shape: (n_samples, prediction_length, n_channels) — Moirai multivariate format
    forecast = forecasts[0]
    pred = np.median(forecast.samples, axis=0)  # (prediction_length, n_channels)

    if pred.shape[0] < prediction_length:
        pad = prediction_length - pred.shape[0]
        pred = np.pad(pred, ((0, pad), (0, 0)), mode="edge")

    return pred[:prediction_length]


def run_station(
    model,
    device: str,
    station_id: str,
    data: np.ndarray,
    output_dir: Path,
    model_id: str,
) -> dict:
    """Run prediction for a single station."""
    splits = split_data(data)
    input_window = splits["input_window"]
    test_target = splits["test_target"]

    t0 = time.time()
    prediction = predict(model, device, input_window, prediction_length=168)
    elapsed = time.time() - t0

    # Apply physical clamp and integer rounding
    prediction = apply_physical_clamp(prediction, KPI_NAMES)

    # Compute metrics
    metrics = compute_metrics(prediction, test_target, input_window=input_window)

    # Save results
    return save_station_result(
        station_id=station_id,
        model_name=model_id,
        prediction=prediction,
        truth=test_target,
        metrics=metrics,
        elapsed_sec=elapsed,
        output_dir=output_dir,
        extra_info={"multivariate_mode": "any_variate_attention"},
    )


def main():
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} Evaluation")
    parser.add_argument("--data-dir", required=True, help="Directory with station CSVs")
    parser.add_argument("--output-dir", default="output/moirai", help="Output directory")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID, help=f"Model ID (default: {DEFAULT_MODEL_ID})"
    )
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--stations", nargs="*", help="Specific stations to run")
    parser.add_argument("--skip-existing", action="store_true", help="Skip completed stations")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find station files
    csv_files = sorted(data_dir.glob("station_*.csv"))
    if args.stations:
        csv_files = [f for f in csv_files if f.stem in args.stations]
    if args.skip_existing:
        csv_files = [f for f in csv_files if not (output_dir / f"{f.stem}.json").exists()]

    print(f"\n{MODEL_NAME} Evaluation (Native Multivariate)")
    print(f"  Model:    {args.model}")
    print(f"  Device:   {args.device}")
    print(f"  Stations: {len(csv_files)}")
    print(f"  Output:   {output_dir}")
    print("  Multivariate forecast: VALID (Any-Variate Attention)\n")

    # Load model
    model, device = load_model(args.model, args.device)

    # Run predictions
    results = []
    total_t0 = time.time()

    for idx, csv_path in enumerate(csv_files, 1):
        station_id = csv_path.stem
        print(f"[{idx}/{len(csv_files)}] {station_id} ... ", end="", flush=True)

        try:
            data = load_station_data(csv_path)
            result = run_station(model, device, station_id, data, output_dir, args.model)
            results.append(result)
            print(f"OK (MAE={result['metrics']['overall_MAE']:.2f}, {result['elapsed_sec']:.1f}s)")
        except Exception as e:
            logger.error(f"Failed {station_id}: {e}", exc_info=args.verbose)
            results.append({"station_id": station_id, "status": "error", "error": str(e)})
            print(f"FAIL ({e})")

    total_elapsed = time.time() - total_t0

    # Save summary
    summary = save_batch_summary(
        results=results,
        model_name=args.model,
        output_dir=output_dir,
        total_elapsed=total_elapsed,
        extra_config={"device": args.device, "multivariate_mode": "native"},
    )
    print_batch_report(summary, output_dir)


if __name__ == "__main__":
    main()
