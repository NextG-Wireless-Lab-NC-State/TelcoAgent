#!/usr/bin/env python3
"""MOMENT Foundation Model Evaluation Script.

CMU's open-source foundation model for time series (ICML'24).
Uses channel-independent approach for multivariate forecasting.

Usage:
    python scripts/run_moment.py --data-dir data/station --output-dir output/moment
    python scripts/run_moment.py --data-dir data/station --model AutonLab/MOMENT-1-large
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

MODEL_NAME = "MOMENT"
DEFAULT_MODEL_ID = "AutonLab/MOMENT-1-large"


def load_model(model_id: str, device: str):
    """Load MOMENT model."""
    from momentfm import MOMENTPipeline

    logger.info(f"Loading MOMENT: {model_id}")

    model = MOMENTPipeline.from_pretrained(
        model_id,
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": 168,
        },
    )
    model.init()

    # Move to device - handle different API versions
    if device == "cuda" and torch.cuda.is_available():
        try:
            # Try new API first
            model = model.to(device)
        except AttributeError:
            try:
                # Try accessing internal model
                if hasattr(model, "model"):
                    model.model = model.model.to(device)
                elif hasattr(model, "backbone"):
                    model.backbone = model.backbone.to(device)
            except Exception as e:
                logger.warning(f"Could not move model to {device}: {e}")

    return model, device


def predict(
    model,
    device: str,
    input_data: np.ndarray,
    prediction_length: int = 168,
) -> np.ndarray:
    """Run MOMENT prediction (channel-independent).

    Args:
        model: MOMENT pipeline
        device: Device string
        input_data: Input array (hours, channels) - normalized
        prediction_length: Forecast horizon

    Returns:
        Prediction array (prediction_length, channels) - normalized
    """
    n_hours, n_channels = input_data.shape
    predictions = []

    # MOMENT requires seq_len=512, pad input if needed
    moment_seq_len = 512

    for c in range(n_channels):
        # Get channel data
        channel_data = input_data[:, c]

        # Pad to MOMENT's expected length (512)
        if len(channel_data) < moment_seq_len:
            # Pad at the beginning with edge values
            pad_len = moment_seq_len - len(channel_data)
            channel_data = np.pad(channel_data, (pad_len, 0), mode="edge")
        elif len(channel_data) > moment_seq_len:
            # Use last 512 points
            channel_data = channel_data[-moment_seq_len:]

        # Shape: (1, 1, 512) for MOMENT
        x = torch.tensor(channel_data, dtype=torch.float32)
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, 512)

        if device == "cuda":
            x = x.to(device)

        with torch.no_grad():
            output = model(x_enc=x)

        # Extract forecast
        forecast = output.forecast.squeeze().cpu().numpy()
        if len(forecast) > prediction_length:
            forecast = forecast[:prediction_length]
        elif len(forecast) < prediction_length:
            # Pad if needed
            forecast = np.pad(forecast, (0, prediction_length - len(forecast)), mode="edge")

        predictions.append(forecast)

    return np.stack(predictions, axis=1)  # (prediction_length, n_channels)


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
    )


def main():
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} Evaluation")
    parser.add_argument("--data-dir", required=True, help="Directory with station CSVs")
    parser.add_argument("--output-dir", default="output/moment", help="Output directory")
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

    print(f"\n{MODEL_NAME} Evaluation")
    print(f"  Model:    {args.model}")
    print(f"  Device:   {args.device}")
    print(f"  Stations: {len(csv_files)}")
    print(f"  Output:   {output_dir}\n")

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
        extra_config={"device": args.device},
    )
    print_batch_report(summary, output_dir)


if __name__ == "__main__":
    main()
