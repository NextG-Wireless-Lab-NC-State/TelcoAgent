#!/usr/bin/env python3
"""Mamba4Cast zero-shot baseline (channel-independent, all 7 KPIs).

Mamba4Cast is a zero-shot foundation model for time series forecasting based
on the Mamba SSM architecture (https://github.com/automl/Mamba4Cast). It is
channel-independent: each KPI is forecast on its own. We stack the 7 KPIs as
the batch dimension so a single ``model(...)`` call produces all 7 forecasts.

Single-context runner. For a context-length sweep see
``scripts/baselines/run_mamba4cast_context_sweep.py``.

Prereqs:
    1. Clone Mamba4Cast at the repo root (already done):
         git clone https://github.com/automl/Mamba4Cast.git
    2. Place pretrained weights at:
         models/mamba4cast_2l_1024_conv_i5e5.pth
       Download:
         https://www.dropbox.com/scl/fi/s7l0razs4xdctcbxkik5t/mamba4cast_2l_1024_conv_i5e5.pth?rlkey=khd27b5tw4ipp06pt13vgl1l5&st=a3bfak3v&dl=1
    3. Install Mamba4Cast deps in the conda env:
         pip install mamba-ssm==2.2.2 causal-conv1d==1.4.0

Usage:
    conda run -n telcoagent python scripts/baselines/run_mamba4cast.py \\
        --context-days 28
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
# Mamba4Cast lives at repo root. Its package layout assumes both
# ``src_torch`` (so ``src_torch.training.models`` works) AND ``src_torch``
# itself (so ``training.utils`` works as a top-level package — that is
# how the upstream code imports its own submodules) are on sys.path.
sys.path.insert(0, str(project_root / "Mamba4Cast"))
sys.path.insert(0, str(project_root / "Mamba4Cast" / "src_torch"))

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
from telcoagent.config import N_CHANNELS
from telcoagent.config import PREDICTION_LENGTH_H as PREDICTION_LENGTH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = project_root / "models" / "mamba4cast_2l_1024_conv_i5e5.pth"
DEFAULT_SCALER = "min_max"  # matches the Mamba4Cast example notebook
SUB_DAY = True  # hourly data => use the 7-d time feature

# Architecture config from the official Mamba4Cast_example.ipynb (do not edit)
SSM_CONFIG = {
    "bidirectional": False,
    "enc_conv": True,
    "init_dil_conv": True,
    "enc_conv_kernel": 5,
    "init_conv_kernel": 5,
    "init_conv_max_dilation": 3,
    "global_residual": False,
    "in_proj_norm": False,
    "initial_gelu_flag": True,
    "linear_seq": 15,
    "mamba2": True,
    "norm": True,
    "norm_type": "layernorm",
    "num_encoder_layers": 2,
    "d_state": 128,
    "residual": False,
    "token_embed_len": 1024,
}

# Anchor for synthetic timestamps (CSVs have only "Day N + hour"; we map
# that to an arbitrary hourly timeline so the time-feature embedding works).
SYNTHETIC_START = pd.Timestamp(date(2020, 1, 1))


def _adapt_state_dict_keys(old_state_dict):
    """Notebook helper: rename ``linear_layer`` -> ``stage_2_layer.0``."""
    new_state_dict = {}
    for key, val in old_state_dict.items():
        if "linear_layer" in key:
            new_key = key.replace("linear_layer", "stage_2_layer.0")
            new_state_dict[new_key] = val
        else:
            new_state_dict[key] = val
    return new_state_dict


def _scale_data(output, scaler):
    """Notebook helper: invert the chosen scaler."""
    if scaler == "custom_robust":
        return (output["result"] * output["scale"][1].squeeze(-1)) + output["scale"][0].squeeze(-1)
    if scaler == "min_max":
        return (
            output["result"] * (output["scale"][0].squeeze(-1) - output["scale"][1].squeeze(-1))
        ) + output["scale"][1].squeeze(-1)
    if scaler == "identity":
        return output["result"]
    raise ValueError(f"Unknown scaler: {scaler}")


def load_model(weights_path: Path, device: torch.device, scaler: str = DEFAULT_SCALER):
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Mamba4Cast weights not found at {weights_path}.\n"
            "Download from the Dropbox link in the script docstring and "
            f"place the file at {weights_path}."
        )
    from src_torch.training.models import SSMModelMulti

    logger.info("Loading Mamba4Cast: %s", weights_path)
    model = SSMModelMulti(scaler=scaler, sub_day=SUB_DAY, **SSM_CONFIG).to(device)
    raw = torch.load(weights_path, map_location=device)
    state = _adapt_state_dict_keys(raw["model_state_dict"])
    model.load_state_dict(state)
    model.eval()
    return model


def _make_time_features(start: pd.Timestamp, n_hours: int) -> np.ndarray:
    """Return a (n_hours, 7) array of [year, month, day, dow+1, doy, hour, minute]."""
    ts_pd = pd.date_range(start=start, periods=n_hours, freq="h")
    return np.stack(
        [
            ts_pd.year.values,
            ts_pd.month.values,
            ts_pd.day.values,
            ts_pd.day_of_week.values + 1,
            ts_pd.day_of_year.values,
            ts_pd.hour.values,
            ts_pd.minute.values,
        ],
        axis=-1,
    ).astype(np.int64)


def _build_batch(
    context: np.ndarray,
    pred_len: int,
    device: torch.device,
) -> Tuple[dict, int]:
    """Build the dict the model expects for a single station.

    Args:
        context:  (context_h, C=7) raw KPI history.
        pred_len: forecast horizon in hours.
        device:   torch device.

    Returns:
        (batch_dict, n_channels). batch tensors live on ``device``.
    """
    context_h, n_channels = context.shape
    assert n_channels == N_CHANNELS, f"expected {N_CHANNELS} channels, got {n_channels}"

    full_ts = _make_time_features(SYNTHETIC_START, context_h + pred_len)
    ctx_ts = full_ts[:context_h]
    tgt_ts = full_ts[context_h:]

    # Mamba4Cast's SSMModelMulti expects a per-channel batch:
    #   history:      (B=C, context_h)
    #   ts:           (B=C, context_h, 7)
    #   target_dates: (B=C, pred_len, 7)
    #   task:         (B=C, pred_len) int — placeholder zeros (notebook uses 4)
    history = torch.tensor(context.T, dtype=torch.float32, device=device)
    ts = (
        torch.tensor(ctx_ts, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .repeat(n_channels, 1, 1)
    )
    target_dates = (
        torch.tensor(tgt_ts, dtype=torch.float32, device=device)
        .unsqueeze(0)
        .repeat(n_channels, 1, 1)
    )
    task = torch.zeros(n_channels, pred_len, dtype=torch.int32, device=device)

    return {"history": history, "ts": ts, "target_dates": target_dates, "task": task}, n_channels


def predict(
    model,
    input_data: np.ndarray,
    context_h: int,
    pred_len: int,
    device: torch.device,
    scaler: str = DEFAULT_SCALER,
) -> np.ndarray:
    """Mamba4Cast zero-shot forecast for a single station.

    Args:
        model:      Loaded ``SSMModelMulti``.
        input_data: Full available input window, shape (L, C).
        context_h:  Trailing window length fed to the model.
        pred_len:   Prediction horizon (hours).
        device:     torch device.
        scaler:     Scaler used during pretraining (default ``min_max``).

    Returns:
        Prediction (pred_len, C).
    """
    if input_data.shape[0] < context_h:
        raise ValueError(f"input_data has only {input_data.shape[0]}h, need {context_h}h")
    context = input_data[-context_h:]  # (context_h, C)
    batch, _ = _build_batch(context, pred_len, device)

    with torch.no_grad():
        out = model(batch, pred_len)
        pred_scaled = _scale_data(out, scaler).detach().cpu().numpy()  # (C, pred_len)

    return pred_scaled.T  # (pred_len, C)


def run_station(
    model,
    station_id: str,
    data: np.ndarray,
    context_h: int,
    output_dir: Path,
    device: torch.device,
    weights_path: Path,
    scaler: str,
) -> dict:
    splits = split_data(data)
    input_window = splits["input_window"]
    test_target = splits["test_target"]

    t0 = time.time()
    prediction = predict(model, input_window, context_h, PREDICTION_LENGTH, device, scaler)
    elapsed = time.time() - t0

    prediction = apply_physical_clamp(prediction, KPI_NAMES)
    metrics = compute_metrics(prediction, test_target, input_window=input_window)

    return save_station_result(
        station_id=station_id,
        model_name=f"Mamba4Cast/{weights_path.stem}",
        prediction=prediction,
        truth=test_target,
        metrics=metrics,
        elapsed_sec=elapsed,
        output_dir=output_dir,
        extra_info={
            "context_h": context_h,
            "context_d": context_h / 24.0,
            "scaler": scaler,
            "sub_day": SUB_DAY,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Mamba4Cast zero-shot baseline")
    parser.add_argument(
        "--context-days", type=int, default=28, help="Trailing context length in days (default 28)"
    )
    parser.add_argument("--data-dir", default="data/station")
    parser.add_argument("--output-dir", default="output/mamba4cast")
    parser.add_argument(
        "--weights", default=str(DEFAULT_WEIGHTS), help="Path to mamba4cast_2l_1024_conv_i5e5.pth"
    )
    parser.add_argument(
        "--scaler", default=DEFAULT_SCALER, choices=["min_max", "custom_robust", "identity"]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stations", nargs="*", help="Subset of station stems")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    context_h = args.context_days * 24
    if context_h < 24 or context_h > 81 * 24:
        parser.error("context-days must be in [1, 81]")

    device = torch.device(
        args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    weights_path = Path(args.weights)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("station_*.csv"))
    if args.stations:
        csv_files = [f for f in csv_files if f.stem in args.stations]
    if not csv_files:
        parser.error(f"No station CSVs found under {data_dir}")
    if args.skip_existing:
        csv_files = [f for f in csv_files if not (output_dir / f"{f.stem}.json").exists()]

    model = load_model(weights_path, device, args.scaler)

    print("\nMamba4Cast Zero-Shot Baseline (channel-independent across 7 KPIs)")
    print(f"  Weights:  {weights_path}")
    print(f"  Scaler:   {args.scaler}")
    print(f"  Context:  {context_h}h ({args.context_days}d)")
    print(f"  Horizon:  {PREDICTION_LENGTH}h (7d)")
    print(f"  Stations: {len(csv_files)}")
    print(f"  Device:   {device}")
    print(f"  Output:   {output_dir}")

    results = []
    t0 = time.time()
    for idx, csv_path in enumerate(csv_files, 1):
        sid = csv_path.stem
        print(f"  [{idx:3d}/{len(csv_files)}] {sid} ... ", end="", flush=True)
        try:
            data = load_station_data(csv_path)
            result = run_station(
                model,
                sid,
                data,
                context_h,
                output_dir,
                device,
                weights_path,
                args.scaler,
            )
            results.append(result)
            tp_smape = result["metrics"]["per_kpi"].get("Throughput", {}).get("sMAPE", 0)
            print(f"OK  sMAPE(TP)={tp_smape:.2f}  {result['elapsed_sec']:.1f}s")
        except Exception as exc:
            logger.error("Failed %s: %s", sid, exc, exc_info=args.verbose)
            results.append({"station_id": sid, "status": "error", "error": str(exc)})
            print(f"FAIL ({exc})")

    summary = save_batch_summary(
        results,
        f"Mamba4Cast/{weights_path.stem}",
        output_dir,
        time.time() - t0,
        extra_config={
            "context_h": context_h,
            "context_d": context_h / 24.0,
            "scaler": args.scaler,
        },
    )
    print_batch_report(summary, output_dir)


if __name__ == "__main__":
    main()
