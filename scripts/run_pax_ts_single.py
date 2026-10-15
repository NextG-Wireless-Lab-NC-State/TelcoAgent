#!/usr/bin/env python3
"""Standalone PAX-TS sensitivity batch for a single station.

Loads Chronos-2 once, runs the canonical PAX-TS Algorithm 2 (4
perturbation types: localized + mean + variance + trend) on the full
1944-hour input window of one station, and writes the resulting
:class:`SensitivityResult` to disk for later integration into the
explainer report and the paper Table II.

This script is **independent** of the explainer ablation runner so
PAX-TS sensitivity can be computed offline whenever GPU time is
available, without re-running the explainer's GPT-4o-mini calls.

Usage
-----
    PYTHONPATH=. python scripts/run_pax_ts_single.py \\
        --station station_A_10 \\
        --output output/pax_ts_smoke
"""

from __future__ import annotations

import os

# Disable TensorFlow inside HuggingFace transformers BEFORE any transformers
# import — chronos pulls transformers via Chronos2Pipeline, which would
# otherwise crash on the local numpy/TF mismatch (bfloat16 ABI).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telcoagent.cli_utils import bootstrap

logger = bootstrap(suppress_tf_warnings=True, logger_name=__name__)

import torch  # noqa: E402

from telcoagent.config import (  # noqa: E402
    CONTEXT_LENGTH_H,
    CORE_KPI_NAMES,
    KPI_NORMALIZATION_MAX,
    PREDICTION_LENGTH_H,
)
from telcoagent.explainer.anomaly_detector import (  # noqa: E402
    detect_anomaly_events,
    events_to_dicts,
)
from telcoagent.explainer.pax_ts import compute_sensitivity  # noqa: E402
from telcoagent.explainer.sensitivity_format import (  # noqa: E402
    write_sensitivity_artefacts,
)

_INPUT_WINDOW_HOURS: int = 1944
_KPI_COLS: List[str] = list(CORE_KPI_NAMES)
_STATION_KPI_COLS_CSV: List[str] = [
    "RRC_Conn_Count_Avg",
    "DL_CQI",
    "DL_iBler",
    "DL_rBler",
    "MAC DL Eff TP",
    "DL PRB Utilization",
    "DL_IP_Throughput",
]


def _load_station_input(station_csv: Path) -> np.ndarray:
    df = pd.read_csv(station_csv)
    arr = df[_STATION_KPI_COLS_CSV].values.astype(np.float64)
    return arr[:_INPUT_WINDOW_HOURS]


def _load_prediction(pred_csv: Path) -> np.ndarray:
    df = pd.read_csv(pred_csv)
    forecast = np.zeros((PREDICTION_LENGTH_H, len(_KPI_COLS)))
    for i, kpi in enumerate(_KPI_COLS):
        forecast[:, i] = df[f"{kpi}_pred"].values.astype(float)
    return forecast


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run PAX-TS Algorithm 2 on a single station's Chronos-2 "
            "forecast (full 1944-hour perturbation window)."
        ),
    )
    parser.add_argument(
        "--station",
        default="station_A_10",
        help="Station id (without .csv extension).",
    )
    parser.add_argument(
        "--stations-dir",
        default="data/station",
        help="Directory with the 81-day input CSVs.",
    )
    parser.add_argument(
        "--predictions-dir",
        default="output/amazon_chronos-2_h7d_ctx_sweep/ctx_672h/csv",
        help="Cached Chronos-2 forecast CSV directory.",
    )
    parser.add_argument(
        "--model",
        default="amazon/chronos-2",
        help="Chronos-2 model id.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for Chronos-2 (cuda or cpu).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for sensitivity artefacts.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Batch chunk size for compute_sensitivity (0 = single batch).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output) / args.station
    out_dir.mkdir(parents=True, exist_ok=True)

    station_csv = Path(args.stations_dir) / f"{args.station}.csv"
    pred_csv = Path(args.predictions_dir) / f"{args.station}.csv"
    if not station_csv.exists():
        raise FileNotFoundError(station_csv)
    if not pred_csv.exists():
        raise FileNotFoundError(pred_csv)

    # ── Load Chronos-2 ────────────────────────────────────────────────
    logger.info(
        "Loading Chronos-2 (%s) on %s — context length = %dh, " "perturbation window = full input",
        args.model,
        args.device,
        CONTEXT_LENGTH_H,
    )
    t0 = time.time()
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.float32,
    )
    logger.info("Chronos-2 ready in %.1fs", time.time() - t0)

    # ── Load station data ─────────────────────────────────────────────
    input_history = _load_station_input(station_csv)
    forecast = _load_prediction(pred_csv)
    logger.info(
        "Station %s loaded: input %s, forecast %s",
        args.station,
        input_history.shape,
        forecast.shape,
    )

    # ── Anomaly events (needed for Type-1 localized centers) ─────────
    baseline = input_history[-PREDICTION_LENGTH_H:]
    events = detect_anomaly_events(
        forecast=forecast,
        baseline=baseline,
        kpi_names=_KPI_COLS,
    )
    anomaly_dicts = events_to_dicts(events)
    logger.info("Detected %d anomaly events", len(anomaly_dicts))

    # ── PAX-TS Algorithm 2 ────────────────────────────────────────────
    logger.info(
        "Computing PAX-TS sensitivity (chunk_size=%d) ...",
        args.chunk_size,
    )
    t1 = time.time()
    result = compute_sensitivity(
        pipeline=pipeline,
        input_kpis=input_history,
        baseline_forecast=forecast,
        anomaly_events=anomaly_dicts,
        kpi_names=_KPI_COLS,
        cap_table=KPI_NORMALIZATION_MAX,
        chunk_size=args.chunk_size,
    )
    elapsed = time.time() - t1
    logger.info(
        "PAX-TS done in %.1fs (n_perturbations=%d, batch=%s)",
        elapsed,
        result.n_perturbations,
        result.batch_strategy,
    )

    # ── Persist artefacts ─────────────────────────────────────────────
    paths = write_sensitivity_artefacts(
        out_dir=str(out_dir),
        result=result,
        kpi_names=_KPI_COLS,
        anomaly_events=anomaly_dicts,
    )
    np.savez_compressed(
        out_dir / "sensitivity_tensors.npz",
        S_global_raw=result.S_global_raw,
        S_global_norm=result.S_global_norm,
        S_local_raw=result.S_local_raw,
        S_local_norm=result.S_local_norm,
    )
    summary = {
        "station": args.station,
        "context_length_h": CONTEXT_LENGTH_H,
        "perturbation_window_h": _INPUT_WINDOW_HOURS,
        "n_perturbations": result.n_perturbations,
        "elapsed_sec_pax_ts": round(elapsed, 1),
        "batch_strategy": result.batch_strategy,
        "scales": list(result.perturbation_scales),
        "n_events": len(anomaly_dicts),
        "n_event_centers_in_window": len(result.event_centers),
        "artefacts": paths,
    }
    with (out_dir / "pax_ts_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info(
        "Wrote artefacts to %s (sensitivity_tensors.npz + CSVs + markdown)",
        out_dir,
    )


if __name__ == "__main__":
    main()
