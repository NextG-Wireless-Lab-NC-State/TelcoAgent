#!/usr/bin/env python3
"""Batch PAX-TS sensitivity computation across every station.

Loads Chronos-2 once and runs the canonical PAX-TS Algorithm 2 over
every station in ``--stations-dir``. Per station, writes a
:class:`SensitivityResult` (raw + cap-normalized global / local
matrices) and the summary JSON to ``<output>/<station>/``.

The run is resume-safe: any station whose
``<output>/<station>/pax_ts_summary.json`` already exists is skipped
(set ``--force`` to redo).

Usage
-----
    PYTHONPATH=. python scripts/run_pax_ts_batch.py \\
        --output output/pax_ts_full_<timestamp>
        [--limit 5]
        [--stations station_A_10,station_B_5]
"""

from __future__ import annotations

import os

# Disable HuggingFace transformers' optional TensorFlow path before any
# transformers/chronos import; the local TF/numpy combo would otherwise
# crash on bfloat16 ABI mismatch.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

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


def _process_station(
    station_id: str,
    station_csv: Path,
    pred_csv: Path,
    pipeline,
    out_root: Path,
    chunk_size: int,
    force: bool,
) -> Dict[str, object]:
    """PAX-TS for one station; reuses the already-loaded pipeline."""
    out_dir = out_root / station_id
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "pax_ts_summary.json"
    if summary_path.exists() and not force:
        with summary_path.open() as f:
            existing = json.load(f)
        logger.info("[%s] resume — skipping", station_id)
        return existing

    input_history = _load_station_input(station_csv)
    forecast = _load_prediction(pred_csv)

    baseline = input_history[-PREDICTION_LENGTH_H:]
    events = detect_anomaly_events(
        forecast=forecast,
        baseline=baseline,
        kpi_names=_KPI_COLS,
    )
    anomaly_dicts = events_to_dicts(events)

    t0 = time.time()
    result = compute_sensitivity(
        pipeline=pipeline,
        input_kpis=input_history,
        baseline_forecast=forecast,
        anomaly_events=anomaly_dicts,
        kpi_names=_KPI_COLS,
        cap_table=KPI_NORMALIZATION_MAX,
        chunk_size=chunk_size,
    )
    elapsed = time.time() - t0

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
    summary: Dict[str, object] = {
        "station": station_id,
        "context_length_h": CONTEXT_LENGTH_H,
        "perturbation_window_h": _INPUT_WINDOW_HOURS,
        "n_perturbations": result.n_perturbations,
        "elapsed_sec_pax_ts": round(elapsed, 2),
        "batch_strategy": result.batch_strategy,
        "scales": list(result.perturbation_scales),
        "n_events": len(anomaly_dicts),
        "n_event_centers_in_window": len(result.event_centers),
        "artefacts": paths,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info(
        "[%s] PAX-TS done in %.2fs (B=%d, events=%d)",
        station_id,
        elapsed,
        result.n_perturbations,
        len(anomaly_dicts),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run PAX-TS Algorithm 2 across every station, reusing one " "Chronos-2 pipeline."
        ),
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
        help="Output root for sensitivity artefacts.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Batch chunk size for compute_sensitivity (0 = single batch).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N stations (smoke test).",
    )
    parser.add_argument(
        "--stations",
        default=None,
        help="Comma-separated explicit station_id list (overrides --limit).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redo even when pax_ts_summary.json already exists.",
    )
    args = parser.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    stations_dir = Path(args.stations_dir)
    pred_dir = Path(args.predictions_dir)

    if args.stations:
        station_ids = [s.strip() for s in args.stations.split(",") if s.strip()]
    else:
        all_stations = sorted(p.stem for p in stations_dir.glob("station_*.csv"))
        station_ids = all_stations[: args.limit] if args.limit else all_stations
    logger.info("Will process %d stations", len(station_ids))

    # ── Load Chronos-2 once ───────────────────────────────────────────
    logger.info(
        "Loading Chronos-2 (%s) on %s — context length = %dh, "
        "perturbation window = full input (%dh)",
        args.model,
        args.device,
        CONTEXT_LENGTH_H,
        _INPUT_WINDOW_HOURS,
    )
    t_load = time.time()
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.float32,
    )
    logger.info("Chronos-2 ready in %.1fs", time.time() - t_load)

    # ── Loop ──────────────────────────────────────────────────────────
    rows: List[Dict[str, object]] = []
    started = time.time()
    for s_idx, station_id in enumerate(station_ids, 1):
        station_csv = stations_dir / f"{station_id}.csv"
        pred_csv = pred_dir / f"{station_id}.csv"
        if not station_csv.exists() or not pred_csv.exists():
            logger.warning(
                "[%s] skipping — missing input or prediction CSV",
                station_id,
            )
            continue
        try:
            summary = _process_station(
                station_id=station_id,
                station_csv=station_csv,
                pred_csv=pred_csv,
                pipeline=pipeline,
                out_root=out_root,
                chunk_size=args.chunk_size,
                force=args.force,
            )
            rows.append(summary)
        except Exception as exc:
            logger.error("[%s] FAILED: %s", station_id, exc)
            raise

        elapsed_total = time.time() - started
        avg = elapsed_total / s_idx
        eta = avg * (len(station_ids) - s_idx)
        logger.info(
            "Progress: %d/%d (avg %.1fs/station, ETA %.1f min)",
            s_idx,
            len(station_ids),
            avg,
            eta / 60.0,
        )

    aggregate_path = out_root / "pax_ts_aggregate.json"
    with aggregate_path.open("w") as f:
        json.dump(
            {
                "stations_dir": str(stations_dir),
                "predictions_dir": str(pred_dir),
                "n_stations": len(station_ids),
                "context_length_h": CONTEXT_LENGTH_H,
                "perturbation_window_h": _INPUT_WINDOW_HOURS,
                "rows": rows,
                "total_elapsed_sec": round(time.time() - started, 1),
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    logger.info("Aggregate written to %s", aggregate_path)


if __name__ == "__main__":
    main()
