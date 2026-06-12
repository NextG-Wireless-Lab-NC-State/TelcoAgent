#!/usr/bin/env python3
"""Plain-LLM explainer baseline — −Explainer ablation arm.

Generates a KPI-forecast explanation using a **single LLM call** with no
ReAct loop, no tool use, and no KG context.  Anomaly detection reuses the
identical ``detect_anomaly_events`` function used by the full explainer so
the comparison is fair (same events, different explainer).

Output layout per the §1.2 interface spec::

    <output>/<station_id>/minus_explainer/
    ├── explanation.md    # judge-parseable 5-section report
    └── result.json       # §1.3 schema (condition=minus_explainer,
                          #   retrieved_contexts=[], ablate_kg=True)

Usage
-----
    PYTHONPATH=. python scripts/ablation/plain_llm_explainer.py \\
        --predictions-dir output/amazon_chronos-2_h7d_ctx_sweep/ctx_672h/csv \\
        --stations-dir    data/station \\
        --output          output/ablation/component \\
        [--stations       station_A_10,station_B_5] \\
        [--limit          3] \\
        [--explainer-model openai/gpt-4o-mini] \\
        [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv  # noqa: E402

# Load API keys from .env (OPENAI_API_KEY etc.) — the ReAct explainer path
# pulls this in transitively, but this single-LLM-call baseline does not.
load_dotenv()

from telcoagent.config import CORE_KPI_NAMES, PREDICTION_LENGTH_H  # noqa: E402
from telcoagent.explainer.anomaly_detector import (  # noqa: E402
    AnomalyEvent,
    detect_anomaly_events,
)
from telcoagent.llm.api import invoke_llm_text  # noqa: E402

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────

_INPUT_WINDOW_HOURS: int = 1944
_FORECAST_HORIZON_H: int = PREDICTION_LENGTH_H
_KPI_COLS: List[str] = list(CORE_KPI_NAMES)
_CONDITION: str = "minus_explainer"

#: Raw EMS column names in data/station CSVs — 1:1 order with _KPI_COLS.
_STATION_KPI_COLS_CSV: List[str] = [
    "RRC_Conn_Count_Avg",
    "DL_CQI",
    "DL_iBler",
    "DL_rBler",
    "MAC DL Eff TP",
    "DL PRB Utilization",
    "DL_IP_Throughput",
]


# ─── IO helpers ───────────────────────────────────────────────────────────


def _load_station_input(
    station_csv: Path,
) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
    df = pd.read_csv(station_csv)
    arr = df[_STATION_KPI_COLS_CSV].values.astype(np.float64)
    if arr.shape[0] < _INPUT_WINDOW_HOURS:
        raise ValueError(
            f"{station_csv.name}: only {arr.shape[0]} rows; " f"need ≥ {_INPUT_WINDOW_HOURS}"
        )
    input_window = arr[:_INPUT_WINDOW_HOURS]
    lat = float(df["latitude"].iloc[0]) if "latitude" in df.columns else None
    lon = float(df["longitude"].iloc[0]) if "longitude" in df.columns else None
    return input_window, lat, lon


def _load_prediction_csv(pred_csv: Path) -> np.ndarray:
    """Return the 168-hour forecast array (168, 7)."""
    df = pd.read_csv(pred_csv)
    if len(df) != _FORECAST_HORIZON_H:
        raise ValueError(f"{pred_csv.name}: expected {_FORECAST_HORIZON_H} rows, got {len(df)}")
    forecast = np.zeros((_FORECAST_HORIZON_H, len(_KPI_COLS)))
    for i, kpi in enumerate(_KPI_COLS):
        forecast[:, i] = df[f"{kpi}_pred"].values.astype(float)
    return forecast


# ─── Prompt builder ───────────────────────────────────────────────────────


def _build_messages(
    station_id: str,
    forecast: np.ndarray,
    events: List[AnomalyEvent],
) -> List[Dict[str, str]]:
    """Build the single-turn chat messages for the plain-LLM explainer.

    The prompt contains only raw numerical summaries derived from the
    forecast and the anomaly event list.  No domain knowledge, causal
    chains, or KG content is embedded — the model must reason from
    numbers alone.
    """
    # §1 forecast table — min/mean/max per KPI
    kpi_rows: List[str] = []
    for i, kpi in enumerate(_KPI_COLS):
        col = forecast[:, i]
        kpi_rows.append(f"  {kpi}: min={col.min():.2f}, mean={col.mean():.2f}, max={col.max():.2f}")
    forecast_summary = "\n".join(kpi_rows)

    # anomaly event list
    if events:
        event_lines = []
        for ev in events:
            kpis = ", ".join(ev.co_occurring_kpis) if ev.co_occurring_kpis else ev.kpi
            event_lines.append(
                f"  - type={ev.type}, kpi={kpis}, "
                f"hours={ev.t_start}-{ev.t_end}, "
                f"z_score={ev.z_score:.2f}, direction={ev.direction}"
            )
        events_text = "\n".join(event_lines)
    else:
        events_text = "  (none)"

    system_content = (
        "You are a network analyst writing a structured forecast explanation report. "
        "Your report must be based solely on the numerical data provided. "
        "Do not invent measurements or reference external documentation. "
        "Format the report in Markdown with the exact section headings and list formats shown below. "
        "Follow the formatting rules precisely — they are required by the downstream evaluation pipeline."
    )

    user_content = f"""\
Station: {station_id}

7-day KPI forecast (168 hours):
{forecast_summary}

Detected anomaly events:
{events_text}

Write a structured explanation report with these exact sections:

## §0 Executive Summary
(2-4 bullet points summarising key findings)

### §1 Forecast Summary
(Markdown table: KPI | Trend | Notes)

### §2 Cross-KPI Couplings
(Observed co-movements from the data above; if none, write "No couplings observed.")

### §3 Spatial & Traffic Context
(General notes; no location data available — write "Location data not available.")

## §4 Anomaly Root-Cause Analysis
For each detected anomaly event write ONE bullet point in this exact format:
  - <concise description of the anomaly, its KPI, and inferred cause> [Anomaly]
If there are no anomaly events write exactly: "No anomaly events detected."

### §5 Actionable Recommendations
Markdown table with columns in this order: Action | Priority | #
(Action text in the first column; one row per recommendation)
"""

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ─── Core runner ─────────────────────────────────────────────────────────


def run_plain_llm_explain(
    station_id: str,
    forecast: np.ndarray,
    input_history: np.ndarray,
    output_dir: Path,
    explainer_model: str = "openai/gpt-4o-mini",
    force: bool = False,
) -> Dict[str, object]:
    """Generate and persist one plain-LLM explanation for one station.

    This is the public entry point used by tests and the CLI main().
    All LLM calls go through ``invoke_llm_text``; no direct
    ``litellm.completion`` call is made here.

    Parameters
    ----------
    station_id
        Station identifier (e.g. ``"station_A_10"``).
    forecast
        Chronos-2 predicted KPI series, shape ``(168, 7)``.
    input_history
        Observed input window, shape ``(T_baseline, 7)``.
    output_dir
        Root output directory; outputs land under
        ``output_dir / station_id / "minus_explainer" /``.
    explainer_model
        LiteLLM model string.
    force
        When True, re-run even if outputs already exist.

    Returns
    -------
    dict
        The §1.3 result.json payload.
    """
    cond_dir = Path(output_dir) / station_id / _CONDITION
    cond_dir.mkdir(parents=True, exist_ok=True)
    report_path = cond_dir / "explanation.md"
    result_path = cond_dir / "result.json"

    if report_path.exists() and result_path.exists() and not force:
        logger.info("[%s/%s] resume — skipping (already exists)", station_id, _CONDITION)
        with result_path.open() as fh:
            return json.load(fh)

    # Anomaly detection — identical function to the full explainer
    baseline = input_history[-_FORECAST_HORIZON_H:]
    events: List[AnomalyEvent] = detect_anomaly_events(
        forecast=forecast,
        baseline=baseline,
        kpi_names=_KPI_COLS,
    )

    messages = _build_messages(station_id, forecast, events)

    t0 = time.time()
    report_text = invoke_llm_text(
        model=explainer_model,
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
    )
    elapsed = time.time() - t0

    report_path.write_text(report_text, encoding="utf-8")

    payload: Dict[str, object] = {
        "station_id": station_id,
        "condition": _CONDITION,
        "ablate_kg": True,
        "explainer_model": explainer_model,
        "elapsed_sec": round(elapsed, 2),
        "n_anomaly_events": len(events),
        "report_path": str(report_path),
        "retrieved_contexts": [],
        "report_chars": len(report_text),
    }
    with result_path.open("w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    logger.info(
        "[%s/%s] done in %.1fs, %d anomaly events, %d chars",
        station_id,
        _CONDITION,
        elapsed,
        len(events),
        len(report_text),
    )
    return payload


# ─── CLI ─────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Plain-LLM explainer baseline (−Explainer ablation arm). "
            "Single LLM call per station; no ReAct, no tools, no KG."
        )
    )
    parser.add_argument(
        "--predictions-dir",
        default="output/amazon_chronos-2_h7d_ctx_sweep/ctx_672h/csv",
        help="Directory with cached Chronos-2 forecast CSVs.",
    )
    parser.add_argument(
        "--stations-dir",
        default="data/station",
        help="Directory with 81-day input-window CSVs.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Root output directory for this ablation arm.",
    )
    parser.add_argument(
        "--explainer-model",
        default=os.environ.get("EXPLAINER_MODEL", "openai/gpt-4o-mini"),
        help="LiteLLM model string (default: openai/gpt-4o-mini).",
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
        help="Comma-separated station IDs (overrides --limit).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when outputs already exist.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    stations_dir = Path(args.stations_dir)
    pred_dir = Path(args.predictions_dir)

    if args.stations:
        station_ids = [s.strip() for s in args.stations.split(",") if s.strip()]
    else:
        all_stations = sorted(p.stem for p in stations_dir.glob("station_*.csv"))
        station_ids = all_stations[: args.limit] if args.limit else all_stations

    logger.info("Will process %d stations", len(station_ids))

    summary_rows: List[Dict[str, object]] = []
    started = time.time()

    for s_idx, station_id in enumerate(station_ids, 1):
        station_csv = stations_dir / f"{station_id}.csv"
        pred_csv = pred_dir / f"{station_id}.csv"
        if not station_csv.exists() or not pred_csv.exists():
            logger.warning("[%s] skipping — missing input or prediction CSV", station_id)
            continue

        input_history, _lat, _lon = _load_station_input(station_csv)
        forecast = _load_prediction_csv(pred_csv)

        try:
            payload = run_plain_llm_explain(
                station_id=station_id,
                forecast=forecast,
                input_history=input_history,
                output_dir=out_dir,
                explainer_model=args.explainer_model,
                force=args.force,
            )
            summary_rows.append(payload)
        except Exception as exc:
            logger.error("[%s] FAILED: %s", station_id, exc)
            raise

        elapsed_total = time.time() - started
        avg = elapsed_total / s_idx
        eta = avg * (len(station_ids) - s_idx)
        logger.info(
            "Progress: %d/%d stations (avg %.1fs/station, ETA %.1f min)",
            s_idx,
            len(station_ids),
            avg,
            eta / 60.0,
        )

    summary_path = out_dir / "ablation_summary_minus_explainer.json"
    with summary_path.open("w") as fh:
        json.dump(
            {
                "condition": _CONDITION,
                "predictions_dir": str(pred_dir),
                "stations_dir": str(stations_dir),
                "n_stations": len(station_ids),
                "explainer_model": args.explainer_model,
                "rows": summary_rows,
                "total_elapsed_sec": round(time.time() - started, 1),
            },
            fh,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
