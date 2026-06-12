#!/usr/bin/env python3
"""−Predictor ablation runner: seasonal-naive forecast → KG-grounded explainer.

Replaces the Chronos-2 prediction with a seasonal-naive(s=24) forecast while
keeping the KG-grounded ReAct explainer intact. This isolates the contribution
of the predictor component within the full TelcoAgent pipeline.

Naive forecast construction (mirrors ``foundation_utils.py:306-325``):
    last_season = input_history[-24:]
    naive_pred = np.tile(last_season, (ceil(168/24), 1))[:168]

Output layout per station (interface spec §1.2):
    <output>/<station_id>/minus_predictor/
        explanation.md
        result.json          # §1.3 schema, condition="minus_predictor"
        anomaly_events.csv

The runner is resume-safe: if both explanation.md and result.json already
exist for a station, the station is skipped unless --force is passed.

Usage
-----
    PYTHONPATH=. python scripts/ablation/run_naive_predictor_explain.py \\
        --kg-path output/kg_sweep/theta_0.90/enriched_kg.json \\
        --stations-dir data/station \\
        --output output/ablation/component \\
        [--limit 1]                          # smoke test on first station
        [--stations station_A_10,station_B_5]
        [--force]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from telcoagent.cli_utils import bootstrap

logger = bootstrap(suppress_tf_warnings=True, logger_name=__name__)

from telcoagent.config import (  # noqa: E402
    CORE_KPI_NAMES,
    PREDICTION_LENGTH_H,
)
from telcoagent.explainer import TelcoAgentExplainer  # noqa: E402
from telcoagent.ontology.core import get_default_ontology  # noqa: E402

# ─── Constants ────────────────────────────────────────────────────────────

#: Canonical 3GPP TS 28.554 KPI names; order is fixed by the interface spec §1.1.
_KPI_COLS: List[str] = list(CORE_KPI_NAMES)

#: 81-day input window matching TelcoAgent's standard split (days 1-81, 1944 h).
_INPUT_WINDOW_HOURS: int = 1944

#: 7-day forecast horizon.
_FORECAST_HORIZON_H: int = PREDICTION_LENGTH_H

#: Seasonal period for the naive predictor (24 h = one diurnal cycle).
_SEASON: int = 24

#: Raw EMS column names in data/station CSVs; aligned 1:1 with ``_KPI_COLS``.
_STATION_KPI_COLS_CSV: List[str] = [
    "RRC_Conn_Count_Avg",
    "DL_CQI",
    "DL_iBler",
    "DL_rBler",
    "MAC DL Eff TP",
    "DL PRB Utilization",
    "DL_IP_Throughput",
]

#: Ablation condition label written into result.json and the output subdirectory.
_CONDITION: str = "minus_predictor"


# ─── In-memory ontology store (shared, no Neo4j) ──────────────────────────

from telcoagent.stores.in_memory_ontology_store import (  # noqa: E402
    InMemoryOntologyStore as _InMemoryOntologyStore,
)

# ─── Public helpers ───────────────────────────────────────────────────────


def naive_forecast_from_history(
    input_history: np.ndarray,
    season: int = _SEASON,
    horizon: int = _FORECAST_HORIZON_H,
) -> np.ndarray:
    """Build a seasonal-naive(s=``season``) forecast of length ``horizon``.

    Tiles the last ``season`` rows of ``input_history`` across the requested
    horizon. Mirrors the logic in ``scripts/baselines/foundation_utils.py:306-325``.

    Parameters
    ----------
    input_history:
        Observed input window of shape ``(T, C)``.  Must have ``T >= season``.
    season:
        Seasonal period in hours (default 24).
    horizon:
        Forecast horizon in hours (default 168).

    Returns
    -------
    np.ndarray
        Shape ``(horizon, C)``, dtype float64.
    """
    last_season = input_history[-season:]  # (season, C)
    n_repeats = math.ceil(horizon / season)
    tiled = np.tile(last_season, (n_repeats, 1))  # (n_repeats*season, C)
    return tiled[:horizon].astype(np.float64)


# ─── IO helpers ───────────────────────────────────────────────────────────


def _load_station_input(
    station_csv: Path,
) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
    """Load the 81-day input window and optional station coordinates.

    Column mapping: EMS names → canonical KPI order via ``_STATION_KPI_COLS_CSV``.
    """
    df = pd.read_csv(station_csv)
    arr = df[_STATION_KPI_COLS_CSV].values.astype(np.float64)
    if arr.shape[0] < _INPUT_WINDOW_HOURS:
        raise ValueError(
            f"{station_csv.name}: only {arr.shape[0]} rows; " f"need >= {_INPUT_WINDOW_HOURS}"
        )
    input_window = arr[:_INPUT_WINDOW_HOURS]
    lat = float(df["latitude"].iloc[0]) if "latitude" in df.columns else None
    lon = float(df["longitude"].iloc[0]) if "longitude" in df.columns else None
    return input_window, lat, lon


# ─── Core per-station runner ──────────────────────────────────────────────


def _run_one(
    station_id: str,
    input_history: np.ndarray,
    ontology_store: object,
    osm: object,
    out_dir: Path,
    explainer_model: str,
    force: bool,
) -> Dict[str, object]:
    """Generate and persist one −Predictor explanation for ``station_id``.

    The naive forecast is built from ``input_history`` and fed into the
    KG-grounded explainer (``ablate_kg=False``).  Output is written under
    ``out_dir/<station_id>/minus_predictor/``.
    """
    cond_dir = out_dir / station_id / _CONDITION
    cond_dir.mkdir(parents=True, exist_ok=True)
    report_path = cond_dir / "explanation.md"
    result_path = cond_dir / "result.json"

    # Resume-safe: skip if both artefacts already present.
    if report_path.exists() and result_path.exists() and not force:
        logger.info(
            "[%s/%s] resume — skipping (already exists)",
            station_id,
            _CONDITION,
        )
        with result_path.open() as f:
            return json.load(f)

    # Build the seasonal-naive forecast (shape (168, 7)).
    naive_pred = naive_forecast_from_history(
        input_history, season=_SEASON, horizon=_FORECAST_HORIZON_H
    )
    pred_dict: Dict[str, List[float]] = {
        kpi: naive_pred[:, i].tolist() for i, kpi in enumerate(_KPI_COLS)
    }

    explainer = TelcoAgentExplainer(
        ontology_store=ontology_store,
        osm=osm,
        model_name=explainer_model,
        station_id=station_id,
        relevance_model=None,  # skip OSM-relevance harness
        sensitivity_pipeline=None,  # PAX-TS deferred
        sensitivity_result=None,
    )

    t0 = time.time()
    result = explainer.explain(
        forecast=naive_pred,
        input_history=input_history,
        prediction=pred_dict,
        output_dir=str(cond_dir),
        lat=None,
        lon=None,
        forecast_metrics=None,
        ablate_kg=False,  # KG-grounded explainer is kept
    )
    elapsed = time.time() - t0

    payload: Dict[str, object] = {
        "station_id": station_id,
        "condition": _CONDITION,
        "ablate_kg": False,
        "explainer_model": explainer_model,
        "elapsed_sec": round(elapsed, 2),
        "n_anomaly_events": len(result.anomaly_events),
        "report_path": str(report_path),
        "retrieved_contexts": result.retrieved_contexts,
        "report_chars": len(result.report),
    }
    with result_path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    logger.info(
        "[%s/%s] done in %.1fs, %d anomaly events, %d chars",
        station_id,
        _CONDITION,
        elapsed,
        len(result.anomaly_events),
        len(result.report),
    )
    return payload


# ─── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "−Predictor ablation: replace Chronos-2 with seasonal-naive(s=24) "
            "forecast and run the KG-grounded explainer. "
            "One of the four arms in the component leave-one-out ablation."
        ),
    )
    parser.add_argument(
        "--kg-path",
        default="output/kg_sweep/theta_0.90/enriched_kg.json",
        help="Path to the spec-extracted enriched_kg.json.",
    )
    parser.add_argument(
        "--stations-dir",
        default="data/station",
        help="Directory with the 81-day input-window CSVs.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output root for the ablation run.",
    )
    parser.add_argument(
        "--explainer-model",
        default=os.environ.get("EXPLAINER_MODEL", "openai/gpt-4o-mini"),
        help="LLM model for the ExplainerAgent (default: openai/gpt-4o-mini).",
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
        help="Redo even when explanation.md + result.json already exist.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load KG.
    kg_path = Path(args.kg_path).resolve()
    if not kg_path.exists():
        raise FileNotFoundError(f"KG file not found: {kg_path}")
    os.environ["TELCOAGENT_KG_PATH"] = str(kg_path)
    ontology = get_default_ontology()
    logger.info(
        "Loaded KG from %s (%d KPIs, %d directional rules)",
        kg_path,
        len(getattr(ontology, "kpis", {}) or {}),
        len(getattr(ontology, "directional_rules", []) or []),
    )
    ontology_store = _InMemoryOntologyStore(ontology)

    # Resolve station list.
    stations_dir = Path(args.stations_dir)
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
        if not station_csv.exists():
            logger.warning("[%s] skipping — missing station CSV", station_id)
            continue

        try:
            input_history, lat, lon = _load_station_input(station_csv)
            payload = _run_one(
                station_id=station_id,
                input_history=input_history,
                ontology_store=ontology_store,
                osm=None,  # TelcoAgentExplainer uses default OpenStreetMapMCP
                out_dir=out_dir,
                explainer_model=args.explainer_model,
                force=args.force,
            )
            summary_rows.append(payload)
        except Exception as exc:  # surface, never silent
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

    summary_path = out_dir / "ablation_summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "condition": _CONDITION,
                "kg_path": str(kg_path),
                "stations_dir": str(stations_dir),
                "n_stations": len(station_ids),
                "explainer_model": args.explainer_model,
                "rows": summary_rows,
                "total_elapsed_sec": round(time.time() - started, 1),
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
