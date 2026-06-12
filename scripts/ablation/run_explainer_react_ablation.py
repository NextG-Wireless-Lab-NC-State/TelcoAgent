#!/usr/bin/env python3
"""Q3 ReAct on/off ablation runner: ReAct loop toggled, KG fixed on.

Answers the third ablation question — *does the explainer's ReAct loop have
a real effect?* — by running the KG-grounded explainer over every station
under two arms that differ ONLY in the ReAct knob:

    react_on   explain(ablate_react=False, ablate_kg=False)
    react_off  explain(ablate_react=True,  ablate_kg=False)

``ablate_kg=False`` is fixed on BOTH arms — the KG context, candidate
causes, and GraphRAG evidence are identical; only the report-authoring
dispatch changes. ``ablate_react=True`` routes every authoring turn
through ``single_shot_generate`` (tool-less); ``ablate_react=False`` routes
through the ReAct loop with the 5-tool registry. This cleanly isolates the
ReAct loop from the KG, unlike the leave-one-out −Explainer arm (plain-LLM:
ReAct + tools + KG removed together).

Both arms explain the SAME anomaly events derived from the SAME cached
Chronos-2 forecast, so the comparison is fair.

Output layout per station × arm (interface spec §1.2):
    <output>/<station_id>/<react_on|react_off>/
        explanation.md
        result.json          # §1.3 schema, condition="react_on"|"react_off"
        anomaly_events.csv

The runner is resume-safe: if both explanation.md and result.json already
exist for a station/arm, that arm is skipped unless --force is passed.

Usage
-----
    PYTHONPATH=. python scripts/ablation/run_explainer_react_ablation.py \\
        --kg-path output/kg_sweep/theta_0.90/enriched_kg.json \\
        --predictions-dir output/amazon_chronos-2_h7d_ctx_sweep/ctx_672h/csv \\
        --stations-dir data/station \\
        --output output/ablation/three_axis/q3_react \\
        [--limit 1]                          # smoke test on first station
        [--stations station_A_5]             # explicit station_-prefixed stems
        [--arms react_on,react_off]
        [--force]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv  # noqa: E402

# Load gpt-4o-mini (OPENAI_API_KEY) before any LLM call. The prior
# plain_llm_explainer arm failed because it omitted this — keep it here.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from telcoagent.cli_utils import bootstrap  # noqa: E402

logger = bootstrap(suppress_tf_warnings=True, logger_name=__name__)

from telcoagent.config import (  # noqa: E402
    CORE_KPI_NAMES,
    PREDICTION_LENGTH_H,
)
from telcoagent.explainer import TelcoAgentExplainer  # noqa: E402
from telcoagent.ontology.core import get_default_ontology  # noqa: E402

# ─── Constants ────────────────────────────────────────────────────────────

#: Canonical 3GPP TS 28.554 KPI names; order is fixed by interface spec §1.1.
_KPI_COLS: List[str] = list(CORE_KPI_NAMES)

#: 81-day input window matching TelcoAgent's standard split (days 1-81, 1944 h).
_INPUT_WINDOW_HOURS: int = 1944

#: 7-day forecast horizon.
_FORECAST_HORIZON_H: int = PREDICTION_LENGTH_H

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

#: The two ReAct-axis arms. ablate_kg is False on both — only ablate_react
#: differs.
_ARMS: Dict[str, bool] = {
    "react_on": False,  # ablate_react=False → ReAct loop
    "react_off": True,  # ablate_react=True  → single-shot dispatch
}


# ─── In-memory ontology store (shared, no Neo4j) ──────────────────────────

from telcoagent.stores.in_memory_ontology_store import (  # noqa: E402
    InMemoryOntologyStore as _InMemoryOntologyStore,
)

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


def _load_prediction_csv(
    pred_csv: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[float]]]:
    """Load the 168-hour Chronos-2 forecast + held-out ground truth."""
    df = pd.read_csv(pred_csv)
    if len(df) != _FORECAST_HORIZON_H:
        raise ValueError(
            f"{pred_csv.name}: expected {_FORECAST_HORIZON_H} forecast rows, " f"got {len(df)}"
        )
    forecast = np.zeros((_FORECAST_HORIZON_H, len(_KPI_COLS)))
    target = np.zeros_like(forecast)
    pred_dict: Dict[str, List[float]] = {}
    for i, kpi in enumerate(_KPI_COLS):
        forecast[:, i] = df[f"{kpi}_pred"].values.astype(float)
        target[:, i] = df[f"{kpi}_true"].values.astype(float)
        pred_dict[kpi] = forecast[:, i].tolist()
    return forecast, target, pred_dict


def _compute_per_kpi_metrics(
    forecast: np.ndarray,
    target: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Per-KPI sMAPE / MASE / MAE / RMSE for the §1 forecast table."""
    metrics: Dict[str, Dict[str, float]] = {}
    eps = 1e-8
    for i, kpi in enumerate(_KPI_COLS):
        y = target[:, i]
        f = forecast[:, i]
        err = np.abs(y - f)
        mae = float(err.mean())
        rmse = float(np.sqrt(((y - f) ** 2).mean()))
        smape = float((200.0 * err / (np.abs(y) + np.abs(f) + eps)).mean())
        if len(y) > 24:
            seasonal_diff = np.abs(y[24:] - y[:-24]).mean() + eps
        else:
            seasonal_diff = np.abs(np.diff(y)).mean() + eps
        mase = float(mae / seasonal_diff)
        metrics[kpi] = {"MAE": mae, "RMSE": rmse, "sMAPE": smape, "MASE": mase}
    return metrics


# ─── Core per-station × arm runner ────────────────────────────────────────


def _run_one(
    station_id: str,
    input_history: np.ndarray,
    forecast: np.ndarray,
    ontology_store: object,
    osm: object,
    out_dir: Path,
    arm: str,
    explainer_model: str,
    force: bool,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    forecast_metrics: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, object]:
    """Generate and persist one ReAct-ablation explanation for ``station_id``.

    ``arm`` selects the ReAct knob: ``react_on`` keeps the ReAct loop,
    ``react_off`` routes report authoring through ``single_shot_generate``.
    ``ablate_kg=False`` is fixed on both arms — the KG-grounded explainer is
    always kept. Output is written under ``out_dir/<station_id>/<arm>/``.
    """
    if arm not in _ARMS:
        raise ValueError(f"unknown arm {arm!r}; must be one of {sorted(_ARMS)}")
    ablate_react = _ARMS[arm]

    arm_dir = out_dir / station_id / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    report_path = arm_dir / "explanation.md"
    result_path = arm_dir / "result.json"

    # Resume-safe: skip if both artefacts already present.
    if report_path.exists() and result_path.exists() and not force:
        logger.info("[%s/%s] resume — skipping (already exists)", station_id, arm)
        with result_path.open() as f:
            return json.load(f)

    pred_dict: Dict[str, List[float]] = {
        kpi: forecast[:, i].tolist() for i, kpi in enumerate(_KPI_COLS)
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
        forecast=forecast,
        input_history=input_history,
        prediction=pred_dict,
        output_dir=str(arm_dir),
        lat=lat,
        lon=lon,
        forecast_metrics=forecast_metrics,
        ablate_kg=False,  # KG-grounded explainer is kept on BOTH arms
        ablate_react=ablate_react,  # the only knob that differs
    )
    elapsed = time.time() - t0

    payload: Dict[str, object] = {
        "station_id": station_id,
        "condition": arm,
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
        arm,
        elapsed,
        len(result.anomaly_events),
        len(result.report),
    )
    return payload


# ─── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Q3 ReAct on/off ablation: toggle the explainer's ReAct loop "
            "(react_on vs react_off) with the KG fixed on for both arms. "
            "The judge step runs separately afterwards."
        ),
    )
    parser.add_argument(
        "--kg-path",
        default="output/kg_sweep/theta_0.90/enriched_kg.json",
        help="Path to the spec-extracted enriched_kg.json.",
    )
    parser.add_argument(
        "--predictions-dir",
        default="output/amazon_chronos-2_h7d_ctx_sweep/ctx_672h/csv",
        help="Directory with cached Chronos-2 forecast CSVs.",
    )
    parser.add_argument(
        "--stations-dir",
        default="data/station",
        help="Directory with the 81-day input-window CSVs.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output root for the ReAct-ablation run.",
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
        help="Comma-separated explicit station_id list (station_-prefixed stems).",
    )
    parser.add_argument(
        "--arms",
        default="react_on,react_off",
        help="Comma-separated subset of {react_on, react_off}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redo even when explanation.md + result.json already exist.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load KG (honoured via TELCOAGENT_KG_PATH before the ontology singleton).
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

    # Resolve station + arm lists.
    stations_dir = Path(args.stations_dir)
    pred_dir = Path(args.predictions_dir)
    if args.stations:
        station_ids = [s.strip() for s in args.stations.split(",") if s.strip()]
    else:
        all_stations = sorted(p.stem for p in stations_dir.glob("station_*.csv"))
        station_ids = all_stations[: args.limit] if args.limit else all_stations
    logger.info("Will process %d stations", len(station_ids))

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in _ARMS:
            raise ValueError(f"unknown arm {a!r}; must be react_on or react_off")

    summary_rows: List[Dict[str, object]] = []
    started = time.time()

    for s_idx, station_id in enumerate(station_ids, 1):
        station_csv = stations_dir / f"{station_id}.csv"
        pred_csv = pred_dir / f"{station_id}.csv"
        if not station_csv.exists() or not pred_csv.exists():
            logger.warning("[%s] skipping — missing input or prediction CSV", station_id)
            continue

        try:
            input_history, lat, lon = _load_station_input(station_csv)
            forecast, target, _ = _load_prediction_csv(pred_csv)
            forecast_metrics = _compute_per_kpi_metrics(forecast, target)
        except Exception as exc:  # surface, never silent
            logger.error("[%s] FAILED to load inputs: %s", station_id, exc)
            raise

        for arm in arms:
            try:
                payload = _run_one(
                    station_id=station_id,
                    input_history=input_history,
                    forecast=forecast,
                    ontology_store=ontology_store,
                    osm=None,  # TelcoAgentExplainer uses default OpenStreetMapMCP
                    out_dir=out_dir,
                    arm=arm,
                    explainer_model=args.explainer_model,
                    force=args.force,
                    lat=lat,
                    lon=lon,
                    forecast_metrics=forecast_metrics,
                )
                summary_rows.append(payload)
            except Exception as exc:  # surface, never silent
                logger.error("[%s/%s] FAILED: %s", station_id, arm, exc)
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

    summary_path = out_dir / "react_ablation_summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "kg_path": str(kg_path),
                "predictions_dir": str(pred_dir),
                "stations_dir": str(stations_dir),
                "n_stations": len(station_ids),
                "arms": arms,
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
