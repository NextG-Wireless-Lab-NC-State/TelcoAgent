#!/usr/bin/env python3
"""Batch explainer driver for the with-KG / no-KG ablation across all stations.

Distinct from the legacy single-station grid runner
``scripts/run_explainer_kg_ablation.py`` (which sweeps the full
KG × ReAct 2×2 grid on one station). This script generates explanations
for **every** station under two conditions only — ``kg_on`` and
``kg_off`` — with PAX-TS sensitivity deferred (sensitivity_pipeline=None)
and no in-line judge. The judge step (Faithfulness + Answer Relevancy
via ORANSight) is invoked separately by ``scripts/run_explainer_kg_judge.py``
after this script finishes.

Pipeline (per station × condition):
    1. Load the cached Chronos-2 prediction CSV (forecast vs ground
       truth on the held-out 7-day window).
    2. Load the station's input window from data/station/<station>.csv
       (TelcoAgent's standard 1944-hour input).
    3. Run TelcoAgentExplainer.explain() with
        - sensitivity_pipeline=None  (PAX-TS deferred)
        - relevance_model=None       (skip OSM-relevance harness)
        - ablate_kg=False / True     (the two ablation arms)
       The explainer model is GPT-4o-mini per the project default.
    4. Save explanation.md, retrieved_contexts (in result.json),
       anomaly_events.csv, and result.json under
       ``<output>/<station>/{kg_on|kg_off}/``.

The runner is resume-safe: a station/condition output directory
already containing both explanation.md and result.json is skipped
(set ``--force`` to redo).

Usage
-----
    PYTHONPATH=. python scripts/run_explainer_kg_batch.py \\
        --kg-path output/kg_sweep/theta_0.90/enriched_kg.json \\
        --predictions-dir output/amazon_chronos-2_h7d_ctx_sweep/ctx_672h/csv \\
        --stations-dir data/station \\
        --output output/kg_ablation_<timestamp>
        [--limit 3]              # smoke test on first 3 stations
        [--stations station_A_10,station_B_5]   # explicit list
        [--conditions kg_on,kg_off]
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telcoagent.cli_utils import bootstrap

logger = bootstrap(suppress_tf_warnings=True, logger_name=__name__)

from telcoagent.config import (  # noqa: E402
    CORE_KPI_NAMES,
    PREDICTION_LENGTH_H,
)
from telcoagent.explainer import TelcoAgentExplainer  # noqa: E402
from telcoagent.explainer.pax_ts import SensitivityResult  # noqa: E402
from telcoagent.ontology.core import get_default_ontology  # noqa: E402

# ─── Constants ────────────────────────────────────────────────────────────

#: 81-day input window matching TelcoAgent's standard split.
_INPUT_WINDOW_HOURS: int = 1944

#: 7-day forecast horizon from telcoagent.config.PREDICTION_LENGTH_H.
_FORECAST_HORIZON_H: int = PREDICTION_LENGTH_H

#: Canonical 3GPP TS 28.554 KPI names (in order).
_KPI_COLS: List[str] = list(CORE_KPI_NAMES)

#: Raw column names used in ``data/station/<station>.csv`` — these
#: differ from the canonical names because the CSVs preserve the
#: operator's element-management-system schema. Order matches
#: ``_KPI_COLS`` 1:1.
_STATION_KPI_COLS_CSV: List[str] = [
    "RRC_Conn_Count_Avg",
    "DL_CQI",
    "DL_iBler",
    "DL_rBler",
    "MAC DL Eff TP",
    "DL PRB Utilization",
    "DL_IP_Throughput",
]


# ─── In-memory ontology store (shared, no Neo4j required) ─────────────────

from telcoagent.stores.in_memory_ontology_store import (  # noqa: E402
    InMemoryOntologyStore as _InMemoryOntologyStore,
)

# ─── IO helpers ───────────────────────────────────────────────────────────


def _load_station_input(
    station_csv: Path,
) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
    """Load the 81-day input window plus station coordinates.

    Station CSVs use the operator's EMS column names (e.g.
    ``RRC_Conn_Count_Avg``, ``MAC DL Eff TP``); we map them to the
    canonical 3GPP TS 28.554 ordering by selecting via
    :data:`_STATION_KPI_COLS_CSV` and trusting the 1:1 ordering.
    """
    df = pd.read_csv(station_csv)
    arr = df[_STATION_KPI_COLS_CSV].values.astype(np.float64)
    if arr.shape[0] < _INPUT_WINDOW_HOURS + _FORECAST_HORIZON_H:
        raise ValueError(
            f"{station_csv.name}: only {arr.shape[0]} rows; "
            f"need ≥ {_INPUT_WINDOW_HOURS + _FORECAST_HORIZON_H}"
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


# ─── PAX-TS sensitivity loader (offline-cached results) ──────────────────


def _load_sensitivity_result(
    station_id: str,
    pax_ts_dir: Path,
) -> Optional[SensitivityResult]:
    """Load a previously computed PAX-TS :class:`SensitivityResult`.

    The PAX-TS batch runner (``scripts/run_pax_ts_batch.py``) writes
    ``<pax_ts_dir>/<station_id>/sensitivity_tensors.npz`` plus a
    ``pax_ts_summary.json``. We materialize a SensitivityResult from
    those two files so the ExplainerToolsConfig can attach the
    sensitivity tensor and the ``query_sensitivity_matrix`` /
    ``query_localized_sensitivity`` ReAct tools surface real
    PAX-TS-derived numbers (instead of returning their "no sensitivity
    result attached" error payload, which forces the LLM to fall back
    on KG mechanism descriptions and risks hallucinated couplings).

    Returns None when the per-station directory is missing — caller
    decides whether to skip or to fail loudly.
    """
    npz_path = pax_ts_dir / station_id / "sensitivity_tensors.npz"
    summary_path = pax_ts_dir / station_id / "pax_ts_summary.json"
    if not (npz_path.exists() and summary_path.exists()):
        return None
    arrays = np.load(npz_path)
    with summary_path.open() as f:
        summary = json.load(f)
    return SensitivityResult(
        S_global_raw=arrays["S_global_raw"],
        S_global_norm=arrays["S_global_norm"],
        S_local_raw=arrays["S_local_raw"],
        S_local_norm=arrays["S_local_norm"],
        perturbation_scales=tuple(summary.get("scales", ())),
        event_centers=[],
        n_perturbations=int(summary.get("n_perturbations", 0)),
        elapsed_sec=float(summary.get("elapsed_sec_pax_ts", 0.0)),
        batch_strategy=str(summary.get("batch_strategy", "joint")),
    )


# ─── Per-station × condition explain ─────────────────────────────────────


def _run_one(
    station_id: str,
    station_csv: Path,
    pred_csv: Path,
    ontology_store: _InMemoryOntologyStore,
    out_dir: Path,
    ablate_kg: bool,
    explainer_model: str,
    force: bool,
    sensitivity_result: Optional[SensitivityResult] = None,
) -> Dict[str, object]:
    """Generate and persist one explanation for one station × condition."""
    condition = "kg_off" if ablate_kg else "kg_on"
    cond_dir = out_dir / station_id / condition
    cond_dir.mkdir(parents=True, exist_ok=True)
    report_path = cond_dir / "explanation.md"
    result_path = cond_dir / "result.json"

    if report_path.exists() and result_path.exists() and not force:
        logger.info(
            "[%s/%s] resume — skipping (already exists)",
            station_id,
            condition,
        )
        with result_path.open() as f:
            return json.load(f)

    input_history, lat, lon = _load_station_input(station_csv)
    forecast, target, pred_dict = _load_prediction_csv(pred_csv)
    forecast_metrics = _compute_per_kpi_metrics(forecast, target)

    explainer = TelcoAgentExplainer(
        ontology_store=ontology_store,
        osm=None,  # default OpenStreetMapMCP
        model_name=explainer_model,
        station_id=station_id,
        relevance_model=None,  # skip OSM-relevance harness
        sensitivity_pipeline=None,  # don't recompute online
        sensitivity_result=sensitivity_result,  # attach PAX-TS cache (or None)
    )

    t0 = time.time()
    result = explainer.explain(
        forecast=forecast,
        input_history=input_history,
        prediction=pred_dict,
        output_dir=str(cond_dir),
        lat=lat,
        lon=lon,
        forecast_metrics=forecast_metrics,
        ablate_kg=ablate_kg,
    )
    elapsed = time.time() - t0

    payload: Dict[str, object] = {
        "station_id": station_id,
        "condition": condition,
        "ablate_kg": ablate_kg,
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
        condition,
        elapsed,
        len(result.anomaly_events),
        len(result.report),
    )
    return payload


# ─── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate explanations for every station under with-KG / "
            "no-KG conditions (no PAX-TS, no judge). Phase-1 of the "
            "KG-ablation pipeline; the Faithfulness + Answer Relevancy "
            "judge runs separately afterwards."
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
        help="Output root for the ablation run.",
    )
    parser.add_argument(
        "--explainer-model",
        default=os.environ.get("EXPLAINER_MODEL", "openai/gpt-4o-mini"),
        help="LLM model for the ExplainerAgent (default: openai/gpt-4o-mini).",
    )
    parser.add_argument(
        "--pax-ts-dir",
        default="output/pax_ts_full",
        help=(
            "Directory containing per-station PAX-TS sensitivity_tensors.npz "
            "+ pax_ts_summary.json (output of scripts/run_pax_ts_batch.py). "
            "Set to '' or 'none' to skip attaching sensitivity (the §2 "
            "Cross-KPI table will then fall back on KG mechanisms only)."
        ),
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
        "--conditions",
        default="kg_on,kg_off",
        help="Comma-separated subset of {kg_on, kg_off}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redo even when explanation.md already exists.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load KG ────────────────────────────────────────────────────────
    # ``get_default_ontology`` honours the ``TELCOAGENT_KG_PATH`` env
    # var (telcoagent.config.DEFAULT_KG_PATH) so we point it at the
    # sweep-specific KG before instantiating the ontology singleton.
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

    # ── Resolve station + condition lists ─────────────────────────────
    stations_dir = Path(args.stations_dir)
    pred_dir = Path(args.predictions_dir)

    if args.stations:
        station_ids = [s.strip() for s in args.stations.split(",") if s.strip()]
    else:
        all_stations = sorted(p.stem for p in stations_dir.glob("station_*.csv"))
        station_ids = all_stations[: args.limit] if args.limit else all_stations
    logger.info("Will process %d stations", len(station_ids))

    pax_ts_root: Optional[Path] = None
    pax_ts_arg = (args.pax_ts_dir or "").strip().lower()
    if args.pax_ts_dir and pax_ts_arg not in ("", "none", "skip"):
        pax_ts_root = Path(args.pax_ts_dir)
        if not pax_ts_root.exists():
            raise FileNotFoundError(f"--pax-ts-dir {pax_ts_root} does not exist")
        logger.info(
            "PAX-TS sensitivity cache attached: %s",
            pax_ts_root,
        )
    else:
        logger.info(
            "PAX-TS sensitivity cache disabled (--pax-ts-dir empty); "
            "§2 will fall back on KG mechanisms only",
        )

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in ("kg_on", "kg_off"):
            raise ValueError(f"unknown condition {c!r}; must be kg_on or kg_off")

    # ── Loop ───────────────────────────────────────────────────────────
    summary_rows: List[Dict[str, object]] = []
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

        sensitivity_result = None
        if pax_ts_root is not None:
            sensitivity_result = _load_sensitivity_result(
                station_id,
                pax_ts_root,
            )
            if sensitivity_result is None:
                logger.warning(
                    "[%s] PAX-TS cache missing under %s — §2 will fall "
                    "back on KG mechanisms only",
                    station_id,
                    pax_ts_root,
                )

        for cond in conditions:
            ablate_kg = cond == "kg_off"
            try:
                payload = _run_one(
                    station_id=station_id,
                    station_csv=station_csv,
                    pred_csv=pred_csv,
                    ontology_store=ontology_store,
                    out_dir=out_dir,
                    ablate_kg=ablate_kg,
                    explainer_model=args.explainer_model,
                    force=args.force,
                    sensitivity_result=sensitivity_result,
                )
                summary_rows.append(payload)
            except Exception as exc:  # surface, never silent
                logger.error("[%s/%s] FAILED: %s", station_id, cond, exc)
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
                "kg_path": str(kg_path),
                "predictions_dir": str(pred_dir),
                "stations_dir": str(stations_dir),
                "n_stations": len(station_ids),
                "conditions": conditions,
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
