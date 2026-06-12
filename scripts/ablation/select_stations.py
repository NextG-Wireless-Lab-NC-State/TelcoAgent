#!/usr/bin/env python3
"""Station selection for component-ablation (n=30) — Step 1 prep.

Deterministic: no LLM calls. Loads Chronos-2 forecast CSVs from the
ctx_672h sweep output and the matching station input CSVs, runs
``detect_anomaly_events`` for every station, and selects 30 stations
prioritising those with at least one anomaly event.

Selection rule
--------------
1. Count anomaly events per station using the deterministic detector
   (same thresholds as the Explainer runtime).
2. Sort by event count (descending), then station ID (ascending) for
   deterministic tie-breaking.
3. Take the top 30.  If fewer than 30 stations exist, take all.
4. If event-bearing stations < 30, fill the remainder with event=0
   stations (fewest-to-most alphabetically for reproducibility).

Forecast source
---------------
``output/amazon_chronos-2_h7d_ctx_sweep/ctx_672h/csv/<station>.csv``
  columns: station_id, day, hour, <kpi>_pred, <kpi>_true  (168 rows)

Input baseline (last 168 h of the input window = days 75-81)
------------------------------------------------------------
``data/station/<station>.csv``
  columns: DATE, HOUR, eNB ID, CELL No, <ems_cols...>

  The Explainer derives its anomaly-detection baseline from
  ``input_history[-_HISTORY_BASELINE_HOURS:]`` (agent.py:782), i.e. the
  *last 168 hours* of the 1944-hour input window — NOT all 1944 hours.
  This script must slice the identical window so that its event counts
  agree with the live Explainer runs.

Outputs
-------
``output/ablation/component/station_selection.json``  — machine-readable
``output/ablation/component/stations_n30.txt``        — one station ID per line

Usage
-----
    PYTHONPATH=. python scripts/ablation/select_stations.py
    PYTHONPATH=. python scripts/ablation/select_stations.py \\
        --forecast-dir output/amazon_chronos-2_h7d_ctx_sweep/ctx_672h/csv \\
        --station-dir  data/station \\
        --n 30 \\
        --out-dir output/ablation/component
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# ── repo root on sys.path ────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Reuse the *exact* loaders the live Explainer ablation driver uses, so
# this selector's anomaly counts are guaranteed to match
# scripts/run_explainer_kg_ablation.py. ``_load_station_input`` returns
# the 1944-hour input window; ``_load_prediction_csv`` returns the
# 168-hour Chronos-2 forecast in CORE_KPI_NAMES order.
import importlib.util as _ilu  # noqa: E402

from telcoagent.config import CORE_KPI_NAMES  # noqa: E402
from telcoagent.explainer.anomaly_detector import detect_anomaly_events  # noqa: E402
from telcoagent.explainer.orchestrator import _HISTORY_BASELINE_HOURS  # noqa: E402

_ABL_DRIVER = _REPO_ROOT / "scripts" / "run_explainer_kg_ablation.py"
_spec = _ilu.spec_from_file_location("_kg_ablation_driver", _ABL_DRIVER)
_kg_ablation_driver = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_kg_ablation_driver)
_load_station_input = _kg_ablation_driver._load_station_input
_load_prediction_csv = _kg_ablation_driver._load_prediction_csv

# ─── Constants ───────────────────────────────────────────────────────────────

#: Forecast horizon (hours).
_HORIZON_H: int = 168

#: Input window for baseline statistics (days 1-81 = 1944 h).
_INPUT_WINDOW_H: int = 1944

#: Existing n=20 station IDs (for overlap reporting).
_N20_STATIONS: List[str] = [
    "A_10",
    "A_12",
    "A_2",
    "A_4",
    "A_5",
    "A_6",
    "A_8",
    "A_9",
    "C_12",
    "C_14",
    "C_15",
    "C_16",
    "C_18",
    "C_1",
    "C_2",
    "C_4",
    "C_7",
    "D_10",
    "D_12",
    "D_1",
]


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _load_forecast(forecast_csv: Path) -> np.ndarray:
    """Load the 168-row Chronos-2 forecast CSV → ndarray (168, 7).

    Delegates to ``run_explainer_kg_ablation._load_prediction_csv`` so
    the forecast array is byte-for-byte identical to the live Explainer
    ablation run (same ``<kpi>_pred`` columns, same CORE_KPI_NAMES order).
    """
    forecast, _target, _pred_dict = _load_prediction_csv(forecast_csv)
    arr = forecast.astype(np.float64)
    if arr.shape != (_HORIZON_H, len(CORE_KPI_NAMES)):
        raise ValueError(
            f"{forecast_csv.name}: expected shape ({_HORIZON_H}, {len(CORE_KPI_NAMES)}), "
            f"got {arr.shape}"
        )
    return arr


def _load_baseline(station_csv: Path) -> np.ndarray:
    """Load the Explainer's anomaly-detection baseline → ndarray (168, 7).

    The live Explainer derives its baseline from
    ``input_history[-_HISTORY_BASELINE_HOURS:]`` (agent.py:782) where
    ``input_history`` is the 1944-hour input window. We reproduce that
    exactly: load the same 1944-hour window through the Explainer's own
    ``_load_station_input`` loader, then take its last 168 hours
    (days 75-81). Using the full 1944-hour window here was the bug — it
    yields a different baseline σ and spurious anomaly events.
    """
    input_window, _lat, _lon = _load_station_input(station_csv)
    arr = input_window.astype(np.float64)
    if arr.shape[0] < _INPUT_WINDOW_H:
        raise ValueError(f"{station_csv.name}: only {arr.shape[0]} rows; need >= {_INPUT_WINDOW_H}")
    return arr[-_HISTORY_BASELINE_HOURS:]


def _station_id_from_csv_name(name: str) -> str:
    """'station_A_10.csv' → 'A_10'."""
    stem = name.replace(".csv", "").replace(".json", "")
    prefix = "station_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def _count_events(
    forecast: np.ndarray,
    baseline: np.ndarray,
) -> int:
    """Run the deterministic detector; return number of events found."""
    events = detect_anomaly_events(
        forecast=forecast,
        baseline=baseline,
        kpi_names=list(CORE_KPI_NAMES),
    )
    return len(events)


# ─── Main ────────────────────────────────────────────────────────────────────


def run(
    forecast_dir: Path,
    station_dir: Path,
    n: int,
    out_dir: Path,
) -> None:
    forecast_csvs = sorted(forecast_dir.glob("station_*.csv"))
    if not forecast_csvs:
        raise FileNotFoundError(f"No station_*.csv found in {forecast_dir}")

    print(f"Found {len(forecast_csvs)} forecast CSVs in {forecast_dir}")

    records: List[Dict] = []
    errors: List[str] = []

    for fc in forecast_csvs:
        sid = _station_id_from_csv_name(fc.name)
        station_csv = station_dir / fc.name

        if not station_csv.exists():
            errors.append(f"{sid}: station CSV not found at {station_csv}")
            continue

        try:
            forecast_arr = _load_forecast(fc)
            baseline_arr = _load_baseline(station_csv)
            n_events = _count_events(forecast_arr, baseline_arr)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{sid}: {exc}")
            continue

        records.append(
            {
                "station_id": sid,
                "n_events": n_events,
                "forecast_csv": str(fc.relative_to(_REPO_ROOT)),
                "station_csv": str(station_csv.relative_to(_REPO_ROOT)),
                "chronos2_available": True,
                "naive_derivable": True,  # always: tile last 24h of baseline
            }
        )

    if errors:
        print(f"\nWARNING: {len(errors)} station(s) skipped:")
        for e in errors:
            print(f"  {e}")

    if not records:
        raise RuntimeError("No stations could be processed.")

    # ── Selection ────────────────────────────────────────────────────────────
    # Sort: event count desc, then station_id asc (deterministic tie-break)
    records.sort(key=lambda r: (-r["n_events"], r["station_id"]))

    with_events = [r for r in records if r["n_events"] >= 1]
    without_events = [r for r in records if r["n_events"] == 0]

    if len(with_events) >= n:
        selected = with_events[:n]
        fill_note = None
    else:
        selected = with_events + without_events[: n - len(with_events)]
        fill_note = (
            f"Only {len(with_events)} stations with events; "
            f"filled {n - len(with_events)} from event=0 pool."
        )

    # ── n=20 overlap analysis ─────────────────────────────────────────────────
    selected_ids = {r["station_id"] for r in selected}
    n20_set = set(_N20_STATIONS)
    n20_with_events = {
        r["station_id"] for r in records if r["station_id"] in n20_set and r["n_events"] >= 1
    }
    n20_in_selected = selected_ids & n20_set

    # ── Summaries ─────────────────────────────────────────────────────────────
    selected_event_count = sum(r["n_events"] for r in selected)
    n_event_bearing_selected = sum(1 for r in selected if r["n_events"] >= 1)
    # AR valid sample = number of selected stations with >= 1 event
    ar_valid_sample = n_event_bearing_selected

    print("\n" + "=" * 64)
    print(f"Total stations processed : {len(records)}")
    print(f"Event-bearing (>=1 event): {len(with_events)}")
    print(f"Selected n={len(selected)}")
    print(f"  Event-bearing in selection: {n_event_bearing_selected}")
    print(f"  AR valid sample (expected): {ar_valid_sample}")
    print("\nn=20 overlap:")
    print(f"  n=20 stations with events  : {len(n20_with_events)}")
    print(f"  n=20 stations in n=30      : {len(n20_in_selected)}")
    if fill_note:
        print(f"\nFill note: {fill_note}")

    print("\nSelected stations (sorted by event count desc):")
    print(f"  {'station_id':<15} {'n_events':>8}")
    print(f"  {'-'*15} {'-'*8}")
    for r in selected:
        print(f"  {r['station_id']:<15} {r['n_events']:>8}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    selection_json = {
        "n_requested": n,
        "n_selected": len(selected),
        "n_processed": len(records),
        "n_event_bearing_total": len(with_events),
        "n_event_bearing_selected": n_event_bearing_selected,
        "ar_valid_sample_expected": ar_valid_sample,
        "fill_note": fill_note,
        "n20_overlap": {
            "n20_event_bearing": sorted(n20_with_events),
            "n20_in_n30": sorted(n20_in_selected),
            "n20_dropped": sorted(n20_set - selected_ids),
        },
        "mde_note": (
            "Faithfulness: paired stdev(n20)=0.1018 -> SE(n30)=0.0186 -> MDE(80%,a.05)~=0.052. "
            "Observed KG-faithfulness gap=-0.044 < MDE: underpowered for faithfulness. "
            "-Predictor/-Explainer effects expected larger -> n=30 sufficient. "
            "AR binding on event-bearing count."
        ),
        "selected": selected,
        "all_stations": records,
        "errors": errors,
    }

    json_path = out_dir / "station_selection.json"
    with open(json_path, "w") as f:
        json.dump(selection_json, f, indent=2)
    print(f"\nWrote {json_path}")

    txt_path = out_dir / "stations_n30.txt"
    with open(txt_path, "w") as f:
        for r in selected:
            f.write(r["station_id"] + "\n")
    print(f"Wrote {txt_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--forecast-dir",
        type=Path,
        default=_REPO_ROOT / "output" / "amazon_chronos-2_h7d_ctx_sweep" / "ctx_672h" / "csv",
        help="Directory with per-station Chronos-2 forecast CSVs (default: ctx_672h/csv).",
    )
    p.add_argument(
        "--station-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "station",
        help="Directory with raw station CSVs (default: data/station).",
    )
    p.add_argument(
        "--n",
        type=int,
        default=30,
        help="Number of stations to select (default: 30).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "output" / "ablation" / "component",
        help="Output directory for selection artefacts.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        forecast_dir=args.forecast_dir,
        station_dir=args.station_dir,
        n=args.n,
        out_dir=args.out_dir,
    )
