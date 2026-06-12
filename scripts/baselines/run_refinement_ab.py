#!/usr/bin/env python3
"""A/B test: baseline (Chronos-2 only) vs. v1 (enriched 1-shot) vs. v2 (2-step harness).

All three modes share the same Chronos-2 median forecast per station; only
the LLM post-correction layer differs.  Requires OPENAI_API_KEY in .env (or
environment) for v1 / v2 modes.

Optimal context from the context-length sweep: 10 days (240h).

Outputs (output/refinement_ab/):
    ab_results.csv          one row per (station, mode, kpi)
    ab_summary.json         per-mode aggregate stats
    plots/ab_smape.png      bar chart — avg sMAPE per mode
    plots/ab_mase.png       bar chart — avg MASE  per mode

Usage:
    conda run -n telcoagent python scripts/baselines/run_refinement_ab.py
    conda run -n telcoagent python scripts/baselines/run_refinement_ab.py \\
        --stations station_A_10 station_C_2 --context-days 10
    conda run -n telcoagent python scripts/baselines/run_refinement_ab.py \\
        --n-stations 20 --context-days 10
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

from scripts.baselines.foundation_utils import (
    KPI_NAMES,
    compute_metrics,
    split_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_ID = "amazon/chronos-2"
DEFAULT_CONTEXT_DAYS = 10  # optimal from context-length sweep
EXPLAINER_MODEL = "openai/gpt-4o-mini"

MODES = ["baseline", "v1", "v2"]


# ── In-memory ontology store (shared, no Neo4j) ──────────────────────────────

from telcoagent.stores.in_memory_ontology_store import (  # noqa: E402
    InMemoryOntologyStore as _InMemoryOntologyStore,
)

# ── Agent builder ─────────────────────────────────────────────────────────────


def _build_agent(device: str):
    from chronos import Chronos2Pipeline

    from telcoagent.llm.config import EnhancedLLMConfig
    from telcoagent.prediction import TelcoAgent

    logger.info("Loading Chronos-2: %s on %s", MODEL_ID, device)
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_ID,
        device_map=device,
        dtype=torch.float32,
    )

    agent = object.__new__(TelcoAgent)
    agent.n_shots = 3
    agent.n_channels = 7
    agent.llm_config = EnhancedLLMConfig()
    agent._model_id = MODEL_ID
    agent._device_map = device
    agent._explainer_model = EXPLAINER_MODEL
    agent._enable_explainer = False
    agent._ontology_store = _InMemoryOntologyStore()
    agent._chronos_pipeline = pipeline

    logger.info("TelcoAgent assembled (in-memory ontology, model=%s)", EXPLAINER_MODEL)
    return agent


# ── Per-station run ───────────────────────────────────────────────────────────


def _load_station(csv_path: Path, context_h: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Return (input_ctx, test_target, metadata) for a station CSV."""
    df = pd.read_csv(csv_path)
    kpi_cols = [
        "RRC_Conn_Count_Avg",
        "DL_CQI",
        "DL_iBler",
        "DL_rBler",
        "MAC DL Eff TP",
        "DL PRB Utilization",
        "DL_IP_Throughput",
    ]
    data = df[kpi_cols].values.astype(np.float64)
    splits = split_data(data)
    full_input = splits["input_window"]
    test_target = splits["test_target"]

    # Trim to the desired context window
    input_ctx = full_input[-context_h:] if full_input.shape[0] > context_h else full_input

    lat = float(df["latitude"].iloc[0]) if "latitude" in df.columns else None
    lon = float(df["longitude"].iloc[0]) if "longitude" in df.columns else None
    eNB = str(df["eNB ID"].iloc[0]) if "eNB ID" in df.columns else csv_path.stem
    cell = str(df["CELL No"].iloc[0]) if "CELL No" in df.columns else "0"

    metadata = {
        "station_id": csv_path.stem,
        "eNB_id": eNB,
        "cell_no": cell,
        "latitude": lat,
        "longitude": lon,
    }
    return input_ctx, test_target, metadata, full_input


def run_station(
    agent,
    csv_path: Path,
    context_h: int,
) -> List[Dict[str, Any]]:
    """Run baseline (Chronos-2 only), v1, v2 for one station.

    baseline — raw Chronos-2 median + physical clamp, no LLM call.
    v1       — same raw forecast + enriched 1-shot LLM post-correction.
    v2       — same raw forecast + 2-step analyze→adjust LLM post-correction.
    """
    from telcoagent.config import PREDICTION_LENGTH_H
    from telcoagent.context.mcp_tools import OpenStreetMapMCP

    input_ctx, test_target, metadata, full_input = _load_station(csv_path, context_h)
    station_id = metadata["station_id"]
    lat = metadata.get("latitude")
    lon = metadata.get("longitude")

    # Attach a fresh OSM instance registered for this station (needed by v1/v2)
    osm = OpenStreetMapMCP()
    if lat is not None and lon is not None:
        osm.register_station(station_id, lat, lon)
    agent._osm = osm

    # ── Step A: Chronos-2 forward pass (shared, no LLM) ───────────────────────
    t0 = time.time()
    predict_fn = agent._make_predict_fn(input_ctx)
    raw_pred = predict_fn(input_ctx)[:PREDICTION_LENGTH_H]
    chronos_sec = time.time() - t0

    # baseline: physical clamp only, no LLM
    pred_baseline = agent._apply_physical_clamp(raw_pred.copy())
    m_base = compute_metrics(pred_baseline, test_target, input_window=full_input)

    results: List[Dict[str, Any]] = []
    for kpi in KPI_NAMES:
        v = m_base["per_kpi"].get(kpi, {})
        results.append(
            {
                "station_id": station_id,
                "mode": "baseline",
                "kpi": kpi,
                "sMAPE": v.get("sMAPE"),
                "MASE": v.get("MASE"),
                "MAE": v.get("MAE"),
                "elapsed_sec": round(chronos_sec, 2),
            }
        )

    # ── Step B: OSM + KG context (shared for v1 and v2) ───────────────────────
    osm_text = agent._fetch_osm_context(station_id, lat, lon)
    kg_text = agent._format_kg_context(list(KPI_NAMES))

    # ── Step C: v1 and v2 refinements on the same raw_pred ────────────────────
    # Refinement variants now live in scripts.baselines.refinement_variants
    # per Stage-2 commit 2.2a (pure refactor-move; canonical location).
    from scripts.baselines.refinement_variants import (
        _agentic_refine_v1,
        _agentic_refine_v2,
    )

    for mode in ("v1", "v2"):
        t1 = time.time()
        try:
            if mode == "v1":
                pred_raw, _ = _agentic_refine_v1(
                    agent,
                    raw_pred,
                    input_ctx,
                    osm_text,
                    kg_text,
                    station_id,
                )
            else:
                pred_raw, _ = _agentic_refine_v2(
                    agent,
                    raw_pred,
                    input_ctx,
                    osm_text,
                    kg_text,
                    station_id,
                )
            pred_final = agent._apply_physical_clamp(pred_raw)
            elapsed = time.time() - t1

            m = compute_metrics(pred_final, test_target, input_window=full_input)
            for kpi in KPI_NAMES:
                v = m["per_kpi"].get(kpi, {})
                results.append(
                    {
                        "station_id": station_id,
                        "mode": mode,
                        "kpi": kpi,
                        "sMAPE": v.get("sMAPE"),
                        "MASE": v.get("MASE"),
                        "MAE": v.get("MAE"),
                        "elapsed_sec": round(elapsed, 2),
                    }
                )
        except Exception as exc:
            logger.error("[%s] %s failed: %s", mode, station_id, exc, exc_info=True)
            for kpi in KPI_NAMES:
                results.append(
                    {
                        "station_id": station_id,
                        "mode": mode,
                        "kpi": kpi,
                        "sMAPE": None,
                        "MASE": None,
                        "MAE": None,
                        "elapsed_sec": None,
                        "error": str(exc),
                    }
                )

    return results


# ── Summary helpers ───────────────────────────────────────────────────────────


def _aggregate(rows: List[Dict]) -> Dict:
    """Per-mode aggregate stats over all stations and KPIs."""
    df = pd.DataFrame(rows)
    for col in ("sMAPE", "MASE"):
        if col not in df.columns:
            df[col] = np.nan
    df = df.dropna(subset=["sMAPE", "MASE"])
    summary = {}
    for mode in MODES:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        per_kpi = {}
        for kpi in KPI_NAMES:
            ks = sub[sub["kpi"] == kpi]
            per_kpi[kpi] = {
                "avg_sMAPE": round(float(ks["sMAPE"].mean()), 3) if not ks.empty else None,
                "avg_MASE": round(float(ks["MASE"].mean()), 4) if not ks.empty else None,
            }
        summary[mode] = {
            "avg_sMAPE_all": round(float(sub["sMAPE"].mean()), 3),
            "avg_MASE_all": round(float(sub["MASE"].mean()), 4),
            "n_stations": int(sub["station_id"].nunique()),
            "per_kpi": per_kpi,
        }
    return summary


def _print_summary(summary: Dict):
    print(f"\n{'=' * 65}")
    print("A/B REFINEMENT SUMMARY  (avg over all stations & KPIs)")
    print(f"{'=' * 65}")
    print(f"  {'Mode':10s}  {'avg sMAPE (%)':>14s}  {'avg MASE':>10s}  {'n stations':>12s}")
    print(f"  {'-' * 55}")
    for mode in MODES:
        if mode not in summary:
            continue
        s = summary[mode]
        print(
            f"  {mode:10s}  {s['avg_sMAPE_all']:>14.3f}  {s['avg_MASE_all']:>10.4f}  "
            f"{s['n_stations']:>12d}"
        )
    print()
    print("  Per-KPI avg sMAPE:")
    print(f"  {'KPI':16s}" + "".join(f"  {m:>10s}" for m in MODES))
    print(f"  {'-' * (16 + len(MODES) * 12)}")
    for kpi in KPI_NAMES:
        row = f"  {kpi:16s}"
        for mode in MODES:
            if mode in summary:
                v = summary[mode]["per_kpi"].get(kpi, {}).get("avg_sMAPE")
                row += f"  {v:>10.3f}" if v is not None else f"  {'N/A':>10s}"
        print(row)
    print(f"{'=' * 65}\n")


def _plot_summary(summary: Dict, out_dir: Path):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )

    COLORS = {"baseline": "#1f77b4", "v1": "#ff7f0e", "v2": "#2ca02c"}
    x = np.arange(len(KPI_NAMES))
    width = 0.25

    for metric, ylabel, filename in [
        ("avg_sMAPE", "Avg. sMAPE (%)", "ab_smape.png"),
        ("avg_MASE", "Avg. MASE", "ab_mase.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.16, 2.8))
        for i, mode in enumerate(MODES):
            if mode not in summary:
                continue
            vals = [summary[mode]["per_kpi"].get(kpi, {}).get(metric) or 0.0 for kpi in KPI_NAMES]
            ax.bar(x + (i - 1) * width, vals, width, label=mode, color=COLORS[mode], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(KPI_NAMES, rotation=25, ha="right", fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Refinement A/B: {ylabel} per KPI (115 stations)")
        ax.legend(frameon=True, loc="upper right", fontsize=8)
        fig.tight_layout(pad=0.5)
        out = out_dir / filename
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Refinement A/B test: baseline vs v1 vs v2")
    parser.add_argument("--data-dir", default="data/station")
    parser.add_argument("--output-dir", default="output/refinement_ab")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--context-days",
        type=int,
        default=DEFAULT_CONTEXT_DAYS,
        help=f"Context fed to Chronos-2 (default: {DEFAULT_CONTEXT_DAYS}d = sweep optimum)",
    )
    parser.add_argument(
        "--stations", nargs="*", help="Explicit station stems (e.g. station_A_10 station_C_2)"
    )
    parser.add_argument(
        "--n-stations", type=int, default=None, help="Run first N stations only (for quick test)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    context_h = args.context_days * 24
    device = args.device if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)

    csv_files = sorted(data_dir.glob("station_*.csv"))
    if args.stations:
        csv_files = [f for f in csv_files if f.stem in args.stations]
    if args.n_stations:
        csv_files = csv_files[: args.n_stations]
    if not csv_files:
        parser.error(f"No station CSVs found under {data_dir}")

    agent = _build_agent(device)

    print("\nRefinement A/B Test")
    print(f"  Model:       {MODEL_ID}")
    print(f"  Refiner:     {EXPLAINER_MODEL}")
    print(f"  Context:     {args.context_days}d ({context_h}h)")
    print(f"  Stations:    {len(csv_files)}")
    print(f"  Modes:       {MODES}")
    print(f"  Output:      {out_dir}")

    all_rows: List[Dict] = []
    t_total = time.time()

    for idx, csv_path in enumerate(csv_files, 1):
        sid = csv_path.stem
        print(f"\n  [{idx:3d}/{len(csv_files)}] {sid}")
        try:
            rows = run_station(agent, csv_path, context_h)
            all_rows.extend(rows)
            # Quick inline summary per station
            df_s = pd.DataFrame(rows).dropna(subset=["sMAPE"])
            for mode in MODES:
                ms = df_s[df_s["mode"] == mode]["sMAPE"].mean()
                print(f"    {mode:10s}: avg sMAPE = {ms:.2f}%")
        except Exception as exc:
            logger.error("Station %s failed: %s", sid, exc, exc_info=args.verbose)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_out = out_dir / "ab_results.csv"
    if all_rows:
        fieldnames = ["station_id", "mode", "kpi", "sMAPE", "MASE", "MAE", "elapsed_sec"]
        with open(csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nSaved: {csv_out}  ({len(all_rows)} rows)")

    # ── Aggregate + print ─────────────────────────────────────────────────────
    summary = _aggregate(all_rows)
    _print_summary(summary)

    json_out = out_dir / "ab_summary.json"
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved: {json_out}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        _plot_summary(summary, out_dir / "plots")
    except Exception as exc:
        logger.warning("Plot generation failed: %s", exc)

    total_min = (time.time() - t_total) / 60
    print(f"\nTotal time: {total_min:.1f} min  ({len(csv_files)} stations)")


if __name__ == "__main__":
    main()
