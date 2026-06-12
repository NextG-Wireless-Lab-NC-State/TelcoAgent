#!/usr/bin/env python3
"""Run the TelcoAgent pipeline on a single station (no Neo4j required).

Pipeline:
    1. TelcoAgent.predict(): TSFM-only — Chronos-2 single forward pass +
       physical clamp (no KG consumption, no refinement).
    2. ExplainerAgent: anomaly events + 3GPP GraphRAG + OSM context +
       (optional) PAX-TS perturbation sensitivity → 6-section report
       (§0–§5).
    3. Explanation evaluation: Faithfulness + anomaly-aware Answer
       Relevancy, computed directly from the paper's §IV-A4 definitions
       (no RAGAS-library dependency). Judge: ORANSight, decoupled from
       the GPT-4o-mini explainer.

Neo4j is replaced with an in-memory ontology store that delegates to
the 3GPP ontology directly -- all other components are real.

Usage:
    PYTHONPATH=/path/to/telcoagent python3 scripts/run_telcoagent_single.py \\
        --station data/station/station_A_10.csv \\
        --output  output/telcoagent_single
"""

import argparse
import json
import sys  # noqa: E402 — bootstrap() mutates sys.path before heavy imports
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from telcoagent.cli_utils import bootstrap  # noqa: E402

logger = bootstrap(suppress_tf_warnings=True, logger_name=__name__)
project_root = Path(__file__).resolve().parent.parent

import numpy as np  # noqa: E402

# =============================================================================
# In-memory ontology store (no Neo4j required)
# =============================================================================
# Single source of truth for the Neo4j-free runtime store. It exposes both
# ``.ontology`` and a real ``.retrieve(target_kpis=...)``, so the explainer's
# ``query_graphrag_mechanism`` tool returns grounded content on the default
# ReAct loop. The module-level alias keeps the historical name the canonical
# wiring (and tests) reference while delegating to the shared class.
from telcoagent.stores.in_memory_ontology_store import (  # noqa: E402
    InMemoryOntologyStore as _InMemoryOntologyStore,
)

# =============================================================================
# Patched TelcoAgent.__init__ that skips Neo4j
# =============================================================================


def _build_agent(model_id: str, device: str):
    """Instantiate the TSFM-only TelcoAgent (no live Neo4j required).

    Uses the sanctioned constructor with an injected in-memory ontology
    store: after task 07, ``TelcoAgent.__init__`` no longer opens a Neo4j
    connection on the forward-pass path, so the ``object.__new__`` bypass is
    no longer needed here. The Chronos-2 backbone is loaded by the
    constructor; the in-memory store satisfies the explainer-facing helpers
    without requiring Neo4j.
    """
    from telcoagent.prediction import TelcoAgent

    logger.info("Loading Chronos-2: %s on %s", model_id, device)
    agent = TelcoAgent(
        model_id=model_id,
        device_map=device,
        enable_explainer=False,
        ontology_store=_InMemoryOntologyStore(),
    )

    logger.info("TelcoAgent assembled (in-memory ontology, Chronos-2 backbone)")
    return agent


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="TelcoAgent single-station run + RAGAS eval")
    parser.add_argument("--station", default="data/station/station_A_10.csv")
    parser.add_argument("--output", default="output/telcoagent_single")
    parser.add_argument("--model", default="amazon/chronos-2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--explainer-model",
        default=None,
        help="LLM model for ExplainerAgent (default: from AGENT_MODELS)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    station_path = Path(args.station)
    station_id = station_path.stem

    # --- Load data (single CSV read) ---
    import pandas as pd

    from scripts.baselines.foundation_utils import KPI_COLS_CSV, compute_metrics, split_data

    raw_df = pd.read_csv(station_path)
    data = raw_df[KPI_COLS_CSV].values.astype(np.float64)
    splits = split_data(data)
    input_kpis = splits["input_window"]  # (1944, 7)
    test_target = splits["test_target"]  # (168, 7)
    logger.info("Station: %s | input=%s test=%s", station_id, input_kpis.shape, test_target.shape)

    lat = float(raw_df["latitude"].iloc[0]) if "latitude" in raw_df.columns else None
    lon = float(raw_df["longitude"].iloc[0]) if "longitude" in raw_df.columns else None

    # --- Build agent ---
    agent = _build_agent(args.model, args.device)

    # Register station coordinates for OSM spatial context
    if lat is not None and lon is not None:
        from telcoagent.context.mcp_tools import OpenStreetMapMCP

        agent._osm = OpenStreetMapMCP()
        agent._osm.register_station(station_id, lat, lon)
        logger.info("Registered station %s at (%.4f, %.4f)", station_id, lat, lon)

    # =========================================================================
    # Phase 1: TelcoAgent.predict() -- TSFM-only (Chronos-2) + physical clamp
    # =========================================================================
    logger.info("Running TelcoAgent.predict()...")
    t0 = time.time()
    prediction_array, result_info = agent.predict(
        input_kpis=input_kpis,
        metadata={"station_id": station_id},
    )
    predict_elapsed = time.time() - t0
    logger.info("predict() done in %.1fs", predict_elapsed)

    # --- Compute forecast metrics ---
    from scripts.baselines.foundation_utils import (
        KPI_NAMES,
        apply_physical_clamp,
        save_prediction_csv,
    )

    prediction_clamped = apply_physical_clamp(prediction_array, KPI_NAMES)
    metrics = compute_metrics(prediction_clamped, test_target, input_window=input_kpis)

    # Save per-station prediction CSV (same format as baseline models)
    csv_path = output_dir / "csv" / f"{station_id}.csv"
    save_prediction_csv(prediction_clamped, test_target, station_id, csv_path)

    # --- Raw TSFM (no refinement) A/B baseline ---
    raw_array = result_info.get("prediction_raw_tsfm")
    metrics_raw = None
    if raw_array is not None:
        raw_clamped = apply_physical_clamp(raw_array, KPI_NAMES)
        metrics_raw = compute_metrics(raw_clamped, test_target, input_window=input_kpis)

    print("\n" + "=" * 60)
    print(f"TelcoAgent Forecast Metrics - {station_id}")
    print("=" * 60)
    if metrics_raw is not None:
        print(f"  {'KPI':<20} {'raw_MAE':>10} {'ref_MAE':>10} {'delta':>10}")
        for kpi, kpi_m in metrics.get("per_kpi", {}).items():
            raw_m = metrics_raw.get("per_kpi", {}).get(kpi, {})
            raw_mae = raw_m.get("MAE", float("nan"))
            ref_mae = kpi_m["MAE"]
            d = ref_mae - raw_mae
            marker = " !" if d > 1e-6 else ("" if d < -1e-6 else "  =")
            print(f"  {kpi:<20} {raw_mae:>10.4f} {ref_mae:>10.4f} {d:>+10.4f}{marker}")
        raw_overall = metrics_raw.get("overall_MAE", float("nan"))
        ref_overall = metrics.get("overall_MAE", float("nan"))
        print(
            f"  {'Overall':<20} {raw_overall:>10.4f} {ref_overall:>10.4f} {ref_overall - raw_overall:>+10.4f}"
        )
    else:
        for kpi, kpi_m in metrics.get("per_kpi", {}).items():
            print(
                f"  {kpi:<20} MAE={kpi_m['MAE']:.4f}   RMSE={kpi_m['RMSE']:.4f}   sMAPE={kpi_m.get('sMAPE', float('nan')):.2f}%"
            )
        print(f"  {'Overall':<20} MAE={metrics.get('overall_MAE', float('nan')):.4f}")

    # =========================================================================
    # Phase 2: Explainer Agent — causal analysis + report generation
    # =========================================================================
    print("\n" + "=" * 60)
    print("Explainer Agent — Report Generation")
    print("=" * 60)

    from telcoagent.explainer import TelcoAgentExplainer
    from telcoagent.llm.config import get_active_agent_models

    explainer_model = args.explainer_model or get_active_agent_models().get(
        "explainer", "openai/gpt-4o-mini"
    )

    explainer = TelcoAgentExplainer(
        ontology_store=agent._ontology_store,
        osm=getattr(agent, "_osm", None),
        model_name=explainer_model,
        station_id=station_id,
    )

    # Build prediction dict from prediction_array for Explainer.
    prediction_dict = {
        KPI_NAMES[i]: prediction_clamped[:, i].tolist() for i in range(prediction_clamped.shape[1])
    }

    # Per-KPI metrics computed against the held-out test target. These are
    # surfaced ONLY in §1 of the explanation report — the prompt template
    # forbids §2–§5 from referencing them.
    metrics_per_kpi = metrics.get("per_kpi", {}) if isinstance(metrics, dict) else {}

    t1 = time.time()
    explanation_result = explainer.explain(
        forecast=prediction_clamped,
        input_history=input_kpis,
        prediction=prediction_dict,
        output_dir=str(output_dir),
        lat=lat,
        lon=lon,
        forecast_metrics=metrics_per_kpi,
    )
    explain_elapsed = time.time() - t1
    total_elapsed = time.time() - t0

    report = explanation_result.report
    retrieved_contexts = explanation_result.retrieved_contexts

    print(f"  Report length: {len(report)} chars")
    print(f"  Retrieved contexts: {len(retrieved_contexts)}")
    print(f"  Explainer elapsed: {explain_elapsed:.1f}s")

    # =========================================================================
    # Phase 3: Explanation evaluation on Explainer Agent's report.
    #
    # Faithfulness and Answer Relevancy are computed directly from the
    # paper's §IV-A4 definitions (no RAGAS-library dependency). The
    # ORANSight judge is decoupled from the GPT-4o-mini explainer to
    # avoid same-family bias. See telcoagent/evaluation/explanation_metrics.py.
    # =========================================================================
    print("\n" + "=" * 60)
    print("Explanation Evaluation (Faithfulness + Anomaly-aware Answer Relevancy)")
    print("=" * 60)

    from telcoagent.evaluation.explanation_metrics import ExplanationEvaluator

    expl_evaluator = ExplanationEvaluator()
    expl_scores = expl_evaluator.evaluate_single(
        station_id=station_id,
        report=report,
        retrieved_contexts=retrieved_contexts,
        anomaly_events=explanation_result.anomaly_events,
    )

    print(
        f"\n  Faithfulness:        {expl_scores.faithfulness:.4f}  "
        f"({expl_scores.n_scored}/{expl_scores.n_claims} claims scored)"
    )
    print(
        f"  Answer Relevancy:    {expl_scores.answer_relevancy:.4f}  "
        f"(mean over {len(expl_scores.per_event_alignment)} events)"
    )
    if expl_scores.error:
        print(f"  Error: {expl_scores.error}")

    # --- Save results ---
    result = {
        "station_id": station_id,
        "model": args.model,
        "explainer_model": explainer_model,
        "predict_elapsed_sec": round(predict_elapsed, 2),
        "explain_elapsed_sec": round(explain_elapsed, 2),
        "total_elapsed_sec": round(total_elapsed, 2),
        "forecast_metrics": metrics,
        "explanation_scores": expl_scores.to_dict(),
        "report": report,
        "retrieved_contexts": retrieved_contexts,
        "sensitivity_attached": explanation_result.sensitivity_result is not None,
    }

    out_path = output_dir / f"{station_id}_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Results saved to %s", out_path)

    # --- Save human-readable markdown report ---
    md_lines = [
        f"# TelcoAgent Report — {station_id}",
        "",
        f"**Model**: {args.model}  ",
        f"**Explainer Model**: {explainer_model}  ",
        f"**Predict Elapsed**: {predict_elapsed:.1f}s  ",
        f"**Explain Elapsed**: {explain_elapsed:.1f}s  ",
        "",
        "---",
        "",
        "## Forecast Metrics",
        "",
        "| KPI | MAE | RMSE | sMAPE |",
        "|-----|-----|------|-------|",
    ]
    for kpi, kpi_m in metrics.get("per_kpi", {}).items():
        md_lines.append(
            f"| {kpi} | {kpi_m['MAE']:.4f} | {kpi_m['RMSE']:.4f} | {kpi_m.get('sMAPE', float('nan')):.2f}% |"
        )
    md_lines += [
        f"| **Overall** | **{metrics.get('overall_MAE', float('nan')):.4f}** | "
        f"**{metrics.get('overall_RMSE', float('nan')):.4f}** | — |",
        "",
        "---",
        "",
        "## Explanation Evaluation (Direct Faithfulness + Answer Relevancy)",
        "",
        "| Metric | Score | Detail |",
        "|--------|-------|--------|",
        f"| Faithfulness | {expl_scores.faithfulness:.4f} | "
        f"{expl_scores.n_scored}/{expl_scores.n_claims} claims scored |",
        f"| Answer Relevancy | {expl_scores.answer_relevancy:.4f} | "
        f"mean over {len(expl_scores.per_event_alignment)} anomaly events |",
        "",
        "---",
        "",
        "# Technical Report",
        "",
        report,
    ]

    md_path = output_dir / f"{station_id}_report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    logger.info("Markdown report saved to %s", md_path)

    print(f"\n  Saved: {out_path}")
    print(f"  Saved: {md_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
