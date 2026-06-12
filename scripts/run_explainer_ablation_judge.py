"""ORANSight Faithfulness + Answer Relevancy judge runner for the
KG-on / KG-off ablation.

Walks an ablation output directory produced by
``scripts/run_explainer_kg_ablation.py`` (one folder per station, with
``kg_on/`` and ``kg_off/`` subfolders each containing
``explanation.md`` + ``anomaly_events.csv``), invokes
:class:`telcoagent.evaluation.explanation_metrics.ExplanationEvaluator`
on every (station, arm) pair, and writes:

* ``judge_per_station.csv`` — one row per (station, arm) with columns
  ``station_id, arm, faithfulness, answer_relevancy, n_claims,
  n_scored, n_events``.
* ``judge_summary.csv`` — arm-level mean ± stdev for both metrics, plus
  the paired kg_on − kg_off difference.
* ``judge_per_station.json`` — same per-station rows in JSON for
  programmatic post-processing.

The judge is decoupled from the GPT-4o-mini explainer (defaults to
ORANSight 14B served via vLLM) so the evaluator does NOT inherit any
explainer-side bias. Run ``--judge-model`` to override; run
``--api-base`` to point at a different vLLM host.

Example
-------

::

    PYTHONPATH=. python scripts/run_explainer_ablation_judge.py \\
        --ablation-dir output/kg_ablation_full_v2 \\
        --output output/kg_ablation_full_v2/judge

    # Smoke on 2 stations:
    PYTHONPATH=. python scripts/run_explainer_ablation_judge.py \\
        --ablation-dir output/kg_ablation_full_v2 \\
        --output output/kg_ablation_full_v2/judge_smoke \\
        --stations station_L_8,station_G_9
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── Anomaly-events CSV → list[dict] ──────────────────────────────────────


_NUMERIC_FIELDS = {"t_start", "t_end", "duration_h", "magnitude", "z_score"}
_LITERAL_FIELDS = {"candidate_causes", "environmental_context"}


_SECTION_HEADER_RE = re.compile(r"^#{1,6}\s*(§\d)\b", re.MULTILINE)


def _extract_evidence_contexts(report: str) -> List[str]:
    """Pull the report's evidence sections (§1, §2, §3) into a list.

    These three sections carry the forecast summary, the cross-KPI
    coupling table (KG mechanisms + PAX-TS sensitivity rows + OSM
    factors), and the spatial context — i.e. what the explainer
    "knew" about the world before authoring §4 / §5 claims. Passing
    them to the Faithfulness judge implements RAGAS-style grounding-
    aware faithfulness: each §4 / §5 claim is scored on whether the
    evidence visible in §1–§3 supports it.

    Sections are returned as separate context entries so the judge
    can localise grounding signals (one §-block per entry).
    """
    if not report:
        return []
    matches = list(_SECTION_HEADER_RE.finditer(report))
    chunks: List[str] = []
    keep = {"§1", "§2", "§3"}
    for idx, m in enumerate(matches):
        tag = m.group(1)
        if tag not in keep:
            continue
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(report)
        chunks.append(report[start:end].rstrip())
    return chunks


def _parse_anomaly_csv(path: Path) -> List[Dict[str, Any]]:
    """Parse the CSV emitted by ``TelcoAgentExplainer._write_csv_artefacts``.

    Numeric columns are coerced to int/float; ``candidate_causes`` /
    ``environmental_context`` columns hold ``repr()`` of nested
    Python objects, parsed via :func:`ast.literal_eval`.

    A missing or empty CSV returns ``[]`` — the judge handles
    ``answer_relevancy = 0.0`` for the event-free case.
    """
    if not path.exists() or path.stat().st_size == 0:
        return []
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            event: Dict[str, Any] = {}
            for key, raw in row.items():
                if raw is None or raw == "":
                    event[key] = None
                    continue
                if key in _NUMERIC_FIELDS:
                    try:
                        event[key] = float(raw) if "." in raw else int(raw)
                    except ValueError:
                        event[key] = raw
                elif key in _LITERAL_FIELDS:
                    try:
                        event[key] = ast.literal_eval(raw)
                    except (ValueError, SyntaxError):
                        event[key] = raw
                elif key == "co_occurring_kpis":
                    event[key] = [item.strip() for item in raw.split(",") if item.strip()]
                else:
                    event[key] = raw
            events.append(event)
    return events


# ─── Ablation discovery ───────────────────────────────────────────────────


def _discover_records(
    ablation_dir: Path,
    arms: List[str],
    stations_filter: Optional[List[str]],
) -> List[Dict[str, Any]]:
    """Locate every ``(station, arm)`` pair that has an explanation report.

    Returns a list of dicts the evaluator's batch interface accepts.
    """
    records: List[Dict[str, Any]] = []
    if not ablation_dir.exists():
        raise FileNotFoundError(f"ablation-dir does not exist: {ablation_dir}")

    station_dirs = sorted(
        d for d in ablation_dir.iterdir() if d.is_dir() and d.name.startswith("station_")
    )
    if stations_filter:
        wanted = set(stations_filter)
        station_dirs = [d for d in station_dirs if d.name in wanted]
        missing = wanted - {d.name for d in station_dirs}
        if missing:
            logger.warning(
                "Requested stations not found under %s: %s",
                ablation_dir,
                sorted(missing),
            )

    for station_dir in station_dirs:
        for arm in arms:
            arm_dir = station_dir / arm
            report_path = arm_dir / "explanation.md"
            if not report_path.exists():
                logger.warning(
                    "skip %s/%s: no explanation.md",
                    station_dir.name,
                    arm,
                )
                continue
            anomaly_path = arm_dir / "anomaly_events.csv"
            records.append(
                {
                    "station_id": station_dir.name,
                    "arm": arm,
                    "report_path": report_path,
                    "anomaly_events": _parse_anomaly_csv(anomaly_path),
                }
            )
    return records


# ─── Score writers ────────────────────────────────────────────────────────


def _write_per_station_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = [
        "station_id",
        "arm",
        "faithfulness",
        "answer_relevancy",
        "n_claims",
        "n_scored",
        "n_events",
        "elapsed_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_per_station_json(rows: List[Dict[str, Any]], path: Path) -> None:
    path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _write_summary_csv(rows: List[Dict[str, Any]], arms: List[str], path: Path) -> None:
    """Arm-level mean ± stdev plus paired kg_on − kg_off diff."""
    import statistics

    by_station_arm: Dict[tuple, Dict[str, float]] = {(r["station_id"], r["arm"]): r for r in rows}
    metrics = ("faithfulness", "answer_relevancy")
    summary: List[Dict[str, Any]] = []

    for arm in arms:
        per_arm = [r for r in rows if r["arm"] == arm]
        for metric in metrics:
            values = [float(r[metric]) for r in per_arm if r.get(metric) is not None]
            if not values:
                continue
            summary.append(
                {
                    "metric": metric,
                    "arm": arm,
                    "n": len(values),
                    "mean": statistics.fmean(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                }
            )

    if {"kg_on", "kg_off"}.issubset(arms):
        for metric in metrics:
            paired_diffs: List[float] = []
            for r in rows:
                if r["arm"] != "kg_on":
                    continue
                pair = by_station_arm.get((r["station_id"], "kg_off"))
                if pair is None:
                    continue
                a = r.get(metric)
                b = pair.get(metric)
                if a is None or b is None:
                    continue
                paired_diffs.append(float(a) - float(b))
            if paired_diffs:
                summary.append(
                    {
                        "metric": metric,
                        "arm": "kg_on - kg_off (paired)",
                        "n": len(paired_diffs),
                        "mean": statistics.fmean(paired_diffs),
                        "stdev": (statistics.stdev(paired_diffs) if len(paired_diffs) > 1 else 0.0),
                        "min": min(paired_diffs),
                        "max": max(paired_diffs),
                    }
                )

    if summary:
        fieldnames = ["metric", "arm", "n", "mean", "stdev", "min", "max"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary:
                writer.writerow(row)


# ─── Main ─────────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run ORANSight Faithfulness + Answer Relevancy judge on a "
            "TelcoAgent KG-ablation output directory."
        ),
    )
    parser.add_argument(
        "--ablation-dir",
        required=True,
        help="Path to the kg-ablation output dir (containing station_*/ folders).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to write judge_per_station.csv / judge_summary.csv / judge_per_station.json.",
    )
    parser.add_argument(
        "--arms",
        default="kg_on,kg_off",
        help="Comma-separated list of arm subfolder names to evaluate.",
    )
    parser.add_argument(
        "--stations",
        default=None,
        help="Comma-separated station_id filter (default: all stations under ablation-dir).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N (station, arm) pairs (smoke / dry-run).",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Override the LiteLLM judge model (default: ORANSight 14B via vLLM).",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Override the vLLM api_base URL (default: ORANSight defaults).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help=(
            "Number of (station, arm) pairs evaluated in parallel. vLLM "
            "auto-batches concurrent HTTP requests, so increasing this "
            "raises judge throughput by ≈4–8× on a single 4090 with "
            "ORANSight 14B (4-bit). Set to 1 for strictly sequential."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (DEBUG / INFO / WARNING).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ablation_dir = Path(args.ablation_dir).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    stations_filter = (
        [s.strip() for s in args.stations.split(",") if s.strip()] if args.stations else None
    )

    records = _discover_records(ablation_dir, arms, stations_filter)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        logger.error("No (station, arm) pairs found under %s", ablation_dir)
        return 1
    logger.info(
        "Discovered %d (station, arm) pairs across %d stations under %s",
        len(records),
        len({r["station_id"] for r in records}),
        ablation_dir,
    )

    # Defer import so a misconfigured judge env does not break --help.
    from telcoagent.evaluation.explanation_metrics import ExplanationEvaluator

    evaluator = ExplanationEvaluator(
        judge_model=args.judge_model,
        api_base=args.api_base,
    )

    # Per-record evaluation closure — used by both sequential and
    # thread-pool dispatch paths.
    def _eval_one(idx_rec: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
        idx, rec = idx_rec
        report_text = Path(rec["report_path"]).read_text(encoding="utf-8")
        station_arm_id = f"{rec['station_id']}/{rec['arm']}"
        logger.info(
            "[%d/%d] judging %s (events=%d, report=%d chars)",
            idx,
            len(records),
            station_arm_id,
            len(rec["anomaly_events"]),
            len(report_text),
        )
        t_start = time.time()
        # Grounding-aware faithfulness: feed §1/§2/§3 of the report
        # to the judge as the evidence the explainer had access to.
        # In kg_on, §2 carries [GraphRAG] mechanism strings + [Cause]
        # rows; in kg_off, §2 only has [Sensitivity]/[Parametric]
        # signals — so KG-grounded claims receive higher scores
        # under the same judge rubric.
        contexts = _extract_evidence_contexts(report_text)
        try:
            scores = evaluator.evaluate_single(
                station_id=station_arm_id,
                report=report_text,
                retrieved_contexts=contexts,
                anomaly_events=rec["anomaly_events"],
            )
        except Exception as exc:
            logger.exception(
                "Evaluator failed for %s; skipping that pair",
                station_arm_id,
            )
            return {
                "_order": idx,
                "station_id": rec["station_id"],
                "arm": rec["arm"],
                "faithfulness": None,
                "answer_relevancy": None,
                "n_claims": 0,
                "n_scored": 0,
                "n_events": len(rec["anomaly_events"]),
                "elapsed_s": round(time.time() - t_start, 2),
                "error": str(exc),
            }
        return {
            "_order": idx,
            "station_id": rec["station_id"],
            "arm": rec["arm"],
            "faithfulness": (
                float(scores.faithfulness) if scores.faithfulness is not None else None
            ),
            "answer_relevancy": (
                float(scores.answer_relevancy) if scores.answer_relevancy is not None else None
            ),
            "n_claims": int(scores.n_claims),
            "n_scored": int(scores.n_scored),
            "n_events": len(rec["anomaly_events"]),
            "elapsed_s": round(time.time() - t_start, 2),
        }

    rows: List[Dict[str, Any]] = []
    workers = max(1, int(args.workers))
    pairs = list(enumerate(records, start=1))
    if workers == 1:
        for pair in pairs:
            rows.append(_eval_one(pair))
    else:
        # vLLM auto-batches concurrent HTTP requests, so we just need
        # to fire ``workers`` of them in parallel against the same
        # endpoint. ThreadPoolExecutor is sufficient — litellm is
        # thread-safe and the bottleneck is the GPU, not Python GIL.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info(
            "Dispatching %d (station, arm) pairs across %d worker threads",
            len(pairs),
            workers,
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_eval_one, pair): pair for pair in pairs}
            for fut in as_completed(futures):
                rows.append(fut.result())

    # Restore discovery order so per-station CSV remains diff-friendly.
    rows.sort(key=lambda r: r.get("_order", 0))
    for r in rows:
        r.pop("_order", None)

    _write_per_station_csv(rows, output_dir / "judge_per_station.csv")
    _write_per_station_json(rows, output_dir / "judge_per_station.json")
    _write_summary_csv(rows, arms, output_dir / "judge_summary.csv")

    logger.info(
        "Wrote %d rows to %s",
        len(rows),
        output_dir / "judge_per_station.csv",
    )

    # Print compact terminal summary.
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    for arm in arms:
        per = by_arm.get(arm, [])
        if not per:
            continue
        n = len(per)
        f_vals = [r["faithfulness"] for r in per if r["faithfulness"] is not None]
        r_vals = [r["answer_relevancy"] for r in per if r["answer_relevancy"] is not None]
        if not f_vals or not r_vals:
            continue
        logger.info(
            "[summary] %s: n=%d | Faithfulness=%.3f | AnswerRelevancy=%.3f",
            arm,
            n,
            sum(f_vals) / len(f_vals),
            sum(r_vals) / len(r_vals),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
