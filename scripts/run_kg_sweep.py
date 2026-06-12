"""KG parameter sweep driver — optimal Θ (quality_threshold) search.

Runs KGConstructionPipeline for 9 Θ values with Stage 1-3 cache reuse.
Then for each resulting KG, probes 5 min_confidence values (post-hoc loader filter)
and writes aggregated metrics to output/sweep/metrics.csv.

Environment (set by caller):
  LLM_BACKEND=vllm|openai
  # when LLM_BACKEND=vllm:
  VLLM_BASE_URL=http://localhost:8000/v1
  VLLM_MODEL=<model id served by vLLM>
  # when LLM_BACKEND=openai:
  OPENAI_MODEL=gpt-4o-mini
  OPENAI_API_KEY=<key>  # loaded from .env
  TELCOAGENT_RPM_DELAY=0
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import time
from pathlib import Path

from telcoagent.kg_construction.pipeline import KGConstructionPipeline
from telcoagent.ontology.core import get_default_ontology
from telcoagent.ontology.kg_loader import load_directional_rules_from_kg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("kg_sweep")

ROOT = Path("/home/gkim26/Desktop/workplace/telcoagent")
SWEEP_ROOT = ROOT / "output" / "sweep"
MASTER_DIR = SWEEP_ROOT / "master"

THETA_VALUES = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
MIN_CONF_VALUES = [0.5, 0.6, 0.7, 0.8, 0.9]

PDFS = [
    str(ROOT / "data/specs/TS_28.552.pdf"),
    str(ROOT / "data/specs/TS_28.554.pdf"),
    # TS_38.133 (45 MB, 900+ pages) excluded from this sweep — too long to
    # parse and the RRM measurement procedures are not directly needed for
    # the 7 forecast KPIs.  Re-add later if the KG audit shows missing
    # measurements that only TS 38.133 defines.
    str(ROOT / "data/specs/TS_38.211.pdf"),
    str(ROOT / "data/specs/TS_38.212.pdf"),
    str(ROOT / "data/specs/TS_38.213.pdf"),
    str(ROOT / "data/specs/TS_38.214.pdf"),
    str(ROOT / "data/specs/TS_38.215.pdf"),
    str(ROOT / "data/specs/TS_38.300.pdf"),
    str(ROOT / "data/specs/TS_38.314.pdf"),
    str(ROOT / "data/specs/TS_38.321.pdf"),
    str(ROOT / "data/specs/TS_38.322.pdf"),
    str(ROOT / "data/specs/TS_38.331.pdf"),
    str(ROOT / "data/specs/TS_38.901.pdf"),
]

# All caches except KG output are Θ-agnostic and therefore symlinked from the
# master directory into each Θ variant.  ``cache_validated.json`` holds the
# raw 3-signal scores; the orchestrator re-applies each variant's threshold
# at load time, so no LLM calls are needed for the variants.
SHARED_CACHES = (
    "cache_sections.json",
    "cache_extracted.json",
    "cache_aligned.json",
    "cache_validated.json",
)


def _theta_dir(theta: float) -> Path:
    return SWEEP_ROOT / f"theta_{theta:.2f}"


def _link_shared_caches(target_dir: Path) -> None:
    """Symlink stage 1-3 caches from master into the per-theta dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in SHARED_CACHES:
        src = MASTER_DIR / name
        dst = target_dir / name
        if not src.exists():
            raise FileNotFoundError(
                f"Expected shared cache {src} — master run must complete first."
            )
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        logger.info("symlinked %s -> %s", dst, src)


def _build_pipeline(
    output_dir: Path, theta: float, model: str, use_cache: bool
) -> KGConstructionPipeline:
    kg_path = output_dir / "enriched_kg.json"
    return KGConstructionPipeline(
        model=model,
        quality_threshold=theta,
        output_dir=str(output_dir),
        use_cache=use_cache,
        build_kg=True,
        kg_output_path=str(kg_path),
        reader_delta=0.3,
        enable_conflict_resolution=True,
        use_embedding_align=True,
    )


def run_master(model: str) -> Path:
    """Run the master pipeline with Θ=0.5 (lowest). Produces shared caches + first KG.

    Honors any pre-existing per-stage caches in MASTER_DIR (sections,
    extracted, aligned).  This lets a crashed run resume from the latest
    successful stage instead of wasting hours re-extracting.  Set
    ``KG_SWEEP_FRESH=1`` in the environment to force a fresh master build.
    """
    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    theta = THETA_VALUES[0]
    use_cache = os.environ.get("KG_SWEEP_FRESH", "0") not in ("1", "true", "True")
    logger.info("=== MASTER RUN (Θ=%.2f, use_cache=%s) ===", theta, use_cache)
    pipe = _build_pipeline(MASTER_DIR, theta, model, use_cache=use_cache)
    result = pipe.run(PDFS)
    logger.info(
        "master done: %s approved triples",
        len(result.approved_triples) if hasattr(result, "approved_triples") else "?",
    )
    # Copy the master's KG into theta_0.50/ so downstream metrics loop sees it.
    # ``_link_shared_caches`` already symlinks every entry in SHARED_CACHES
    # (including ``cache_validated.json`` post-Phase E), so we only need the
    # KG copy here — the previous explicit copyfile of cache_validated.json
    # was redundant and crashed with SameFileError on resume.
    target = _theta_dir(theta)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MASTER_DIR / "enriched_kg.json", target / "enriched_kg.json")
    _link_shared_caches(target)
    return MASTER_DIR


def run_theta_variant(theta: float, model: str) -> Path:
    """Reuse Stage 1-3 caches; run only Evaluator + feedback + KG build."""
    target = _theta_dir(theta)
    _link_shared_caches(target)
    logger.info("=== THETA RUN Θ=%.2f ===", theta)
    pipe = _build_pipeline(target, theta, model, use_cache=True)
    t0 = time.time()
    pipe.run(PDFS)
    logger.info("Θ=%.2f elapsed %.1fs", theta, time.time() - t0)
    return target


def aggregate_metrics() -> Path:
    """Build the final metrics CSV: (theta, min_conf) -> KG-level metrics."""
    out_csv = SWEEP_ROOT / "metrics.csv"
    known = {k.short_name for k in get_default_ontology().kpis.values()}
    canonical = {"INCREASES", "DECREASES", "DETERMINES", "LIMITS", "CAUSES"}

    rows = []
    for theta in THETA_VALUES:
        kg = _theta_dir(theta) / "enriched_kg.json"
        if not kg.exists():
            logger.warning("missing KG: %s", kg)
            continue

        data = json.load(open(kg))
        edges = data.get("edges") or data.get("links") or []
        rel_edges = [
            e for e in edges if e.get("triple_type") in ("relationship", "ontology_relationship")
        ]
        canonical_edges = [
            e for e in rel_edges if (e.get("relation_type") or "").upper() in canonical
        ]
        quality_sum = sum(float(e.get("quality_score", 0.0)) for e in rel_edges)
        avg_quality = quality_sum / len(rel_edges) if rel_edges else 0.0

        for conf in MIN_CONF_VALUES:
            rules = load_directional_rules_from_kg(str(kg), known, min_confidence=conf)
            rows.append(
                {
                    "theta": theta,
                    "min_confidence": conf,
                    "total_edges": len(edges),
                    "relationship_edges": len(rel_edges),
                    "canonical_edges": len(canonical_edges),
                    "directional_rules": len(rules),
                    "avg_quality_score": round(avg_quality, 4),
                }
            )

    with open(out_csv, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    logger.info("wrote %s with %d rows", out_csv, len(rows))
    return out_csv


def main():
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    backend = os.environ.get("LLM_BACKEND", "openai").lower()
    if backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set (check .env)")
        model = os.environ.get("OPENAI_MODEL", "gpt-5-nano")
        model_string = f"openai/{model}"
        logger.info("Using OpenAI API backend with model=%s", model)
    else:
        model = os.environ.get("VLLM_MODEL")
        if not model:
            raise RuntimeError("Set VLLM_MODEL env var (e.g. NextGLab/ORANSight_Qwen_14B_Instruct)")
        model_string = f"openai/{model}"
        logger.info("Using vLLM backend with model=%s", model)

    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

    # Phase 1: master run produces shared caches + theta_0.50 KG
    run_master(model_string)

    # Phase 2: remaining Θ variants reuse shared caches
    for theta in THETA_VALUES[1:]:
        try:
            run_theta_variant(theta, model_string)
        except Exception as e:
            logger.error("Θ=%.2f failed: %s", theta, e)

    # Phase 3: aggregate metrics CSV
    aggregate_metrics()


if __name__ == "__main__":
    main()
