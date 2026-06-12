"""KG-construction pipeline orchestrator (paper Sec. III-A).

Coordinates the 3-agent pipeline:
  PDF → Extractor → Aligner → Evaluator → Enriched Ontology

Features:
- JSON cache at each stage (output/ directory)
- Cost tracking via EnhancedLLMConfig.SessionMetrics
- Progress display via tqdm
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

# Rate-limit delay (seconds) between LLM calls for free-tier APIs
_RPM_DELAY = float(os.environ.get("TELCOAGENT_RPM_DELAY", "15"))

from tqdm import tqdm

from ..llm.config import EnhancedLLMConfig, get_active_agent_models
from ..ontology.core import get_default_ontology
from .aligner import AlignerAgent
from .evaluator import EvaluatorAgent
from .extractor import ExtractorAgent
from .kg_builder import KnowledgeGraphBuilder, save_graph_json
from .pdf_parser import parse_multiple_pdfs
from .schema import (
    AlignedEntity,
    ExtractedTriple,
    PipelineResult,
    ReprocessingFlag,
    SectionChunk,
    ValidatedTriple,
)

logger = logging.getLogger(__name__)


class KGConstructionPipeline:
    """Orchestrates the 3-agent (Extractor → Aligner → Evaluator) KG build."""

    def __init__(
        self,
        model: str = None,
        quality_threshold: float = 0.9,
        output_dir: str = "output",
        use_cache: bool = True,
        build_kg: bool = False,
        kg_output_path: Optional[str] = None,
        reader_delta: float = 0.0,
        summarize_threshold: int = 6000,
        max_section_chars: Optional[int] = 15000,
        enable_conflict_resolution: bool = True,
        use_embedding_align: bool = False,
        push_to_neo4j: bool = False,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        """Initialize the pipeline.

        Args:
            model: LiteLLM model identifier for all agents.
            quality_threshold: KARMA Θ — approval threshold on integrate_score = (C+Cl+R)/3.
            output_dir: Directory for cache and output files.
            use_cache: Whether to use cached intermediate results.
            build_kg: Whether to build a knowledge graph from approved triples.
            kg_output_path: Path for the knowledge graph JSON output.
            reader_delta: KARMA Reader δ threshold. 0.0 disables the filter (v1 behaviour).
            summarize_threshold: Body length above which the Summarizer is invoked.
                Setting a very large value effectively disables the Summarizer.
            enable_conflict_resolution: Run KARMA CRA on aligned pairs when True.
            use_embedding_align: Use sentence-transformers for KARMA SAA when True.
                Falls back silently to name-matching if the package is missing.
            push_to_neo4j: Push the built KG to Neo4j after Stage 5.  Requires
                ``build_kg=True``.  Connection params default to env vars
                NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD.
            neo4j_uri: Neo4j bolt URI (overrides NEO4J_URI env var).
            neo4j_user: Neo4j username (overrides NEO4J_USER env var).
            neo4j_password: Neo4j password (overrides NEO4J_PASSWORD env var).
        """
        self.quality_threshold = quality_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self.build_kg = build_kg
        self.kg_output_path = kg_output_path
        self.reader_delta = reader_delta
        self.summarize_threshold = summarize_threshold
        self.max_section_chars = max_section_chars
        self.enable_conflict_resolution = enable_conflict_resolution
        self.use_embedding_align = use_embedding_align
        self.push_to_neo4j = push_to_neo4j
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

        # Configure LLM with the specified model for all agents
        if model is None:
            model = get_active_agent_models()["kg_build"]
        agent_models = {
            "kg_build": model,
        }
        fallback_chains = {
            "kg_build": [model],
        }
        self.llm_config = EnhancedLLMConfig(
            agent_models=agent_models,
            fallback_chains=fallback_chains,
        )

        # KG build pipeline: must import ontology even when KG is absent.
        self.ontology = get_default_ontology(strict=False)
        self.extractor = ExtractorAgent(
            self.llm_config,
            ontology=self.ontology,
            reader_delta=reader_delta,
            summarize_threshold=summarize_threshold,
        )
        self.aligner = AlignerAgent(
            self.llm_config,
            self.ontology,
            enable_conflict_resolution=enable_conflict_resolution,
            use_embedding_align=use_embedding_align,
        )
        self.evaluator = EvaluatorAgent(
            self.llm_config,
            quality_threshold,
            ontology=self.ontology,
        )

    def _cache_path(self, stage: str) -> Path:
        """Get cache file path for a pipeline stage."""
        return self.output_dir / f"cache_{stage}.json"

    def _load_cache(self, stage: str) -> Optional[list]:
        """Load cached results for a stage if available."""
        if not self.use_cache:
            return None
        path = self._cache_path(stage)
        if path.exists():
            logger.info(f"Loading cached {stage} from {path}")
            with open(path) as f:
                return json.load(f)
        return None

    def _save_cache(self, stage: str, data: list):
        """Save stage results to cache."""
        path = self._cache_path(stage)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Cached {stage} to {path}")

    # ─── Stage 1: Parse PDFs ─────────────────────────────────────────────

    def _stage_parse(self, pdf_paths: List[str]) -> List[SectionChunk]:
        """Stage 1: Parse PDFs into section chunks."""
        cached = self._load_cache("sections")
        if cached is not None:
            return [SectionChunk(**c) for c in cached]

        chunks = parse_multiple_pdfs(
            pdf_paths,
            max_section_chars=self.max_section_chars,
        )

        self._save_cache("sections", [c.model_dump() for c in chunks])
        return chunks

    # ─── Stage 2: Extract Triples ────────────────────────────────────────

    def _stage_extract(self, chunks: List[SectionChunk]) -> List[ExtractedTriple]:
        """Stage 2: Extract triples from all chunks."""
        cached = self._load_cache("extracted")
        if cached is not None:
            return [ExtractedTriple(**t) for t in cached]

        all_triples = []
        for i, chunk in enumerate(tqdm(chunks, desc="Extracting triples")):
            if i > 0 and _RPM_DELAY > 0:
                time.sleep(_RPM_DELAY)
            triples = self.extractor.extract(chunk)
            all_triples.extend(triples)

        self._save_cache("extracted", [t.model_dump() for t in all_triples])
        return all_triples

    # ─── Stage 3: Align to Ontology ──────────────────────────────────────

    def _stage_align(self, triples: List[ExtractedTriple]):
        """Stage 3: Align triples to existing ontology."""
        cached = self._load_cache("aligned")
        if cached is not None:
            pairs = []
            for item in cached:
                t = ExtractedTriple(**item["triple"])
                a = AlignedEntity(**item["alignment"])
                pairs.append((t, a))
            return pairs

        pairs = self.aligner.align(triples)

        cache_data = [{"triple": t.model_dump(), "alignment": a.model_dump()} for t, a in pairs]
        self._save_cache("aligned", cache_data)
        return pairs

    # ─── Stage 4: Evaluate ───────────────────────────────────────────────

    def _stage_evaluate(self, aligned_pairs):
        """Stage 4: Evaluate aligned triples.

        The Evaluator produces Θ-agnostic 3-signal scores; ``approved`` is
        derived from ``self.quality_threshold`` and is therefore re-applied
        whenever a cached validated set is loaded.  This lets a Θ sweep
        re-use the expensive scoring pass across all variants.

        Returns:
            Tuple of (validated triples, reprocessing flags).
        """
        cached = self._load_cache("validated")
        if cached is not None:
            validated = [ValidatedTriple(**v) for v in cached]
            self._reapply_threshold(validated)
            return validated, []

        validated, flags = self.evaluator.evaluate(aligned_pairs)

        self._save_cache("validated", [v.model_dump() for v in validated])
        return validated, flags

    def _reapply_threshold(self, validated: list) -> None:
        """Recompute ``approved`` from cached scores using the current Θ.

        ``cache_validated.json`` is shared across Θ variants in
        ``run_kg_sweep.py``; only the threshold filter changes per variant.
        """
        thr = self.quality_threshold
        for v in validated:
            v.approved = bool(v.integrate_score + 1e-9 >= thr)

    # ─── Stage 5: Build Knowledge Graph ─────────────────────────────────

    def _stage_build_kg(self, result: PipelineResult, kg_output_path: Optional[str] = None):
        """Stage 5: Build knowledge graph from approved triples."""
        builder = KnowledgeGraphBuilder()
        graph = builder.build_from_pipeline_result(result)

        output_path = kg_output_path or str(self.output_dir / "knowledge_graph.json")
        save_graph_json(graph, output_path)
        print(f"  → {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"  → Saved to {output_path}")

        result.kg_graph_path = output_path

        stats = builder.get_stats()
        logger.info("KG stats: %s", stats)

    # ─── Main Pipeline ───────────────────────────────────────────────────

    def run(
        self,
        pdf_paths: List[str],
        output_path: Optional[str] = None,
    ) -> PipelineResult:
        """Run the full TelcoAgent-RAG pipeline.

        Args:
            pdf_paths: Paths to 3GPP PDF files.
            output_path: Path for the enriched ontology JSON output.

        Returns:
            PipelineResult with all validated triples and statistics.
        """
        start_time = time.time()

        total_stages = 4
        if self.build_kg:
            total_stages += 1
        if self.push_to_neo4j and self.build_kg:
            total_stages += 1

        print("=" * 60)
        print("TelcoAgent-RAG: 3GPP Ontology Builder")
        print("=" * 60)
        print(f"Input PDFs: {len(pdf_paths)}")
        print(f"Quality threshold: {self.quality_threshold}")
        print(f"Output dir: {self.output_dir}")
        if self.build_kg:
            print("Knowledge graph: enabled")
        print()

        # Stage 1: Parse
        print(f"[1/{total_stages}] Parsing PDFs...")
        chunks = self._stage_parse(pdf_paths)
        print(f"  → {len(chunks)} section chunks")

        # Stage 2: Extract
        print(f"[2/{total_stages}] Extracting triples...")
        triples = self._stage_extract(chunks)
        print(f"  → {len(triples)} raw triples")

        # Stage 3: Align
        print(f"[3/{total_stages}] Aligning to ontology...")
        aligned_pairs = self._stage_align(triples)
        print(f"  → {len(aligned_pairs)} aligned pairs")

        # Stage 4: Evaluate
        print(f"[4/{total_stages}] Evaluating quality...")
        validated, flags = self._stage_evaluate(aligned_pairs)

        # ── Feedback loop (max 1 iteration) ──────────────────────────────
        feedback_iterations = 0
        if flags and len(flags) > 0:
            feedback_iterations = 1
            print(f"  [Feedback] {len(flags)} triples flagged for reprocessing")

            # Invalidate evaluation cache for the reprocessing pass
            eval_cache = self._cache_path("validated")
            if eval_cache.exists():
                eval_cache.unlink()

            extractor_flags = [f for f in flags if f["target_agent"] == "extractor"]
            aligner_flags = [f for f in flags if f["target_agent"] == "aligner"]

            # Re-extract flagged chunks
            if extractor_flags:
                flagged_specs = set()
                for f in extractor_flags:
                    idx = f["triple_index"]
                    if idx < len(aligned_pairs):
                        t, _ = aligned_pairs[idx]
                        flagged_specs.add((t.source_spec, t.source_clause))

                re_extracted = []
                for chunk in chunks:
                    if (chunk.spec, chunk.clause) in flagged_specs:
                        new_triples = self.extractor.extract(chunk)
                        re_extracted.extend(new_triples)

                if re_extracted:
                    print(f"  [Feedback] Re-extracted {len(re_extracted)} triples")
                    # Replace old triples from those chunks and re-align
                    old_subjects = {(s, c) for s, c in flagged_specs}
                    triples = [
                        t for t in triples if (t.source_spec, t.source_clause) not in old_subjects
                    ] + re_extracted
                    aligned_pairs = self.aligner.align(triples)

            # Re-align flagged triples
            if aligner_flags:
                flagged_triples = []
                unflagged_pairs = []
                flagged_indices = {f["triple_index"] for f in aligner_flags}
                for idx, pair in enumerate(aligned_pairs):
                    if idx in flagged_indices:
                        flagged_triples.append(pair[0])
                    else:
                        unflagged_pairs.append(pair)

                if flagged_triples:
                    re_aligned = self.aligner.align(flagged_triples)
                    print(f"  [Feedback] Re-aligned {len(re_aligned)} triples")
                    aligned_pairs = unflagged_pairs + re_aligned

            # Re-evaluate
            print("  [Feedback] Re-evaluating...")
            validated, _ = self._stage_evaluate(aligned_pairs)

        # Build result
        approved = [v for v in validated if v.approved]
        rejected = [v for v in validated if not v.approved]
        new_entities = [v.alignment for v in approved if v.alignment.is_new]
        mapped_entities = [v.alignment for v in approved if not v.alignment.is_new]

        reprocessing_flag_models = [ReprocessingFlag(**f) for f in flags] if flags else []

        result = PipelineResult(
            total_sections=len(chunks),
            total_extracted=len(triples),
            total_aligned=len(aligned_pairs),
            total_validated=len(validated),
            total_approved=len(approved),
            approved_triples=approved,
            rejected_triples=rejected,
            new_entities=new_entities,
            mapped_entities=mapped_entities,
            reprocessing_flags=reprocessing_flag_models,
            feedback_iteration_count=feedback_iterations,
        )

        # Stage 5: Build Knowledge Graph (optional)
        if self.build_kg:
            print(f"[5/{total_stages}] Building knowledge graph...")
            self._stage_build_kg(result, self.kg_output_path)

        # Stage 6: Push KG to Neo4j (optional)
        if self.push_to_neo4j and self.build_kg:
            print(f"[6/{total_stages}] Pushing knowledge graph to Neo4j...")
            try:
                from ..stores import Neo4jConnection, Neo4jOntologyStore

                conn = Neo4jConnection(
                    uri=self.neo4j_uri,
                    user=self.neo4j_user,
                    password=self.neo4j_password,
                )
                store = Neo4jOntologyStore(conn=conn)
                kg_path = result.kg_graph_path or str(self.output_dir / "knowledge_graph.json")
                push_stats = store.push_kg_json(kg_path)
                print(
                    f"  → {push_stats['nodes_merged']} nodes, "
                    f"{push_stats['edges_merged']} edges merged into Neo4j"
                )
            except Exception as exc:
                logger.warning("Neo4j push failed: %s", exc)
                print(f"  [Warning] Neo4j push failed: {exc}")
        elif self.push_to_neo4j and not self.build_kg:
            logger.warning("push_to_neo4j=True requires build_kg=True; skipping Neo4j push")

        # Save output
        if output_path is None:
            output_path = str(self.output_dir / "enriched_ontology.json")
        self._save_output(result, output_path)

        # Print summary
        elapsed = time.time() - start_time
        self._print_summary(result, elapsed)

        return result

    def _save_output(self, result: PipelineResult, output_path: str):
        """Save enriched ontology JSON."""
        output = {
            "metadata": {
                "total_sections": result.total_sections,
                "total_extracted": result.total_extracted,
                "total_aligned": result.total_aligned,
                "total_validated": result.total_validated,
                "total_approved": result.total_approved,
                "quality_threshold": self.quality_threshold,
            },
            "approved_triples": [v.model_dump() for v in result.approved_triples],
            "new_entities": [e.model_dump() for e in result.new_entities],
            "mapped_entities": [e.model_dump() for e in result.mapped_entities],
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nOutput saved to: {output_path}")

    def _print_summary(self, result: PipelineResult, elapsed: float):
        """Print pipeline execution summary."""
        print()
        print("=" * 60)
        print("Pipeline Summary")
        print("=" * 60)
        print(f"Sections parsed:    {result.total_sections}")
        print(f"Triples extracted:  {result.total_extracted}")
        print(f"Triples aligned:    {result.total_aligned}")
        print(f"Triples validated:  {result.total_validated}")
        print(f"Triples approved:   {result.total_approved}")
        print(f"Triples rejected:   {result.total_validated - result.total_approved}")
        print(f"New entities:       {len(result.new_entities)}")
        print(f"Mapped to existing: {len(result.mapped_entities)}")

        # KARMA per-stage stats
        print()
        print("KARMA pipeline stats")
        print("-" * 60)
        print(f"Reader δ:               {self.reader_delta}")
        print(f"Sections skipped (RA):  {self.extractor.skipped_by_reader}")
        print(f"Sections summarized:    {self.extractor.summarized_count}")
        print(f"CRA conflicts seen:     {self.aligner.conflicts_seen}")
        print(f"CRA conflicts resolved: {self.aligner.conflicts_resolved}")
        print(f"Embedding alignment:    {'on' if self.aligner.use_embedding_align else 'off'}")

        if result.total_validated > 0:
            approved = [v for v in result.approved_triples]
            if approved:
                avg_c = sum(v.confidence for v in approved) / len(approved)
                avg_cl = sum(v.clarity for v in approved) / len(approved)
                avg_r = sum(v.relevance for v in approved) / len(approved)
                avg_int = sum(v.integrate_score for v in approved) / len(approved)
                print(f"Avg confidence (approved): {avg_c:.3f}")
                print(f"Avg clarity    (approved): {avg_cl:.3f}")
                print(f"Avg relevance  (approved): {avg_r:.3f}")
                print(f"Avg integrate  (approved): {avg_int:.3f}  (Θ={self.quality_threshold})")

        if result.feedback_iteration_count > 0:
            print(f"Feedback iterations: {result.feedback_iteration_count}")
            print(f"Reprocessing flags: {len(result.reprocessing_flags)}")
        print(f"Elapsed time:       {elapsed:.1f}s")

        # LLM cost summary
        self.llm_config.print_session_summary()
