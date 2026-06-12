"""Aligner agent — raw triples → ontology-aligned entities (paper Sec. III-A, step 2).

Second of the three KG-construction agents. Maps every extracted
entity onto the canonical 3GPP ontology (normalizing inconsistent
terminology into single nodes), optionally assisted by embedding
nearest-neighbour alignment (KARMA SAA), and resolves contradictory
triples through the KARMA Conflict-Resolution debate (CRA).

Input:  :class:`telcoagent.kg_construction.schema.ExtractedTriple` list
        from the Extractor agent.
Output: ``(ExtractedTriple, AlignedEntity)`` pairs, consumed by the
        Evaluator agent.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

from telcoagent.agents.base import BaseTrainingFreeAgent

# Task-target KPIs (the columns the Predictor forecasts).  This is *task
# config*, not telecom knowledge — listing which fields the dataset CSV
# carries, no thresholds or causality.  We use it only to make sure the
# Aligner classifies these short_names as ``KPIDefinition`` so downstream
# KG queries find them as KPI nodes instead of generic Entity/Measurement
# nodes.
from telcoagent.config import CORE_KPI_NAMES

from ..llm.config import EnhancedLLMConfig
from ..ontology.core import ThreeGPPOntology
from .prompts import (
    ALIGNER_SYSTEM,
    ALIGNER_USER_TEMPLATE,
    CONFLICT_RESOLVER_SYSTEM,
    CONFLICT_RESOLVER_USER_TEMPLATE,
)
from .schema import AlignedEntity, ExtractedTriple
from .shared import _RPM_DELAY, _parse_json_response
from .tools import make_aligner_tools, make_conflict_resolver_tools

logger = logging.getLogger(__name__)

CORE_KPI_TARGETS: frozenset = frozenset(CORE_KPI_NAMES)


def _enforce_core_kpi_alignment(a: "AlignedEntity") -> None:
    """If an alignment names one of the 7 task-target KPIs, make sure it
    is classified as ``KPIDefinition`` and pinned to the canonical short
    name.  No-op for any other entity.

    The match is intentionally permissive (aligned_name OR mapped_to OR
    original_name): the LLM Aligner sometimes labels DL_iBler-derived
    entities as ``Measurement`` or ``Entity`` and we want them surfaced
    as KPI nodes downstream.
    """
    candidates = {a.aligned_name, a.mapped_to, a.original_name}
    canonical = next(
        (c for c in candidates if c and c in CORE_KPI_TARGETS),
        None,
    )
    if canonical is None:
        return
    a.entity_type = "KPIDefinition"
    a.aligned_name = canonical
    a.mapped_to = canonical
    a.is_new = False
    if a.confidence < 0.95:
        a.confidence = 0.95


class AlignerAgent:
    """Aligns extracted entities to the existing 3GPP ontology schema.

    Also handles KARMA Conflict Resolution (CRA) via `resolve_conflicts`
    and optional embedding-based alignment via `_embed_align`.
    """

    def __init__(
        self,
        llm_config: EnhancedLLMConfig,
        ontology: ThreeGPPOntology,
        enable_conflict_resolution: bool = True,
        use_embedding_align: bool = False,
        embedding_model: Optional[str] = None,
    ):
        self.llm = llm_config
        self.ontology = ontology
        self.enable_conflict_resolution = enable_conflict_resolution
        self.use_embedding_align = use_embedding_align
        # Resolution order: explicit arg > env var > sentence-transformers default.
        # Use "openai/<model>" (e.g. "openai/text-embedding-3-large") to route
        # through the OpenAI Embeddings API instead of a local SBERT model.
        self._embedding_model_name = (
            embedding_model
            or os.environ.get("TELCOAGENT_EMBEDDING_MODEL")
            or "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._ontology_summary = self._build_ontology_summary()
        self._react_agent: Optional[BaseTrainingFreeAgent] = None
        self._tool_registry = None
        # CRA stats — read by orchestrator summary
        self.conflicts_seen = 0
        self.conflicts_resolved = 0

        # Embedding backend (lazily initialised when use_embedding_align == True)
        self._embedder = None  # SBERT model OR OpenAI client
        self._embedder_kind: Optional[str] = None  # "sbert" | "openai"
        self._embedder_model_id: Optional[str] = None  # actual model id sent to backend
        self._ontology_index = None  # list of (short_name, vector)

        model = llm_config.agent_models["kg_build"]
        self._react_agent = BaseTrainingFreeAgent(model)
        self._tool_registry = make_aligner_tools(ontology)
        self._cra_tool_registry = make_conflict_resolver_tools(ontology)

    def _embed_texts(self, texts: List[str]):
        """Encode texts → L2-normalised numpy array. Backend-agnostic.

        Returns an empty (0, dim) array when ``texts`` is empty so callers
        don't accidentally fire an HTTP request with input=[] (the OpenAI
        embeddings endpoint rejects empty arrays with HTTP 400).
        """
        if not texts:
            return self._np.zeros((0, 0), dtype=self._np.float32)
        if self._embedder_kind == "openai":
            resp = self._embedder.embeddings.create(
                model=self._embedder_model_id,
                input=texts,
            )
            vecs = self._np.array([d.embedding for d in resp.data], dtype=self._np.float32)
            norms = self._np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            return vecs / norms
        # SBERT path (already normalised)
        return self._embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def _ensure_embedder(self) -> bool:
        """Lazily load the embedding backend. Returns False if unavailable.

        Routes by ``self._embedding_model_name`` prefix:
          - "openai/<model>"  → OpenAI Embeddings API (no GPU, requires OPENAI_API_KEY)
          - anything else     → local sentence-transformers model
        """
        if self._embedder is not None:
            return True

        try:
            import numpy as np  # type: ignore
        except ImportError:
            logger.warning("numpy missing; embedding alignment disabled.")
            self.use_embedding_align = False
            return False
        self._np = np

        name = self._embedding_model_name
        if name.startswith("openai/"):
            try:
                from openai import OpenAI  # type: ignore
            except ImportError:
                logger.warning(
                    "openai package not installed; embedding alignment disabled. "
                    "Install with `pip install openai`."
                )
                self.use_embedding_align = False
                return False
            if not os.environ.get("OPENAI_API_KEY"):
                logger.warning("OPENAI_API_KEY not set; embedding alignment disabled.")
                self.use_embedding_align = False
                return False
            self._embedder = OpenAI()
            self._embedder_kind = "openai"
            self._embedder_model_id = name.split("/", 1)[1]
            logger.info("Embedder backend: OpenAI %s", self._embedder_model_id)
        else:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed; embedding alignment disabled. "
                    "Install with `pip install sentence-transformers` to enable."
                )
                self.use_embedding_align = False
                return False
            self._embedder = SentenceTransformer(name)
            self._embedder_kind = "sbert"
            self._embedder_model_id = name
            logger.info("Embedder backend: SBERT %s", name)

        # Build ontology index using whichever backend was selected.
        # KPIDefinition is the slim KG-loaded shape: it exposes short_name,
        # description, formula, unit — no separate `name`, no `category`,
        # no `kpi_type`.  During a KG *build* run the ontology is empty
        # (the KG doesn't exist yet), in which case we disable SAA entirely
        # because there is nothing to align against.
        docs: List[str] = []
        names: List[str] = []
        for short_name, kpi in self.ontology.kpis.items():
            desc = kpi.description or ""
            docs.append(f"{short_name}. {desc}".strip())
            names.append(short_name)
        if not docs:
            logger.warning(
                "Ontology has 0 KPIDefinition entries; embedding alignment "
                "is disabled for this run.  This is expected during a KG "
                "build before any KG exists.",
            )
            self.use_embedding_align = False
            return False
        vecs = self._embed_texts(docs)
        self._ontology_index = list(zip(names, vecs))
        return True

    def _embed_align(
        self,
        entity_names: List[str],
    ) -> Dict[str, Tuple[str, float]]:
        """KARMA SAA: for each entity, ê = argmin d(φ(e), ψ(v)) over the ontology.

        Returns a dict mapping entity_name -> (best_short_name, cosine_distance)
        where distance = 1 - cos_sim. Returns empty dict when the embedder is
        unavailable.
        """
        if not self._ensure_embedder():
            return {}
        vecs = self._embed_texts(entity_names)
        ont_matrix = self._np.stack([v for _, v in self._ontology_index])
        result: Dict[str, Tuple[str, float]] = {}
        for name, v in zip(entity_names, vecs):
            sims = ont_matrix @ v  # already normalized
            best_idx = int(self._np.argmax(sims))
            best_name = self._ontology_index[best_idx][0]
            distance = float(1.0 - sims[best_idx])
            result[name] = (best_name, distance)
        return result

    def _build_ontology_summary(self) -> str:
        """Build a concise summary of existing ontology KPIs.

        KPIDefinition exposes only KG-derived fields (short_name, description,
        unit, ...).  The legacy ``name`` and ``category`` fields were retired
        with the rest of the hand-curated ontology.
        """
        lines = []
        for short_name, kpi in self.ontology.kpis.items():
            desc = kpi.description or "(no description)"
            unit = kpi.unit or "unitless"
            lines.append(f"- {short_name}: {desc} [{unit}]")
        return "\n".join(lines)

    def align(
        self,
        triples: List[ExtractedTriple],
        batch_size: int = 20,
    ) -> List[Tuple[ExtractedTriple, AlignedEntity]]:
        """Align extracted triples to the existing ontology.

        Runs schema alignment first, then KARMA conflict resolution (CRA) if
        `enable_conflict_resolution` is True.
        """
        unique_subjects = list({t.subject for t in triples})
        logger.info(f"Aligning {len(unique_subjects)} unique entities")

        # Optional KARMA SAA: pre-compute embedding-nearest ontology entity
        emb_map: Dict[str, Tuple[str, float]] = {}
        if self.use_embedding_align and unique_subjects:
            emb_map = self._embed_align(unique_subjects)

        alignment_map: Dict[str, AlignedEntity] = {}
        for i in range(0, len(unique_subjects), batch_size):
            if i > 0 and _RPM_DELAY > 0:
                time.sleep(_RPM_DELAY)
            batch = unique_subjects[i : i + batch_size]
            batch_alignments = self._align_batch(batch)
            for a in batch_alignments:
                # Stamp embedding distance when available
                if a.original_name in emb_map:
                    a.embedding_distance = emb_map[a.original_name][1]
                _enforce_core_kpi_alignment(a)
                alignment_map[a.original_name] = a

        results = []
        for triple in triples:
            alignment = alignment_map.get(triple.subject)
            if alignment is None:
                emb_dist = emb_map.get(triple.subject, (None, None))[1]
                alignment = AlignedEntity(
                    original_name=triple.subject,
                    aligned_name=triple.subject.replace(" ", "_"),
                    entity_type="NEW_TYPE",
                    is_new=True,
                    mapped_to=None,
                    confidence=0.3,
                    embedding_distance=emb_dist,
                )
                _enforce_core_kpi_alignment(alignment)
            results.append((triple, alignment))

        if self.enable_conflict_resolution:
            results = self.resolve_conflicts(results)

        return results

    # ─── KARMA CRA: LLM debate over contradictory triples ───────────────

    @staticmethod
    def _triple_id(t: ExtractedTriple) -> str:
        return f"{t.source_spec}|{t.source_clause}|{t.subject}"

    def resolve_conflicts(
        self,
        pairs: List[Tuple[ExtractedTriple, AlignedEntity]],
    ) -> List[Tuple[ExtractedTriple, AlignedEntity]]:
        """Detect triples sharing (aligned_subject, predicate) and drop contradictions.

        For each conflict pair, runs the KARMA CRA LLM debate. If the verdict is
        "contradict", the lower-relation_prob triple is dropped; ties fall back
        to preferring the more specific (longer dotted) clause. `conflicts_with`
        is not stored on ExtractedTriple (which is immutable-in-intent); instead
        the EvaluatorAgent sees the reduced set and the surviving triple's
        quality signals reflect that it won the debate.
        """
        # Group by (aligned_name, predicate)
        groups: Dict[Tuple[str, str], List[int]] = {}
        for idx, (t, a) in enumerate(pairs):
            key = (a.aligned_name, t.predicate)
            groups.setdefault(key, []).append(idx)

        to_drop: set = set()
        for key, indices in groups.items():
            if len(indices) < 2:
                continue
            # Cap pairs per group to avoid O(n²) explosion on large document sets.
            # Top-10 by relation_prob; ties broken by clause specificity.
            if len(indices) > 10:
                indices = sorted(
                    indices,
                    key=lambda idx: (
                        -(pairs[idx][0].relation_prob or 0.0),
                        -len(pairs[idx][0].source_clause or ""),
                    ),
                )[:10]
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    if indices[i] in to_drop or indices[j] in to_drop:
                        continue
                    t_i, _ = pairs[indices[i]]
                    t_j, _ = pairs[indices[j]]
                    # Same object → just a restatement, skip
                    if t_i.object.strip() == t_j.object.strip():
                        continue
                    self.conflicts_seen += 1
                    verdict, reject_local = self._debate(t_i, t_j)
                    if verdict == "contradict":
                        reject_abs = indices[i] if reject_local == 0 else indices[j]
                        to_drop.add(reject_abs)
                        self.conflicts_resolved += 1
                        logger.info(
                            f"CRA: dropped {self._triple_id(pairs[reject_abs][0])} "
                            f"(conflict with {self._triple_id(pairs[indices[j] if reject_local == 0 else indices[i]][0])})"
                        )

        if to_drop:
            logger.info(
                f"CRA: resolved {self.conflicts_resolved} conflicts, kept {len(pairs) - len(to_drop)} / {len(pairs)} pairs"
            )
        return [p for idx, p in enumerate(pairs) if idx not in to_drop]

    def _debate(
        self,
        t0: ExtractedTriple,
        t1: ExtractedTriple,
    ) -> Tuple[str, int]:
        """Run the KARMA CRA debate on a candidate conflict pair.

        Returns (verdict, reject_index). reject_index is meaningful only when
        verdict == "contradict".
        """

        # Heuristic tie-break helper — used if LLM fails or doesn't decide.
        def heuristic_reject() -> int:
            # Prefer dropping the lower relation_prob, else the shorter clause.
            if t0.relation_prob != t1.relation_prob:
                return 0 if t0.relation_prob < t1.relation_prob else 1
            depth0 = t0.source_clause.count(".")
            depth1 = t1.source_clause.count(".")
            return 0 if depth0 < depth1 else 1

        triple_0_repr = json.dumps(
            {
                "subject": t0.subject,
                "predicate": t0.predicate,
                "object": t0.object,
                "triple_type": t0.triple_type,
                "source_spec": t0.source_spec,
                "source_clause": t0.source_clause,
                "relation_prob": t0.relation_prob,
            },
            indent=2,
        )
        triple_1_repr = json.dumps(
            {
                "subject": t1.subject,
                "predicate": t1.predicate,
                "object": t1.object,
                "triple_type": t1.triple_type,
                "source_spec": t1.source_spec,
                "source_clause": t1.source_clause,
                "relation_prob": t1.relation_prob,
            },
            indent=2,
        )
        prompt = CONFLICT_RESOLVER_USER_TEMPLATE.format(
            triple_0=triple_0_repr,
            triple_1=triple_1_repr,
        )

        try:
            if self._react_agent and self._cra_tool_registry:
                try:
                    text, _ = self._react_agent.react_generate(
                        system_prompt=CONFLICT_RESOLVER_SYSTEM,
                        user_prompt=prompt,
                        tool_registry=self._cra_tool_registry,
                        max_steps=5,
                        temperature=0.2,
                        step_delay=_RPM_DELAY,
                    )
                except Exception as e:
                    logger.warning(f"CRA ReAct failed, falling back to one-shot: {e}")
                    text, _ = self.llm.call_with_retry(
                        prompt=prompt,
                        agent_type="kg_build",
                        role="kg_build",
                        system_message=CONFLICT_RESOLVER_SYSTEM,
                    )
            else:
                text, _ = self.llm.call_with_retry(
                    prompt=prompt,
                    agent_type="kg_build",
                    role="kg_build",
                    system_message=CONFLICT_RESOLVER_SYSTEM,
                )

            stripped = text.strip()
            if stripped.startswith("```"):
                stripped = "\n".join(
                    ln for ln in stripped.split("\n") if not ln.strip().startswith("```")
                )
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start == -1 or end == -1:
                return ("agree", 0)
            obj = json.loads(stripped[start : end + 1])
            verdict = str(obj.get("verdict", "agree")).lower().strip()
            if verdict != "contradict":
                return ("agree", 0)
            reject = obj.get("reject_index")
            if reject not in (0, 1):
                reject = heuristic_reject()
            return ("contradict", int(reject))
        except Exception as e:
            logger.warning(f"CRA debate failed: {e}; treating as agree")
            return ("agree", 0)

    def _align_batch(self, entity_names: List[str]) -> List[AlignedEntity]:
        """Align a batch of entity names to the ontology."""
        entities_json = json.dumps(
            [{"name": name} for name in entity_names],
            indent=2,
        )

        prompt = ALIGNER_USER_TEMPLATE.format(
            ontology_summary=self._ontology_summary,
            entities_json=entities_json,
        )

        try:
            response_text = self._call_llm(ALIGNER_SYSTEM, prompt)
            raw_alignments = _parse_json_response(response_text)

            alignments = []
            for ra in raw_alignments:
                alignments.append(
                    AlignedEntity(
                        original_name=ra.get("original_name", ""),
                        aligned_name=ra.get("aligned_name", ""),
                        entity_type=ra.get("entity_type", "NEW_TYPE"),
                        is_new=ra.get("is_new", True),
                        mapped_to=ra.get("mapped_to"),
                        confidence=float(ra.get("confidence", 0.5)),
                    )
                )

            logger.info(f"Aligned {len(alignments)} entities")
            return alignments

        except Exception as e:
            logger.error(f"Alignment batch failed: {e}")
            return [
                AlignedEntity(
                    original_name=name,
                    aligned_name=name.replace(" ", "_"),
                    entity_type="NEW_TYPE",
                    is_new=True,
                    mapped_to=None,
                    confidence=0.1,
                )
                for name in entity_names
            ]

    def _call_llm(self, system: str, prompt: str) -> str:
        """Call LLM via ReAct."""
        if not self._react_agent or not self._tool_registry:
            raise RuntimeError(
                "AlignerAgent requires a react_agent and tool_registry; "
                "pass them to the constructor."
            )
        text, _ = self._react_agent.react_generate(
            system_prompt=system,
            user_prompt=prompt,
            tool_registry=self._tool_registry,
            max_steps=5,
            temperature=0.3,
            step_delay=_RPM_DELAY,
        )
        _parse_json_response(text)  # validate parseable
        return text
