"""Pydantic data models for TelcoAgent-RAG pipeline.

Defines intermediate data structures flowing between the 3 LLM agents:
  Extractor → Aligner → Evaluator

Based on KARMA-inspired extraction/schema-mapping separation.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

# =============================================================================
# PDF Parser Output
# =============================================================================


class SectionChunk(BaseModel):
    """A parsed section from a 3GPP specification PDF."""

    spec: str = Field(description="Spec identifier, e.g. 'TS 28.552'")
    clause: str = Field(description="Section number, e.g. '5.1.1.2.1'")
    title: str = Field(description="Section title")
    body: str = Field(description="Section body text")
    tables: List[str] = Field(default_factory=list, description="Tables as text")
    page_start: int = Field(default=0, description="Starting page number")


# =============================================================================
# Agent 1: Extractor Output
# =============================================================================


class ExtractedTriple(BaseModel):
    """Raw triple extracted from a PDF section by the ExtractorAgent."""

    subject: str = Field(description="Entity name, e.g. 'DL PRB Usage'")
    predicate: str = Field(description="Relation, e.g. 'calculated_by'")
    object: str = Field(description="Target, e.g. 'PRBUsedDl.Avg / PRBTotalDl'")
    triple_type: str = Field(
        description="One of: kpi_definition, measurement, relationship, causal_chain"
    )
    source_spec: str = Field(description="Source specification, e.g. 'TS 28.552'")
    source_clause: str = Field(description="Source clause, e.g. '5.1.1.2.1'")
    raw_text: str = Field(description="Original text snippet from the PDF")
    # KARMA REA: p(r | ê_i, ê_j). 1.0 when the LLM does not emit a probability.
    relation_prob: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="KARMA relationship probability p(r|ê_i, ê_j).",
    )
    # Causal-strength fields populated only for relationship / causal_chain triples.
    min_pct_source: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="% change in source below which the effect is negligible (KARMA causal threshold).",
    )
    min_pct_target: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="% change in target that qualifies as a consistent reaction.",
    )
    mechanism: Optional[str] = Field(
        default=None,
        description="One short sentence (<=20 words) describing the physical cause.",
    )
    fine_clause: Optional[str] = Field(
        default=None,
        description="Finer spec clause (e.g., '7.1.4') when more specific than source_clause.",
    )


# =============================================================================
# Agent 2: Aligner Output
# =============================================================================


class AlignedEntity(BaseModel):
    """Entity aligned to the existing ontology schema by the AlignerAgent."""

    original_name: str = Field(description="Name as extracted from PDF")
    aligned_name: str = Field(
        description="Normalized name following ontology short_name convention"
    )
    entity_type: str = Field(description="One of: KPIDefinition, Measurement, NEW_TYPE")
    is_new: bool = Field(description="True if not found in existing ontology")
    mapped_to: Optional[str] = Field(
        default=None, description="Existing KPI short_name if mapped, None if new"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Alignment confidence")
    # KARMA SAA: d(φ(e), ψ(v)) — distance between extracted entity embedding and
    # ontology entity embedding. None when embedding alignment is disabled.
    embedding_distance: Optional[float] = Field(
        default=None,
        description="Distance in embedding space to the aligned ontology entity.",
    )


# =============================================================================
# Agent 3: Evaluator Output
# =============================================================================


class ValidatedTriple(BaseModel):
    """Final validated triple from the EvaluatorAgent.

    Uses KARMA 3-signal scoring: confidence (M_Con), clarity (M_Cla),
    relevance (M_Rel). integrate_score = (C+Cl+R)/3, and the legacy
    quality_score is mirrored to integrate_score for kg_builder compatibility.
    """

    triple: ExtractedTriple
    alignment: AlignedEntity
    quality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Legacy overall score, mirrored from integrate_score.",
    )
    issues: List[str] = Field(default_factory=list, description="Identified issues")
    approved: bool = Field(description="True if integrate_score >= Θ")
    # KARMA EA 3-signal decomposition
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="C(t) — factual correctness / confidence (KARMA M_Con).",
    )
    clarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cl(t) — structural/semantic clarity (KARMA M_Cla).",
    )
    relevance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="R(t) — relevance to the 3GPP ontology scope (KARMA M_Rel).",
    )
    integrate_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="(C + Cl + R) / 3 — KARMA integrate(t).",
    )
    conflicts_with: List[str] = Field(
        default_factory=list,
        description="Identifiers (spec|clause|subject) of triples found to contradict this one by the AlignerAgent CRA step.",
    )


# =============================================================================
# Feedback / Reprocessing
# =============================================================================


class ReprocessingFlag(BaseModel):
    """Flag raised by EvaluatorAgent requesting reprocessing of a triple."""

    triple_index: int = Field(description="Index of the flagged triple in the batch")
    reason: str = Field(description="Why reprocessing is needed")
    target_agent: str = Field(description="'extractor' or 'aligner'")


# =============================================================================
# Pipeline Summary
# =============================================================================


class PipelineResult(BaseModel):
    """Aggregated result of the full TelcoAgent-RAG pipeline."""

    total_sections: int = 0
    total_extracted: int = 0
    total_aligned: int = 0
    total_validated: int = 0
    total_approved: int = 0
    approved_triples: List[ValidatedTriple] = Field(default_factory=list)
    rejected_triples: List[ValidatedTriple] = Field(default_factory=list)
    new_entities: List[AlignedEntity] = Field(default_factory=list)
    mapped_entities: List[AlignedEntity] = Field(default_factory=list)
    reprocessing_flags: List[ReprocessingFlag] = Field(
        default_factory=list, description="Flags raised by evaluator for reprocessing"
    )
    feedback_iteration_count: int = Field(
        default=0, description="Number of feedback iterations executed"
    )
    kg_graph_path: Optional[str] = Field(
        default=None, description="Path to the serialized knowledge graph JSON"
    )
