"""Knowledge-graph construction pipeline (paper Sec. III-A).

KARMA-inspired three-agent pipeline that automatically extracts,
aligns, and validates KPI/measurement knowledge from 3GPP
specification PDFs into ``data/enriched_kg.json``. Runs offline; the
Prediction and Explanation pipelines only ever read its output.

Agents:
    ExtractorAgent: PDF section → raw triples            (extractor.py)
    AlignerAgent:   raw triples + ontology → aligned     (aligner.py)
    EvaluatorAgent: aligned triples → validated triples  (evaluator.py)

Usage:
    from telcoagent.kg_construction import KGConstructionPipeline

    pipeline = KGConstructionPipeline()
    result = pipeline.run(pdf_paths=["data/specs/TS_28.554.pdf"])
"""

from .aligner import AlignerAgent
from .evaluator import EvaluatorAgent
from .extractor import ExtractorAgent
from .kg_builder import (
    KnowledgeGraphBuilder,
    graph_to_json,
    json_to_graph,
    load_graph_json,
    save_graph_json,
)
from .pdf_parser import parse_multiple_pdfs, parse_pdf
from .pipeline import KGConstructionPipeline
from .schema import (
    AlignedEntity,
    ExtractedTriple,
    PipelineResult,
    ReprocessingFlag,
    SectionChunk,
    ValidatedTriple,
)
from .tools import make_aligner_tools, make_evaluator_tools, make_extractor_tools

__all__ = [
    # Schema
    "SectionChunk",
    "ExtractedTriple",
    "AlignedEntity",
    "ValidatedTriple",
    "PipelineResult",
    "ReprocessingFlag",
    # Parser
    "parse_pdf",
    "parse_multiple_pdfs",
    # Agents
    "ExtractorAgent",
    "AlignerAgent",
    "EvaluatorAgent",
    # Pipeline
    "KGConstructionPipeline",
    # ReAct Tools
    "make_extractor_tools",
    "make_aligner_tools",
    "make_evaluator_tools",
    # KG Builder
    "KnowledgeGraphBuilder",
    "graph_to_json",
    "json_to_graph",
    "save_graph_json",
    "load_graph_json",
]
