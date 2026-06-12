"""ReAct Tool Factories for TelcoAgent-RAG agents.

Follows the ToolRegistry / _schema pattern from telcoagent.agents.registry.
Each factory binds tools to a ThreeGPPOntology instance via closures.
"""

import logging
from typing import Any, Dict, List, Optional

from ..agents.registry import ToolRegistry, _schema
from ..ontology.core import ThreeGPPOntology

logger = logging.getLogger(__name__)


# =============================================================================
# Shared helpers (deduplicate across factories)
# =============================================================================


def _search_ontology_impl(ontology: ThreeGPPOntology, query: str) -> Dict[str, Any]:
    """Fuzzy-search KPI definitions in the spec-extracted ontology."""
    query_lower = query.lower()
    matches = []
    for short_name, kpi in ontology.kpis.items():
        score = 0
        if query_lower in short_name.lower():
            score += 2
        if kpi.description and query_lower in kpi.description.lower():
            score += 1
        if any(query_lower in alias.lower() for alias in kpi.original_names):
            score += 1
        if score > 0:
            matches.append(
                {
                    "short_name": short_name,
                    "aliases": list(kpi.original_names),
                    "score": score,
                }
            )
    matches.sort(key=lambda m: m["score"], reverse=True)
    return {"query": query, "matches": matches[:10]}


def _get_kpi_details_impl(ontology: ThreeGPPOntology, kpi_name: str) -> Dict[str, Any]:
    """Return KG-known details for a KPI by short_name."""
    kpi = ontology.get_kpi(kpi_name)
    if kpi is None:
        return {"error": f"KPI '{kpi_name}' not found in ontology"}
    related = ontology.get_related_kpis(kpi_name)
    return {
        "short_name": kpi.short_name,
        "description": kpi.description,
        "formula": kpi.formula,
        "unit": kpi.unit,
        "aliases": list(kpi.original_names),
        "source_specs": list(kpi.source_specs),
        "source_clauses": list(kpi.source_clauses),
        "confidence": kpi.confidence,
        "affects": [
            {"target": r.target_kpi, "type": r.relation_type, "strength": r.strength}
            for r in related["affects"]
        ],
        "affected_by": [
            {"source": r.source_kpi, "type": r.relation_type, "strength": r.strength}
            for r in related["affected_by"]
        ],
    }


_SEARCH_SCHEMA = _schema(
    "search_ontology",
    "Fuzzy-search the 3GPP KPI ontology by name, description, or measurement counter. "
    "Use this to find the canonical short_name before extraction or alignment.",
    {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (KPI name, counter name, or keyword)",
            }
        },
        "required": ["query"],
    },
)

_KPI_NAMES_LIST: Optional[List[str]] = None


def _get_kpi_names(ontology: ThreeGPPOntology) -> List[str]:
    global _KPI_NAMES_LIST
    if _KPI_NAMES_LIST is None:
        _KPI_NAMES_LIST = list(ontology.kpis.keys())
    return _KPI_NAMES_LIST


def _details_schema(ontology: ThreeGPPOntology) -> Dict:
    return _schema(
        "get_kpi_details",
        "Get full details for a KPI: formula, unit, constraints, relationships.",
        {
            "type": "object",
            "properties": {
                "kpi_name": {
                    "type": "string",
                    "description": "KPI short_name (e.g. 'Throughput', 'PRB_Util')",
                    "enum": _get_kpi_names(ontology),
                }
            },
            "required": ["kpi_name"],
        },
    )


# =============================================================================
# Extractor Tools (2)
# =============================================================================


def make_extractor_tools(ontology: ThreeGPPOntology) -> ToolRegistry:
    """Create tools for the ExtractorAgent.

    Tools:
      1. search_ontology  — fuzzy search KPI names
      2. get_kpi_details  — full KPI info
    """
    registry = ToolRegistry()

    registry.register(
        "search_ontology",
        lambda query: _search_ontology_impl(ontology, query),
        _SEARCH_SCHEMA,
    )
    registry.register(
        "get_kpi_details",
        lambda kpi_name: _get_kpi_details_impl(ontology, kpi_name),
        _details_schema(ontology),
    )
    return registry


# =============================================================================
# Aligner Tools (3)
# =============================================================================


def make_aligner_tools(ontology: ThreeGPPOntology) -> ToolRegistry:
    """Create tools for the AlignerAgent.

    Tools:
      1. search_ontology — fuzzy search
      2. get_kpi_details — KG-known KPI info

    The previous ``list_kpi_category`` tool has been removed: the spec
    extraction does not currently populate a category attribute on KPI
    definitions, and the in-Python KPICategory enum was hand-curated.
    """
    registry = ToolRegistry()

    registry.register(
        "search_ontology",
        lambda query: _search_ontology_impl(ontology, query),
        _SEARCH_SCHEMA,
    )
    registry.register(
        "get_kpi_details",
        lambda kpi_name: _get_kpi_details_impl(ontology, kpi_name),
        _details_schema(ontology),
    )
    return registry


# =============================================================================
# Conflict Resolver Tools (2) — KARMA CRA
# =============================================================================


def _check_formula_consistency_impl(
    ontology: ThreeGPPOntology,
    kpi_name: str,
    formula: str,
) -> Dict[str, Any]:
    """Shared formula-consistency check used by evaluator + conflict resolver."""
    kpi = ontology.get_kpi(kpi_name)
    if kpi is None:
        return {"consistent": False, "reason": f"KPI '{kpi_name}' not found in ontology"}
    issues = []
    canonical = kpi.formula or ""
    if canonical:
        canonical_tokens = set(canonical.replace("/", " ").replace("×", " ").split())
        formula_tokens = set(formula.replace("/", " ").replace("×", " ").split())
        if not (canonical_tokens & formula_tokens):
            issues.append(
                f"No token overlap between extracted formula and canonical "
                f"formula '{canonical}'"
            )
    return {
        "kpi_name": kpi_name,
        "canonical_formula": canonical,
        "extracted_formula": formula,
        "consistent": len(issues) == 0,
        "issues": issues,
    }


_FORMULA_SCHEMA = _schema(
    "check_formula_consistency",
    "Check whether an extracted formula is consistent with the canonical "
    "3GPP formula for a KPI.",
    {
        "type": "object",
        "properties": {
            "kpi_name": {"type": "string", "description": "KPI short_name"},
            "formula": {"type": "string", "description": "Extracted formula string"},
        },
        "required": ["kpi_name", "formula"],
    },
)


def make_conflict_resolver_tools(ontology: ThreeGPPOntology) -> ToolRegistry:
    """Create tools for the AlignerAgent's conflict-resolution step.

    Tools:
      1. get_kpi_details            — look up canonical KPI details
      2. check_formula_consistency  — compare formulas against the ontology
    """
    registry = ToolRegistry()
    registry.register(
        "get_kpi_details",
        lambda kpi_name: _get_kpi_details_impl(ontology, kpi_name),
        _details_schema(ontology),
    )
    registry.register(
        "check_formula_consistency",
        lambda kpi_name, formula: _check_formula_consistency_impl(ontology, kpi_name, formula),
        _FORMULA_SCHEMA,
    )
    return registry


# =============================================================================
# Evaluator Tools (3)
# =============================================================================


def make_evaluator_tools(
    ontology: ThreeGPPOntology,
    reprocessing_flags: List[Dict],
) -> ToolRegistry:
    """Create tools for the EvaluatorAgent.

    Tools:
      1. get_kpi_details           — verify entities against ontology
      2. check_formula_consistency — verify formula correctness
      3. flag_for_reprocessing     — flag triples for re-extraction/re-alignment

    Args:
        ontology: The 3GPP ontology instance.
        reprocessing_flags: Mutable list; flag_for_reprocessing appends to it.
    """
    registry = ToolRegistry()

    registry.register(
        "get_kpi_details",
        lambda kpi_name: _get_kpi_details_impl(ontology, kpi_name),
        _details_schema(ontology),
    )

    # 2. check_formula_consistency
    def check_formula_consistency(kpi_name: str, formula: str) -> Dict[str, Any]:
        return _check_formula_consistency_impl(ontology, kpi_name, formula)

    registry.register(
        "check_formula_consistency",
        check_formula_consistency,
        _schema(
            "check_formula_consistency",
            "Check whether an extracted formula is consistent with the canonical "
            "3GPP formula for a KPI.",
            {
                "type": "object",
                "properties": {
                    "kpi_name": {
                        "type": "string",
                        "description": "KPI short_name",
                    },
                    "formula": {
                        "type": "string",
                        "description": "Extracted formula string to verify",
                    },
                },
                "required": ["kpi_name", "formula"],
            },
        ),
    )

    # 3. flag_for_reprocessing (closure over mutable list)
    def flag_for_reprocessing(triple_index: int, reason: str, target_agent: str) -> Dict[str, Any]:
        if target_agent not in ("extractor", "aligner"):
            return {"error": f"target_agent must be 'extractor' or 'aligner', got '{target_agent}'"}
        flag = {
            "triple_index": triple_index,
            "reason": reason,
            "target_agent": target_agent,
        }
        reprocessing_flags.append(flag)
        logger.info(f"  [Feedback] Flagged triple {triple_index} for {target_agent}: {reason}")
        return {"status": "flagged", **flag}

    registry.register(
        "flag_for_reprocessing",
        flag_for_reprocessing,
        _schema(
            "flag_for_reprocessing",
            "Flag a triple for reprocessing by the extractor or aligner. "
            "Use this when you find critical errors that require re-extraction or re-alignment.",
            {
                "type": "object",
                "properties": {
                    "triple_index": {
                        "type": "integer",
                        "description": "Index of the triple in the current evaluation batch",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this triple needs reprocessing",
                    },
                    "target_agent": {
                        "type": "string",
                        "enum": ["extractor", "aligner"],
                        "description": "Which agent should reprocess this triple",
                    },
                },
                "required": ["triple_index", "reason", "target_agent"],
            },
        ),
    )

    return registry
