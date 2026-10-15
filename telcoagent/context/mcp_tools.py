"""MCP-style runtime tools used by the Predictor and Explainer agents.

One tool lives here:

1. :class:`StandardKnowledgeGraphMCP` — KG-driven retrieval over the
   spec-extracted ontology (``ThreeGPPOntology``).  Optionally queries a
   Neo4j store for the same content; there is no hand-curated fallback.

Hand-curated validators, the sine/cosine periodicity helper, the
``DataAnalystMCP`` 3-sigma critic, and the ``physics_3gpp`` Python fallback
have been retired and must not be reintroduced.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from ..ontology.core import get_default_ontology


class BaseMCP:
    """Base class for MCP-style runtime tools."""

    def __init__(self, name: str):
        self.name = name


# =============================================================================
# StandardKnowledgeGraphMCP — KG retrieval, no in-Python ontology fallback
# =============================================================================


class StandardKnowledgeGraphMCP(BaseMCP):
    """KG-backed retrieval of KPI definitions, relationships, and causal chains.

    All knowledge comes from :class:`ThreeGPPOntology`, which itself loads
    everything from the enriched KG JSON.  When a Neo4j store is provided
    via the constructor, physics-style queries are routed through it; if
    Neo4j is absent, those queries report unavailable rather than falling
    back to a hand-curated table.
    """

    def __init__(self, neo4j_store=None):
        super().__init__("StandardKnowledgeGraphMCP")
        self._3gpp_ontology = get_default_ontology()
        self._neo4j_store = neo4j_store
        self.ontology = self._build_ontology_dict()

    def _build_ontology_dict(self) -> Dict[str, Any]:
        """Project the typed ontology into a dict for legacy dict-based callers."""
        ont = self._3gpp_ontology

        constraints: Dict[str, Any] = {}
        for kpi in ont.kpis.values():
            entry: Dict[str, Any] = {
                "description": kpi.description,
            }
            if kpi.unit:
                entry["unit"] = kpi.unit
            if kpi.formula:
                entry["formula"] = kpi.formula
            if kpi.source_specs:
                entry["source_specs"] = list(kpi.source_specs)
            if kpi.source_clauses:
                entry["source_clauses"] = list(kpi.source_clauses)
            constraints[kpi.short_name] = entry

        relationships: List[Dict[str, Any]] = []
        for rel in ont.relationships:
            r: Dict[str, Any] = {
                "source": rel.source_kpi,
                "relation": rel.relation_type,
                "target": rel.target_kpi,
                "strength": rel.strength,
                "reason": rel.reason,
            }
            if rel.condition:
                r["condition"] = rel.condition
            if rel.spec_reference:
                r["spec_reference"] = rel.spec_reference
            relationships.append(r)

        causal_chains: List[Dict[str, Any]] = [
            {
                "name": ch.name,
                "chain": ch.steps,
                "description": ch.description,
            }
            for ch in ont.causal_chains
        ]

        return {
            "constraints": constraints,
            "relationships": relationships,
            "causal_chains": causal_chains,
        }

    # -- queries ----------------------------------------------------------

    def query_impact_analysis(self, kpi_name: str) -> List[str]:
        """Return KG-backed lines describing a KPI plus its in/out edges."""
        results: List[str] = []
        ont = self._3gpp_ontology

        kpi_def = ont.get_kpi(kpi_name)
        if kpi_def:
            if kpi_def.description:
                results.append(f"[Definition] {kpi_def.description}")
            if kpi_def.formula:
                results.append(f"[Formula] {kpi_def.formula}")
            if kpi_def.unit:
                results.append(f"[Unit] {kpi_def.unit}")
            if kpi_def.source_specs:
                results.append(
                    f"[Reference] {', '.join(kpi_def.source_specs)}"
                    + (f" §{', '.join(kpi_def.source_clauses)}" if kpi_def.source_clauses else "")
                )

        related = ont.get_related_kpis(kpi_name)
        for rel in related.get("affects", []):
            piece = f"[Relation] {kpi_name} {rel.relation_type} {rel.target_kpi} ({rel.strength})"
            if rel.condition:
                piece += f" when {rel.condition}"
            if rel.reason:
                piece += f" — {rel.reason}"
            results.append(piece)

        for rel in related.get("affected_by", []):
            piece = f"[Relation] {rel.source_kpi} {rel.relation_type} {kpi_name} ({rel.strength})"
            if rel.condition:
                piece += f" when {rel.condition}"
            if rel.reason:
                piece += f" — {rel.reason}"
            results.append(piece)

        for chain in ont.get_causal_chains_for_kpi(kpi_name):
            results.append(
                f"[CausalChain] {chain.name}: {' → '.join(chain.steps)} — {chain.description}"
            )

        return results

    def get_constraints(self, kpi_name: str) -> Dict[str, Any]:
        return self.ontology["constraints"].get(kpi_name, {})

    def get_causal_chains(self) -> List[Dict[str, Any]]:
        return self.ontology["causal_chains"]

    def get_ontology_summary(self) -> str:
        return self._3gpp_ontology.get_ontology_summary()

    def get_kpi_info(self, kpi_name: str) -> str:
        return self._3gpp_ontology.format_kpi_info(kpi_name)

    def query_physics_context(self, environment_type: str) -> str:
        """Physics context query.  Requires Neo4j; no in-Python fallback."""
        if self._neo4j_store is not None:
            return self._neo4j_store.query_physics_context(environment_type)
        return (
            f"Physics context for '{environment_type}' unavailable: "
            "no Neo4j store configured and the in-Python physics table "
            "has been retired."
        )

    def query_cqi_context(self, cqi_value: Optional[int] = None) -> str:
        """CQI table query.  Requires Neo4j; no in-Python fallback."""
        if self._neo4j_store is not None:
            return self._neo4j_store.query_cqi_context(cqi_value)
        return (
            "CQI table unavailable: no Neo4j store configured and the "
            "in-Python CQI table has been retired."
        )
