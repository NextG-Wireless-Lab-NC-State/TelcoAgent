"""InMemoryOntologyStore — Neo4j-free 3GPP ontology store.

The single, shared in-memory ontology store used everywhere the runtime
predictor and explainer need KG-grounded context without a live Neo4j. It
honours the public interface of ``Neo4jOntologyStore`` that the explainer's
tool registry depends on:

    - ``.ontology``  — read-only ``ThreeGPPOntology`` reference (duck-typed by
      ``explainer/orchestrator.py`` and ``tools/explainer_tools.py``)
    - ``.retrieve(target_kpis, prediction_stats=None, iteration_history=None)``
      — returns a multi-section formatted string (never ``None``), identical in
      shape to ``Neo4jOntologyStore.retrieve`` (ontology_store.py:900-927).

All data is read from the in-memory ``ThreeGPPOntology``; there is no Neo4j
connection and no network access. This module is the *only* place an in-memory
store is defined — scripts and the predictor import it rather than rolling
their own stub (task 08 consolidation).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from telcoagent.config import CORE_KPI_NAMES
from telcoagent.ontology.core import ThreeGPPOntology

logger = logging.getLogger(__name__)


class InMemoryOntologyStore:
    """In-memory 3GPP ontology store honouring the Neo4jOntologyStore interface.

    Exposes:
        ``.ontology`` — ``ThreeGPPOntology`` read-only reference.
        ``.retrieve(target_kpis, prediction_stats=None, iteration_history=None)``
            -> ``str`` — formatted 3GPP causal context.

    No Neo4j connection; every operation reads from the in-memory ontology.
    """

    def __init__(self, ontology: Optional[ThreeGPPOntology] = None) -> None:
        if ontology is None:
            from telcoagent.ontology.core import get_default_ontology

            ontology = get_default_ontology()
        self.ontology = ontology

    def retrieve(
        self,
        target_kpis: Optional[List[str]] = None,
        prediction_stats: Optional[Dict[str, Dict[str, Any]]] = None,
        iteration_history: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Retrieve formatted 3GPP causal context for ``target_kpis``.

        Returns a multi-section string containing:
            - ``[KPI METADATA]`` unit, formula, spec references per target KPI
            - ``[ACTIVE RELATIONSHIPS]`` scored edges (source --type--> target)
            - ``[CAUSAL CHAINS]`` matching causal chains

        Mirrors ``Neo4jOntologyStore.retrieve`` (ontology_store.py:900-927):
        always returns a ``str`` (never ``None``); may be empty when no
        ontology content matches the requested KPIs.

        ``prediction_stats`` and ``iteration_history`` are accepted for
        interface parity with the Neo4j store; the in-memory store has no
        condition-detection layer, so they do not alter the result.
        """
        if target_kpis is None:
            target_kpis = list(CORE_KPI_NAMES)

        if not target_kpis:
            return ""

        sections: List[str] = []

        metadata_lines = self._format_kpi_metadata(target_kpis)
        if metadata_lines:
            sections.append("[KPI METADATA]\n" + "\n".join(metadata_lines))

        rel_lines = self._format_relationships(target_kpis)
        if rel_lines:
            sections.append("[ACTIVE RELATIONSHIPS]\n" + "\n".join(rel_lines))

        chain_lines = self._format_causal_chains(target_kpis)
        if chain_lines:
            sections.append("[CAUSAL CHAINS]\n" + "\n".join(chain_lines))

        return "\n\n".join(sections)

    # -- helpers --------------------------------------------------------

    def _format_kpi_metadata(self, target_kpis: List[str]) -> List[str]:
        lines: List[str] = []
        for kpi_name in target_kpis:
            kpi = self.ontology.get_kpi(kpi_name)
            if kpi is None:
                continue
            spec = ", ".join(kpi.source_specs) if kpi.source_specs else ""
            parts = [f"{kpi.short_name}: {kpi.description or ''}".rstrip()]
            if kpi.formula:
                parts.append(f"formula={kpi.formula}")
            if kpi.unit:
                parts.append(f"unit={kpi.unit}")
            if spec:
                parts.append(f"spec={spec}")
            lines.append("  " + " | ".join(parts))
        return lines

    def _format_relationships(self, target_kpis: List[str]) -> List[str]:
        lines: List[str] = []
        seen: set = set()
        for kpi_name in target_kpis:
            related = self.ontology.get_related_kpis(kpi_name)
            for direction in ("affects", "affected_by"):
                for rel in related.get(direction, []):
                    key = (rel.source_kpi, rel.relation_type, rel.target_kpi)
                    if key in seen:
                        continue
                    seen.add(key)
                    ref = f" [{rel.spec_reference}]" if rel.spec_reference else ""
                    lines.append(
                        f"  {rel.source_kpi} --{rel.relation_type}--> "
                        f"{rel.target_kpi} ({rel.strength}){ref}"
                    )
        return lines

    def _format_causal_chains(self, target_kpis: List[str]) -> List[str]:
        lines: List[str] = []
        seen: set = set()
        for kpi_name in target_kpis:
            for chain in self.ontology.get_causal_chains_for_kpi(kpi_name):
                steps = " -> ".join(chain.steps)
                if steps in seen:
                    continue
                seen.add(steps)
                lines.append(f"  {steps}")
        return lines
