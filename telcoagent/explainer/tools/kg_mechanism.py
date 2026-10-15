"""``query_graphrag_mechanism`` — 3GPP causal mechanism for a KPI pair.

Retrieves the knowledge-graph context, KPI definitions, and the
directional coupling rule for a directed (source, target) KPI pair so
§2/§4 causal claims cite spec-grounded mechanisms instead of free-form
generation. Returns an explicit error payload when no ontology store is
attached (the KG-off ablation arm).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from telcoagent.agents.registry import ToolRegistry, _capture_context, _schema

if TYPE_CHECKING:
    from .factory import ExplainerToolsConfig

logger = logging.getLogger(__name__)


def _register_query_graphrag_mechanism(
    registry: ToolRegistry,
    cfg: "ExplainerToolsConfig",
    ctx: List[str],
    n_channels: int,
) -> None:
    kpi_names = cfg.kpi_names
    ontology_store = cfg.ontology_store

    def query_graphrag_mechanism(source_kpi: str, target_kpi: str) -> Dict:
        """Retrieve the 3GPP mechanism for a directed KPI pair."""
        if ontology_store is None:
            return {"error": "No ontology store available"}

        # Guard: explicit .retrieve() check before invocation. A store that
        # exposes .ontology but not a usable .retrieve() is a degraded store;
        # degrade EXPLICITLY with a logged error (principle 1 — fail loud),
        # never rely on the ToolRegistry catch-all turning the AttributeError
        # into a silent generic error dict.
        retrieve_fn = getattr(ontology_store, "retrieve", None)
        if not callable(retrieve_fn):
            logger.error(
                "ontology_store does not expose a callable .retrieve("
                "target_kpis=...); store type=%s, has .ontology=%s — "
                "GraphRAG mechanism degraded",
                type(ontology_store).__name__,
                hasattr(ontology_store, "ontology"),
            )
            return {"error": "ontology_store missing .retrieve() method (degraded)"}

        try:
            rag_context = retrieve_fn(target_kpis=[source_kpi, target_kpi])
        except Exception as exc:
            logger.error("ontology_store.retrieve() raised: %s", exc)
            return {"error": f"retrieve() failed: {exc}"}

        kpi_definitions: Dict[str, Dict[str, Optional[str]]] = {}
        ontology = getattr(ontology_store, "ontology", None)
        if ontology is not None:
            for kpi in (source_kpi, target_kpi):
                kpi_def = ontology.get_kpi(kpi)
                if not kpi_def:
                    continue
                spec = None
                if kpi_def.source_specs:
                    spec = ", ".join(kpi_def.source_specs)
                    if kpi_def.source_clauses:
                        spec += f" §{', '.join(kpi_def.source_clauses)}"
                kpi_definitions[kpi] = {
                    "name": kpi_def.short_name,
                    "description": kpi_def.description or None,
                    "formula": kpi_def.formula or None,
                    "spec": spec,
                    "unit": kpi_def.unit or None,
                }

        coupling_info: Dict[str, Any] = {}
        # Prefer the per-run ontology (so the AX-4 q_TH sweep's
        # ``min_confidence`` filter applies) and fall back to the
        # process-wide default only when the explainer was constructed
        # without an ontology_store.
        run_ontology = getattr(ontology_store, "ontology", None)
        if run_ontology is None:
            from telcoagent.ontology.core import get_default_ontology

            run_ontology = get_default_ontology()
        for rule in run_ontology.directional_rules:
            if rule.source_kpi == source_kpi and rule.target_kpi == target_kpi:
                expectation = "increase" if rule.coupling == "positive" else "decrease"
                coupling_info = {
                    "coupling": rule.coupling,
                    "strength": rule.strength,
                    "directional_expectation": (
                        f"When {source_kpi} increases, {target_kpi} is expected to "
                        f"{expectation} ({rule.coupling} coupling, {rule.spec_clause})"
                    ),
                    "mechanism": rule.mechanism,
                }
                break

        result: Dict[str, Any] = {
            "source_kpi": source_kpi,
            "target_kpi": target_kpi,
            "graphrag_context": rag_context,
            "kpi_definitions": kpi_definitions,
        }
        if coupling_info:
            result["coupling"] = coupling_info

        if isinstance(rag_context, str):
            for line in rag_context.splitlines():
                line = line.strip()
                if line:
                    _capture_context(ctx, line)
        elif isinstance(rag_context, list):
            for entry in rag_context:
                _capture_context(ctx, str(entry))
        for kpi, definition in kpi_definitions.items():
            parts = [
                f"{key}={value}"
                for key, value in definition.items()
                if value and key in ("name", "description", "formula", "spec", "unit")
            ]
            _capture_context(ctx, f"KPI definition {kpi}: {', '.join(parts)}")

        return result

    registry.register(
        "query_graphrag_mechanism",
        query_graphrag_mechanism,
        _schema(
            "query_graphrag_mechanism",
            "Query 3GPP domain knowledge for the causal mechanism between a source "
            "and target KPI. Returns relationship type, causal chains, spec "
            "references, KPI definitions, and the directional coupling rule.",
            {
                "type": "object",
                "properties": {
                    "source_kpi": {
                        "type": "string",
                        "description": "Source KPI name.",
                        "enum": kpi_names,
                    },
                    "target_kpi": {
                        "type": "string",
                        "description": "Target KPI name.",
                        "enum": kpi_names,
                    },
                },
                "required": ["source_kpi", "target_kpi"],
            },
        ),
    )
