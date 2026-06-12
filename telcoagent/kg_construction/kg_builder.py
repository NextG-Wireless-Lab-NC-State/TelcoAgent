"""Knowledge Graph Builder for TelcoAgent-RAG pipeline.

Converts LLM-extracted PipelineResult.approved_triples into a NetworkX DiGraph.

Node types:
  KPIDefinition, Measurement, CausalChain, Entity

Edge types:
  kpi_definition, measurement, relationship, causal_chain
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
from networkx.readwrite import json_graph

from .schema import PipelineResult, ValidatedTriple

logger = logging.getLogger(__name__)


# =============================================================================
# Node type resolution
# =============================================================================


def _resolve_node_type(triple: ValidatedTriple) -> str:
    """Determine node_type from alignment entity_type and triple_type."""
    entity_type = triple.alignment.entity_type
    if entity_type in ("KPIDefinition", "Measurement"):
        return entity_type

    triple_type = triple.triple.triple_type
    if triple_type == "kpi_definition":
        return "KPIDefinition"
    if triple_type == "measurement":
        return "Measurement"
    if triple_type == "causal_chain":
        return "CausalChain"
    return "Entity"


def _node_id(triple: ValidatedTriple) -> str:
    """Determine the canonical node ID: mapped_to if available, else aligned_name."""
    return triple.alignment.mapped_to or triple.alignment.aligned_name


# =============================================================================
# Node enrichment helpers
# =============================================================================


def _ensure_node(G: nx.DiGraph, node_id: str, node_type: str):
    """Add node if missing, with default attributes."""
    if node_id not in G:
        G.add_node(
            node_id,
            node_type=node_type,
            aligned_name=node_id,
            original_names=[],
            source_specs=[],
            source_clauses=[],
            quality_scores=[],
            provenance=[],
        )


def _enrich_node(G: nx.DiGraph, node_id: str, triple: ValidatedTriple):
    """Accumulate provenance data onto an existing node."""
    attrs = G.nodes[node_id]

    original = triple.alignment.original_name
    if original not in attrs.get("original_names", []):
        attrs.setdefault("original_names", []).append(original)

    spec = triple.triple.source_spec
    if spec not in attrs.get("source_specs", []):
        attrs.setdefault("source_specs", []).append(spec)

    clause = triple.triple.source_clause
    if clause not in attrs.get("source_clauses", []):
        attrs.setdefault("source_clauses", []).append(clause)

    attrs.setdefault("quality_scores", []).append(triple.quality_score)

    prov = f"{spec}:{clause}"
    if prov not in attrs.get("provenance", []):
        attrs.setdefault("provenance", []).append(prov)

    # Set confidence to max seen
    existing = attrs.get("confidence", 0.0)
    attrs["confidence"] = max(existing, triple.alignment.confidence)

    # Preserve mapped_to
    if triple.alignment.mapped_to:
        attrs["mapped_to"] = triple.alignment.mapped_to


# =============================================================================
# Edge construction helpers
# =============================================================================


def _make_edge_attrs(
    triple: ValidatedTriple, relation_type: Optional[str] = None
) -> Dict[str, Any]:
    """Build edge attribute dict from a ValidatedTriple."""
    attrs: Dict[str, Any] = {
        "relation_type": relation_type or triple.triple.predicate,
        "triple_type": triple.triple.triple_type,
        "strength": "extracted",
        "source_spec": triple.triple.source_spec,
        "source_clause": triple.triple.source_clause,
        "quality_score": triple.quality_score,
        "confidence": triple.alignment.confidence,
        "raw_text": triple.triple.raw_text,
    }
    # Propagate causal-strength metadata when present (relationship / causal_chain).
    t = triple.triple
    if t.min_pct_source is not None:
        attrs["min_pct_source"] = t.min_pct_source
    if t.min_pct_target is not None:
        attrs["min_pct_target"] = t.min_pct_target
    if t.mechanism:
        attrs["mechanism"] = t.mechanism
    if t.fine_clause:
        attrs["fine_clause"] = t.fine_clause
    return attrs


# =============================================================================
# Triple-type handlers
# =============================================================================


def _coerce_float(value: Any) -> Optional[float]:
    """Best-effort numeric coercion for has_min_value / has_max_value objects."""
    if value is None:
        return None
    try:
        return float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None


def _split_counter_list(value: str) -> List[str]:
    """Parse a comma/semicolon-separated counter list into a clean list."""
    if not value:
        return []
    parts = [p.strip() for p in str(value).replace(";", ",").split(",")]
    return [p for p in parts if p]


def _handle_kpi_definition(G: nx.DiGraph, triple: ValidatedTriple):
    """Handle kpi_definition triples.

    Subject → KPI node with KG-extracted attributes.  Recognized predicates
    set node attributes directly so the Predictor can read spec-grounded
    metadata (category, kpi_object, kpi_type, min/max, formula, etc.)
    without any in-Python prior.

    `uses_counter` is the only predicate that emits an edge — it links the
    KPIDefinition node to its Measurement counter so multi-counter formulas
    stay traceable in the graph.
    """
    subj_id = _node_id(triple)
    _ensure_node(G, subj_id, "KPIDefinition")
    _enrich_node(G, subj_id, triple)

    predicate = triple.triple.predicate.lower()
    obj_val = triple.triple.object

    if predicate in ("calculated_by", "has_formula"):
        G.nodes[subj_id]["formula"] = obj_val
    elif predicate == "has_physical_formula":
        G.nodes[subj_id]["physical_formula"] = obj_val
    elif predicate in ("defined_as", "has_description"):
        G.nodes[subj_id]["description"] = obj_val
    elif predicate == "has_unit":
        G.nodes[subj_id]["unit"] = obj_val
    elif predicate == "has_category":
        G.nodes[subj_id]["category"] = obj_val
    elif predicate == "has_kpi_object":
        G.nodes[subj_id]["kpi_object"] = obj_val
    elif predicate == "has_kpi_type":
        G.nodes[subj_id]["kpi_type"] = obj_val
    elif predicate == "has_min_value":
        coerced = _coerce_float(obj_val)
        if coerced is not None:
            G.nodes[subj_id]["min_value"] = coerced
    elif predicate == "has_max_value":
        coerced = _coerce_float(obj_val)
        if coerced is not None:
            G.nodes[subj_id]["max_value"] = coerced
    elif predicate == "has_measurements":
        counters = _split_counter_list(obj_val)
        existing = list(G.nodes[subj_id].get("measurements", []))
        for c in counters:
            if c not in existing:
                existing.append(c)
            _ensure_node(G, c, "Measurement")
            G.add_edge(subj_id, c, **_make_edge_attrs(triple, "uses_counter"))
        G.nodes[subj_id]["measurements"] = existing
    elif predicate == "uses_counter":
        _ensure_node(G, obj_val, "Measurement")
        G.add_edge(subj_id, obj_val, **_make_edge_attrs(triple, "uses_counter"))
        existing = list(G.nodes[subj_id].get("measurements", []))
        if obj_val not in existing:
            existing.append(obj_val)
            G.nodes[subj_id]["measurements"] = existing
    else:
        # Unknown predicate — fall back to generic entity edge so we don't
        # silently drop the triple.
        _ensure_node(G, obj_val, "Entity")
        G.add_edge(subj_id, obj_val, **_make_edge_attrs(triple))


def _handle_measurement(G: nx.DiGraph, triple: ValidatedTriple):
    """Handle measurement triples: subject → Measurement node, edge to object."""
    subj_id = _node_id(triple)
    _ensure_node(G, subj_id, "Measurement")
    _enrich_node(G, subj_id, triple)

    obj_val = triple.triple.object
    _ensure_node(G, obj_val, "Entity")
    G.add_edge(subj_id, obj_val, **_make_edge_attrs(triple))


def _handle_relationship(G: nx.DiGraph, triple: ValidatedTriple):
    """Handle relationship triples: subject → edge (predicate) → object."""
    subj_id = _node_id(triple)
    node_type = _resolve_node_type(triple)
    _ensure_node(G, subj_id, node_type)
    _enrich_node(G, subj_id, triple)

    obj_val = triple.triple.object
    _ensure_node(G, obj_val, "Entity")
    G.add_edge(subj_id, obj_val, **_make_edge_attrs(triple))


def _handle_causal_chain(G: nx.DiGraph, triple: ValidatedTriple):
    """Handle causal_chain triples: cause → causal edge → effect."""
    subj_id = _node_id(triple)
    _ensure_node(G, subj_id, "CausalChain")
    _enrich_node(G, subj_id, triple)

    obj_val = triple.triple.object
    _ensure_node(G, obj_val, "Entity")
    G.add_edge(subj_id, obj_val, **_make_edge_attrs(triple))


_TRIPLE_HANDLERS = {
    "kpi_definition": _handle_kpi_definition,
    "measurement": _handle_measurement,
    "relationship": _handle_relationship,
    "causal_chain": _handle_causal_chain,
}


# =============================================================================
# Main builder class
# =============================================================================


class KnowledgeGraphBuilder:
    """Builds a NetworkX DiGraph from TelcoAgent-RAG pipeline results."""

    def __init__(self):
        self._graph: nx.DiGraph = nx.DiGraph()

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    def build_from_pipeline_result(self, result: PipelineResult) -> nx.DiGraph:
        """Build KG from a full PipelineResult (uses approved_triples)."""
        return self.build_from_triples(result.approved_triples)

    def build_from_triples(self, triples: List[ValidatedTriple]) -> nx.DiGraph:
        """Build KG from a list of ValidatedTriples (LLM-extracted from PDFs)."""
        seen: set = set()
        for triple in triples:
            # Deduplicate by content hash
            key = (triple.triple.subject, triple.triple.predicate, triple.triple.object)
            if key in seen:
                continue
            seen.add(key)

            triple_type = triple.triple.triple_type
            handler = _TRIPLE_HANDLERS.get(triple_type, _handle_relationship)
            handler(self._graph, triple)

        logger.info(
            "Added LLM triples: %d nodes, %d edges from %d triples",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
            len(triples),
        )
        return self._graph

    def get_stats(self) -> Dict[str, int]:
        """Return node/edge statistics for the current graph."""
        G = self._graph
        node_types: Dict[str, int] = {}
        for _, attrs in G.nodes(data=True):
            nt = attrs.get("node_type", "Unknown")
            node_types[nt] = node_types.get(nt, 0) + 1

        edge_types: Dict[str, int] = {}
        for _, _, attrs in G.edges(data=True):
            tt = attrs.get("triple_type", "unknown")
            edge_types[tt] = edge_types.get(tt, 0) + 1

        return {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
        }


# =============================================================================
# JSON serialization
# =============================================================================


def graph_to_json(graph: nx.DiGraph) -> dict:
    """Serialize a NetworkX DiGraph to a JSON-compatible dict."""
    data = json_graph.node_link_data(graph, edges="edges")
    data["metadata"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
    }
    return data


def json_to_graph(data: dict) -> nx.DiGraph:
    """Deserialize a dict back to a NetworkX DiGraph."""
    return json_graph.node_link_graph(data, directed=True, multigraph=False, edges="edges")


def save_graph_json(graph: nx.DiGraph, path: str):
    """Serialize and save a graph to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = graph_to_json(graph)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(
        "Saved graph to %s (%d nodes, %d edges)",
        path,
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )


def load_graph_json(path: str) -> nx.DiGraph:
    """Load a graph from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    graph = json_to_graph(data)
    logger.info(
        "Loaded graph from %s (%d nodes, %d edges)",
        path,
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph
