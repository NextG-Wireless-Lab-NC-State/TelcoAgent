"""KG-sourced loader for KPIDefinition + KPIRelationship + CausalChain
+ DirectionalRule.

Reads `data/enriched_kg.json` (NetworkX node-link format produced by
``kg_construction.kg_builder.graph_to_json``) and returns the dataclass shapes used
by ``ontology_3gpp``.  The KG is the single source of truth — there are
no hand-curated fallback registries.

Design notes (telecom domain):
- Ontology edges with ``strength == "extracted"`` come from LLM-driven 3GPP
  spec extraction. They are admitted only when ``confidence >= min_confidence``
  so noisy extractions don't pollute spec-grounded relationships.
- Causal chains are stored as standalone ``CausalChain`` nodes; the
  per-step ``causal_chain`` edges in the KG are redundant for our purposes.
- KPIDefinition shape mirrors what the KARMA pipeline actually emits:
  ``description``, ``formula``, ``unit``, ``original_names``,
  ``source_specs``, ``source_clauses``, ``confidence``, ``mapped_to``.
  Anything the spec extraction does not provide (categories, kpi_type,
  numeric ranges, etc.) is intentionally absent — the Predictor must
  operate on what the KG actually contains.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

# Re-export DEFAULT_KG_PATH at this module's import path: several callers
# (ontology.core, stores.ontology_store, scripts/, tests/) do
# `from telcoagent.ontology.kg_loader import DEFAULT_KG_PATH`. Ruff F401
# would otherwise mark this as unused — see __all__ below.
from telcoagent.config import DEFAULT_KG_PATH

if TYPE_CHECKING:
    from ..stores import Neo4jConnection
    from .core import (
        CausalChain,
        DirectionalRule,
        KPIDefinition,
        KPIRelationship,
    )

__all__ = [
    "DEFAULT_KG_PATH",
    "load_kpi_definitions_from_kg",
    "load_relations_from_kg",
    "load_directional_rules_from_kg",
    "load_directional_rules_from_neo4j",
]


# Coupling direction derived from RDF-style relation type.
_COUPLING_MAP = {
    "INCREASES": "positive",
    "DETERMINES": "positive",
    "CAUSES": "positive",
    "DECREASES": "negative",
    "LIMITS": "negative",
}

# Default hourly-violation thresholds keyed by edge strength.
# Tighter for critical/strong rules, looser for weak/extracted ones.
_STRENGTH_DEFAULTS = {
    "critical": (5.0, 5.0),
    "strong": (8.0, 10.0),
    "moderate": (10.0, 10.0),
    "weak": (15.0, 15.0),
    "extracted": (12.0, 12.0),
}

logger = logging.getLogger(__name__)


VALID_RELATION_TYPES = {"INCREASES", "DECREASES", "DETERMINES", "LIMITS", "CAUSES"}
VALID_STRENGTHS = {"critical", "strong", "moderate", "weak", "extracted"}

# Spec-extracted causal relation_type aliases → canonical 3GPP coupling.
# The corpus-grounded KG (data/enriched_kg.json) phrases the same causal
# couplings with richer verbs than the 5 canonical types. These 5 aliases are
# semantically equivalent to a canonical type (mapping fixed in Step 1,
# .omc/plans/step1-relation-type-mapping.md, by coupling sign + the edge's
# own ``reason`` field). Normalizing them lets the loader read corpus edges
# it already carries — it does NOT invent new edges.
#
# Scope: causal aliases ONLY. Ontology/schema relation types (USES_COUNTER,
# DEPENDS_ON, DEFINED_AS, …) are NOT couplings and are intentionally absent —
# they must still fail the VALID_RELATION_TYPES check and be dropped.
_RELATION_TYPE_ALIASES = {
    "INVERSELY_AFFECTS": "DECREASES",
    "POSITIVELY_AFFECTS": "INCREASES",
    "SEVERELY_DECREASES": "DECREASES",  # severity is preserved via strength
    "DIRECTLY_PROPORTIONAL": "INCREASES",
    "ENABLES": "INCREASES",
}


def _canonical_relation_type(relation_type: str) -> str:
    """Normalize a (already upper-cased) relation_type through the causal
    alias table. Non-alias types pass through unchanged."""
    return _RELATION_TYPE_ALIASES.get(relation_type, relation_type)


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _is_ontology_relationship_edge(edge: dict) -> bool:
    return edge.get("triple_type") in ("ontology_relationship", "relationship")


def _is_causal_chain_node(node: dict) -> bool:
    return node.get("node_type") == "CausalChain"


def _build_relationship(
    edge: dict,
    known_kpis: Set[str],
    min_confidence: float,
) -> Optional["KPIRelationship"]:
    from .core import KPIRelationship

    src = edge.get("source")
    tgt = edge.get("target")
    if not src or not tgt:
        logger.warning("KG edge missing source/target: %s", edge)
        return None
    if src not in known_kpis or tgt not in known_kpis:
        logger.debug("Skipping KG edge with unknown KPI: %s -> %s", src, tgt)
        return None

    relation_type = _canonical_relation_type((edge.get("relation_type") or "").upper())
    if relation_type not in VALID_RELATION_TYPES:
        logger.warning(
            "KG edge %s -> %s has invalid relation_type %r; skipping",
            src,
            tgt,
            relation_type,
        )
        return None

    strength = edge.get("strength", "moderate")
    if strength not in VALID_STRENGTHS:
        logger.warning(
            "KG edge %s -> %s has unknown strength %r; coercing to 'moderate'",
            src,
            tgt,
            strength,
        )
        strength = "moderate"

    confidence = float(edge.get("confidence", 1.0))
    if strength == "extracted" and confidence < min_confidence:
        logger.debug(
            "Dropping low-confidence extracted edge %s -> %s (conf=%.2f < %.2f)",
            src,
            tgt,
            confidence,
            min_confidence,
        )
        return None

    condition = edge.get("condition") or None
    if condition == "":
        condition = None

    spec_reference = edge.get("spec_reference") or None
    if spec_reference == "":
        spec_reference = None

    return KPIRelationship(
        source_kpi=src,
        target_kpi=tgt,
        relation_type=relation_type,
        strength=strength,
        reason=edge.get("reason", ""),
        condition=condition,
        spec_reference=spec_reference,
    )


def _build_causal_chain(node: dict) -> Optional["CausalChain"]:
    from .core import CausalChain

    name = node.get("name")
    steps = node.get("steps")
    if not name or not isinstance(steps, list) or not steps:
        logger.warning("Malformed CausalChain node (missing name/steps): %s", node.get("id"))
        return None

    trigger = node.get("trigger_condition") or None
    if trigger == "":
        trigger = None

    return CausalChain(
        name=name,
        description=node.get("description", ""),
        steps=list(steps),
        trigger_condition=trigger,
    )


def _parse_spec_clause(spec_reference: Optional[str]) -> Tuple[str, str]:
    """Split ``'TS 38.214 §7.1'`` into (``'TS 38.214'``, ``'7.1'``)."""
    if not spec_reference:
        return "", ""
    if "§" in spec_reference:
        parts = spec_reference.split("§", 1)
        return parts[0].strip(), parts[1].strip()
    return spec_reference.strip(), ""


def _build_directional_rule(
    edge: dict,
    known_kpis: Set[str],
    min_confidence: float,
) -> Optional["DirectionalRule"]:
    from .core import DirectionalRule

    src = edge.get("source")
    tgt = edge.get("target")
    if not src or not tgt:
        return None
    if src not in known_kpis or tgt not in known_kpis:
        return None

    relation_type = _canonical_relation_type((edge.get("relation_type") or "").upper())
    coupling = _COUPLING_MAP.get(relation_type)
    if coupling is None:
        return None

    strength = edge.get("strength", "moderate")
    if strength not in _STRENGTH_DEFAULTS:
        strength = "moderate"

    confidence = float(edge.get("confidence", 1.0))
    if strength == "extracted" and confidence < min_confidence:
        return None

    default_src, default_tgt = _STRENGTH_DEFAULTS[strength]
    min_pct_source = float(edge.get("min_pct_source", default_src))
    min_pct_target = float(edge.get("min_pct_target", default_tgt))

    spec_ref_full = edge.get("spec_reference") or ""
    spec_ref, parsed_clause = _parse_spec_clause(spec_ref_full)
    spec_clause = edge.get("spec_clause") or parsed_clause

    condition = edge.get("condition") or None
    if condition == "":
        condition = None

    mechanism = edge.get("mechanism") or edge.get("reason", "")
    correction_hint = edge.get("correction_hint") or (
        f"Adjust {tgt} in the "
        f"{'same' if coupling == 'positive' else 'opposite'} "
        f"direction as {src}."
    )

    return DirectionalRule(
        source_kpi=src,
        target_kpi=tgt,
        coupling=coupling,
        strength=strength if strength != "extracted" else "moderate",
        min_pct_source=min_pct_source,
        min_pct_target=min_pct_target,
        mechanism=mechanism,
        spec_reference=spec_ref or "TS 28.554",
        spec_clause=spec_clause,
        correction_hint=correction_hint,
        condition=condition,
    )


def load_directional_rules_from_kg(
    path: str,
    known_kpis: Set[str],
    min_confidence: float = 0.7,
) -> List["DirectionalRule"]:
    """Build the directional-consistency rule list from the enriched KG.

    Every ``ontology_relationship`` edge whose ``relation_type`` maps to a
    known coupling ({INCREASES, DETERMINES, CAUSES}→positive;
    {DECREASES, LIMITS}→negative) is admitted.  Thresholds come from the
    edge itself if ``min_pct_source``/``min_pct_target`` are present, else
    from strength-level defaults.

    The 7 canonical TS 28.554 KPIs are unioned into ``known_kpis`` so that
    relationship edges naming them survive even when the RAG Aligner failed
    to register them as ``KPIDefinition`` nodes (which happens when the
    extractor wrote them under a synonymous label).  Without this union the
    Predictor's ``refine_prediction`` tool would silently see zero violations
    and the KG ablation would degenerate into the no-KG baseline.
    """
    from ..config import CORE_KPI_NAMES

    known_kpis = set(known_kpis) | set(CORE_KPI_NAMES)

    data = _load_json(path)
    edges = data.get("edges") or data.get("links") or []

    rules: List["DirectionalRule"] = []
    seen_pairs: Set[Tuple[str, str]] = set()
    for edge in edges:
        if not _is_ontology_relationship_edge(edge):
            continue
        rule = _build_directional_rule(edge, known_kpis, min_confidence)
        if rule is None:
            continue
        pair = (rule.source_kpi, rule.target_kpi)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        rules.append(rule)

    rules.sort(key=lambda r: (r.source_kpi, r.target_kpi))
    return rules


def load_relations_from_kg(
    path: str,
    known_kpis: Set[str],
    min_confidence: float = 0.7,
) -> Tuple[List["KPIRelationship"], List["CausalChain"]]:
    """Load relationships and causal chains from a KG JSON file.

    Parameters
    ----------
    path
        Path to the NetworkX node-link JSON (``data/enriched_kg.json``).
    known_kpis
        Set of KPI short names to validate against. Edges with endpoints
        outside this set are silently skipped (KG may legitimately grow
        ahead of the typed registry).
    min_confidence
        Minimum confidence for ``strength == "extracted"`` edges. Hardcoded
        edges (``strength`` in {critical, strong, moderate, weak}) are
        always admitted regardless of confidence.

    Returns
    -------
    Tuple of ``(relationships, causal_chains)``, both deterministically
    sorted (relationships by ``(source, target)``, chains by ``name``).

    Raises
    ------
    FileNotFoundError
        If the KG file cannot be opened.
    ValueError
        If the JSON cannot be parsed.
    """
    # Mirror ``load_directional_rules_from_kg``: 7 forecast KPIs are unioned
    # into known_kpis so spec-extracted edges naming them survive even when
    # the Aligner failed to register them as ``KPIDefinition`` nodes (e.g.
    # ``MAC_DL_Eff`` and ``Throughput`` only appear as edge endpoints in the
    # KARMA-extracted KG, not as standalone definition nodes).
    from ..config import CORE_KPI_NAMES

    known_kpis = set(known_kpis) | set(CORE_KPI_NAMES)

    data = _load_json(path)

    edges = data.get("edges") or data.get("links") or []
    nodes = data.get("nodes") or []

    relationships: List["KPIRelationship"] = []
    seen_pairs: Set[Tuple[str, str, str]] = set()
    for edge in edges:
        if not _is_ontology_relationship_edge(edge):
            continue
        rel = _build_relationship(edge, known_kpis, min_confidence)
        if rel is None:
            continue
        key = (rel.source_kpi, rel.target_kpi, rel.relation_type)
        if key in seen_pairs:
            logger.debug(
                "Duplicate KG edge %s -> %s [%s]; keeping first",
                rel.source_kpi,
                rel.target_kpi,
                rel.relation_type,
            )
            continue
        seen_pairs.add(key)
        relationships.append(rel)

    chains: List["CausalChain"] = []
    seen_chain_names: Set[str] = set()
    for node in nodes:
        if not _is_causal_chain_node(node):
            continue
        chain = _build_causal_chain(node)
        if chain is None:
            continue
        if chain.name in seen_chain_names:
            logger.debug("Duplicate CausalChain %r; keeping first", chain.name)
            continue
        seen_chain_names.add(chain.name)
        chains.append(chain)

    # The paper KG (data/enriched_kg.json) labels seven RRC-message entities
    # with node_type="CausalChain" but never populates name/steps on them —
    # the actual chains live in the top-level ``causal_chains`` array. A KG
    # may carry chains in either place or both; we dedup by name across the
    # two populations.
    for top_chain in data.get("causal_chains") or []:
        chain = _build_causal_chain(top_chain)
        if chain is None:
            continue
        if chain.name in seen_chain_names:
            continue
        seen_chain_names.add(chain.name)
        chains.append(chain)

    relationships.sort(key=lambda r: (r.source_kpi, r.target_kpi, r.relation_type))
    chains.sort(key=lambda c: c.name)

    logger.info(
        "Loaded ontology from KG %s: %d relationships, %d causal chains",
        path,
        len(relationships),
        len(chains),
    )
    return relationships, chains


def load_directional_rules_from_neo4j(
    conn: "Neo4jConnection",
    known_kpis: Set[str],
    min_confidence: float = 0.7,
) -> List["DirectionalRule"]:
    """Build DirectionalRule list by querying semantic edges from Neo4j.

    Queries each of the five canonical relationship types (INCREASES,
    DECREASES, DETERMINES, LIMITS, CAUSES) directly as typed Neo4j
    relationship labels — which is the format written by
    ``Neo4jOntologyStore.push_kg_json()``.

    This is the Neo4j-native counterpart to ``load_directional_rules_from_kg``
    and returns identical dataclass shapes.

    Args:
        conn: Active ``Neo4jConnection`` instance.
        known_kpis: KPI short names to validate endpoints against.
        min_confidence: Drop ``strength='extracted'`` edges below this value.

    Returns:
        Sorted list of ``DirectionalRule`` objects (by source_kpi, target_kpi).
    """
    from .core import DirectionalRule

    rules: List["DirectionalRule"] = []
    seen_pairs: Set[Tuple[str, str]] = set()

    for rel_label, coupling in _COUPLING_MAP.items():
        try:
            rows = conn.run(
                f"MATCH (a)-[r:{rel_label}]->(b) "
                "RETURN a.name AS source, b.name AS target, "
                "r.confidence AS confidence, r.strength AS strength, "
                "r.reason AS reason, r.condition AS condition, "
                "r.spec_reference AS spec_reference, "
                "r.spec_clause AS spec_clause, "
                "r.mechanism AS mechanism, "
                "r.correction_hint AS correction_hint, "
                "r.min_pct_source AS min_pct_source, "
                "r.min_pct_target AS min_pct_target"
            )
        except Exception as exc:
            logger.warning(
                "load_directional_rules_from_neo4j: query for :%s failed: %s",
                rel_label,
                exc,
            )
            continue

        for row in rows:
            src, tgt = row.get("source"), row.get("target")
            if not src or not tgt:
                continue
            if src not in known_kpis or tgt not in known_kpis:
                continue

            pair = (src, tgt)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            strength = row.get("strength") or "moderate"
            if strength not in _STRENGTH_DEFAULTS:
                strength = "moderate"

            confidence = float(row.get("confidence") or 1.0)
            if strength == "extracted" and confidence < min_confidence:
                continue

            default_src, default_tgt = _STRENGTH_DEFAULTS[strength]
            min_pct_source = float(row.get("min_pct_source") or default_src)
            min_pct_target = float(row.get("min_pct_target") or default_tgt)

            spec_ref_full = row.get("spec_reference") or ""
            spec_ref, parsed_clause = _parse_spec_clause(spec_ref_full)
            spec_clause = row.get("spec_clause") or parsed_clause

            condition = row.get("condition") or None
            mechanism = row.get("mechanism") or row.get("reason") or ""
            correction_hint = row.get("correction_hint") or (
                f"Adjust {tgt} in the "
                f"{'same' if coupling == 'positive' else 'opposite'} "
                f"direction as {src}."
            )

            rules.append(
                DirectionalRule(
                    source_kpi=src,
                    target_kpi=tgt,
                    coupling=coupling,
                    strength=strength if strength != "extracted" else "moderate",
                    min_pct_source=min_pct_source,
                    min_pct_target=min_pct_target,
                    mechanism=mechanism,
                    spec_reference=spec_ref or "TS 28.554",
                    spec_clause=spec_clause,
                    correction_hint=correction_hint,
                    condition=condition,
                )
            )

    rules.sort(key=lambda r: (r.source_kpi, r.target_kpi))
    logger.info("load_directional_rules_from_neo4j: loaded %d rules", len(rules))
    return rules


# =============================================================================
# KPIDefinition loader (KG → typed dict)
# =============================================================================

# Tokens in ``unit`` that indicate a count / discrete quantity.  The Predictor
# uses this to decide whether a forecast must be rounded to an integer.  Kept
# as a data-derived heuristic — the KG does not carry an explicit is_integer
# flag, so we infer from the unit field that the spec extraction produces.
_INTEGER_UNIT_TOKENS = (
    "count",
    "number",
    "ue",
    "integer",
    "attempts",
    "events",
)
# Tokens that indicate a percent / ratio KPI bounded to [0, 100].
_PERCENT_UNIT_TOKENS = ("%", "percent", "percentage")
# Tokens that indicate a [0, 1] probability / rate.
_PROBABILITY_UNIT_TOKENS = ("ratio", "probability", "fraction")


def _infer_kpi_numeric_meta(
    unit: str,
    description: str,
) -> Tuple[Optional[Tuple[float, float]], bool]:
    """Derive (valid_range, is_integer) from KG-provided metadata.

    The KG does not explicitly emit physical ranges or an integer flag, but
    the spec-extracted ``unit`` field is expressive enough to infer both
    safely for well-known unit classes.  Returns ``(None, False)`` when the
    unit is unrecognized; the Predictor must then fall back to whatever the
    station dataset itself reveals (e.g. empirical min/max).
    """
    u = (unit or "").lower().strip()
    d = (description or "").lower()

    is_integer = any(tok in u for tok in _INTEGER_UNIT_TOKENS)
    # 3GPP CQI index is officially integer 0..15 — derivable from description
    # alone when unit is the vague 'index'.
    if not is_integer and u == "index" and "cqi" in d:
        is_integer = True

    valid_range: Optional[Tuple[float, float]] = None
    if any(tok in u for tok in _PERCENT_UNIT_TOKENS):
        valid_range = (0.0, 100.0)
    elif any(tok in u for tok in _PROBABILITY_UNIT_TOKENS):
        valid_range = (0.0, 1.0)
    elif u in {"kbit/s", "kbps", "bit/s", "bps", "mbit/s", "mbps"}:
        valid_range = (0.0, float("inf"))
    elif is_integer:
        valid_range = (0.0, float("inf"))
    elif u == "index" and "cqi" in d:
        valid_range = (0.0, 15.0)

    return valid_range, is_integer


def _build_kpi_definition(node: dict) -> Optional["KPIDefinition"]:
    from .core import KPIDefinition

    short_name = node.get("aligned_name") or node.get("id")
    if not short_name:
        return None

    unit = node.get("unit") or ""
    description = node.get("description") or ""
    valid_range, is_integer = _infer_kpi_numeric_meta(unit, description)

    return KPIDefinition(
        short_name=short_name,
        description=description,
        formula=node.get("formula") or "",
        unit=unit,
        original_names=list(node.get("original_names") or []),
        source_specs=list(node.get("source_specs") or []),
        source_clauses=list(node.get("source_clauses") or []),
        confidence=float(node.get("confidence", 1.0)),
        mapped_to=node.get("mapped_to") or None,
        valid_range=valid_range,
        is_integer=is_integer,
    )


def load_kpi_definitions_from_kg(
    path: str,
) -> Dict[str, "KPIDefinition"]:
    """Load every ``KPIDefinition`` node from the enriched KG.

    Returns a dict keyed by KPI short_name (= aligned_name).  KPI nodes
    that the LLM placed under ``Entity`` or ``Measurement`` are NOT
    promoted here — the caller sees only what the KG explicitly typed
    as a KPI definition.  This is intentional: thesis policy says the
    Predictor must operate on what the spec extraction actually produced.
    """
    data = _load_json(path)
    nodes = data.get("nodes") or []

    kpis: Dict[str, "KPIDefinition"] = {}
    for node in nodes:
        if node.get("node_type") != "KPIDefinition":
            continue
        kpi = _build_kpi_definition(node)
        if kpi is None:
            continue
        if kpi.short_name in kpis:
            logger.debug("Duplicate KPIDefinition node %s; keeping first", kpi.short_name)
            continue
        kpis[kpi.short_name] = kpi

    logger.info(
        "Loaded %d KPIDefinition entries from KG %s",
        len(kpis),
        path,
    )
    return kpis
