"""3GPP KPI ontology — KG-driven, no hand-curated knowledge.

Thesis policy
-------------
Every piece of telecom domain knowledge consumed by the Predictor must come
from the spec-extracted Knowledge Graph (``data/enriched_kg.json`` by default,
overridable via ``TELCOAGENT_KG_PATH``).  This module exposes only:

* Pure data dataclasses that mirror what the KG actually emits.
* A single class, :class:`ThreeGPPOntology`, that loads KPIs, relationships,
  causal chains, and directional rules from the KG.
* KG-driven analytics (``check_directional_consistency_hourly``) that read
  the loaded rules — they hold no hand-written rules of their own.

There is no in-Python KPI registry, no relationship/causal-chain fallback,
no temporal-pattern dict, no ``validate_prediction`` threshold logic, and no
``KPIDefinition`` field that the KG does not actually populate.  Hand-curated
content must not be reintroduced.

If the KG cannot be loaded, this module raises at import time.  That is the
intended behavior — the codebase is supposed to be unusable without the KG.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Dataclasses — shape matches what the KARMA pipeline actually writes
# =============================================================================


@dataclass
class KPIDefinition:
    """A KPI as it exists in the KG.

    Only fields the KARMA pipeline actually emits are present.  Anything
    semantic about *what is good or bad* (typical/warning/critical thresholds,
    direction, environment-specific ranges, hardcoded min/max) is deliberately
    absent.

    ``valid_range`` and ``is_integer`` are **KG-derived** interpretations of
    the spec-extracted ``unit`` field — they are not hand-authored.  See
    ``ontology_kg_loader._infer_kpi_numeric_meta``.
    """

    short_name: str
    description: str = ""
    formula: str = ""
    unit: str = ""
    original_names: List[str] = field(default_factory=list)
    source_specs: List[str] = field(default_factory=list)
    source_clauses: List[str] = field(default_factory=list)
    confidence: float = 1.0
    mapped_to: Optional[str] = None
    valid_range: Optional[Tuple[float, float]] = None
    is_integer: bool = False


@dataclass
class KPIRelationship:
    """Directed causal relationship between two KPIs (KG-loaded)."""

    source_kpi: str
    target_kpi: str
    relation_type: str  # INCREASES | DECREASES | DETERMINES | LIMITS | CAUSES
    strength: str  # critical | strong | moderate | weak | extracted
    reason: str = ""
    condition: Optional[str] = None
    spec_reference: Optional[str] = None


@dataclass
class CausalChain:
    """Multi-step causal chain across KPIs (KG-loaded)."""

    name: str
    description: str
    steps: List[str]
    trigger_condition: Optional[str] = None


@dataclass
class DirectionalRule:
    """3GPP-grounded directional coupling rule, materialized from a KG edge."""

    source_kpi: str
    target_kpi: str
    coupling: str  # "positive" | "negative"
    strength: str  # "critical" | "strong" | "moderate"
    min_pct_source: float
    min_pct_target: float
    mechanism: str
    spec_reference: str
    spec_clause: str
    correction_hint: str
    condition: Optional[str] = None


# =============================================================================
# Module-level KG-loaded artifacts (resolved at import time)
# =============================================================================


def _bootstrap_directional_rules() -> List[DirectionalRule]:
    """Load directional rules from the KG at import time.

    Returns an empty list (with a warning) when the KG file is missing or
    contains no usable rules.  This is what enables the KG *build* itself
    to import this module before any KG exists; the Predictor still
    raises at runtime if it tries to use directional rules that aren't
    there, but ``pipeline.KGConstructionPipeline`` (which builds the
    KG) doesn't need them.
    """
    from .kg_loader import (
        DEFAULT_KG_PATH,
        load_directional_rules_from_kg,
        load_kpi_definitions_from_kg,
    )

    try:
        kpis = load_kpi_definitions_from_kg(DEFAULT_KG_PATH)
    except FileNotFoundError:
        logger.warning(
            "KG file %s missing at import time; directional rules empty. "
            "This is expected during a KG build run; downstream Predictor "
            "use will fail until the build finishes.",
            DEFAULT_KG_PATH,
        )
        return []
    try:
        rules = load_directional_rules_from_kg(DEFAULT_KG_PATH, set(kpis))
    except FileNotFoundError:
        return []
    if not rules:
        logger.warning(
            "KG at %s produced 0 directional rules — Predictor will have "
            "no causal calibration until the KG is rebuilt with relations.",
            DEFAULT_KG_PATH,
        )
    else:
        logger.info(
            "Loaded %d directional rules from KG %s",
            len(rules),
            DEFAULT_KG_PATH,
        )
    return rules


_DIRECTIONAL_RULES: List[DirectionalRule] = _bootstrap_directional_rules()


# =============================================================================
# KG-driven causal-consistency analytics
# =============================================================================


def _hour_of_week_baseline(input_arr: np.ndarray, horizon: int = 168) -> np.ndarray:
    """Collapse a 1-D input series into a 168-hour hour-of-week median baseline."""
    arr = np.asarray(input_arr, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.zeros(horizon, dtype=float)
    if arr.size < horizon:
        reps = int(np.ceil(horizon / arr.size))
        arr = np.tile(arr, reps)[: horizon * max(1, arr.size // horizon or 1)]
    n_full = arr.size // horizon
    if n_full == 0:
        return arr[:horizon]
    truncated = arr[: n_full * horizon].reshape(n_full, horizon)
    return np.median(truncated, axis=0)


def check_directional_consistency_hourly(
    input_series: Dict[str, Any],
    pred_series: Dict[str, Any],
    min_violation_hours: int = 8,
) -> List[Dict[str, Any]]:
    """Hourly directional-consistency check against KG-loaded causal rules."""
    horizon = 168
    violations: List[Dict[str, Any]] = []

    baselines: Dict[str, np.ndarray] = {}
    preds: Dict[str, np.ndarray] = {}
    input_means: Dict[str, float] = {}
    for kpi, series in input_series.items():
        arr = np.asarray(series, dtype=float).reshape(-1)
        if arr.size == 0:
            continue
        baselines[kpi] = _hour_of_week_baseline(arr, horizon)
        input_means[kpi] = float(arr.mean())
    for kpi, series in pred_series.items():
        arr = np.asarray(series, dtype=float).reshape(-1)
        if arr.size == 0:
            continue
        if arr.size < horizon:
            reps = int(np.ceil(horizon / arr.size))
            arr = np.tile(arr, reps)[:horizon]
        preds[kpi] = arr[:horizon]

    for rule in _DIRECTIONAL_RULES:
        src, tgt = rule.source_kpi, rule.target_kpi
        if src not in baselines or src not in preds:
            continue
        if tgt not in baselines or tgt not in preds:
            continue

        if rule.condition:
            try:
                parts = rule.condition.split()
                cond_kpi = parts[0].replace("_baseline", "")
                cond_op = parts[1]
                cond_val = float(parts[2])
                bv = input_means.get(cond_kpi, 0.0)
                if cond_op == ">" and bv <= cond_val:
                    continue
                if cond_op == "<" and bv >= cond_val:
                    continue
            except Exception:
                pass

        src_base = baselines[src]
        tgt_base = baselines[tgt]
        src_pred = preds[src]
        tgt_pred = preds[tgt]

        src_pct = 100.0 * (src_pred - src_base) / (np.abs(src_base) + 1e-8)
        tgt_pct = 100.0 * (tgt_pred - tgt_base) / (np.abs(tgt_base) + 1e-8)

        active = np.abs(src_pct) >= rule.min_pct_source
        if not np.any(active):
            continue

        if rule.coupling == "positive":
            wrong = ((src_pct > 0) & (tgt_pct < -rule.min_pct_target)) | (
                (src_pct < 0) & (tgt_pct > rule.min_pct_target)
            )
        else:
            wrong = ((src_pct > 0) & (tgt_pct > rule.min_pct_target)) | (
                (src_pct < 0) & (tgt_pct < -rule.min_pct_target)
            )

        violating_mask = active & wrong
        n_violating = int(violating_mask.sum())
        if n_violating < min_violation_hours:
            continue

        idxs = np.where(violating_mask)[0]
        src_pct_at = src_pct[idxs]
        tgt_pct_at = tgt_pct[idxs]
        mean_src_pct = float(np.mean(src_pct_at))
        mean_tgt_pct = float(np.mean(tgt_pct_at))
        if rule.coupling == "positive":
            expected_dir = "↑" if mean_src_pct > 0 else "↓"
            actual_dir = "↓" if mean_src_pct > 0 else "↑"
        else:
            expected_dir = "↓" if mean_src_pct > 0 else "↑"
            actual_dir = "↑" if mean_src_pct > 0 else "↓"

        violations.append(
            {
                "rule": (
                    f"{src} {'→+' if rule.coupling == 'positive' else '→−'} {tgt}"
                    f" [{rule.strength}] ({rule.spec_reference}"
                    + (f" §{rule.spec_clause})" if rule.spec_clause else ")")
                ),
                "source_kpi": src,
                "target_kpi": tgt,
                "violating_hours": n_violating,
                "first_violating_hour": int(idxs[0]),
                "last_violating_hour": int(idxs[-1]),
                "mean_source_pct_change": round(mean_src_pct, 1),
                "mean_target_pct_change": round(mean_tgt_pct, 1),
                "expected_target_direction": expected_dir,
                "actual_target_direction": actual_dir,
                "severity": rule.strength,
                "mechanism": rule.mechanism,
                "correction_hint": rule.correction_hint,
            }
        )

    return violations


# =============================================================================
# Main Ontology Class — every member is KG-loaded
# =============================================================================


class ThreeGPPOntology:
    """KPI ontology loaded from the spec-extracted KG.

    Raises if the KG cannot be loaded — there is no in-Python fallback.
    """

    def __init__(
        self,
        kg_path: Optional[str] = None,
        strict: bool = True,
        min_confidence: float = 0.7,
        drop_fraction: float = 0.0,
    ) -> None:
        """Load ontology from the spec-extracted KG.

        ``strict=True`` (default) enforces the thesis policy: the Predictor must
        not run against an empty ontology.  Raises ``RuntimeError`` if the KG
        file is missing or yields zero KPIs.  ``strict=False`` is reserved for
        the KG build pipeline itself (``telcoagent.kg_construction.pipeline``), which
        must be able to import this module *before* any KG has been produced.

        ``min_confidence`` is the q_TH threshold applied to ``strength ==
        "extracted"`` KG edges (kg_loader._build_relationship,
        _build_directional_rule).  Hardcoded edges
        (``strength`` in {critical, strong, moderate, weak}) are always
        admitted regardless of confidence.  The default 0.7 matches both
        ``load_relations_from_kg`` and ``load_directional_rules_from_kg``
        defaults; AX-4 (Phase A — confidence semantic) sweeps this value
        in [0.1, 0.9].

        ``drop_fraction`` is the AX-4 Phase B (rank-based drop) semantic.
        After the standard ``min_confidence`` load, every surviving
        relationship is scored on the strength × confidence axis:

            score(rel) = STRENGTH_RANK[rel.strength]

        where ``STRENGTH_RANK = {critical: 1.0, strong: 0.9, moderate: 0.7,
        weak: 0.5, extracted: <confidence-from-KG>}``.  Edges are sorted
        ascending by ``score`` (weakest first) and the bottom
        ``round(drop_fraction × N)`` are removed.  ``drop_fraction=0.0``
        (default) is a no-op; ``drop_fraction=1.0`` removes every edge.
        Mutually exclusive with non-default ``min_confidence`` — passing
        both raises ``ValueError`` to keep the AX-4 Phase A / Phase B
        arms cleanly separated.

        Note: the extracted-edge confidence used in scoring is NOT
        preserved on the ``KPIRelationship`` dataclass (kg_loader strips
        it).  We re-read the KG edges JSON here only when
        ``drop_fraction > 0`` so the rank-based filter has the original
        confidence available; the read is cheap (~ms) compared to the
        explainer's LLM call.
        """
        # Mutual exclusivity check between the two AX-4 semantics — keeps
        # confusing "what does q=0.5 mean here" questions out of the
        # paper.  Phase A uses ``min_confidence``; Phase B uses
        # ``drop_fraction``; the default arm reproduces AX-2 KG-on.
        if drop_fraction > 0.0 and min_confidence != 0.7:
            raise ValueError(
                "ThreeGPPOntology: min_confidence and drop_fraction are "
                "mutually exclusive AX-4 semantics — set only one. "
                f"Got min_confidence={min_confidence}, drop_fraction={drop_fraction}."
            )
        if not 0.0 <= drop_fraction <= 1.0:
            raise ValueError(
                f"ThreeGPPOntology: drop_fraction must be in [0, 1]; got {drop_fraction}."
            )
        from .kg_loader import (
            DEFAULT_KG_PATH,
            load_directional_rules_from_kg,
            load_kpi_definitions_from_kg,
            load_relations_from_kg,
        )

        path = kg_path or DEFAULT_KG_PATH
        self._ontology_source = f"kg:{path}"

        try:
            self.kpis: Dict[str, KPIDefinition] = load_kpi_definitions_from_kg(path)
        except FileNotFoundError:
            if strict:
                raise RuntimeError(
                    f"KG file {path} missing. The Predictor requires a loaded KG. "
                    f"Rebuild via telcoagent.kg_construction.pipeline. If you are the KG "
                    f"build pipeline itself, instantiate with strict=False."
                ) from None
            logger.warning(
                "KG file %s missing — ThreeGPPOntology starts empty (strict=False). "
                "Only the KG build pipeline should use this mode.",
                path,
            )
            self.kpis = {}
            self.relationships = []
            self.causal_chains = []
            self.directional_rules = []
            return

        if strict and not self.kpis:
            raise RuntimeError(
                f"KG file {path} loaded but contains 0 KPI definitions. "
                f"The Predictor cannot operate without KPI metadata."
            )

        try:
            relationships, chains = load_relations_from_kg(
                path, set(self.kpis), min_confidence=min_confidence
            )
        except FileNotFoundError:
            relationships, chains = [], []

        self.relationships: List[KPIRelationship] = relationships
        self.causal_chains: List[CausalChain] = chains
        # Reload directional rules per-instance so ``min_confidence`` (AX-4
        # q_TH sweep) actually affects what survives.  At the default 0.7 we
        # reuse the import-time bootstrap to avoid the second KG round-trip;
        # any other threshold forces a fresh load.
        if min_confidence == 0.7:
            self.directional_rules: List[DirectionalRule] = list(_DIRECTIONAL_RULES)
        else:
            try:
                self.directional_rules = load_directional_rules_from_kg(
                    path, set(self.kpis), min_confidence=min_confidence
                )
            except FileNotFoundError:
                self.directional_rules = []

        # AX-4 Phase B — rank-based drop_fraction filter.  Applied AFTER
        # the standard load so the ranking universe is the same set the
        # confidence-semantic baseline would expose at min_confidence=0.7.
        if drop_fraction > 0.0:
            self._apply_drop_fraction(path, drop_fraction)

        if strict and not self.directional_rules:
            logger.warning(
                "KG %s yielded 0 directional rules. Causal refinement will be "
                "a no-op; consider rebuilding the KG with richer relation edges.",
                path,
            )

    # AX-4 Phase B helper — kept as a method (not a free function) so the
    # ranking logic lives next to the dataclass it mutates.
    _STRENGTH_RANK: Dict[str, float] = {
        "critical": 1.0,
        "strong": 0.9,
        "moderate": 0.7,
        "weak": 0.5,
        "extracted": 0.0,  # actual score comes from the KG-edge confidence
    }

    def _apply_drop_fraction(self, kg_path: str, drop_fraction: float) -> None:
        """Drop the bottom ``drop_fraction`` of edges by strength × confidence.

        See :meth:`__init__` for the ranking formula.  Mutates
        ``self.relationships`` and ``self.directional_rules`` in place so the
        downstream explainer + cause_inference layers see the pruned set.
        """
        import json

        try:
            with open(kg_path) as f:
                kg_data = json.load(f)
        except FileNotFoundError:
            logger.warning("drop_fraction filter skipped — KG file %s missing", kg_path)
            return

        from .kg_loader import _canonical_relation_type

        edges = kg_data.get("edges") or kg_data.get("links") or []
        # Build a lookup: (src, tgt, relation_type) → original confidence.
        # The relation_type is upper-cased AND alias-normalized exactly the
        # way kg_loader does before constructing KPIRelationship — otherwise
        # an alias-typed extracted edge (e.g. POSITIVELY_AFFECTS) would key
        # under the alias here but be looked up under the canonical type,
        # silently scoring 0.0 and being dropped first.
        conf_lookup: Dict[Tuple[str, str, str], float] = {}
        for edge in edges:
            if edge.get("triple_type") not in ("ontology_relationship", "relationship"):
                continue
            src = edge.get("source")
            tgt = edge.get("target")
            rt = _canonical_relation_type((edge.get("relation_type") or "").upper())
            if not (src and tgt and rt):
                continue
            key = (src, tgt, rt)
            # Keep the maximum confidence if multiple edges with the same key.
            conf_lookup[key] = max(conf_lookup.get(key, 0.0), float(edge.get("confidence", 1.0)))

        def _rel_score(rel: KPIRelationship) -> float:
            base = self._STRENGTH_RANK.get(rel.strength, 0.0)
            if rel.strength == "extracted":
                return conf_lookup.get((rel.source_kpi, rel.target_kpi, rel.relation_type), 0.0)
            return base

        def _dir_score(rule: DirectionalRule) -> float:
            base = self._STRENGTH_RANK.get(rule.strength, 0.0)
            # Directional rules don't carry strength == "extracted" in
            # practice (kg_loader maps them to coupling/critical-tier
            # buckets) — score on strength alone.
            return base

        # Rank ASCENDING (weakest first) so dropping the bottom slice
        # removes the least-supported edges first.
        n_rels = len(self.relationships)
        if n_rels:
            ranked = sorted(self.relationships, key=_rel_score)
            n_drop = int(round(drop_fraction * n_rels))
            kept = ranked[n_drop:]
            # Preserve the original (source, target, relation_type) sort
            # order kg_loader establishes; downstream cause_inference
            # iterates relationships in deterministic order.
            kept.sort(key=lambda r: (r.source_kpi, r.target_kpi, r.relation_type))
            self.relationships = kept
            logger.info(
                "drop_fraction=%.2f: %d → %d relationships (dropped %d weakest)",
                drop_fraction,
                n_rels,
                len(kept),
                n_drop,
            )

        n_dirs = len(self.directional_rules)
        if n_dirs:
            ranked = sorted(self.directional_rules, key=_dir_score)
            n_drop = int(round(drop_fraction * n_dirs))
            kept_dir = ranked[n_drop:]
            self.directional_rules = kept_dir
            logger.info(
                "drop_fraction=%.2f: %d → %d directional rules (dropped %d weakest)",
                drop_fraction,
                n_dirs,
                len(kept_dir),
                n_drop,
            )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_kpi(self, short_name: str) -> Optional[KPIDefinition]:
        return self.kpis.get(short_name)

    def get_related_kpis(self, kpi_name: str) -> Dict[str, List[KPIRelationship]]:
        return {
            "affects": [r for r in self.relationships if r.source_kpi == kpi_name],
            "affected_by": [r for r in self.relationships if r.target_kpi == kpi_name],
        }

    def get_causal_chains_for_kpi(self, kpi_name: str) -> List[CausalChain]:
        return [c for c in self.causal_chains if any(kpi_name in s for s in c.steps)]

    # ------------------------------------------------------------------
    # KG-derived numeric metadata helpers
    # ------------------------------------------------------------------
    # These methods expose the ``valid_range`` / ``is_integer`` fields that
    # ``ontology_kg_loader._infer_kpi_numeric_meta`` populated from the
    # spec-extracted unit.  No hand-curated KPI catalog is consulted — if the
    # KG does not carry enough metadata for a predictor KPI, these helpers
    # return ``None`` / ``False`` and the caller must degrade gracefully.

    def unit(self, kpi_name: str) -> str:
        kpi = self.get_kpi(kpi_name)
        return kpi.unit if kpi else ""

    def is_integer_kpi(self, kpi_name: str) -> bool:
        kpi = self.get_kpi(kpi_name)
        return bool(kpi and kpi.is_integer)

    def integer_kpis(self, for_names: Optional[List[str]] = None) -> frozenset:
        """Return the subset of ``for_names`` (or all KG KPIs) that are integers.

        The decision is made from KG-provided units (see loader).
        """
        candidates = for_names if for_names is not None else list(self.kpis.keys())
        return frozenset(n for n in candidates if self.is_integer_kpi(n))

    def physical_range(self, kpi_name: str) -> Optional[Tuple[float, float]]:
        """Return the KG-derived valid range for a KPI, or ``None`` if unknown."""
        kpi = self.get_kpi(kpi_name)
        return kpi.valid_range if kpi else None

    # ------------------------------------------------------------------
    # Formatting (purely structural — no thresholds/typical/critical fields)
    # ------------------------------------------------------------------

    def format_kpi_info(self, kpi_name: str) -> str:
        kpi = self.get_kpi(kpi_name)
        if not kpi:
            return f"KPI '{kpi_name}' not found in ontology."

        lines = [
            f"=== {kpi.short_name} ===",
            f"Description: {kpi.description}" if kpi.description else "",
            f"Formula    : {kpi.formula}" if kpi.formula else "",
            f"Unit       : {kpi.unit}" if kpi.unit else "",
        ]
        if kpi.source_specs:
            lines.append(f"Source specs : {', '.join(kpi.source_specs)}")
        if kpi.source_clauses:
            lines.append(f"Source clauses: {', '.join(kpi.source_clauses)}")
        if kpi.original_names and kpi.original_names != [kpi.short_name]:
            lines.append(f"Aliases    : {', '.join(kpi.original_names)}")

        related = self.get_related_kpis(kpi_name)
        if related["affects"]:
            lines += ["", "Affects:"]
            for r in related["affects"]:
                cond = f" (when {r.condition})" if r.condition else ""
                lines.append(f"  → {r.target_kpi} [{r.relation_type}, {r.strength}]{cond}")
        if related["affected_by"]:
            lines += ["", "Affected by:"]
            for r in related["affected_by"]:
                cond = f" (when {r.condition})" if r.condition else ""
                lines.append(f"  ← {r.source_kpi} [{r.relation_type}, {r.strength}]{cond}")

        return "\n".join(line for line in lines if line)

    def get_ontology_summary(self) -> str:
        lines = [
            "=" * 60,
            "3GPP KPI Ontology (KG-driven, spec-extracted)",
            "=" * 60,
            "",
            f"Total KPIs          : {len(self.kpis)}",
            f"Total Relationships : {len(self.relationships)}",
            f"Total Causal Chains : {len(self.causal_chains)}",
            f"Directional Rules   : {len(self.directional_rules)}",
            f"Ontology source     : {self._ontology_source}",
        ]
        return "\n".join(lines)


# =============================================================================
# Utility
# =============================================================================


@functools.lru_cache(maxsize=32)
def get_default_ontology(
    kg_path: Optional[str] = None,
    strict: bool = True,
    min_confidence: float = 0.7,
    drop_fraction: float = 0.0,
) -> ThreeGPPOntology:
    """Return a cached ThreeGPPOntology. Strict by default (Predictor policy).

    Pass ``strict=False`` only from the KG build pipeline.

    ``min_confidence`` and ``drop_fraction`` extend the cache key so the AX-4
    q_TH sweep can pre-warm one ontology per sweep point.  Both default to
    the Predictor's existing baseline behaviour; only the Explainer's q_TH
    sweep should pass non-default values.  See
    :meth:`ThreeGPPOntology.__init__` for the AX-4 Phase A / Phase B
    semantics.
    """
    return ThreeGPPOntology(
        kg_path=kg_path,
        strict=strict,
        min_confidence=min_confidence,
        drop_fraction=drop_fraction,
    )
