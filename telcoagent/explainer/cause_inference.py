"""KG-grounded cause inference for forecast anomaly events.

Given the deterministic events emitted by
:func:`telcoagent.explainer.anomaly_detector.detect_anomaly_events`, this
module enriches each event with a ranked list of candidate upstream
causes drawn purely from:

* the spec-extracted 3GPP KG (``ThreeGPPOntology.get_related_kpis``), and
* concurrent-event co-occurrence in the same forecast horizon, with an
  optional fallback to background-value deviation against the input
  baseline.

No statistical estimator (e.g., Transfer Entropy) is applied here. The
function is pure (no I/O besides a single INFO log line) and
JSON-serializable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..ontology.core import KPIRelationship, ThreeGPPOntology
from .anomaly_detector import AnomalyEvent, events_to_dicts

logger = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────────────────

#: KG strength → numeric ordering for ranking candidate causes.
STRENGTH_ORDER: Dict[str, int] = {
    "critical": 4,
    "strong": 3,
    "moderate": 2,
    "weak": 1,
    "extracted": 1,
}

#: Default top-K cap on candidate causes per event.
_DEFAULT_TOP_K: int = 5

#: Z-deviation threshold for background-value evidence (only when no
#: concurrent source-KPI event was found).
_BACKGROUND_Z_THRESHOLD: float = 1.5

#: Relation types where source and target are expected to move in
#: the same direction (positive coupling).
_POSITIVE_RELATIONS = {"INCREASES", "DETERMINES", "CAUSES"}

#: Relation types where source and target are expected to move in
#: opposite directions (negative coupling).
_NEGATIVE_RELATIONS = {"DECREASES", "LIMITS"}


# ─── Helpers ──────────────────────────────────────────────────────────────


def _windows_overlap(
    s_start: int,
    s_end: int,
    t_start: int,
    t_end: int,
    window: int,
) -> bool:
    """``[s_start - W, s_end + W] ∩ [t_start, t_end]`` non-empty."""
    return (s_end + window) >= t_start and (s_start - window) <= t_end


def _direction_consistency(
    relation_type: str,
    source_dir: str,
    target_dir: str,
) -> Optional[bool]:
    """Compare KG-stated relation polarity vs. observed event directions.

    Returns ``True`` / ``False`` for positive/negative couplings, or
    ``None`` for unknown relation types or ``mixed`` directions.
    """
    if source_dir == "mixed" or target_dir == "mixed":
        return None
    if relation_type in _NEGATIVE_RELATIONS:
        return source_dir != target_dir
    if relation_type in _POSITIVE_RELATIONS:
        return source_dir == target_dir
    return None


def _find_concurrent_source_event(
    source_kpi: str,
    target: AnomalyEvent,
    events: List[AnomalyEvent],
    window: int,
) -> Optional[AnomalyEvent]:
    """Pick the strongest source-KPI event overlapping the target window.

    "Strongest" is the largest |z_score|. ``MULTI`` events are skipped
    because they are not associated with a single channel.
    """
    candidates = [
        ev
        for ev in events
        if ev is not target
        and ev.kpi == source_kpi
        and ev.kpi != "MULTI"
        and _windows_overlap(ev.t_start, ev.t_end, target.t_start, target.t_end, window)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda ev: abs(ev.z_score))


def _build_concurrent_evidence(
    source_event: AnomalyEvent,
    target: AnomalyEvent,
) -> Dict[str, Any]:
    return {
        "type": source_event.type,
        "t_start": int(source_event.t_start),
        "t_end": int(source_event.t_end),
        "magnitude": round(float(source_event.magnitude), 4),
        "z_score": round(float(source_event.z_score), 3),
        "direction": source_event.direction,
        "lag_hours": int(source_event.t_start - target.t_start),
    }


def _background_evidence(
    source_kpi: str,
    target: AnomalyEvent,
    forecast: np.ndarray,
    kpi_names: List[str],
    baseline: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    """Compute background-value z-deviation for the source KPI.

    Returns ``None`` if the deviation is below threshold, the KPI is
    not present in ``kpi_names``, or the inputs are missing.
    """
    if baseline is None:
        return None
    if source_kpi not in kpi_names:
        return None
    source_idx = kpi_names.index(source_kpi)
    if source_idx >= forecast.shape[1] or source_idx >= baseline.shape[1]:
        return None

    segment = forecast[target.t_start : target.t_end + 1, source_idx]
    if segment.size == 0:
        return None

    base_window = (
        baseline[-168:, source_idx] if baseline.shape[0] >= 168 else baseline[:, source_idx]
    )
    if base_window.size == 0:
        return None

    base_mean = float(base_window.mean())
    base_std = float(base_window.std()) or 1.0
    z_dev = float((segment.mean() - base_mean) / base_std)

    if abs(z_dev) < _BACKGROUND_Z_THRESHOLD:
        return None
    return {
        "mean_during_event": float(segment.mean()),
        "baseline_mean": base_mean,
        "z_deviation": z_dev,
    }


def _rank_key(cause: Dict[str, Any]) -> tuple:
    """Sort key for candidate causes (descending rank)."""
    has_concurrent = 1 if cause.get("concurrent_evidence") else 0
    strength = STRENGTH_ORDER.get(cause.get("strength", ""), 0)
    concurrent = cause.get("concurrent_evidence") or {}
    z_abs = abs(float(concurrent.get("z_score", 0.0))) if concurrent else 0.0
    lag = concurrent.get("lag_hours", 0) if concurrent else 0
    consistency = 1 if cause.get("direction_consistency") is True else 0
    return (has_concurrent, strength, z_abs, -abs(int(lag)), consistency)


def _candidate_for_relationship(
    rel: KPIRelationship,
    target: AnomalyEvent,
    events: List[AnomalyEvent],
    forecast: np.ndarray,
    kpi_names: List[str],
    baseline: Optional[np.ndarray],
    window: int,
) -> Optional[Dict[str, Any]]:
    """Build one cause record from a single ``affected_by`` edge."""
    source_kpi = rel.source_kpi
    if source_kpi == target.kpi:
        return None  # self-edges aren't informative.

    source_event = _find_concurrent_source_event(source_kpi, target, events, window)
    concurrent_evidence: Optional[Dict[str, Any]] = None
    background_value: Optional[Dict[str, Any]] = None
    direction_consistency: Optional[bool] = None

    if source_event is not None:
        concurrent_evidence = _build_concurrent_evidence(source_event, target)
        direction_consistency = _direction_consistency(
            rel.relation_type,
            source_event.direction,
            target.direction,
        )
    else:
        background_value = _background_evidence(
            source_kpi,
            target,
            forecast,
            kpi_names,
            baseline,
        )
        if background_value is None:
            # No concurrent event AND no notable background deviation — skip.
            return None

    return {
        "source_kpi": source_kpi,
        "relation_type": rel.relation_type,
        "strength": rel.strength,
        "spec_reference": rel.spec_reference,
        "concurrent_evidence": concurrent_evidence,
        "background_value": background_value,
        "direction_consistency": direction_consistency,
        "shared_mechanism": False,
    }


def _multi_event_causes(
    target: AnomalyEvent,
    ontology: ThreeGPPOntology,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Surface KG-known causal pairs among co-occurring KPIs."""
    co_kpis = list(target.co_occurring_kpis or [])
    if len(co_kpis) < 2:
        return []

    seen_pairs: set = set()
    candidates: List[Dict[str, Any]] = []
    for a in co_kpis:
        for b in co_kpis:
            if a == b:
                continue
            related = ontology.get_related_kpis(a)
            for rel in related.get("affects", []):
                if rel.target_kpi != b:
                    continue
                pair_key = (rel.source_kpi, rel.target_kpi)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                candidates.append(
                    {
                        "source_kpi": rel.source_kpi,
                        "relation_type": rel.relation_type,
                        "strength": rel.strength,
                        "spec_reference": rel.spec_reference,
                        "concurrent_evidence": None,
                        "background_value": None,
                        "direction_consistency": None,
                        "shared_mechanism": True,
                    }
                )

    candidates.sort(
        key=lambda c: STRENGTH_ORDER.get(c.get("strength", ""), 0),
        reverse=True,
    )
    return candidates[:top_k]


# ─── Public API ───────────────────────────────────────────────────────────


def infer_anomaly_causes(
    events: List[AnomalyEvent],
    ontology: ThreeGPPOntology,
    forecast: np.ndarray,
    kpi_names: List[str],
    baseline: Optional[np.ndarray] = None,
    window_hours: int = 6,
    top_k: int = _DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """Enrich each event with KG-grounded candidate causes.

    Returns a list of dicts in the same order as :func:`events_to_dicts`,
    each augmented with ``candidate_causes``: KG-grounded upstream-KPI
    cause records (may be empty). See module docstring for the ranking
    key.
    """
    enriched = events_to_dicts(events)

    if ontology is None:
        for event_dict in enriched:
            event_dict["candidate_causes"] = []
        logger.info(
            "Cause inference skipped: ontology unavailable (%d events).",
            len(events),
        )
        return enriched

    n_with_causes = 0
    for event, event_dict in zip(events, enriched):
        if event.kpi == "MULTI" or event.type == "multi_kpi":
            causes = _multi_event_causes(event, ontology, top_k)
        else:
            related = ontology.get_related_kpis(event.kpi)
            relationships: List[KPIRelationship] = related.get("affected_by", [])
            causes_raw: List[Dict[str, Any]] = []
            for rel in relationships:
                cause = _candidate_for_relationship(
                    rel,
                    event,
                    events,
                    forecast,
                    kpi_names,
                    baseline,
                    window_hours,
                )
                if cause is not None:
                    causes_raw.append(cause)
            causes_raw.sort(key=_rank_key, reverse=True)
            causes = causes_raw[:top_k]
        event_dict["candidate_causes"] = causes
        if causes:
            n_with_causes += 1

    logger.info(
        "Cause inference: %d/%d events have KG-grounded candidate causes",
        n_with_causes,
        len(events),
    )
    return enriched
