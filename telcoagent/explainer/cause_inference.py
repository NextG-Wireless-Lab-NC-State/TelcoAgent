"""KG-grounded cause inference for forecast anomaly events.

Given the deterministic events emitted by
:func:`telcoagent.explainer.anomaly_detector.detect_anomaly_events`, this
module enriches each event with a ranked list of candidate upstream
causes drawn purely from:

* the spec-extracted 3GPP KG (``ThreeGPPOntology.get_related_kpis``), and
* concurrent-event co-occurrence in the same forecast horizon, with an
  optional fallback to background-value deviation against the input
  baseline.

When an OSM spatial-context summary is supplied, each event also carries
an ``environmental_context`` payload with the parsed station factors
(area_type, poi_density_band, building_height_class, transit_proximity,
telecom_infra, land_use). No telecom-domain causal interpretation is
hardcoded -- the LLM harness ``infer_environmental_relevance_with_llm``
(when invoked) consumes these raw factors and produces its own per-event
environmental assessment, which is attached as
``environmental_context.llm_assessment``.

No statistical estimator (e.g., Transfer Entropy) is applied here. The
deterministic part of the function is pure (no I/O besides a single
INFO log line) and JSON-serializable. The optional LLM harness step
runs only when ``relevance_model`` is provided to
:func:`infer_anomaly_causes`.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

from ..llm.api import invoke_llm
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


# ─── OSM context parsing ──────────────────────────────────────────────────


_OSM_DEFAULTS: Dict[str, Any] = {
    "location": None,
    "area_type": None,
    "poi_density_band": None,
    "poi_count": 0,
    "land_use": [],
    "building_count": 0,
    "avg_building_height_m": 0.0,
    "max_building_height_m": 0.0,
    "transit_hubs": 0,
    "telecom_masts": 0,
    "environment_hint": None,
}

#: Regex patterns for parsing the OpenStreetMapMCP text summary. Each
#: pattern is tolerant to leading whitespace and surrounding punctuation
#: variations so that a missing line never raises.
_OSM_LINE_PATTERNS: Dict[str, re.Pattern] = {
    "location": re.compile(r"^\s*Location\s*:\s*(.+?)\s*$", re.IGNORECASE),
    "area_type": re.compile(r"^\s*Area type\s*:\s*(.+?)\s*$", re.IGNORECASE),
    "poi_density": re.compile(
        r"^\s*POI density\s*:\s*([A-Za-z ]+?)\s*" r"\(\s*(\d+)\s*POIs?\s*within",
        re.IGNORECASE,
    ),
    "land_use": re.compile(r"^\s*Land use\s*:\s*(.+?)\s*$", re.IGNORECASE),
    "buildings": re.compile(
        r"^\s*Buildings\s*:\s*(\d+)\s+within\s+\S+"
        r"(?:.*?avg\s+height\s*=\s*([\d.]+)\s*m?)?"
        r"(?:.*?max\s+height\s*=\s*([\d.]+)\s*m?)?",
        re.IGNORECASE,
    ),
    "transit": re.compile(r"^\s*Transit hubs\s*:\s*(\d+)", re.IGNORECASE),
    "telecom": re.compile(
        r"^\s*Telecom infrastructure\s*:\s*(\d+)\s+mast",
        re.IGNORECASE,
    ),
    "env_hint": re.compile(
        r"^\s*Environment hint(?:\s*\(heuristic\))?\s*:\s*(.+?)\s*$",
        re.IGNORECASE,
    ),
}


def parse_osm_context(osm_text: str) -> Dict[str, Any]:
    """Convert the OpenStreetMapMCP text summary into a structured dict.

    Returns a dict with these keys (each Optional / safely defaulted):
        location:                str            (e.g. "Frisco, Texas, US")
        area_type:               str            (e.g. "residential/suburb")
        poi_density_band:        str            (Low|Medium|High|"Very Low")
        poi_count:               int
        land_use:                List[str]
        building_count:          int
        avg_building_height_m:   float
        max_building_height_m:   float
        transit_hubs:            int
        telecom_masts:           int
        environment_hint:        str            (the heuristic line)

    Missing fields default to ``None`` / ``0`` / ``[]``. Pure parsing —
    no network calls, no logging.
    """
    parsed: Dict[str, Any] = dict(_OSM_DEFAULTS)
    parsed["land_use"] = []  # fresh list per call
    if not osm_text or not isinstance(osm_text, str):
        return parsed

    for raw_line in osm_text.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue

        m = _OSM_LINE_PATTERNS["location"].match(line)
        if m:
            parsed["location"] = m.group(1).strip()
            continue

        m = _OSM_LINE_PATTERNS["area_type"].match(line)
        if m:
            parsed["area_type"] = m.group(1).strip()
            continue

        m = _OSM_LINE_PATTERNS["poi_density"].match(line)
        if m:
            parsed["poi_density_band"] = m.group(1).strip()
            try:
                parsed["poi_count"] = int(m.group(2))
            except (TypeError, ValueError):
                parsed["poi_count"] = 0
            continue

        m = _OSM_LINE_PATTERNS["land_use"].match(line)
        if m:
            parts = [token.strip() for token in re.split(r"[,/;]", m.group(1)) if token.strip()]
            parsed["land_use"] = parts
            continue

        m = _OSM_LINE_PATTERNS["buildings"].match(line)
        if m:
            try:
                parsed["building_count"] = int(m.group(1))
            except (TypeError, ValueError):
                parsed["building_count"] = 0
            try:
                parsed["avg_building_height_m"] = float(m.group(2)) if m.group(2) else 0.0
            except (TypeError, ValueError):
                parsed["avg_building_height_m"] = 0.0
            try:
                parsed["max_building_height_m"] = float(m.group(3)) if m.group(3) else 0.0
            except (TypeError, ValueError):
                parsed["max_building_height_m"] = 0.0
            continue

        m = _OSM_LINE_PATTERNS["transit"].match(line)
        if m:
            try:
                parsed["transit_hubs"] = int(m.group(1))
            except (TypeError, ValueError):
                parsed["transit_hubs"] = 0
            continue

        m = _OSM_LINE_PATTERNS["telecom"].match(line)
        if m:
            try:
                parsed["telecom_masts"] = int(m.group(1))
            except (TypeError, ValueError):
                parsed["telecom_masts"] = 0
            continue

        m = _OSM_LINE_PATTERNS["env_hint"].match(line)
        if m:
            parsed["environment_hint"] = m.group(1).strip()
            continue

    return parsed


# ─── Environment-factor classification ────────────────────────────────────


def _building_height_class(avg_height_m: float) -> str:
    """Map average building height (m) to a categorical class."""
    if avg_height_m >= 25.0:
        return "high_rise"
    if avg_height_m >= 10.0:
        return "mid_rise"
    return "low_rise"


def _transit_proximity(transit_hubs: int) -> str:
    if transit_hubs >= 2:
        return "near"
    if transit_hubs == 1:
        return "moderate"
    return "none"


def _telecom_infra(masts: int) -> str:
    return "present" if masts >= 1 else "none"


def _osm_summary_line(parsed: Dict[str, Any], factors: Dict[str, Any]) -> str:
    """One-line description of the parsed OSM context."""
    parts: List[str] = []
    if parsed.get("area_type"):
        parts.append(str(parsed["area_type"]))

    band = parsed.get("poi_density_band")
    poi_count = parsed.get("poi_count", 0)
    if band:
        parts.append(f"{band} POI density ({poi_count} POIs/500m)")

    height_class = factors.get("building_height_class", "low_rise")
    avg_h = parsed.get("avg_building_height_m", 0.0) or 0.0
    if avg_h > 0:
        height_label = height_class.replace("_", "-")
        parts.append(f"{height_label} (avg {avg_h:.1f}m)")
    else:
        parts.append(height_class.replace("_", "-"))

    transit_hubs = parsed.get("transit_hubs", 0) or 0
    if transit_hubs:
        suffix = "hub" if transit_hubs == 1 else "hubs"
        parts.append(f"{transit_hubs} transit {suffix}")

    masts = parsed.get("telecom_masts", 0) or 0
    if masts:
        suffix = "mast" if masts == 1 else "masts"
        parts.append(f"{masts} telecom {suffix}")

    return ", ".join(parts)


def _build_environmental_context(
    event: AnomalyEvent,
    parsed: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the per-event ``environmental_context`` payload.

    Only the parsed OSM factors are surfaced -- no rule-table relevance
    flags are computed here. When the harness LLM is enabled (see
    :func:`infer_environmental_relevance_with_llm`) its assessment is
    attached afterwards under ``llm_assessment``.
    """
    factors: Dict[str, Any] = {
        "area_type": parsed.get("area_type"),
        "poi_density_band": parsed.get("poi_density_band"),
        "poi_count": parsed.get("poi_count"),
        "building_count": parsed.get("building_count"),
        "avg_building_height_m": parsed.get("avg_building_height_m"),
        "max_building_height_m": parsed.get("max_building_height_m"),
        "building_height_class": _building_height_class(
            parsed.get("avg_building_height_m", 0.0) or 0.0,
        ),
        "transit_hubs": parsed.get("transit_hubs"),
        "transit_proximity": _transit_proximity(
            parsed.get("transit_hubs", 0) or 0,
        ),
        "telecom_masts": parsed.get("telecom_masts"),
        "telecom_infra": _telecom_infra(parsed.get("telecom_masts", 0) or 0),
        "land_use": list(parsed.get("land_use") or []),
        "environment_hint": parsed.get("environment_hint"),
    }
    return {
        "osm_summary": _osm_summary_line(parsed, factors),
        "factors": factors,
        "llm_assessment": None,
    }


# ─── Harness: LLM-driven environmental relevance ──────────────────────────


_HARNESS_DEFAULT_MODEL: str = "openai/gpt-4o-mini"
_HARNESS_MAX_TOKENS: int = 600
_HARNESS_TEMPERATURE: float = 0.2

_HARNESS_SYSTEM_PROMPT = (
    "You are an evidence-grounded reasoner for a telecom KPI explanation "
    "pipeline. Given a list of forecast anomaly events plus the parsed "
    "OpenStreetMap (OSM) factors describing the cell's surroundings, judge "
    "whether the environment plausibly contributes to each anomaly. "
    "Reason only over the supplied factors -- never invent values that are "
    "not present. If the OSM factors give no information relevant to an "
    "event, say so explicitly.\n\n"
    'Output STRICT JSON: {"events": [{"event_id": int, '
    '"assessment": str, "primary_layer": str, "confidence": str, '
    '"cited_factors": [str]}]}.\n'
    "  - assessment: one to two short sentences in your own words.\n"
    '  - primary_layer: one of "kg", "osm", "both", or "none".\n'
    '  - confidence: one of "low", "medium", or "high".\n'
    "  - cited_factors: factor keys you actually referenced from the "
    "supplied factors dict. Empty list if none.\n"
    "Do not output prose outside the JSON. Do not invent factor names."
)


def _harness_user_payload(
    events: List[Dict[str, Any]],
    parsed_osm: Dict[str, Any],
) -> str:
    """Render the per-call input payload for the harness LLM."""
    sanitized_events = []
    for idx, ev in enumerate(events):
        sanitized_events.append(
            {
                "event_id": idx,
                "kpi": ev.get("kpi"),
                "type": ev.get("type"),
                "t_start": ev.get("t_start"),
                "t_end": ev.get("t_end"),
                "direction": ev.get("direction"),
                "z_score": ev.get("z_score"),
                "magnitude": ev.get("magnitude"),
                "co_occurring_kpis": ev.get("co_occurring_kpis", []),
                "candidate_causes_summary": [
                    {
                        "source_kpi": c.get("source_kpi"),
                        "relation_type": c.get("relation_type"),
                        "strength": c.get("strength"),
                        "has_concurrent": c.get("concurrent_evidence") is not None,
                        "has_background": c.get("background_value") is not None,
                    }
                    for c in ev.get("candidate_causes", [])
                ],
            }
        )

    payload = {
        "osm_factors": parsed_osm,
        "events": sanitized_events,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _allowed_factor_keys() -> set:
    """Factor keys the harness may legitimately cite."""
    return {
        "area_type",
        "poi_density_band",
        "poi_count",
        "building_count",
        "avg_building_height_m",
        "max_building_height_m",
        "building_height_class",
        "transit_hubs",
        "transit_proximity",
        "telecom_masts",
        "telecom_infra",
        "land_use",
        "environment_hint",
    }


def _validate_harness_record(
    rec: Dict[str, Any],
    n_events: int,
) -> Optional[Dict[str, Any]]:
    """Validate one harness record; return cleaned dict or None on failure."""
    raw_event_id = rec.get("event_id")
    if raw_event_id is None:
        return None
    try:
        event_id = int(raw_event_id)
    except (TypeError, ValueError):
        return None
    if not (0 <= event_id < n_events):
        return None
    assessment = str(rec.get("assessment") or "").strip()
    if not assessment:
        return None
    primary_layer = str(rec.get("primary_layer") or "none").lower()
    if primary_layer not in {"kg", "osm", "both", "none"}:
        primary_layer = "none"
    confidence = str(rec.get("confidence") or "low").lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"
    cited_factors_raw = rec.get("cited_factors") or []
    if not isinstance(cited_factors_raw, list):
        cited_factors_raw = []
    allowed = _allowed_factor_keys()
    cited_factors = [str(c) for c in cited_factors_raw if str(c) in allowed]
    return {
        "event_id": event_id,
        "assessment": assessment,
        "primary_layer": primary_layer,
        "confidence": confidence,
        "cited_factors": cited_factors,
    }


def infer_environmental_relevance_with_llm(
    events_with_context: List[Dict[str, Any]],
    parsed_osm: Dict[str, Any],
    model_name: str = _HARNESS_DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    """Run the harness LLM to produce a per-event environmental assessment.

    The function returns the input list with each event's
    ``environmental_context.llm_assessment`` populated by an
    LLM-generated record::

        {
            "assessment":     "<one or two sentences in the LLM's own words>",
            "primary_layer":  "kg" | "osm" | "both" | "none",
            "confidence":     "low" | "medium" | "high",
            "cited_factors":  [<factor_key>, ...],
            "model":          <model_name>,
        }

    Validation:
      * cited_factors are restricted to the schema keys present in
        ``factors`` -- the harness cannot invent factor names.
      * Records with malformed JSON, missing event_id, or empty
        assessment are dropped (the field stays ``None``).
      * On total LLM failure the function logs a warning and returns
        the input unchanged.

    The harness adds NO domain-specific rule logic. The LLM reasons
    over the raw factor dict; the code only enforces output shape.
    """
    n_events = len(events_with_context)
    if n_events == 0 or not parsed_osm:
        return events_with_context

    user_payload = _harness_user_payload(events_with_context, parsed_osm)
    resp = invoke_llm(
        model=model_name,
        messages=[
            {"role": "system", "content": _HARNESS_SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ],
        temperature=_HARNESS_TEMPERATURE,
        max_tokens=_HARNESS_MAX_TOKENS,
    )
    text = resp.choices[0].message.content or ""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned).rstrip("`").rstrip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise RuntimeError(
            f"Environmental relevance harness ({model_name}) returned no JSON object. "
            f"Raw output: {text[:200]!r}"
        )
    obj = json.loads(match.group(0))

    records = obj.get("events") or []
    if not isinstance(records, list):
        raise RuntimeError(
            f"Environmental relevance harness ({model_name}) returned 'events' that is "
            f"not a list: {type(records).__name__}"
        )

    n_attached = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        validated = _validate_harness_record(rec, n_events)
        if validated is None:
            continue
        idx = validated.pop("event_id")
        env = events_with_context[idx].get("environmental_context")
        if env is None:
            continue
        env["llm_assessment"] = {**validated, "model": model_name}
        n_attached += 1

    logger.info(
        "Environmental relevance harness: %d/%d events received an LLM assessment (model=%s)",
        n_attached,
        n_events,
        model_name,
    )
    return events_with_context


# ─── Public API ───────────────────────────────────────────────────────────


def infer_anomaly_causes(
    events: List[AnomalyEvent],
    ontology: ThreeGPPOntology,
    forecast: np.ndarray,
    kpi_names: List[str],
    baseline: Optional[np.ndarray] = None,
    window_hours: int = 6,
    osm_context: Optional[str] = None,
    top_k: int = _DEFAULT_TOP_K,
    relevance_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Enrich each event with KG-grounded causes + parsed OSM factors.

    Returns a list of dicts in the same order as :func:`events_to_dicts`,
    each augmented with:

    * ``candidate_causes``: KG-grounded upstream-KPI cause records (may
      be empty). See module docstring for the ranking key.
    * ``environmental_context``: ``None`` when ``osm_context`` is not
      supplied / unparseable, otherwise ``{osm_summary, factors,
      llm_assessment}``. ``llm_assessment`` is populated only when
      ``relevance_model`` is provided and the harness LLM call
      succeeds; otherwise it stays ``None``.

    No telecom-domain rule table is applied. Environmental relevance
    judgement is delegated to the harness LLM (see
    :func:`infer_environmental_relevance_with_llm`).
    """
    enriched = events_to_dicts(events)

    parsed_osm: Optional[Dict[str, Any]] = None
    if osm_context:
        parsed_osm = parse_osm_context(osm_context)
        if parsed_osm is not None:
            avg_h = parsed_osm.get("avg_building_height_m", 0.0) or 0.0
            logger.info(
                "OSM context parsed: %s, building_class=%s, transit=%s, telecom=%s",
                parsed_osm.get("area_type"),
                _building_height_class(avg_h),
                _transit_proximity(parsed_osm.get("transit_hubs", 0) or 0),
                _telecom_infra(parsed_osm.get("telecom_masts", 0) or 0),
            )

    if ontology is None:
        for event, event_dict in zip(events, enriched):
            event_dict["candidate_causes"] = []
            event_dict["environmental_context"] = (
                _build_environmental_context(event, parsed_osm) if parsed_osm is not None else None
            )
        logger.info(
            "Cause inference skipped: ontology unavailable (%d events).",
            len(events),
        )
    else:
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
            event_dict["environmental_context"] = (
                _build_environmental_context(event, parsed_osm) if parsed_osm is not None else None
            )
            if causes:
                n_with_causes += 1

        logger.info(
            "Cause inference: %d/%d events have KG-grounded candidate causes",
            n_with_causes,
            len(events),
        )

    if relevance_model and parsed_osm is not None:
        infer_environmental_relevance_with_llm(
            enriched,
            parsed_osm,
            model_name=relevance_model,
        )

    return enriched
