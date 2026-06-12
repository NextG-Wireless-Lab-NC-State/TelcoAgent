"""``verify_explanation`` — claim cross-check before the report ships.

Validates a natural-language claim against the 3GPP directional rules,
the per-event KG candidate causes, anomaly-KPI coverage, and OSM
grounding. This is the self-verification step of the explanation
pipeline: every causal sentence the agent wants to assert can be
checked here for spec consistency instead of being trusted as-is.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from telcoagent.agents.registry import ToolRegistry, _capture_context, _schema

if TYPE_CHECKING:
    from .factory import ExplainerToolsConfig

logger = logging.getLogger(__name__)


_ARROW_PATTERN = re.compile(r"(\w+)(↑|↓)")
_CAUSAL_PAIR_PATTERN = re.compile(r"(\w+)\s*(?:->|→|=>|⇒|=)\s*(\w+)")

#: Keywords that, when found in a verification claim, indicate the claim
#: is appealing to the OSM spatial / environmental context. The set is
#: deliberately broad so common spatial language ("urban", "low-rise",
#: "POI", "mast", "transit hub") all trigger the OSM grounding check.
_OSM_KEYWORDS: tuple = (
    "transit",
    "residential",
    "urban",
    "suburban",
    "dense",
    "low-rise",
    "low rise",
    "high-rise",
    "high rise",
    "mast",
    "poi",
    "building",
)


def _claim_cites_osm(claim: str) -> bool:
    """Return True if the claim contains any OSM-related keyword."""
    if not claim:
        return False
    lowered = claim.lower()
    return any(keyword in lowered for keyword in _OSM_KEYWORDS)


_OSM_FACTOR_KEYS_FOR_GROUNDING = (
    "area_type",
    "poi_density_band",
    "land_use",
    "building_height_class",
    "transit_proximity",
    "telecom_infra",
    "environment_hint",
)


def _osm_grounding_present(anomaly_events: List[Dict[str, Any]]) -> bool:
    """True if at least one event carries a parsed environmental_context.

    Verification is generic: we only check whether OSM factors were
    successfully attached to any event. The caller's claim itself is
    inspected separately by the per-claim factor scan in
    ``_verify_explanation`` -- here we just confirm OSM evidence is
    available at all.
    """
    for event in anomaly_events:
        env = event.get("environmental_context")
        if not env:
            continue
        factors = env.get("factors") or {}
        if any(factors.get(k) for k in _OSM_FACTOR_KEYS_FOR_GROUNDING):
            return True
    return False


def _verify_causal_pair_supported(
    claim: str,
    anomaly_events: List[Dict[str, Any]],
    kpi_names: List[str],
) -> Optional[Dict[str, Any]]:
    """Check whether a ``source -> target`` claim is backed by candidate_causes.

    Scans the natural-language claim for a directed KPI pair and looks
    up that pair in any event's ``candidate_causes`` list. Returns
    ``None`` if no directed pair is detected.
    """
    pairs: List[tuple] = []
    for src, tgt in _CAUSAL_PAIR_PATTERN.findall(claim):
        if src in kpi_names and tgt in kpi_names and src != tgt:
            pairs.append((src, tgt))
    if not pairs:
        return None

    matches: List[Dict[str, Any]] = []
    unsupported: List[Dict[str, str]] = []
    for src, tgt in pairs:
        found = False
        for event in anomaly_events:
            if event.get("kpi") != tgt:
                continue
            for cause in event.get("candidate_causes", []) or []:
                if cause.get("source_kpi") == src:
                    matches.append(
                        {
                            "source_kpi": src,
                            "target_kpi": tgt,
                            "relation_type": cause.get("relation_type"),
                            "strength": cause.get("strength"),
                            "spec_reference": cause.get("spec_reference"),
                        }
                    )
                    found = True
                    break
            if found:
                break
        if not found:
            unsupported.append({"source_kpi": src, "target_kpi": tgt})

    return {
        "pairs_detected": [{"source_kpi": s, "target_kpi": t} for s, t in pairs],
        "cause_verified": matches,
        "cause_unsupported": unsupported,
    }


def _register_verify_explanation(
    registry: ToolRegistry,
    cfg: "ExplainerToolsConfig",
    ctx: List[str],
    n_channels: int,
) -> None:
    kpi_names = cfg.kpi_names

    def verify_explanation(
        claim: str,
        report_kpis: Optional[List[str]] = None,
        environment_type: str = "",
    ) -> Dict:
        """Cross-reference a natural-language claim against the KG."""
        verification: Dict[str, Any] = {"claim": claim, "checks": []}
        mentioned = [kpi for kpi in kpi_names if kpi in claim]

        cause_check = _verify_causal_pair_supported(
            claim,
            cfg.anomaly_events,
            kpi_names,
        )
        if cause_check is not None:
            verification["cause_verified"] = cause_check["cause_verified"]
            verification["cause_unsupported"] = cause_check["cause_unsupported"]
            verification["pairs_detected"] = cause_check["pairs_detected"]

        if cfg.ontology_store is not None and mentioned:
            try:
                verification["graphrag_verification"] = cfg.ontology_store.retrieve(
                    target_kpis=mentioned[:4]
                )
            except Exception as exc:
                verification["graphrag_error"] = str(exc)

        directional_issues: List[str] = []
        # Prefer the per-run ontology so the AX-4 q_TH filter applies to
        # the directional-coupling check.  Fall back to the process-wide
        # default only when no ontology_store is attached.
        ontology = getattr(cfg.ontology_store, "ontology", None)
        if ontology is None:
            from telcoagent.ontology.core import get_default_ontology

            ontology = get_default_ontology()
        arrows = _ARROW_PATTERN.findall(claim)
        if len(arrows) >= 2:
            directions = {kpi: arrow for kpi, arrow in arrows if kpi in kpi_names}
            for rule in ontology.directional_rules:
                source, target = rule.source_kpi, rule.target_kpi
                if source not in directions or target not in directions:
                    continue
                same_direction = directions[source] == directions[target]
                violates_positive = rule.coupling == "positive" and not same_direction
                violates_negative = rule.coupling == "negative" and same_direction
                if violates_positive or violates_negative:
                    directional_issues.append(
                        f"{source}{directions[source]} → {target}{directions[target]} "
                        f"violates {rule.coupling} coupling (TS {rule.spec_clause})"
                    )
        verification["directional_issues"] = directional_issues

        anomaly_kpis = {event["kpi"] for event in cfg.anomaly_events}
        if report_kpis is not None and anomaly_kpis:
            missing = sorted(anomaly_kpis - set(report_kpis))
            if missing:
                verification["missing_kpis"] = missing
                verification["coverage_warning"] = (
                    f"§4 is missing narratives for: {', '.join(missing)}. "
                    f"These KPIs have detected anomaly events."
                )

        if environment_type in ("RMa", "InH") and "POI" in claim:
            human_label = "rural" if environment_type == "RMa" else "indoor"
            verification["environment_warning"] = (
                f"Claim mentions POI in {environment_type} environment. POI "
                f"analysis is not meaningful for {human_label} environments."
            )

        # OSM grounding check — when the claim cites an OSM factor
        # (transit/residential/urban/POI/mast/building/etc.), at least
        # one event must carry parsed environmental_context.factors so
        # the claim is anchored in actual OSM data and not invented.
        if _claim_cites_osm(claim):
            if _osm_grounding_present(cfg.anomaly_events):
                verification["osm_grounding_present"] = True
            else:
                verification["osm_grounding_missing"] = (
                    "Claim cites an OSM factor (transit / residential / "
                    "urban / POI / mast / building) but no anomaly event "
                    "carries any parsed environmental_context.factors to "
                    "support it."
                )

        verification["mentioned_kpis"] = mentioned
        verification["supported"] = (
            "graphrag_verification" in verification
            or bool(verification["directional_issues"]) is False
            and bool(mentioned)
        )

        if "graphrag_verification" in verification:
            payload = verification["graphrag_verification"]
            if isinstance(payload, str):
                _capture_context(ctx, f"Verification GraphRAG: {payload}")
            elif isinstance(payload, list):
                for entry in payload:
                    _capture_context(ctx, str(entry))

        return verification

    registry.register(
        "verify_explanation",
        verify_explanation,
        _schema(
            "verify_explanation",
            "Cross-reference a natural-language claim against the 3GPP "
            "directional rules and the KG. Checks directional coupling "
            "consistency, anomaly-KPI coverage gaps (when report_kpis is "
            "provided), and POI mentions in rural/indoor environments.",
            {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "Natural-language claim to verify.",
                    },
                    "report_kpis": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "KPIs already covered in §4 narratives.",
                    },
                    "environment_type": {
                        "type": "string",
                        "description": "Environment from query_spatial_context.",
                        "enum": ["UMa", "UMi", "RMa", "InH"],
                    },
                },
                "required": ["claim"],
            },
        ),
    )
