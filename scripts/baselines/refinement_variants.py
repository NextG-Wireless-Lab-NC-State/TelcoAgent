"""Refinement variants — moved out of telcoagent.prediction.predictor (formerly foundation.py) at Stage-2 commit 2.2a.

The paper (TelcoAgent_Globecom.pdf §III.B) describes the predictor as
**pure TSFM (Chronos-2 forward + physical clamp)**; the LLM refinement
step lives ONLY in ablation harnesses. This module hosts the three
refinement variants previously embedded in
the predictor module (now ``telcoagent/prediction/predictor.py``):

    baseline — 1-shot compact-summary prompt + bounded factor adjust.
    v1       — enriched 1-shot prompt (diurnal + CV + cross-KPI verify).
    v2       — 2-step analyze→adjust harness with consistency check.

Plus their shared helpers:

    _parse_adjustments_json — JSON output parser (pre-2.3 collapses
                              4 distinct failure modes into ``({}, "")``;
                              raises on parse failure at commit 2.3).
    _apply_adjustment_gate  — strong-signal gate ``[0.85, 1.15]`` bound.

This module IS NOT imported by the predictor; the architecture guard
``tests/architecture/test_no_core_to_baselines_imports.py`` enforces
that. The baseline runner
``scripts/baselines/run_refinement_ab.py`` imports the three refine
functions here directly.

The refine functions take ``agent`` as a positional first arg so they
can access ``agent._explainer_model``; this preserves the pre-2.2a
behaviour where the runner instantiated a partial ``TelcoAgent`` via
``object.__new__`` and bound ``_explainer_model`` on it.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from telcoagent.config import CORE_KPI_NAMES
from telcoagent.llm.api import invoke_llm
from telcoagent.prediction import TelcoAgent as _TelcoAgent

logger = logging.getLogger(__name__)

# Re-use the canonical 7-KPI order.
KPI_NAMES: List[str] = list(CORE_KPI_NAMES)

# The three context-builder helpers stay class-static on TelcoAgent (they
# have no per-instance state and were @staticmethod on the class pre-2.2a).
# Pre-2.2b they live in both places (duplicated for the pure-move window);
# post-2.2b they are deleted from foundation.py and only this module's
# binding survives.
_summarize_forecast = _TelcoAgent._summarize_forecast
_summarize_history = _TelcoAgent._summarize_history
_build_enriched_context = _TelcoAgent._build_enriched_context


# =============================================================================
# Refinement JSON parser (1-shot LLM output -> bounded adjustment factors)
# =============================================================================


def _parse_adjustments_json(
    text: str,
    kpi_names: List[str],
) -> Tuple[Dict[str, float], str]:
    """Parse the 1-shot LLM JSON output into ``(adjustments, rationale)``.

    Tolerates markdown fences and prose wrapping the JSON. Filters unknown
    KPI names, drops non-finite factors, clips to ``[0.85, 1.15]`` for
    safety, and skips no-op factors so ``refinement_history`` only records
    meaningful changes.
    """
    if not text:
        return {}, ""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = cleaned.rstrip("`").rstrip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(
            f"_parse_adjustments_json: no JSON object found in LLM output "
            f"(first 120 chars: {cleaned[:120]!r})"
        )
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"_parse_adjustments_json: malformed JSON in LLM output: {exc}") from exc
    raw_adj = obj.get("adjustments") or {}
    rationale = str(obj.get("rationale") or "").strip()
    out: Dict[str, float] = {}
    if isinstance(raw_adj, dict):
        for kpi, factor in raw_adj.items():
            if kpi not in kpi_names:
                continue
            try:
                f = float(factor)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(f):
                continue
            f = max(0.85, min(1.15, f))
            if abs(f - 1.0) < 1e-6:
                continue
            out[kpi] = f
    return out, rationale


# =============================================================================
# Adjustment gate (count strong signals; pass-through or zero)
# =============================================================================


def _apply_adjustment_gate(
    adjustments: Dict[str, float],
    s1_analysis: Optional[Dict] = None,
    min_strong: int = 3,
) -> Dict[str, float]:
    """Return empty dict if fewer than min_strong KPIs have a strong signal.

    For v2, uses Step-1 declared confidence + direction.
    For v1, uses factor magnitude (>=5% deviation from 1.0).
    """
    strong = 0
    for kpi, factor in adjustments.items():
        if s1_analysis is not None:
            kpi_meta = s1_analysis.get(kpi, {})
            if kpi_meta.get("confidence") == "high" and kpi_meta.get("direction") in (
                "UP",
                "DOWN",
            ):
                strong += 1
        else:
            if abs(factor - 1.0) >= 0.05:
                strong += 1
    if strong < min_strong:
        logger.info(
            "[gate] %d strong signals < %d required; all factors → 1.0",
            strong,
            min_strong,
        )
        return {}
    return adjustments


# =============================================================================
# Refinement variants — agent passed positionally for _explainer_model access
# =============================================================================


def _agentic_refine(
    agent: Any,
    raw_pred: np.ndarray,
    input_kpis: np.ndarray,
    osm_text: str,
    kg_text: str,
    station_id: str,
    mode: str = "baseline",
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Dispatch to refinement implementation selected by *mode*.

    Modes:
        "baseline" — original 1-shot call with compact summary context.
        "v1"       — enriched 1-shot prompt (diurnal + CV + cross-KPI
                     self-verification instruction).
        "v2"       — 2-step harness: analyze → adjust with explicit
                     cross-KPI consistency check between steps.
    """
    if mode == "v1":
        return _agentic_refine_v1(agent, raw_pred, input_kpis, osm_text, kg_text, station_id)
    if mode == "v2":
        return _agentic_refine_v2(agent, raw_pred, input_kpis, osm_text, kg_text, station_id)
    # baseline (original behaviour)
    forecast_summary = _summarize_forecast(raw_pred, list(KPI_NAMES))
    history_summary = _summarize_history(input_kpis, list(KPI_NAMES))

    kpi_list_str = ", ".join(KPI_NAMES)
    system_prompt = (
        "You are a 5G O-RAN traffic forecast refinement expert. "
        "You will see a raw 168-hour forecast from a time-series foundation model "
        "(Chronos-2), a recent input-history summary, the cell's spatial context from "
        "OpenStreetMap, and a 3GPP knowledge graph of KPI definitions and causal chains.\n\n"
        "Propose SMALL bounded multiplicative adjustment factors per KPI based ONLY on:\n"
        "  (a) cross-KPI causal consistency from the KG (e.g., Throughput is bounded by "
        "MAC_DL_Eff and degrades when DL_iBler is high),\n"
        "  (b) the cell's environment (urban vs suburban density, transit hubs, telecom "
        "infrastructure) which affects realistic load and throughput.\n\n"
        'STRICT JSON OUTPUT: {"adjustments": {"KPI_NAME": factor_float, ...}, '
        '"rationale": "<one short paragraph>"}.\n'
        f"Allowed KPIs: {kpi_list_str}.\n"
        "Each factor must lie in [0.85, 1.15] (max +/-15% per KPI). "
        "Default to 1.0 (no change) for any KPI you cannot justify from KG or OSM context. "
        "Never invent KPI names. Output JSON only -- no prose outside the JSON."
    )
    user_prompt = (
        f"Station: {station_id}\n\n"
        f"=== Raw 168h forecast (Chronos-2) ===\n{forecast_summary}\n\n"
        f"=== Input history summary (last 168h) ===\n{history_summary}\n\n"
        f"=== Spatial context (OSM, 500m radius) ===\n"
        f"{osm_text or '[OSM context unavailable]'}\n\n"
        f"=== 3GPP KPI knowledge graph ===\n{kg_text}\n\n"
        "Return JSON now."
    )

    if not getattr(agent, "_explainer_model", None):
        raise RuntimeError(
            "agent._explainer_model is not set; pass explainer_model= "
            "to the agent constructor before calling _agentic_refine()."
        )
    model_name = agent._explainer_model

    history: List[Dict[str, Any]] = []
    resp = invoke_llm(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    text = resp.choices[0].message.content or ""

    adjustments, rationale = _parse_adjustments_json(text, list(KPI_NAMES))
    if not adjustments:
        logger.info("Refinement LLM returned no usable adjustments; shipping raw forecast")
        return raw_pred, history

    refined = raw_pred.copy()
    for kpi, factor in adjustments.items():
        if kpi not in KPI_NAMES:
            continue
        idx = KPI_NAMES.index(kpi)
        if idx < refined.shape[1]:
            refined[:, idx] = refined[:, idx] * float(factor)

    history.append(
        {
            "step": 1,
            "model": model_name,
            "adjustments": adjustments,
            "rationale": rationale,
        }
    )
    logger.info(
        "Refinement applied: %d KPIs adjusted (%s)",
        len(adjustments),
        ", ".join(f"{k}={v:.3f}" for k, v in adjustments.items()),
    )
    return refined, history


def _agentic_refine_v1(
    agent: Any,
    raw_pred: np.ndarray,
    input_kpis: np.ndarray,
    osm_text: str,
    kg_text: str,
    station_id: str,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Version 1 — enriched 1-shot prompt.

    Extends the baseline with:
    - Diurnal fingerprint + CV uncertainty per KPI
    - 7d daily averages from both input and forecast
    - Explicit cross-KPI self-verification step in the system prompt
    - max_tokens raised to 1024
    """
    enriched = _build_enriched_context(raw_pred, input_kpis)
    history_summary = _summarize_history(input_kpis, list(KPI_NAMES))
    kpi_list_str = ", ".join(KPI_NAMES)

    system_prompt = (
        "You are a 5G O-RAN traffic forecast refinement expert. "
        "You receive a raw 168-hour Chronos-2 forecast, enriched input history "
        "(diurnal fingerprint, coefficient of variation, daily averages, drift ratio), "
        "OpenStreetMap spatial context, and the 3GPP KPI knowledge graph.\n\n"
        "IMPORTANT PRIOR: The Chronos-2 forecaster you are correcting already achieves "
        "~27% sMAPE on its own. Most KPIs do not need adjustment. Default every factor "
        "to 1.000 unless the enriched context (drift ratio, diurnal fingerprint, CV, "
        "OSM, KG) provides a specific named reason the raw forecast is biased. "
        "A well-calibrated answer for a typical station adjusts at most 2–3 KPIs off 1.0.\n\n"
        "Task:\n"
        "1. Examine per-KPI uncertainty (CV) and diurnal pattern deviations.\n"
        "2. Use the KG causal chains to identify cross-KPI consistency constraints "
        "(e.g. rising DL_iBler should suppress MAC_DL_Eff and Throughput).\n"
        "3. Use OSM environment to assess whether forecast load levels are plausible.\n"
        "4. Propose bounded multiplicative adjustment factors per KPI in [0.85, 1.15]. "
        "Use factors outside [0.95, 1.05] only when the drift ratio shows >10% deviation "
        "AND a KG causal chain or OSM factor provides a named explanation.\n"
        "5. SELF-VERIFY: after proposing adjustments, check that every factor pair "
        "that shares a KG causal chain points in the correct direction. "
        "If an inconsistency is found, revise the offending factor before outputting.\n\n"
        "STRICT JSON OUTPUT (no prose outside the JSON):\n"
        '{"adjustments": {"KPI_NAME": factor_float, ...}, '
        '"rationale": "<one paragraph>", '
        '"consistency_check": "<one sentence confirming cross-KPI directions>"}\n'
        f"Allowed KPIs: {kpi_list_str}. "
        "Default to 1.0 for any KPI you cannot justify. "
        "Output JSON only."
    )
    user_prompt = (
        f"Station: {station_id}\n\n"
        f"{enriched}\n\n"
        f"=== Input history summary (last 168h stats) ===\n{history_summary}\n\n"
        f"=== Spatial context (OSM, 500m radius) ===\n"
        f"{osm_text or '[OSM context unavailable]'}\n\n"
        f"=== 3GPP KPI knowledge graph ===\n{kg_text}\n\n"
        "Apply steps 1-5 and return JSON now."
    )

    if not getattr(agent, "_explainer_model", None):
        raise RuntimeError("agent._explainer_model is not set.")
    resp = invoke_llm(
        model=agent._explainer_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    text = resp.choices[0].message.content or ""
    adjustments, rationale = _parse_adjustments_json(text, list(KPI_NAMES))
    adjustments = _apply_adjustment_gate(adjustments)
    if not adjustments:
        logger.info("[v1] No usable adjustments; shipping raw forecast")
        return raw_pred, []

    refined = raw_pred.copy()
    for kpi, factor in adjustments.items():
        idx = KPI_NAMES.index(kpi)
        if idx < refined.shape[1]:
            refined[:, idx] *= float(factor)

    history = [
        {
            "step": 1,
            "mode": "v1",
            "model": agent._explainer_model,
            "adjustments": adjustments,
            "rationale": rationale,
        }
    ]
    logger.info(
        "[v1] %d KPIs adjusted: %s",
        len(adjustments),
        ", ".join(f"{k}={v:.3f}" for k, v in adjustments.items()),
    )
    return refined, history


def _agentic_refine_v2(
    agent: Any,
    raw_pred: np.ndarray,
    input_kpis: np.ndarray,
    osm_text: str,
    kg_text: str,
    station_id: str,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Version 2 — 2-step harness: analyze → adjust.

    Step 1 (Analyze): LLM produces structured per-KPI analysis
        {KPI: {direction, magnitude, confidence, reason}}.
    Step 2 (Adjust): LLM receives Step-1 analysis and outputs final
        adjustment factors after verifying cross-KPI consistency.
    """
    if not getattr(agent, "_explainer_model", None):
        raise RuntimeError("agent._explainer_model is not set.")
    model = agent._explainer_model
    enriched = _build_enriched_context(raw_pred, input_kpis)
    kpi_list_str = ", ".join(KPI_NAMES)

    # ── Step 1: per-KPI directional analysis ──────────────────────
    s1_system = (
        "You are a 5G O-RAN KPI forecast analyst. "
        "Analyze whether each of the 7 KPIs in the raw Chronos-2 forecast "
        "appears over- or under-estimated relative to the input history, "
        "diurnal pattern, and spatial context.\n\n"
        "For each KPI output a JSON analysis:\n"
        "  direction: 'UP' | 'DOWN' | 'NONE'  (should forecast be corrected upward/downward?)\n"
        "  magnitude: 'small' (<5%) | 'medium' (5-15%) | 'large' (>15%, only for noisy KPIs)\n"
        "  confidence: 'high' | 'medium' | 'low'\n"
        "  reason: one sentence grounded in diurnal deviation, CV, OSM, or KG.\n\n"
        'Output ONLY valid JSON: {"per_kpi": {"KPI": {"direction": ..., '
        '"magnitude": ..., "confidence": ..., "reason": ...}, ...}}'
    )
    s1_user = (
        f"Station: {station_id}\n\n"
        f"{enriched}\n\n"
        f"=== Spatial context (OSM) ===\n{osm_text or '[unavailable]'}\n\n"
        f"=== 3GPP KG ===\n{kg_text}\n\n"
        f"Analyze all 7 KPIs ({kpi_list_str}) and return JSON."
    )
    s1_resp = invoke_llm(
        model=model,
        messages=[
            {"role": "system", "content": s1_system},
            {"role": "user", "content": s1_user},
        ],
        temperature=0.1,
        max_tokens=768,
    )
    s1_text = s1_resp.choices[0].message.content or ""
    s1_analysis: Dict = {}
    m = re.search(r"\{.*\}", s1_text, re.DOTALL)
    if m:
        try:
            s1_analysis = json.loads(m.group(0)).get("per_kpi", {})
        except json.JSONDecodeError:
            pass
    if not s1_analysis:
        logger.warning("[v2] Step-1 parse failed; raw_text[:200]=%s", s1_text[:200])
        return raw_pred, [
            {
                "step": 1,
                "mode": "v2_analyze",
                "model": model,
                "analysis": {},
                "error": "step1_parse_failed",
            }
        ]

    # ── Step 2: translate analysis → bounded factors ──────────────
    s2_system = (
        "You are a 5G O-RAN forecast correction engine. "
        "Given a structured per-KPI analysis from Step 1, translate each "
        "direction+magnitude into a multiplicative adjustment factor in [0.85, 1.15].\n\n"
        "IMPORTANT PRIOR: The Chronos-2 forecaster you are correcting already achieves "
        "~27% sMAPE on its own. Most KPIs do not need adjustment. Default every factor "
        "to 1.000 unless the enriched context (drift ratio, diurnal fingerprint, CV, "
        "OSM, KG) provides a specific named reason the raw forecast is biased. "
        "A well-calibrated answer for a typical station adjusts at most 2–3 KPIs off 1.0.\n\n"
        "Rules:\n"
        "- UP direction: factor > 1.0, bounded to [1.0, 1.15]\n"
        "- DOWN direction: factor < 1.0, bounded to [0.85, 1.0]\n"
        "- NONE direction: factor = 1.0\n"
        "Apply a non-trivial factor only when Step-1 confidence is `high`. "
        "For `medium` confidence, keep the factor within 5% of 1.0. "
        "For `low` confidence, force the factor to 1.0.\n\n"
        "CROSS-KPI CONSISTENCY: apply KG causal directions from the analysis "
        "reasons to ensure no directional contradiction between causally linked KPIs "
        "(e.g. if DL_iBler goes UP, MAC_DL_Eff must not go UP).\n\n"
        'Output ONLY JSON: {"adjustments": {"KPI": factor, ...}, '
        '"rationale": "<one paragraph>", '
        '"consistency_check": "<one sentence>"}'
    )
    s2_user = (
        f"Step-1 per-KPI analysis:\n{json.dumps({'per_kpi': s1_analysis}, indent=2)}\n\n"
        f"=== 3GPP KG (causal chains for consistency check) ===\n{kg_text}\n\n"
        f"Allowed KPIs: {kpi_list_str}. Return JSON."
    )
    s2_resp = invoke_llm(
        model=model,
        messages=[
            {"role": "system", "content": s2_system},
            {"role": "user", "content": s2_user},
        ],
        temperature=0.1,
        max_tokens=512,
    )
    s2_text = s2_resp.choices[0].message.content or ""
    adjustments, rationale = _parse_adjustments_json(s2_text, list(KPI_NAMES))
    adjustments = _apply_adjustment_gate(adjustments, s1_analysis=s1_analysis)
    if not adjustments:
        logger.info("[v2] Step-2 returned no usable adjustments; shipping raw forecast")
        return raw_pred, []

    refined = raw_pred.copy()
    for kpi, factor in adjustments.items():
        idx = KPI_NAMES.index(kpi)
        if idx < refined.shape[1]:
            refined[:, idx] *= float(factor)

    history = [
        {"step": 1, "mode": "v2_analyze", "model": model, "analysis": s1_analysis},
        {
            "step": 2,
            "mode": "v2_adjust",
            "model": model,
            "adjustments": adjustments,
            "rationale": rationale,
        },
    ]
    logger.info(
        "[v2] %d KPIs adjusted: %s",
        len(adjustments),
        ", ".join(f"{k}={v:.3f}" for k, v in adjustments.items()),
    )
    return refined, history
