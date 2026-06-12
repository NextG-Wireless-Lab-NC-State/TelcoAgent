"""Explainer tool factory — assembles the ReAct tool registry.

:class:`ExplainerToolsConfig` carries every input the seven tools
close over (anomaly events, prediction, KG store, OSM client,
sensitivity tensor); :func:`make_explainer_tools` registers the tools
on a fresh :class:`telcoagent.agents.registry.ToolRegistry`. Each tool
lives in its own sibling module — see the package docstring for the
catalogue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from telcoagent.agents.registry import ToolRegistry

from .anomaly_events import _register_query_anomaly_events
from .kg_mechanism import _register_query_graphrag_mechanism
from .sensitivity import (
    _register_query_localized_sensitivity,
    _register_query_sensitivity_matrix,
)
from .spatial import _register_query_spatial_context
from .temporal import _register_query_forecast_temporal
from .verification import _register_verify_explanation

logger = logging.getLogger(__name__)


@dataclass
class ExplainerToolsConfig:
    """Inputs required to build the Explainer Agent tool registry.

    Attributes
    ----------
    anomaly_events
        Deterministic anomaly events on the forecast horizon (see
        :func:`telcoagent.explainer.anomaly_detector.events_to_dicts`).
    prediction
        Per-KPI forecast as ``{kpi_name: list[float]}`` of length 168.
    ontology_store
        Object exposing ``retrieve(target_kpis=…)`` for KG queries.
    osm
        :class:`telcoagent.context.mcp_tools.OpenStreetMapMCP` instance.
    station_id, kpi_names
        Identifiers used for context-tagging and KPI lookup.
    input_baseline
        Optional ``{kpi_name: list[float]}`` containing the last 168
        hours of the input window for baseline comparison.
    forecast_start_weekday
        Weekday index (0 = Monday) of forecast hour 0; used to attach
        weekday labels to per-day forecast summaries.
    sensitivity_result
        Optional :class:`telcoagent.explainer.pax_ts.SensitivityResult`
        produced by :func:`compute_sensitivity` upstream. When None,
        ``query_sensitivity_matrix`` and ``query_localized_sensitivity``
        return an explicit ``{"error": ...}`` payload (no synthetic
        fallback).
    """

    anomaly_events: List[Dict[str, Any]]
    prediction: Optional[Dict[str, List[float]]]
    ontology_store: Any
    osm: Any
    station_id: str = ""
    kpi_names: List[str] = field(default_factory=list)
    input_baseline: Optional[Dict[str, List[float]]] = None
    forecast_start_weekday: int = 0
    sensitivity_result: Optional[Any] = None
    enabled_tools: Optional[List[str]] = None


def make_explainer_tools(cfg: ExplainerToolsConfig) -> tuple:
    """Build a :class:`ToolRegistry` populated with the explainer tools.

    When ``cfg.enabled_tools`` is ``None`` (the default) all seven tools
    are registered, preserving the existing behaviour.  Pass an explicit
    list of tool names to restrict the registry to a subset — useful for
    ablation studies that want to measure the contribution of individual
    tools (e.g. ``verify_explanation`` on/off).

    Returns
    -------
    tuple
        ``(registry, state)`` where ``state["retrieved_contexts"]``
        accumulates tool-output snippets used by RAGAS NLI evaluation.
    """
    state: Dict[str, Any] = {"retrieved_contexts": []}
    ctx = state["retrieved_contexts"]
    registry = ToolRegistry()

    kpi_names = list(cfg.kpi_names)
    n_channels = len(kpi_names)

    # ``enabled_tools=None`` means "all tools" for backward compatibility.
    allowed: Optional[set] = None if cfg.enabled_tools is None else set(cfg.enabled_tools)

    def _allowed(name: str) -> bool:
        return allowed is None or name in allowed

    if _allowed("query_graphrag_mechanism"):
        _register_query_graphrag_mechanism(registry, cfg, ctx, n_channels)
    if _allowed("query_spatial_context"):
        _register_query_spatial_context(registry, cfg, ctx)
    if _allowed("query_forecast_temporal"):
        _register_query_forecast_temporal(registry, cfg, ctx, n_channels)
    if _allowed("query_anomaly_events"):
        _register_query_anomaly_events(registry, cfg, ctx, n_channels)
    if _allowed("query_sensitivity_matrix"):
        _register_query_sensitivity_matrix(registry, cfg, ctx, n_channels)
    if _allowed("query_localized_sensitivity"):
        _register_query_localized_sensitivity(registry, cfg, ctx, n_channels)
    if _allowed("verify_explanation"):
        _register_verify_explanation(registry, cfg, ctx, n_channels)

    return registry, state
