"""Explainer Agent ReAct tools (paper Sec. III-C), one module per tool.

Every KPI-causal claim in the generated report must be grounded in a
tool output rather than free-form generation. The factory wires a
:class:`telcoagent.agents.registry.ToolRegistry` with six tools:

    kg_mechanism    -- ``query_graphrag_mechanism``: 3GPP causal
                       mechanism + directional coupling for a
                       (source, target) KPI pair.
    temporal        -- ``query_forecast_temporal``: per-day forecast
                       breakdown, co-occurrences, weekday patterns.
    anomaly_events  -- ``query_anomaly_events``: deterministic anomaly
                       events ranked by |z|.
    sensitivity     -- ``query_sensitivity_matrix`` (PAX-TS global) and
                       ``query_localized_sensitivity`` (per-event).
    verification    -- ``verify_explanation``: claim cross-check against
                       3GPP directional rules and candidate causes.

``factory.make_explainer_tools(cfg)`` assembles the registry;
``cfg.enabled_tools`` restricts the set for tool-ablation studies.
"""

from .factory import ExplainerToolsConfig, make_explainer_tools

__all__ = [
    "ExplainerToolsConfig",
    "make_explainer_tools",
]
