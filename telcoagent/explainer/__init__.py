"""ExplainerAgent and supporting modules: anomaly detection, cause
inference, PAX-TS sensitivity analysis, audit, and HTML report.

Cross-channel sensitivity is computed by
:func:`telcoagent.explainer.pax_ts.compute_sensitivity` and surfaced
to the ReAct agent as the ``[Sensitivity]`` and
``[Sensitivity-Local]`` evidence channels. Sensitivity computation
is optional at explain time — when ``sensitivity_pipeline=None`` is
passed to :class:`TelcoAgentExplainer`, the report omits sensitivity
citations and the per-station sensitivity tensor can be populated
post-hoc via the offline batch script for paper-grade evaluation.

A prior Transfer Entropy estimator was removed because the 168h
forecast horizon was too short for stable surrogate-test p-values;
the directed-causality role is now filled by PAX-TS perturbation +
KG-derived directional rules.
"""

from .anomaly_detector import (
    AnomalyEvent,
    detect_anomaly_events,
    events_to_dicts,
)
from .anomaly_plot import plot_forecast_with_anomalies
from .cause_inference import (
    infer_anomaly_causes,
    infer_environmental_relevance_with_llm,
    parse_osm_context,
)
from .html_report import render_html_report
from .orchestrator import ExplanationResult, TelcoAgentExplainer
from .react_agent import ExplainerAgent

__all__ = [
    "TelcoAgentExplainer",
    "ExplainerAgent",
    "ExplanationResult",
    "AnomalyEvent",
    "detect_anomaly_events",
    "events_to_dicts",
    "infer_anomaly_causes",
    "infer_environmental_relevance_with_llm",
    "parse_osm_context",
    "plot_forecast_with_anomalies",
    "render_html_report",
]
