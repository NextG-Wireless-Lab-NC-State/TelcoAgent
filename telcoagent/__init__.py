"""TelcoAgent -- TSFM (Chronos-2) zero-shot prediction with 3GPP-KG- and
OSM-grounded explainability (PAX-TS perturbation attribution).

Three pipelines, mirroring the paper (see docs/module_map.md):
    kg_construction/  Sec. III-A  offline 3-agent KG build from 3GPP PDFs
    prediction/       Sec. III-B  TSFM-only single-pass forecasting
    explainer/        Sec. III-C  ReAct explanation + sensitivity attribution

PAX-TS (Kreuzer et al., arXiv 2508.18982) computes a cross-channel
sensitivity tensor by perturbing the recent input window's mean,
variance, trend, and per-event localized amplitude, then re-querying
the Chronos-2 forecast in a single batched call. The resulting
``S`` tensor is surfaced to the ReAct explainer through the
``[Sensitivity]`` and ``[Sensitivity-Local]`` evidence channels.

Import policy
-------------
Each pillar's public API is re-exported below, but wrapped in
``try/except ImportError`` so that lightweight environments (CI test
slices, KG-only consumers, docs build) can ``import telcoagent`` and
reach the pillars they actually need without dragging the entire
heavy-ML dependency closure (chronos, torch, gluonts, etc.) into the
import graph.

This is **not** a fallback path in the engineering-principle #1 sense
— there is no alternative runtime code that fires when an import
fails. The pillar is simply unavailable from the top-level namespace,
and a direct ``from telcoagent.X import Y`` still raises the original
``ImportError`` with the missing dep named.
"""

# Always-available primitives (numpy + json + stdlib only).
from .config import KPI_NORMALIZATION_MAX
from .ontology.core import (
    CausalChain,
    DirectionalRule,
    KPIDefinition,
    KPIRelationship,
    ThreeGPPOntology,
    get_default_ontology,
)

# Pillar re-exports — guarded so a missing transitive dep (matplotlib,
# litellm, markdown, sentence_transformers, ...) does not poison
# ``import telcoagent`` for callers that never reach that pillar.
try:
    from .context.mcp_tools import (
        OpenStreetMapMCP,
        StandardKnowledgeGraphMCP,
    )
except ImportError:
    pass

try:
    from .explainer import (
        AnomalyEvent,
        ExplainerAgent,
        ExplanationResult,
        TelcoAgentExplainer,
        detect_anomaly_events,
        events_to_dicts,
        plot_forecast_with_anomalies,
        render_html_report,
    )
except ImportError:
    pass

try:
    from .agents import (
        AgentResult,
        BaseTrainingFreeAgent,
        ToolRegistry,
    )
except ImportError:
    pass

try:
    from .prediction import (
        KPI_NAMES,
        TelcoAgent,
        compute_input_statistics,
        kpis_to_text,
    )
except ImportError:
    pass

try:
    from .llm.config import (
        AGENT_MODELS,
        FALLBACK_CHAINS,
        ROLE_CONFIGS,
        EnhancedLLMConfig,
        LLMCallMetrics,
        SessionMetrics,
        TokenUsage,
    )
    from .llm.prompts import EXPLAINER_SYSTEM_PROMPT
except ImportError:
    pass

try:
    from .stores import (
        Neo4jConnection,
        Neo4jOntologyStore,
    )
except ImportError:
    pass

try:
    from .explainer.tools import make_explainer_tools
except ImportError:
    pass

__all__ = [
    # Main Model
    "TelcoAgent",
    # Agents
    "ExplainerAgent",
    "BaseTrainingFreeAgent",
    # Utilities
    "AgentResult",
    "kpis_to_text",
    "compute_input_statistics",
    "KPI_NAMES",
    "KPI_NORMALIZATION_MAX",
    # MCP Tools (KG-driven retrieval + external runtime tools only)
    "StandardKnowledgeGraphMCP",
    "OpenStreetMapMCP",
    # LLM Configuration
    "EnhancedLLMConfig",
    "AGENT_MODELS",
    "ROLE_CONFIGS",
    "FALLBACK_CHAINS",
    "TokenUsage",
    "LLMCallMetrics",
    "SessionMetrics",
    # Prompt Templates
    "EXPLAINER_SYSTEM_PROMPT",
    # Explainer tools
    "ToolRegistry",
    "make_explainer_tools",
    # 3GPP Ontology (KG-driven; no in-Python KPI registry / no validators)
    "ThreeGPPOntology",
    "KPIDefinition",
    "KPIRelationship",
    "CausalChain",
    "DirectionalRule",
    "get_default_ontology",
    # Neo4j stores
    "Neo4jConnection",
    "Neo4jOntologyStore",
    # Explainability
    "TelcoAgentExplainer",
    "ExplanationResult",
    # Anomaly detection primitives
    "AnomalyEvent",
    "detect_anomaly_events",
    "events_to_dicts",
    # Visualisation + HTML rendering
    "plot_forecast_with_anomalies",
    "render_html_report",
]
