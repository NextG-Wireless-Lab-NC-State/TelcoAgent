"""ToolRegistry + shared helpers for ReAct tool factories.

OpenAI-compatible tool schemas and an in-process dispatch registry.
Used by the Explainer tool factory (``make_explainer_tools``, which
registers the Explainer Agent's seven tools: KG mechanism, spatial
context, forecast temporal, anomaly events, PAX-TS global / per-event
sensitivity, and verify_explanation) and by the KG-construction agent
tool factories.

The :func:`_classify_sensitivity_magnitude` helper is the canonical
post-processor that buckets ``|S|`` into negligible / weak /
meaningful / strong tiers used by the PAX-TS sensitivity tools and
the audit module.
"""

import logging
from typing import Any, Callable, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# JSON Serialization Utility
# =============================================================================


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types and other non-JSON-serializable objects."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    return str(obj)


def _classify_sensitivity_magnitude(value: float) -> dict:
    """Classify sensitivity magnitude per field engineer thresholds.

    Thresholds (dimensionless elasticity):
      < 0.001  — negligible, not usable as causal evidence
      0.001–0.01 — weak but measurable, use with qualifier
      0.01–1.0 — meaningful, usable as primary evidence
      >= 1.0   — strong
    """
    abs_val = abs(value)
    if abs_val < 0.001:
        return {"magnitude": "negligible", "label": f"negligible ({value:.4f} < 0.001)"}
    elif abs_val < 0.01:
        return {"magnitude": "weak", "label": f"weak but measurable ({value:.4f})"}
    elif abs_val < 1.0:
        return {"magnitude": "meaningful", "label": f"meaningful ({value:.4f})"}
    else:
        return {"magnitude": "strong", "label": f"strong ({value:.4f})"}


# =============================================================================
# Tool Registry
# =============================================================================


class ToolRegistry:
    """Registry that stores tool functions and their OpenAI-compatible schemas."""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: List[Dict] = []

    def register(self, name: str, func: Callable, schema: Dict) -> None:
        self._tools[name] = func
        self._schemas.append(schema)

    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self._tools:
            return {
                "error": f"Unknown tool: {name}",
                "hint": f"Available tools: {list(self._tools.keys())}",
            }
        try:
            result = self._tools[name](**arguments)
            return _make_serializable(result)
        except TypeError as exc:
            logger.warning(f"Tool '{name}' argument error: {exc}")
            import inspect

            sig = str(inspect.signature(self._tools[name]))
            return {"error": f"Argument error: {exc}", "hint": f"Expected signature: {name}{sig}"}
        except (KeyError, IndexError) as exc:
            logger.warning(f"Tool '{name}' data access error: {exc}")
            return {"error": f"Data access error: {exc}", "hint": "Check that kpi_name is valid"}
        except Exception as exc:
            logger.warning(f"Tool '{name}' execution failed: {exc}")
            return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    def get_schemas(self) -> List[Dict]:
        return list(self._schemas)

    @property
    def tool_names(self) -> List[str]:
        return list(self._tools.keys())


# =============================================================================
# Schema builder
# =============================================================================


def _schema(name: str, description: str, parameters: Dict) -> Dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


# =============================================================================
# Context capture (for RAGAS NLI evaluation)
# =============================================================================


def _capture_context(ctx_list: List[str], text: str) -> None:
    """Append a context string to the list (for RAGAS NLI evaluation)."""
    if text:
        ctx_list.append(text)
