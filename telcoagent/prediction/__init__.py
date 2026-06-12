"""Prediction pipeline (paper Sec. III-B) -- TSFM-only zero-shot forecasting.

Single Chronos-2 forward pass over the raw multivariate input window,
followed by the mandatory physical clamp. No ReAct loop, no LLM
refinement, no KG consumption on the forward path.

    features        -- KPI_NAMES + pure-NumPy input-window utilities
                       (torch-free; safe for lightweight consumers).
    predictor       -- :class:`TelcoAgent`, the predict entry point
                       (lazy-imports torch/chronos at construction).
    physical_clamp  -- ``apply_physical_clamp_strict``: clip to
                       KPI_NORMALIZATION_MAX + integer-KPI rounding,
                       shared with every baseline runner (ADR 0001).
"""

from .features import KPI_NAMES, compute_input_statistics, kpis_to_text
from .physical_clamp import apply_physical_clamp_strict
from .predictor import TelcoAgent

__all__ = [
    "KPI_NAMES",
    "TelcoAgent",
    "apply_physical_clamp_strict",
    "compute_input_statistics",
    "kpis_to_text",
]
