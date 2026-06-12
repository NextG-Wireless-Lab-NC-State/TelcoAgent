"""Shared ReAct agent infrastructure.

This package is pillar-neutral plumbing used by every LLM agent in the
system; it contains no pipeline logic of its own:

    base      -- :class:`BaseTrainingFreeAgent` (ReAct loop +
                 single-shot ablation arm) and :class:`AgentResult`.
    registry  -- :class:`ToolRegistry` (OpenAI-compatible tool schemas
                 + in-process dispatch) and shared schema helpers.

Consumers:
    - Explanation pipeline (paper Sec. III-C): ``ExplainerAgent`` runs its
      ReAct loop through :class:`BaseTrainingFreeAgent` and registers its
      tools on a :class:`ToolRegistry`.
    - KG-construction pipeline (paper Sec. III-A): the Extractor / Aligner /
      Evaluator agents reuse the same loop and registry.

The prediction pipeline (paper Sec. III-B) is a single TSFM forward pass
and uses none of this package.
"""

from .base import AgentResult, BaseTrainingFreeAgent
from .registry import ToolRegistry

__all__ = [
    "AgentResult",
    "BaseTrainingFreeAgent",
    "ToolRegistry",
]
