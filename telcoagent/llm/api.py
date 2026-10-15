"""Unified LiteLLM invocation wrapper for TelcoAgent.

Single-entry helper that centralises:
    - VLLM / local-API-base routing (``USE_LOCAL_API_BASE``)
    - ``drop_params=True`` default so provider-specific args are silently
      dropped when unsupported
    - Reasoning-model param swap: o1/o3/o4 variants use
      ``max_completion_tokens`` and reject ``temperature``

All three previous call styles (direct ``litellm.completion()``,
``foundation._completion_params()``, and ``EnhancedLLMConfig.call()``)
now delegate here for the raw provider call.

The higher-level ``EnhancedLLMConfig`` keeps handling fallback chains,
retry logic, and token tracking — this wrapper is only the raw call.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from litellm import completion

from telcoagent.llm.config import USE_LOCAL_API_BASE, VLLM_BASE_URL, VLLM_MODEL

logger = logging.getLogger(__name__)

# Models that require `max_completion_tokens` instead of `max_tokens`,
# and reject `temperature`. Pattern matches openai's o1/o3/o4 families
# and anthropic's opus reasoning variants.
_REASONING_MODEL_PATTERN = re.compile(
    r"(?:^|/)o[134](?:-|$)|o[134]-mini|reasoning",
    re.IGNORECASE,
)


def _is_reasoning_model(model: str) -> bool:
    return bool(_REASONING_MODEL_PATTERN.search(model or ""))


def _prepare_params(model: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise generation params for reasoning vs. standard models.

    - Reasoning models (o1/o3/o4*): swap ``max_tokens`` → ``max_completion_tokens``,
      strip ``temperature``.
    - Standard models: keep ``max_tokens`` and ``temperature`` as-is.
    """
    out = dict(params)
    if _is_reasoning_model(model):
        if "max_tokens" in out and "max_completion_tokens" not in out:
            out["max_completion_tokens"] = out.pop("max_tokens")
        out.pop("temperature", None)
    return out


def invoke_llm(
    model: str,
    messages: List[Dict[str, str]],
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> Any:
    """Raw LiteLLM completion call with VLLM + reasoning-model plumbing.

    Args:
        model: LiteLLM model string (e.g. ``"openai/gpt-4o"``).
        messages: Chat messages list.
        tools: Optional tool schemas (ReAct loops pass these).
        tool_choice: Optional tool_choice (e.g. ``"auto"``).
        temperature: Standard-model sampling temperature.
        max_tokens: Standard-model output cap; auto-renamed for reasoning models.
        **kwargs: Passed through to ``litellm.completion`` verbatim.

    Returns:
        The raw LiteLLM response object. Callers that just want the text
        should use ``response.choices[0].message.content``.
    """
    call_params: Dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    call_params = _prepare_params(model, call_params)

    call_params.update(kwargs)
    call_params.setdefault("drop_params", True)

    if tools is not None:
        call_params["tools"] = tools
    if tool_choice is not None:
        call_params["tool_choice"] = tool_choice

    if USE_LOCAL_API_BASE and VLLM_MODEL in model:
        call_params.setdefault("api_base", VLLM_BASE_URL)
        call_params.setdefault("api_key", "none")

    return completion(model=model, messages=messages, **call_params)


def invoke_llm_text(
    model: str,
    messages: List[Dict[str, str]],
    **kwargs: Any,
) -> str:
    """Convenience wrapper returning ``choices[0].message.content`` only.

    Use for single-turn requests where tool calls and metadata are not needed.
    """
    response = invoke_llm(model=model, messages=messages, **kwargs)
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError) as exc:
        logger.warning("invoke_llm_text: empty response (%s)", exc)
        return ""
