"""LiteLLM Configuration Manager for TelcoAgent.

Provides:
- Agent-specific model configuration (Network Inference Agent, Explainer Agent)
- Role-based temperature and token settings
- Retry with exponential backoff
- Fallback model chains
- Optional cost/token tracking

LLM backend: selected by LLM_BACKEND env var.
  - "vllm"   (default): OpenAI-compatible endpoint at VLLM_BASE_URL.
  - "openai": OpenAI API (uses OPENAI_API_KEY, model from OPENAI_MODEL).
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from litellm import completion, completion_cost

logger = logging.getLogger(__name__)


# =============================================================================
# Backend selection
# =============================================================================

LLM_BACKEND = os.environ.get("LLM_BACKEND", "vllm").lower()

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "NextGLab/ORANSight_Qwen_14B_Instruct")

if LLM_BACKEND == "openai":
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    ACTIVE_MODEL_STRING = f"openai/{OPENAI_MODEL}"
    USE_LOCAL_API_BASE = False
else:
    ACTIVE_MODEL_STRING = f"openai/{VLLM_MODEL}"
    USE_LOCAL_API_BASE = True

# Per-agent overrides. Set EXPLAINER_MODEL / PREDICTOR_MODEL / KG_BUILD_MODEL
# in the environment to point a single agent at a different backend (e.g.
# keep the predictor on a local vLLM endpoint while sending the Explainer
# Agent to OpenAI gpt-4o-mini for its richer tool-calling reliability).
EXPLAINER_MODEL = os.environ.get("EXPLAINER_MODEL", "openai/gpt-4o-mini")
PREDICTOR_MODEL = os.environ.get("PREDICTOR_MODEL", ACTIVE_MODEL_STRING)
KG_BUILD_MODEL = os.environ.get("KG_BUILD_MODEL", ACTIVE_MODEL_STRING)

AGENT_MODELS: Dict[str, str] = {
    "predictor": PREDICTOR_MODEL,
    "explainer": EXPLAINER_MODEL,
    "kg_build": KG_BUILD_MODEL,
}

ROLE_CONFIGS: Dict[str, Dict] = {
    "predictor": {"temperature": 0.3, "max_tokens": 4096},
    "explainer": {"temperature": 0.3, "max_tokens": 4096},
    "kg_build": {"temperature": 0.2, "max_tokens": 4096},
}

FALLBACK_CHAINS: Dict[str, List[str]] = {
    "predictor": [PREDICTOR_MODEL],
    "explainer": [EXPLAINER_MODEL],
    "kg_build": [KG_BUILD_MODEL],
}


def get_active_agent_models() -> Dict[str, str]:
    return AGENT_MODELS.copy()


def get_active_role_configs() -> Dict[str, Dict]:
    return ROLE_CONFIGS.copy()


def get_active_fallback_chains() -> Dict[str, List[str]]:
    return FALLBACK_CHAINS.copy()


@dataclass
class TokenUsage:
    """Track token usage and costs."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0


@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM call."""

    model: str
    latency_ms: float
    tokens: TokenUsage
    success: bool
    retries: int = 0
    error: Optional[str] = None


@dataclass
class SessionMetrics:
    """Aggregated metrics for a session."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_retries: int = 0
    total_latency_ms: float = 0.0
    tokens: TokenUsage = field(default_factory=TokenUsage)
    calls_by_model: Dict[str, int] = field(default_factory=dict)

    def add_call(self, metrics: LLMCallMetrics):
        """Add a call's metrics to the session."""
        self.total_calls += 1
        if metrics.success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        self.total_retries += metrics.retries
        self.total_latency_ms += metrics.latency_ms

        self.tokens.prompt_tokens += metrics.tokens.prompt_tokens
        self.tokens.completion_tokens += metrics.tokens.completion_tokens
        self.tokens.total_tokens += metrics.tokens.total_tokens
        self.tokens.estimated_cost += metrics.tokens.estimated_cost

        if metrics.model not in self.calls_by_model:
            self.calls_by_model[metrics.model] = 0
        self.calls_by_model[metrics.model] += 1


class EnhancedLLMConfig:
    """
    Enhanced LLM Configuration Manager.

    Features:
    - Agent-specific model selection
    - Role-based parameter configuration
    - Retry with exponential backoff
    - Fallback model chains
    - Cost/token tracking
    """

    def __init__(
        self,
        agent_models: Optional[Dict[str, str]] = None,
        role_configs: Optional[Dict[str, Dict]] = None,
        fallback_chains: Optional[Dict[str, List[str]]] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        track_usage: bool = True,
    ):
        self.agent_models = agent_models or get_active_agent_models()
        self.role_configs = role_configs or get_active_role_configs()
        self.fallback_chains = fallback_chains or get_active_fallback_chains()

        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.track_usage = track_usage

        self.session_metrics = SessionMetrics()

        if LLM_BACKEND == "openai":
            logger.info("EnhancedLLMConfig initialized (OpenAI API)")
        else:
            logger.info("EnhancedLLMConfig initialized (vLLM at %s)", VLLM_BASE_URL)
        logger.info("  Agent models: %s", self.agent_models)
        logger.info("  Max retries: %d, Base delay: %.1fs", max_retries, base_delay)

    def get_model_for_agent(self, agent_type: str) -> str:
        # "explainer" is a required fallback key; absence is a configuration
        # bug, not a runtime condition to silently degrade through.
        result = self.agent_models.get(agent_type) or self.agent_models["explainer"]
        return result

    def get_config_for_role(self, role: str) -> Dict[str, Any]:
        result = self.role_configs.get(role) or self.role_configs["explainer"]
        return result

    def get_fallback_chain(self, agent_type: str) -> List[str]:
        result = self.fallback_chains.get(agent_type) or self.fallback_chains["explainer"]
        return result

    def _exponential_backoff(self, attempt: int) -> float:
        import random

        delay = min(self.base_delay * (2**attempt), self.max_delay)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return delay + jitter

    def call_with_retry(
        self,
        prompt: str,
        agent_type: str = "explainer",
        role: str = "explainer",
        system_message: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ) -> tuple:
        """
        Make an LLM call with retry and fallback support.

        Returns:
            When tools is None: Tuple of (response_text, call_metrics)
            When tools is provided: Tuple of (full_response_object, call_metrics)
        """
        fallback_chain = self.get_fallback_chain(agent_type)
        role_config = self.get_config_for_role(role)

        call_params = {**role_config, **kwargs}

        if tools:
            call_params["tools"] = tools
            call_params["tool_choice"] = tool_choice

        model = fallback_chain[0]
        start_time = time.time()
        total_retries = 0
        last_error = None

        for attempt in range(self.max_retries):
            try:
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})

                completion_kwargs = {
                    "model": model,
                    "messages": messages,
                    "drop_params": True,
                    **call_params,
                }
                if USE_LOCAL_API_BASE:
                    completion_kwargs["api_base"] = VLLM_BASE_URL
                    completion_kwargs["api_key"] = "none"
                response = completion(**completion_kwargs)

                latency_ms = (time.time() - start_time) * 1000

                usage = response.usage if hasattr(response, "usage") else None
                tokens = TokenUsage(
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                )

                if self.track_usage:
                    content = response.choices[0].message.content or ""
                    tokens.estimated_cost = completion_cost(
                        model=model, prompt=prompt, completion=content
                    )

                metrics = LLMCallMetrics(
                    model=model,
                    latency_ms=latency_ms,
                    tokens=tokens,
                    success=True,
                    retries=total_retries,
                )

                if self.track_usage:
                    self.session_metrics.add_call(metrics)

                if tools:
                    return response, metrics
                return response.choices[0].message.content, metrics

            except Exception as e:
                total_retries += 1
                last_error = str(e)
                logger.warning("LLM call failed (model=%s, attempt=%d): %s", model, attempt + 1, e)

                if attempt < self.max_retries - 1:
                    retry_match = re.search(r"retry in (\d+\.?\d*)s", str(e), re.IGNORECASE)
                    if retry_match:
                        delay = float(retry_match.group(1)) + 1.0
                    else:
                        delay = self._exponential_backoff(attempt)
                    logger.info("Retrying in %.2fs...", delay)
                    time.sleep(delay)

        latency_ms = (time.time() - start_time) * 1000
        metrics = LLMCallMetrics(
            model=model,
            latency_ms=latency_ms,
            tokens=TokenUsage(),
            success=False,
            retries=total_retries,
            error=last_error,
        )

        if self.track_usage:
            self.session_metrics.add_call(metrics)

        raise RuntimeError(
            f"Model {model} failed after {total_retries} retries. Last error: {last_error}"
        )

    def get_session_metrics(self) -> SessionMetrics:
        return self.session_metrics

    def reset_session_metrics(self):
        self.session_metrics = SessionMetrics()

    def print_session_summary(self):
        m = self.session_metrics
        print("\n" + "=" * 60)
        print("LLM Session Summary")
        print("=" * 60)
        print(
            f"Total Calls: {m.total_calls} (Success: {m.successful_calls}, Failed: {m.failed_calls})"
        )
        print(f"Total Retries: {m.total_retries}")
        print(f"Total Latency: {m.total_latency_ms:.2f}ms ({m.total_latency_ms/1000:.2f}s)")
        print(f"Avg Latency: {m.total_latency_ms/max(1, m.total_calls):.2f}ms")
        print("\nToken Usage:")
        print(f"  Prompt Tokens: {m.tokens.prompt_tokens:,}")
        print(f"  Completion Tokens: {m.tokens.completion_tokens:,}")
        print(f"  Total Tokens: {m.tokens.total_tokens:,}")
        print(f"  Estimated Cost: ${m.tokens.estimated_cost:.4f}")
        print("\nCalls by Model:")
        for model, count in m.calls_by_model.items():
            print(f"  {model}: {count}")
        print("=" * 60)
