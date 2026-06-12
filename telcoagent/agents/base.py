"""Training-free ReAct agent base class shared by every LLM agent.

:class:`BaseTrainingFreeAgent` owns the ReAct loop (Thought -> Action ->
Observation against a :class:`~telcoagent.agents.registry.ToolRegistry`)
plus a tool-free ``single_shot_generate`` arm used by the ReAct-off
ablation. Subclasses / consumers:

    - ``telcoagent.explainer`` -- ``ExplainerAgent`` (paper Sec. III-C).
    - ``telcoagent.kg_construction`` -- Extractor / Aligner / Evaluator agents
      (paper Sec. III-A).

All LLM traffic goes through :func:`telcoagent.llm.api.invoke_llm`
(code convention: never call ``litellm.completion`` directly).
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from telcoagent.agents.registry import ToolRegistry
from telcoagent.llm.api import invoke_llm

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Output from a Training-Free Agent."""

    prediction: Dict[str, List[float]]
    analysis: str
    confidence: float
    metadata: Dict[str, Any]
    method_used: str = ""
    key_patterns: List[str] = field(default_factory=list)
    raw_response: str = ""


def _prune_messages(messages: List[Dict], keep_last_n: int = 8) -> List[Dict]:
    """Compress older ReAct steps to reduce context size.

    Keeps system + user prompt (first 2 messages) and the most recent
    *keep_last_n* messages. Middle messages are summarized into a single
    assistant message listing tool names called and key numeric results.
    """
    if len(messages) <= keep_last_n + 4:
        return messages  # not enough to compress

    head = messages[:2]  # system + user prompt
    middle = messages[2:-keep_last_n]
    tail = messages[-keep_last_n:]

    # Build concise summary of middle steps
    summary_parts = []
    for msg in middle:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "tool" and content:
                # Extract tool result summary: first 120 chars
                summary_parts.append(f"  Tool result: {content[:120]}...")
        else:
            # LiteLLM message object with tool_calls
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                names = [tc.function.name for tc in tool_calls]
                summary_parts.append(f"  Called: {', '.join(names)}")

    summary_msg = {
        "role": "assistant",
        "content": (
            f"[Context summary — {len(middle)} earlier messages compressed]\n"
            + "\n".join(summary_parts[:20])  # cap at 20 lines
        ),
    }
    return head + [summary_msg] + tail


class BaseTrainingFreeAgent:
    """Base class with ReAct generation loop."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def react_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_registry: ToolRegistry,
        max_steps: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        step_delay: float = 0,
        agent_state: Optional[Dict] = None,
    ) -> Tuple[str, List[Dict]]:
        """ReAct loop: Thought → Action → Observation, up to *max_steps*.

        Args:
            step_delay: Seconds to sleep between ReAct steps (for rate limiting).
            agent_state: Shared tool state dict. When provided, the loop checks
                ``agent_state["refinement_converged"]`` after each tool call and
                exits early if the refinement has converged (3 consecutive full
                rollbacks with no improvement on any KPI).
        """
        messages: List[Dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tools = tool_registry.get_schemas()

        for step in range(max_steps):
            if step > 0 and step_delay > 0:
                time.sleep(step_delay)
            try:
                response = invoke_llm(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                # Log token usage per step for cost tracking
                usage = getattr(response, "usage", None)
                if usage:
                    logger.info(
                        "  [ReAct] Step %d tokens: in=%d out=%d",
                        step + 1,
                        getattr(usage, "prompt_tokens", 0),
                        getattr(usage, "completion_tokens", 0),
                    )
            except Exception as e:
                logger.error(f"ReAct call failed at step {step}: {e}")
                return self._last_content(messages), messages

            message = response.choices[0].message
            messages.append(message)

            if not message.tool_calls:
                logger.info("  [ReAct] Step %d: Final answer", step + 1)
                return message.content or "", messages

            # Parse all tool calls, then execute in parallel if multiple
            tool_summary: dict = {}
            # (tc_id, name, args, None) on parse success;
            # (tc_id, name, None, error_msg) on parse failure.
            parsed_calls: List[Tuple[Any, str, Optional[Dict], Optional[str]]] = []
            for tc in message.tool_calls:
                name = tc.function.name
                tool_summary[name] = tool_summary.get(name, 0) + 1
                try:
                    args = json.loads(tc.function.arguments)
                    parsed_calls.append((tc.id, name, args, None))
                except Exception as parse_exc:
                    logger.warning(
                        "  [ReAct] Step %d: arg parse error for '%s': %s",
                        step + 1,
                        name,
                        parse_exc,
                    )
                    parsed_calls.append((tc.id, name, None, str(parse_exc)))

            # Execute valid tool calls (parallel if >1, sequential if 1)
            valid_calls: List[Tuple[Any, str, Dict]] = [
                (tc_id, nm, ar)
                for tc_id, nm, ar, err in parsed_calls
                if err is None and ar is not None
            ]

            tool_results = {}
            if len(valid_calls) > 1:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                with ThreadPoolExecutor(max_workers=min(4, len(valid_calls))) as executor:
                    futures = {
                        executor.submit(tool_registry.execute, nm, ar): tc_id
                        for tc_id, nm, ar in valid_calls
                    }
                    for future in as_completed(futures):
                        tc_id = futures[future]
                        try:
                            tool_results[tc_id] = future.result(timeout=30)
                        except Exception as exc:
                            tool_results[tc_id] = {"error": f"execution failed: {exc}"}
            else:
                for tc_id, nm, ar in valid_calls:
                    tool_results[tc_id] = tool_registry.execute(nm, ar)

            # Append results in original order (LLM expects tool messages
            # to match the order of tool_calls)
            for tc_id, name, args, err in parsed_calls:
                if err is not None:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": json.dumps(
                                {"error": f"argument parse error: {err}", "tool": name},
                                ensure_ascii=False,
                            ),
                        }
                    )
                else:
                    result = tool_results.get(tc_id, {"error": "no result"})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": json.dumps(result, ensure_ascii=False, default=str),
                        }
                    )
            calls_str = ", ".join(f"{n}x{c}" if c > 1 else n for n, c in tool_summary.items())
            logger.info("  [ReAct] Step %d: %s", step + 1, calls_str)

            # Prune message history to limit context growth
            if step >= 10 and len(messages) > 30:
                before_len = len(messages)
                messages = _prune_messages(messages, keep_last_n=8)
                if len(messages) < before_len:
                    logger.info(
                        "  [ReAct] Pruned messages: %d -> %d at step %d",
                        before_len,
                        len(messages),
                        step + 1,
                    )

            # Early stop: BOTH refinement paths exhausted.
            #   1) KG-deterministic refine_prediction reported converged.
            #   2) LLM-aided llm_kg_adjustment was either invoked at least
            #      once (apply / abstain / cap) — meaning the agent has
            #      considered the broader KG content.
            # The two-flag gate prevents the deterministic path from
            # short-circuiting the LLM-aided path before it gets a turn.
            if agent_state and agent_state.get("refinement_converged"):
                llm_aided_calls = agent_state.get("llm_kg_adjustment_calls", 0)
                if llm_aided_calls >= 1:
                    logger.info(
                        "  [ReAct] Early stop: deterministic converged + "
                        "llm-aided attempted at step %d",
                        step + 1,
                    )
                    return self._last_content(messages), messages

        logger.warning(f"  [ReAct] Max steps ({max_steps}) reached")
        return self._last_content(messages), messages

    def single_shot_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Tuple[str, List[Dict]]:
        """One-shot LLM call with no tool registry — used for the
        ReAct-off ablation arm.

        The user prompt is expected to already carry every piece of
        evidence the system prompt promises (anomaly events, OSM
        context, causal-mechanism evidence, …) since the agent has no
        way to fetch more data after this single call. Mirrors the
        return shape of :meth:`react_generate` so the explainer code
        path can swap the two transparently.
        """
        messages: List[Dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = invoke_llm(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            usage = getattr(response, "usage", None)
            if usage:
                logger.info(
                    "  [SingleShot] tokens: in=%d out=%d",
                    getattr(usage, "prompt_tokens", 0),
                    getattr(usage, "completion_tokens", 0),
                )
        except Exception as exc:
            logger.error("Single-shot call failed: %s", exc)
            return "", messages

        message = response.choices[0].message
        messages.append(message)
        return (message.content or ""), messages

    @staticmethod
    def _last_content(messages: List) -> str:
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "tool":
                continue
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if content:
                return content
        return ""
