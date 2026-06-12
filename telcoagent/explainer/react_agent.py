"""ReAct report author for the Explanation pipeline (paper Sec. III-C).

:class:`ExplainerAgent` drives the LLM that writes the 5-section
explanation report. It authors the report across three turns
(tables → narrative → action) on top of the shared ReAct loop in
:class:`telcoagent.agents.base.BaseTrainingFreeAgent`, then
``_stitch_sections`` reassembles the per-turn outputs into the
canonical §0–§5 display order.

The agent itself holds no evidence: everything it cites comes either
from ReAct tool calls (ReAct-on) or from the pre-stuffed evidence block
built by :func:`telcoagent.explainer.evidence._prestuff_evidence`
(ReAct-off ablation).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ─── Multi-turn section stitcher ─────────────────────────────────────────

_SECTION_HEADER_PATTERN = re.compile(
    r"^#{1,6}\s*(§\d)\b",
    re.MULTILINE,
)
_DISPLAY_ORDER = ("§0", "§1", "§2", "§3", "§4", "§5")


def _split_into_sections(text: str) -> Dict[str, str]:
    """Carve a markdown blob into ``{"§N": "## §N ..."}`` chunks.

    Headers may use any heading level (`#`–`######`). The chunk for
    ``§N`` extends from its header line up to (but not including) the
    next §-header in the same blob.
    """
    matches = list(_SECTION_HEADER_PATTERN.finditer(text or ""))
    if not matches:
        return {}
    chunks: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        tag = match.group(1)
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunks[tag] = text[start:end].rstrip()
    return chunks


def _stitch_sections(*turn_outputs: str) -> str:
    """Reorder per-turn outputs into the canonical §0–§5 display order.

    Each turn may produce multiple sections (turn 1 = §1+§2+§3, turn 3 =
    §5+§0). The stitcher:
      1. parses every turn for its section headers,
      2. picks the latest non-empty chunk per tag (later turns override
         earlier ones — the model never re-emits an earlier section, but
         this guards against repetition),
      3. concatenates the chunks in §0 → §5 display order with blank
         lines between them.

    A section that does not appear in any turn is simply skipped — the
    downstream sanity check in :class:`TelcoAgentExplainer` already
    handles missing-figure backstops, so callers can detect a missing
    §N by string-searching the stitched report.
    """
    by_tag: Dict[str, str] = {}
    for output in turn_outputs:
        for tag, chunk in _split_into_sections(output).items():
            if chunk.strip():
                by_tag[tag] = chunk
    ordered = [by_tag[tag] for tag in _DISPLAY_ORDER if tag in by_tag]
    return "\n\n".join(ordered)


# ─── ExplainerAgent (ReAct) ───────────────────────────────────────────────


class ExplainerAgent:
    """ReAct agent that generates the 5-section explanation report."""

    _MAX_REACT_STEPS: int = 15
    _TEMPERATURE: float = 0.1
    _MAX_TOKENS: int = 2048

    def __init__(
        self,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        from telcoagent.agents.base import BaseTrainingFreeAgent
        from telcoagent.llm.config import EXPLAINER_MODEL
        from telcoagent.llm.prompts import EXPLAINER_SYSTEM_PROMPT

        resolved = model_name or EXPLAINER_MODEL
        self._agent = BaseTrainingFreeAgent(resolved)
        # ``None`` means "use the default prompt" so callers that omit the
        # argument get unchanged behaviour (backward compatible).
        self._system_prompt = (
            system_prompt if system_prompt is not None else EXPLAINER_SYSTEM_PROMPT
        )
        logger.info("ExplainerAgent initialised (model=%s)", resolved)

    _PER_TURN_REACT_STEPS: int = 5

    def explain_multi_turn(
        self,
        *,
        tool_registry,
        prediction_summary: str,
        evidence_block: str,
        structured_context: str,
        plot_catalogue: str,
        ablate_react: bool,
        user_prompt_prefix: Optional[str] = None,
    ) -> str:
        """Author the report across three turns: tables → narrative → action.

        Each turn produces a narrower output (1–3 sections at a time)
        so format adherence is more reliable on smaller models. The
        same system prompt is sent on every turn, which makes the
        prefix cacheable on providers that support it.

        Parameters
        ----------
        tool_registry
            ReAct tool registry; ignored when ``ablate_react=True``.
        prediction_summary
            Forecast summary block (used in turn 1).
        evidence_block
            Pre-stuffed evidence text from
            :func:`telcoagent.explainer.evidence._prestuff_evidence`.
            Empty string in ReAct-on mode tells the LLM to fetch
            evidence via tool calls instead.
        structured_context
            Anomaly snapshot (used in turn 2).
        plot_catalogue
            Plot path catalogue for §4 figure embeds (used in turn 2).
        ablate_react
            When True, dispatch to ``single_shot_generate``; when
            False, dispatch to the ReAct loop.
        user_prompt_prefix
            Optional string prepended to every turn's user message.
            ``None`` (default) leaves the user prompt unmodified (backward
            compatible). Used by the Q5 ``instruction_placement`` variant to
            move grounding rules from the system prompt into the user turns.
        """
        from telcoagent.llm.prompts import (
            TURN1_USER_TEMPLATE,
            TURN2_USER_TEMPLATE,
            TURN3_USER_TEMPLATE,
        )

        evidence_for_turn1 = evidence_block or (
            "(no pre-stuffed evidence — call the available tools to "
            "fetch causal-mechanism, spatial, and temporal evidence "
            "as needed)"
        )

        # Turn 1: §1, §2, §3
        turn1_user = TURN1_USER_TEMPLATE.format(
            prediction_summary=prediction_summary or "(no prediction summary)",
            evidence_block=evidence_for_turn1,
        )
        turn1_out = self._run_turn(
            label="turn1_tables",
            user_prompt=turn1_user,
            tool_registry=tool_registry,
            ablate_react=ablate_react,
            user_prompt_prefix=user_prompt_prefix,
        )

        # Turn 2: §4
        turn2_user = (
            "## Sections already authored (§1, §2, §3)\n"
            f"{turn1_out}\n\n"
            + TURN2_USER_TEMPLATE.format(
                structured_context=structured_context or "(no anomaly events)",
                plot_catalogue=plot_catalogue or "(no plot catalogue)",
            )
        )
        turn2_out = self._run_turn(
            label="turn2_narrative",
            user_prompt=turn2_user,
            tool_registry=tool_registry,
            ablate_react=ablate_react,
            user_prompt_prefix=user_prompt_prefix,
        )

        # Turn 3: §5 followed by §0
        turn3_user = (
            "## Sections already authored (§1, §2, §3, §4)\n"
            f"{turn1_out}\n\n{turn2_out}\n\n" + TURN3_USER_TEMPLATE
        )
        turn3_out = self._run_turn(
            label="turn3_action",
            user_prompt=turn3_user,
            tool_registry=tool_registry,
            ablate_react=ablate_react,
            user_prompt_prefix=user_prompt_prefix,
        )

        return _stitch_sections(turn1_out, turn2_out, turn3_out)

    def _run_turn(
        self,
        *,
        label: str,
        user_prompt: str,
        tool_registry,
        ablate_react: bool,
        user_prompt_prefix: Optional[str] = None,
    ) -> str:
        """Dispatch a single turn to either single-shot or ReAct.

        Parameters
        ----------
        user_prompt_prefix
            Optional string prepended to ``user_prompt`` before the LLM
            call. ``None`` (default) leaves the prompt unmodified.
        """
        effective_user = user_prompt_prefix + user_prompt if user_prompt_prefix else user_prompt
        if ablate_react:
            logger.info(
                "  [ExplainerAgent] %s single-shot (user=%d chars)",
                label,
                len(effective_user),
            )
            content, _ = self._agent.single_shot_generate(
                system_prompt=self._system_prompt,
                user_prompt=effective_user,
                temperature=self._TEMPERATURE,
                max_tokens=self._MAX_TOKENS,
            )
            return content
        logger.info(
            "  [ExplainerAgent] %s ReAct (user=%d chars, max_steps=%d)",
            label,
            len(effective_user),
            self._PER_TURN_REACT_STEPS,
        )
        content, _ = self._agent.react_generate(
            system_prompt=self._system_prompt,
            user_prompt=effective_user,
            tool_registry=tool_registry,
            max_steps=self._PER_TURN_REACT_STEPS,
            temperature=self._TEMPERATURE,
            max_tokens=self._MAX_TOKENS,
        )
        return content
