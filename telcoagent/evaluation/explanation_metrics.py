"""Direct implementation of Faithfulness and Answer Relevancy.

Paper §IV-A4 defines:
  - Faithfulness: proportion of generated claims that are factually
    correct under 3GPP NR standards. The judge consults only its own
    domain knowledge (no external context supplied) so the metric is
    not biased by retrieved-context coverage.
  - Answer Relevancy: alignment between the generated explanation and
    the forecasting query on four anomaly-specific axes (KPI coverage,
    time window, causal mechanism, operator action).

Both metrics use a 1–5 Likert scale issued by ORANSight (Qwen-14B
fine-tuned on 3GPP specifications) and normalize to [0, 1] via
``(x - 1) / 4``. ORANSight is decoupled from the GPT-4o-mini
explainer to avoid same-family bias.

This module does NOT depend on the RAGAS library; the metric
definitions follow the paper's §IV-A4 formulation directly.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import pandas as pd  # noqa: F401  (used only in return-type annotation)

from telcoagent.llm.api import invoke_llm
from telcoagent.llm.config import VLLM_BASE_URL, VLLM_MODEL

logger = logging.getLogger(__name__)

# ─── Judge configuration ──────────────────────────────────────────────────────

_DEFAULT_JUDGE_MODEL: str = f"openai/{VLLM_MODEL}"
_DEFAULT_API_BASE: str = VLLM_BASE_URL
_DEFAULT_MAX_TOKENS: int = 512
_DEFAULT_TEMPERATURE: float = 0.0

# ─── Prompt constants ─────────────────────────────────────────────────────────

_FAITHFULNESS_JUDGE_SYSTEM_PROMPT = (
    "You are a 3GPP NR domain expert acting as a faithfulness "
    "judge. You will be given ONE atomic claim extracted from a 5G "
    "explanation report, plus the retrieved evidence the explainer "
    "had access to when authoring the claim.\n\n"
    "Faithfulness here is GROUNDING-AWARE: rate the claim primarily "
    "on whether it is supported by the retrieved evidence (when "
    "evidence is supplied). When the retrieved evidence is empty, "
    "fall back to your own 3GPP NR domain knowledge.\n\n"
    "Use this 1–5 Likert scale:\n"
    "  5: Fully grounded — the claim is directly entailed by a "
    "specific item in the retrieved evidence (named mechanism, "
    "exact KPI direction, or quantitative value present in the "
    "evidence). When no evidence is supplied, score 5 only if the "
    "claim cites a specific 3GPP-named mechanism or spec.\n"
    "  4: Mostly grounded — paraphrase of evidence, or a specific "
    "named mechanism that is consistent with evidence but lacks an "
    "exact value match.\n"
    "  3: Partially grounded — generic statement that holds under "
    "3GPP physics but is NOT directly supported by retrieved "
    "evidence (e.g., the explainer wrote a plausible chain without "
    "the evidence to back it).\n"
    "  2: Mostly ungrounded — claim contradicts the retrieved "
    "evidence in a small detail (e.g., direction or magnitude class), "
    "or asserts a specific value absent from evidence.\n"
    "  1: Hallucinated — the claim contradicts retrieved evidence "
    "or 3GPP NR standards, or invents non-existent entities (TS "
    "section, mechanism, threshold).\n\n"
    "Output STRICT JSON only — no prose outside the JSON object:\n"
    '{"score": <1|2|3|4|5>, "rationale": "<one sentence>"}'
)

# Reason-first variant of the faithfulness judge. Follows G-Eval / RubricEval
# best practice: the model emits step-by-step reasoning that contrasts the
# claim against each Likert anchor BEFORE committing to a score. The legacy
# score-first prompt above is retained so the Q6 evaluation-robustness arm can
# compare score-first vs reason-first chain-of-thought order.
_FAITHFULNESS_JUDGE_SYSTEM_PROMPT_REASON_FIRST = (
    "You are a 3GPP NR domain expert acting as a faithfulness "
    "judge. You will be given ONE atomic claim extracted from a 5G "
    "explanation report, plus the retrieved evidence the explainer "
    "had access to when authoring the claim.\n\n"
    "Faithfulness here is GROUNDING-AWARE: rate the claim primarily "
    "on whether it is supported by the retrieved evidence (when "
    "evidence is supplied). When the retrieved evidence is empty, "
    "fall back to your own 3GPP NR domain knowledge.\n\n"
    "Use this 1–5 Likert scale:\n"
    "  5: Fully grounded — the claim is directly entailed by a "
    "specific item in the retrieved evidence (named mechanism, "
    "exact KPI direction, or quantitative value present in the "
    "evidence). When no evidence is supplied, score 5 only if the "
    "claim cites a specific 3GPP-named mechanism or spec.\n"
    "  4: Mostly grounded — paraphrase of evidence, or a specific "
    "named mechanism that is consistent with evidence but lacks an "
    "exact value match.\n"
    "  3: Partially grounded — generic statement that holds under "
    "3GPP physics but is NOT directly supported by retrieved "
    "evidence (e.g., the explainer wrote a plausible chain without "
    "the evidence to back it).\n"
    "  2: Mostly ungrounded — claim contradicts the retrieved "
    "evidence in a small detail (e.g., direction or magnitude class), "
    "or asserts a specific value absent from evidence.\n"
    "  1: Hallucinated — the claim contradicts retrieved evidence "
    "or 3GPP NR standards, or invents non-existent entities (TS "
    "section, mechanism, threshold).\n\n"
    "First reason against each anchor level, then emit the score. "
    "In the 'reasoning' field, walk through the anchors step by step: "
    "state whether the claim meets or fails each Likert level (5→1) "
    "and why, then conclude with the level it lands on. Only after "
    "that reasoning, emit the integer 'score'.\n\n"
    "Output STRICT JSON only — no prose outside the JSON object. The "
    "'reasoning' key MUST come before the 'score' key:\n"
    '{"reasoning": "<step-by-step contrast against each Likert '
    'anchor>", "score": <1|2|3|4|5>}'
)

_ANSWER_RELEVANCY_JUDGE_SYSTEM_PROMPT = (
    "You are a structured evaluation judge for a 5G NR telecom "
    "explanation pipeline. You will be given ONE anomaly event "
    "description and a full explanation report.\n\n"
    "Score the report on FOUR axes for this specific event, using a "
    "1–5 Likert scale (1 = absent / wrong, 5 = explicit and "
    "operationally detailed):\n\n"
    "  1. kpi_mention: Does the report explicitly discuss the event's "
    "KPI by name?\n"
    "     5 = explicit, repeated, and quantitatively detailed.\n"
    "     4 = explicit single mention with detail.\n"
    "     3 = mentioned in passing without quantitative grounding.\n"
    "     2 = vague allusion (e.g., 'throughput-related').\n"
    "     1 = not mentioned.\n\n"
    "  2. time_mention: Does the report mention the event's hour "
    "window (t_start to t_end)?\n"
    "     5 = exact window referenced with hour-of-day anchors.\n"
    "     4 = exact window referenced but without anchors.\n"
    "     3 = same day or rough range mentioned.\n"
    "     2 = generic temporal hint (e.g., 'during the forecast').\n"
    "     1 = no time anchor.\n\n"
    "  3. mechanism: Does the report provide a 5G NR causal mechanism "
    "for this anomaly?\n"
    "     5 = specific 3GPP-grounded chain with spec citation and "
    "quantitative grounding (z-score, magnitude).\n"
    "     4 = specific mechanism (named upstream KPI + RAN function) "
    "without spec citation.\n"
    "     3 = generic mechanism description (e.g., 'channel quality "
    "issue').\n"
    "     2 = vague mention without an actual mechanism.\n"
    "     1 = absent or wrong direction.\n\n"
    "  4. mitigation: Does the report provide a specific operator "
    "action for this anomaly?\n"
    "     5 = specific RAN parameter / threshold with target value or "
    "operating range.\n"
    "     4 = specific RAN parameter named without a target value.\n"
    "     3 = generic operational suggestion (e.g., 'investigate "
    "scheduler').\n"
    "     2 = vague advice (e.g., 'monitor more closely').\n"
    "     1 = absent.\n\n"
    "You have NO access to external knowledge graphs — score only "
    "what is present in the report text.\n\n"
    "Output STRICT JSON only — no prose outside the JSON object:\n"
    '{"kpi_mention": <1|2|3|4|5>, "time_mention": <1|2|3|4|5>, '
    '"mechanism": <1|2|3|4|5>, "mitigation": <1|2|3|4|5>, '
    '"rationale": "<one or two sentences>"}'
)

# Reason-first variant of the answer-relevancy judge. Per G-Eval / RubricEval
# best practice the judge writes per-axis reasoning that contrasts the report
# against each Likert anchor BEFORE emitting the four axis scores. The legacy
# score-first prompt above is retained for the Q6 evaluation-robustness arm.
_ANSWER_RELEVANCY_JUDGE_SYSTEM_PROMPT_REASON_FIRST = (
    "You are a structured evaluation judge for a 5G NR telecom "
    "explanation pipeline. You will be given ONE anomaly event "
    "description and a full explanation report.\n\n"
    "Score the report on FOUR axes for this specific event, using a "
    "1–5 Likert scale (1 = absent / wrong, 5 = explicit and "
    "operationally detailed):\n\n"
    "  1. kpi_mention: Does the report explicitly discuss the event's "
    "KPI by name?\n"
    "     5 = explicit, repeated, and quantitatively detailed.\n"
    "     4 = explicit single mention with detail.\n"
    "     3 = mentioned in passing without quantitative grounding.\n"
    "     2 = vague allusion (e.g., 'throughput-related').\n"
    "     1 = not mentioned.\n\n"
    "  2. time_mention: Does the report mention the event's hour "
    "window (t_start to t_end)?\n"
    "     5 = exact window referenced with hour-of-day anchors.\n"
    "     4 = exact window referenced but without anchors.\n"
    "     3 = same day or rough range mentioned.\n"
    "     2 = generic temporal hint (e.g., 'during the forecast').\n"
    "     1 = no time anchor.\n\n"
    "  3. mechanism: Does the report provide a 5G NR causal mechanism "
    "for this anomaly?\n"
    "     5 = specific 3GPP-grounded chain with spec citation and "
    "quantitative grounding (z-score, magnitude).\n"
    "     4 = specific mechanism (named upstream KPI + RAN function) "
    "without spec citation.\n"
    "     3 = generic mechanism description (e.g., 'channel quality "
    "issue').\n"
    "     2 = vague mention without an actual mechanism.\n"
    "     1 = absent or wrong direction.\n\n"
    "  4. mitigation: Does the report provide a specific operator "
    "action for this anomaly?\n"
    "     5 = specific RAN parameter / threshold with target value or "
    "operating range.\n"
    "     4 = specific RAN parameter named without a target value.\n"
    "     3 = generic operational suggestion (e.g., 'investigate "
    "scheduler').\n"
    "     2 = vague advice (e.g., 'monitor more closely').\n"
    "     1 = absent.\n\n"
    "You have NO access to external knowledge graphs — score only "
    "what is present in the report text.\n\n"
    "First reason against each anchor level, then emit the score. In "
    "the 'reasoning' field, take each of the four axes in turn and "
    "walk step by step through its Likert anchors (5→1), stating "
    "whether the report meets or fails each level and why, then "
    "concluding with the level that axis lands on. Only after "
    "completing the per-axis reasoning, emit the four integer axis "
    "scores.\n\n"
    "Output STRICT JSON only — no prose outside the JSON object. The "
    "'reasoning' key MUST come before the axis score keys:\n"
    '{"reasoning": "<per-axis step-by-step contrast against each '
    'Likert anchor>", "kpi_mention": <1|2|3|4|5>, '
    '"time_mention": <1|2|3|4|5>, "mechanism": <1|2|3|4|5>, '
    '"mitigation": <1|2|3|4|5>}'
)

# ─── Section-splitter (mirrors audit.py pattern) ──────────────────────────────

_SECTION_HEADER_RE = re.compile(r"^#{1,6}\s*(§\d)\b", re.MULTILINE)


def _split_report_sections(report: str) -> Dict[str, str]:
    """Carve a markdown report into ``{"§N": text}`` blocks."""
    if not report:
        return {}
    matches = list(_SECTION_HEADER_RE.finditer(report))
    sections: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        tag = match.group(1)
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(report)
        sections[tag] = report[start:end]
    return sections


# ─── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class EventAlignmentScore:
    """Per-event 4-axis alignment score (each axis is the normalized
    Likert value in [0, 1])."""

    event_idx: int
    kpi: str
    t_start: int
    t_end: int
    kpi_mention: float
    time_mention: float
    mechanism: float
    mitigation: float
    rationale: str

    @property
    def score(self) -> float:
        """Mean of the four normalized axes; in [0, 1]."""
        return (self.kpi_mention + self.time_mention + self.mechanism + self.mitigation) / 4.0


@dataclass
class ClaimFaithfulnessScore:
    """Per-claim Faithfulness score on the normalized [0, 1] scale.

    The raw 1–5 Likert from the judge is mapped via ``(raw - 1) / 4``
    so that 5 → 1.0 and 1 → 0.0. Claims whose judge response failed
    to parse are dropped from the per-station mean and counted only
    in :attr:`telcoagent.evaluation.explanation_metrics.ExplanationScores.n_claims`
    minus :attr:`~telcoagent.evaluation.explanation_metrics.ExplanationScores.n_scored`.
    """

    claim: str
    raw_score: int  # 1–5 (Likert) or 0 if parse failed
    score: float  # normalized to [0, 1]; NaN if parse failed
    rationale: str
    parsed: bool = True  # False when the judge JSON was malformed


@dataclass
class ExplanationScores:
    """Output of :meth:`ExplanationEvaluator.evaluate_single`.

    ``faithfulness`` is the mean over successfully scored claims of
    the per-claim normalized Faithfulness, and ``answer_relevancy``
    is the mean over events of the per-event 4-axis mean. Both are
    bounded in [0, 1] and follow the paper's §IV-A4 definitions.
    """

    station_id: str
    faithfulness: Optional[float]
    answer_relevancy: Optional[float]
    n_claims: int
    n_scored: int
    per_claim: List[ClaimFaithfulnessScore] = field(default_factory=list)
    per_event_alignment: List[EventAlignmentScore] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station_id": self.station_id,
            "faithfulness": (
                round(self.faithfulness, 4) if self.faithfulness is not None else None
            ),
            "answer_relevancy": (
                round(self.answer_relevancy, 4) if self.answer_relevancy is not None else None
            ),
            "n_claims": self.n_claims,
            "n_scored": self.n_scored,
            "error": self.error,
        }


# ─── Claim extraction ─────────────────────────────────────────────────────────

_SECTION2_TABLE_ROW_RE = re.compile(
    r"^\|([^|]+)\|([^|]+)\|([^|]+)\|",
    re.MULTILINE,
)
# NOTE: ``Sensitivity-Local`` must come before ``Sensitivity`` in the
# alternation so the longer prefix is matched first; same for
# Sensitivity-Matrix. ``Parametric`` is the LLM-parametric 3GPP chain
# tag emitted in the kg_off arm so the audit can score those claims.
_TAG_NAMES = (
    "GraphRAG",
    "Cause",
    "Sensitivity-Local",
    "Sensitivity-Matrix",
    "Sensitivity",
    "Anomaly",
    "Temporal",
    "OSM",
    "Forecast",
    "Spatial",
    "Parametric",
)
_TAG_ALT = "|".join(_TAG_NAMES)
_SECTION4_BULLET_TAG_RE = re.compile(
    rf"^\s*[-*]\s+(.+\[(?:{_TAG_ALT})(?:[^\]]*)\].*)$",
    re.MULTILINE,
)
_SECTION5_ACTION_ROW_RE = re.compile(
    r"^\|([^|]+)\|([^|]+)\|([^|]+)\|",
    re.MULTILINE,
)
_TAG_RE = re.compile(rf"\[(?:{_TAG_ALT})(?:[^\]]*)\]")

_MIN_CLAIM_LEN: int = 10


def _extract_atomic_claims(report: str) -> List[str]:
    """Carve the report markdown into atomic claims.

    Sources:
      - §2 table rows: one claim per row, formatted as "(source → target) mechanism"
      - §4 bullet lines with [GraphRAG]/[Cause]/[Sensitivity]/[Anomaly]/[Temporal]/[OSM] tags
      - §5 action rows: one claim per row (Action cell text)
    """
    sections = _split_report_sections(report)
    claims: List[str] = []

    # §2 table rows: source → target + mechanism columns
    sec2 = sections.get("§2", "")
    for match in _SECTION2_TABLE_ROW_RE.finditer(sec2):
        col1 = match.group(1).strip()
        col2 = match.group(2).strip()
        col3 = match.group(3).strip()
        # Skip header rows (contain "Source", "Target", "Mechanism" labels or dashes)
        if re.match(r"^[-: ]+$", col1) or col1.lower() in ("source", "kpi", ""):
            continue
        if col2.lower() in ("target", "relationship", "") and col3.lower() in (
            "mechanism",
            "strength",
            "description",
            "",
        ):
            continue
        # Build claim from the row data
        if col1 and col3:
            claim = f"({col1} → {col2}) {col3}".strip()
        elif col1:
            claim = col1
        else:
            continue
        if len(claim) >= _MIN_CLAIM_LEN:
            claims.append(claim)

    # §4 bullets with recognized tags
    sec4 = sections.get("§4", "")
    for match in _SECTION4_BULLET_TAG_RE.finditer(sec4):
        bullet_text = match.group(1).strip()
        # Strip the tags themselves to get the clean claim text
        clean = _TAG_RE.sub("", bullet_text).strip()
        clean = re.sub(r"\s{2,}", " ", clean)
        if len(clean) >= _MIN_CLAIM_LEN:
            claims.append(clean)

    # §5 action rows
    sec5 = sections.get("§5", "")
    for match in _SECTION5_ACTION_ROW_RE.finditer(sec5):
        col1 = match.group(1).strip()
        if re.match(r"^[-: ]+$", col1) or col1.lower() in ("action", "priority", ""):
            continue
        if len(col1) >= _MIN_CLAIM_LEN:
            claims.append(col1)

    return claims


# ─── Judge helpers ────────────────────────────────────────────────────────────


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").rstrip()
    return text


def _normalize_likert(raw: int) -> float:
    """Map a 1–5 Likert score to [0, 1] via ``(raw - 1) / 4``."""
    return (float(raw) - 1.0) / 4.0


_MAX_CONTEXT_CHARS_FOR_JUDGE: int = 6000


def _judge_faithfulness(
    claim: str,
    judge_model: str,
    api_base: str,
    retrieved_contexts: Optional[List[str]] = None,
    system_prompt: str = _FAITHFULNESS_JUDGE_SYSTEM_PROMPT_REASON_FIRST,
) -> ClaimFaithfulnessScore:
    """Ask the ORANSight judge to rate the claim on a 1–5 Likert scale.

    When ``retrieved_contexts`` is non-empty, it is concatenated into
    the user message and the judge scores the claim primarily on
    whether the retrieved evidence supports it (RAGAS-style grounding-
    aware faithfulness). When empty, the judge falls back to its own
    3GPP NR domain knowledge — preserving backwards-compatible
    behaviour for callers that do not yet pass context.

    ``system_prompt`` selects the judge chain-of-thought order: the
    reason-first variant (default) places step-by-step reasoning ahead
    of the score per G-Eval / RubricEval best practice; the legacy
    score-first variant is :data:`_FAITHFULNESS_JUDGE_SYSTEM_PROMPT`.
    The JSON parser below is key-order independent and accepts either
    a ``rationale`` or a ``reasoning`` field, so both prompt variants
    are handled by the same code path.

    On JSON parse failure or out-of-range score the function logs at
    ERROR and returns a :class:`ClaimFaithfulnessScore` with
    ``parsed=False`` so the caller can drop the claim from the mean
    rather than silently degrading the score.
    """
    if retrieved_contexts:
        # Truncate to keep judge prompts inside the 8K context window.
        ctx_text = "\n\n---\n\n".join(retrieved_contexts)
        if len(ctx_text) > _MAX_CONTEXT_CHARS_FOR_JUDGE:
            ctx_text = ctx_text[:_MAX_CONTEXT_CHARS_FOR_JUDGE] + "\n[...truncated]"
        user_msg = (
            "Retrieved evidence supplied to the explainer:\n"
            "------\n"
            f"{ctx_text}\n"
            "------\n\n"
            f"Claim:\n{claim}"
        )
    else:
        user_msg = f"Claim:\n{claim}"
    resp = invoke_llm(
        model=judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_DEFAULT_MAX_TOKENS,
        api_base=api_base,
    )
    raw = resp.choices[0].message.content or ""
    cleaned = _strip_code_fence(raw)
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not json_match:
        preview = raw[:120].replace("\n", " ")
        logger.error(
            "_judge_faithfulness: no JSON in response for claim %r. Raw: %r",
            claim[:60],
            preview,
        )
        return ClaimFaithfulnessScore(
            claim=claim,
            raw_score=0,
            score=float("nan"),
            rationale=f"judge JSON parse failed: {preview}",
            parsed=False,
        )
    try:
        obj = json.loads(json_match.group(0))
    except json.JSONDecodeError as exc:
        preview = raw[:120].replace("\n", " ")
        logger.error(
            "_judge_faithfulness: JSON decode error for claim %r: %s. Raw: %r",
            claim[:60],
            exc,
            preview,
        )
        return ClaimFaithfulnessScore(
            claim=claim,
            raw_score=0,
            score=float("nan"),
            rationale=f"judge JSON parse failed: {preview}",
            parsed=False,
        )

    # Accept either the score-first ``rationale`` field or the
    # reason-first ``reasoning`` field; both prompt variants share
    # this parser.
    rationale = str(obj.get("rationale") or obj.get("reasoning") or "").strip()
    raw_score_field = obj.get("score")
    try:
        raw_score = int(raw_score_field)
    except (TypeError, ValueError):
        logger.error(
            "_judge_faithfulness: non-integer score %r for claim %r",
            raw_score_field,
            claim[:60],
        )
        return ClaimFaithfulnessScore(
            claim=claim,
            raw_score=0,
            score=float("nan"),
            rationale=f"judge returned non-integer score: {raw_score_field!r}",
            parsed=False,
        )
    if raw_score not in (1, 2, 3, 4, 5):
        logger.error(
            "_judge_faithfulness: out-of-range score %d for claim %r",
            raw_score,
            claim[:60],
        )
        return ClaimFaithfulnessScore(
            claim=claim,
            raw_score=raw_score,
            score=float("nan"),
            rationale=f"judge returned out-of-range score: {raw_score}",
            parsed=False,
        )

    return ClaimFaithfulnessScore(
        claim=claim,
        raw_score=raw_score,
        score=_normalize_likert(raw_score),
        rationale=rationale,
        parsed=True,
    )


def _judge_event_alignment(
    report: str,
    event: Dict[str, Any],
    judge_model: str,
    api_base: str,
    system_prompt: str = _ANSWER_RELEVANCY_JUDGE_SYSTEM_PROMPT_REASON_FIRST,
) -> EventAlignmentScore:
    """Score the report on 4 axes for one anomaly event.

    ``system_prompt`` selects the judge chain-of-thought order: the
    reason-first variant (default) emits per-axis reasoning before the
    four axis scores; the legacy score-first variant is
    :data:`_ANSWER_RELEVANCY_JUDGE_SYSTEM_PROMPT`. The JSON parser is
    key-order independent and accepts either a ``rationale`` or a
    ``reasoning`` field, so both prompt variants share this code path.

    On JSON parse failure: logs ERROR and returns zeros on all 4 axes.
    """
    event_idx = int(event.get("event_idx", event.get("idx", 0)))
    kpi = str(event.get("kpi", ""))
    t_start = int(event.get("t_start", 0))
    t_end = int(event.get("t_end", 0))

    event_desc = json.dumps(
        {
            "kpi": kpi,
            "t_start": t_start,
            "t_end": t_end,
            "type": event.get("type"),
            "direction": event.get("direction"),
            "z_score": event.get("z_score"),
            "magnitude": event.get("magnitude"),
        },
        ensure_ascii=False,
    )
    user_msg = f"Anomaly event:\n{event_desc}\n\nExplanation report:\n{report}"

    resp = invoke_llm(
        model=judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_DEFAULT_MAX_TOKENS,
        api_base=api_base,
    )
    raw = resp.choices[0].message.content or ""
    cleaned = _strip_code_fence(raw)
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not json_match:
        preview = raw[:120].replace("\n", " ")
        logger.error(
            "_judge_event_alignment: no JSON in response for event %r. Raw: %r",
            kpi,
            preview,
        )
        return EventAlignmentScore(
            event_idx=event_idx,
            kpi=kpi,
            t_start=t_start,
            t_end=t_end,
            kpi_mention=0.0,
            time_mention=0.0,
            mechanism=0.0,
            mitigation=0.0,
            rationale=f"judge JSON parse failed: {preview}",
        )
    try:
        obj = json.loads(json_match.group(0))
    except json.JSONDecodeError as exc:
        preview = raw[:120].replace("\n", " ")
        logger.error(
            "_judge_event_alignment: JSON decode error for event %r: %s. Raw: %r",
            kpi,
            exc,
            preview,
        )
        return EventAlignmentScore(
            event_idx=event_idx,
            kpi=kpi,
            t_start=t_start,
            t_end=t_end,
            kpi_mention=0.0,
            time_mention=0.0,
            mechanism=0.0,
            mitigation=0.0,
            rationale=f"judge JSON parse failed: {preview}",
        )

    def _axis(key: str) -> float:
        """Parse a 1–5 Likert value and normalize to [0, 1]."""
        val = obj.get(key)
        try:
            raw = int(val)
        except (TypeError, ValueError):
            logger.error(
                "_judge_event_alignment: non-integer %s=%r for event %r",
                key,
                val,
                kpi,
            )
            return 0.0
        if raw not in (1, 2, 3, 4, 5):
            logger.error(
                "_judge_event_alignment: out-of-range %s=%d for event %r",
                key,
                raw,
                kpi,
            )
            return 0.0
        return _normalize_likert(raw)

    return EventAlignmentScore(
        event_idx=event_idx,
        kpi=kpi,
        t_start=t_start,
        t_end=t_end,
        kpi_mention=_axis("kpi_mention"),
        time_mention=_axis("time_mention"),
        mechanism=_axis("mechanism"),
        mitigation=_axis("mitigation"),
        # Accept score-first ``rationale`` or reason-first ``reasoning``.
        rationale=str(obj.get("rationale") or obj.get("reasoning") or "").strip(),
    )


# ─── Public evaluator ─────────────────────────────────────────────────────────


class ExplanationEvaluator:
    """Direct implementation of the paper's Faithfulness and Answer Relevancy.

    No dependency on the RAGAS library. The judge is ORANSight (Qwen-14B
    fine-tuned on 3GPP specs), decoupled from the GPT-4o-mini explainer
    to avoid same-family bias.
    """

    def __init__(
        self,
        judge_model: Optional[str] = None,
        api_base: Optional[str] = None,
        max_claims_per_call: int = 1,
        judge_cot_mode: str = "reason_first",
    ):
        """Construct an evaluator.

        Args:
            judge_cot_mode: Chain-of-thought order for the judge
                prompts. ``"reason_first"`` (default) emits step-by-step
                reasoning before the score per G-Eval / RubricEval best
                practice; ``"score_first"`` keeps the legacy
                score-then-rationale order for the Q6 evaluation-
                robustness ablation arm. Any other value raises
                ``ValueError`` — there is no silent fallback.
        """
        self._judge_model = judge_model if judge_model is not None else _DEFAULT_JUDGE_MODEL
        self._api_base = api_base if api_base is not None else _DEFAULT_API_BASE
        self._max_claims_per_call = max_claims_per_call
        if judge_cot_mode not in ("reason_first", "score_first"):
            raise ValueError(
                f"judge_cot_mode must be 'reason_first' or 'score_first', "
                f"got {judge_cot_mode!r}"
            )
        self._judge_cot_mode = judge_cot_mode
        if judge_cot_mode == "reason_first":
            self._faithfulness_prompt = _FAITHFULNESS_JUDGE_SYSTEM_PROMPT_REASON_FIRST
            self._answer_relevancy_prompt = _ANSWER_RELEVANCY_JUDGE_SYSTEM_PROMPT_REASON_FIRST
        else:
            self._faithfulness_prompt = _FAITHFULNESS_JUDGE_SYSTEM_PROMPT
            self._answer_relevancy_prompt = _ANSWER_RELEVANCY_JUDGE_SYSTEM_PROMPT

    def evaluate_single(
        self,
        station_id: str,
        report: str,
        retrieved_contexts: Optional[List[str]] = None,
        anomaly_events: Optional[List[Dict[str, Any]]] = None,
    ) -> ExplanationScores:
        """Evaluate one explanation report.

        Args:
            station_id: Base station identifier.
            report: Full markdown explanation report from the ExplainerAgent.
            retrieved_contexts: Evidence blob seen by the explainer
                (KG mechanisms, sensitivity rows, OSM context). Forwarded
                to the Faithfulness judge so claims are scored against
                the actual supplied evidence — RAGAS-style grounding-
                aware faithfulness — rather than against the judge's
                general 3GPP knowledge alone. Pass ``None`` to fall back
                to the original general-knowledge-only behaviour.
            anomaly_events: Anomaly event dicts from the anomaly detector.

        Returns:
            :class:`ExplanationScores` with normalized Faithfulness in
            [0, 1] (mean over successfully scored claims of the
            normalized 1–5 Likert), normalized Answer Relevancy in
            [0, 1] (mean over events of the per-event 4-axis mean) or
            ``None`` when the report has zero anomaly events (so the
            metric is excluded from cross-station means rather than
            biasing them toward zero), and per-claim / per-event
            detail.
        """
        events = anomaly_events or []

        claims = _extract_atomic_claims(report)
        n_claims = len(claims)

        per_claim: List[ClaimFaithfulnessScore] = []
        for claim in claims:
            score = _judge_faithfulness(
                claim=claim,
                judge_model=self._judge_model,
                api_base=self._api_base,
                retrieved_contexts=retrieved_contexts,
                system_prompt=self._faithfulness_prompt,
            )
            per_claim.append(score)

        scored_claims = [c for c in per_claim if c.parsed]
        n_scored = len(scored_claims)
        faithfulness: Optional[float]
        if n_scored > 0:
            faithfulness = sum(c.score for c in scored_claims) / n_scored
        else:
            faithfulness = None

        per_event_alignment: List[EventAlignmentScore] = []
        for idx, event in enumerate(events):
            event_with_idx = dict(event)
            event_with_idx.setdefault("event_idx", idx)
            alignment = _judge_event_alignment(
                report=report,
                event=event_with_idx,
                judge_model=self._judge_model,
                api_base=self._api_base,
                system_prompt=self._answer_relevancy_prompt,
            )
            per_event_alignment.append(alignment)

        answer_relevancy: Optional[float]
        if per_event_alignment:
            answer_relevancy = sum(e.score for e in per_event_alignment) / len(per_event_alignment)
        else:
            # No anomaly events ⇒ metric is undefined for this report;
            # caller should drop None values from cross-station means
            # so reports with zero events do not bias the average.
            answer_relevancy = None

        logger.info(
            "[ExplanationEvaluator] %s: faithfulness=%s (%d/%d claims "
            "scored), answer_relevancy=%s (%d events)",
            station_id,
            f"{faithfulness:.4f}" if faithfulness is not None else "N/A",
            n_scored,
            n_claims,
            f"{answer_relevancy:.4f}" if answer_relevancy is not None else "N/A",
            len(per_event_alignment),
        )

        return ExplanationScores(
            station_id=station_id,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            n_claims=n_claims,
            n_scored=n_scored,
            per_claim=per_claim,
            per_event_alignment=per_event_alignment,
        )

    def evaluate_batch(self, records: List[Dict[str, Any]]) -> "pd.DataFrame":
        """Evaluate multiple stations.

        Args:
            records: List of dicts, each containing the kwargs for evaluate_single:
                ``station_id``, ``report``, ``retrieved_contexts``, ``anomaly_events``.

        Returns:
            pandas DataFrame with one row per station and scalar metric columns.
        """
        import pandas as pd

        results = []
        for rec in records:
            scores = self.evaluate_single(
                station_id=rec["station_id"],
                report=rec["report"],
                retrieved_contexts=rec.get("retrieved_contexts", []),
                anomaly_events=rec.get("anomaly_events", []),
            )
            results.append(scores.to_dict())

        df = pd.DataFrame(results)
        logger.info(
            "[ExplanationEvaluator] batch: %d stations | "
            "mean faithfulness=%.3f | mean answer_relevancy=%.3f",
            len(df),
            df["faithfulness"].mean(),
            df["answer_relevancy"].mean(),
        )
        return df
