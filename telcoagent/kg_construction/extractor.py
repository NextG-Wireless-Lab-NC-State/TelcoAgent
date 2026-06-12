"""Extractor agent — 3GPP PDF section chunks → raw triples (paper Sec. III-A, step 1).

First of the three KG-construction agents. Internally runs the KARMA
Parsing + Extraction phases:

    IA (normalize)  →  RA (relevance δ)  →  SA (summarize if long)
                    →  EEA + REA (entity + relation extraction).

Input:  :class:`telcoagent.kg_construction.schema.SectionChunk` from
        the PDF parser.
Output: :class:`telcoagent.kg_construction.schema.ExtractedTriple` list,
        consumed by the Aligner agent.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import List, Optional

from telcoagent.agents.base import BaseTrainingFreeAgent

from ..llm.config import EnhancedLLMConfig
from ..ontology.core import ThreeGPPOntology
from .prompts import (
    EXTRACTOR_SYSTEM,
    EXTRACTOR_USER_TEMPLATE,
    READER_SYSTEM,
    READER_USER_TEMPLATE,
    SUMMARIZER_SYSTEM,
    SUMMARIZER_USER_TEMPLATE,
)
from .schema import ExtractedTriple, SectionChunk
from .shared import _RPM_DELAY, _parse_json_response
from .tools import make_extractor_tools

logger = logging.getLogger(__name__)

# KARMA Reader δ threshold — sections below this relevance are skipped.
# Default 0.0 preserves v1 behavior (no filtering) for backwards compatibility.
_READER_DELTA = float(os.environ.get("TELCOAGENT_READER_DELTA", "0.0"))

# Body length above which the Summarizer condenses the text instead of
# truncating. Default matches the legacy body[:6000] boundary.
_SUMMARIZE_THRESHOLD = int(os.environ.get("TELCOAGENT_SUMMARIZE_THRESHOLD", "6000"))

_HYPHEN_BREAK_RE = re.compile(r"-\n(\w)")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


class ExtractorAgent:
    """Extracts structured triples from 3GPP PDF section chunks.

    Internally runs the KARMA Parsing+Extraction phases:
      IA (normalize)  →  RA (relevance δ)  →  SA (summarize if long)
                      →  EEA + REA (entity + relation extraction).
    """

    def __init__(
        self,
        llm_config: EnhancedLLMConfig,
        ontology: Optional[ThreeGPPOntology] = None,
        reader_delta: Optional[float] = None,
        summarize_threshold: Optional[int] = None,
    ):
        self.llm = llm_config
        self.ontology = ontology
        self.reader_delta = _READER_DELTA if reader_delta is None else reader_delta
        self.summarize_threshold = (
            _SUMMARIZE_THRESHOLD if summarize_threshold is None else summarize_threshold
        )
        # Counters for orchestrator-side reporting
        self.skipped_by_reader = 0
        self.summarized_count = 0

        self._react_agent: Optional[BaseTrainingFreeAgent] = None
        self._tool_registry = None

        if ontology is not None:
            model = llm_config.agent_models["kg_build"]
            self._react_agent = BaseTrainingFreeAgent(model)
            self._tool_registry = make_extractor_tools(ontology)

    # ─── KARMA IA: deterministic text normalization ─────────────────────

    @staticmethod
    def _normalize_body(text: str) -> str:
        """Fix PDF artifacts: hyphen line-breaks, runs of whitespace."""
        if not text:
            return ""
        text = _HYPHEN_BREAK_RE.sub(r"\1", text)  # "cali-\nbration" -> "calibration"
        text = _WHITESPACE_RE.sub(" ", text)
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)
        return text.strip()

    # ─── KARMA RA: relevance score against ontology scope ───────────────

    def _score_relevance(self, chunk: SectionChunk) -> float:
        """Return KARMA Reader relevance score in [0, 1].

        When reader_delta <= 0 (default), scoring is skipped and 1.0 is returned.
        """
        if self.reader_delta <= 0.0:
            return 1.0
        try:
            prompt = READER_USER_TEMPLATE.format(
                spec=chunk.spec,
                clause=chunk.clause,
                title=chunk.title,
                body_preview=chunk.body[:1500],
            )
            text, _ = self.llm.call_with_retry(
                prompt=prompt,
                agent_type="kg_build",
                role="kg_build",
                system_message=READER_SYSTEM,
            )
            # Locate a JSON object in the response.
            stripped = text.strip()
            if stripped.startswith("```"):
                stripped = "\n".join(
                    ln for ln in stripped.split("\n") if not ln.strip().startswith("```")
                )
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return 1.0
            obj = json.loads(stripped[start : end + 1])
            score = float(obj.get("relevance", 1.0))
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"Reader scoring failed for {chunk.spec} §{chunk.clause}: {e}")
            return 1.0  # fail-open: do not drop the chunk on transient errors

    # ─── KARMA SA: condense long clauses instead of truncating ──────────

    def _summarize_if_long(self, body: str, chunk: SectionChunk) -> str:
        """If body exceeds the threshold, LLM-summarize it; otherwise return as-is."""
        if self.summarize_threshold <= 0 or len(body) <= self.summarize_threshold:
            return body
        try:
            prompt = SUMMARIZER_USER_TEMPLATE.format(
                spec=chunk.spec,
                clause=chunk.clause,
                title=chunk.title,
                body=body,
            )
            text, _ = self.llm.call_with_retry(
                prompt=prompt,
                agent_type="kg_build",
                role="kg_build",
                system_message=SUMMARIZER_SYSTEM,
            )
            condensed = text.strip()
            # Strip markdown fences if the model wrapped the output.
            if condensed.startswith("```"):
                lines = condensed.split("\n")
                condensed = "\n".join(
                    lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                ).strip()
            if not condensed or len(condensed) > len(body):
                # Bad summary (empty or bloated). Fall back to hard cut.
                logger.warning(
                    f"Summarizer produced unusable output for {chunk.spec} §{chunk.clause}; falling back to truncation."
                )
                return body[: self.summarize_threshold]
            self.summarized_count += 1
            logger.info(
                f"Summarized {chunk.spec} §{chunk.clause}: {len(body)} -> {len(condensed)} chars"
            )
            return condensed
        except Exception as e:
            logger.warning(
                f"Summarizer failed for {chunk.spec} §{chunk.clause}: {e}; falling back to truncation."
            )
            return body[: self.summarize_threshold]

    def extract(self, chunk: SectionChunk) -> List[ExtractedTriple]:
        """Extract triples from a single PDF section chunk.

        Flow: normalize → relevance filter (δ) → summarize if long → extract.
        """
        # [1] Normalize PDF artifacts
        body = self._normalize_body(chunk.body)
        if not body:
            return []

        # [2] Relevance filter (KARMA RA)
        relevance = self._score_relevance(chunk)
        if relevance < self.reader_delta:
            self.skipped_by_reader += 1
            logger.info(
                f"Skipping {chunk.spec} §{chunk.clause} (relevance={relevance:.2f} < δ={self.reader_delta:.2f})"
            )
            return []

        # [3] Summarize if too long (KARMA SA) — replaces legacy body[:6000] hard cut
        body = self._summarize_if_long(body, chunk)

        # [4] Build tables section
        tables_section = ""
        if chunk.tables:
            tables_text = "\n\n".join(chunk.tables[:5])
            tables_section = f"**Tables**:\n{tables_text}"

        prompt = EXTRACTOR_USER_TEMPLATE.format(
            spec=chunk.spec,
            clause=chunk.clause,
            title=chunk.title,
            body=body,
            tables_section=tables_section,
        )

        try:
            response_text = self._call_llm(EXTRACTOR_SYSTEM, prompt)
            raw_triples = _parse_json_response(response_text)

            triples = []
            for rt in raw_triples:
                # KARMA REA: use relation_prob when provided, else default 1.0
                try:
                    rel_prob = float(rt.get("relation_prob", 1.0))
                except (TypeError, ValueError):
                    rel_prob = 1.0
                rel_prob = max(0.0, min(1.0, rel_prob))

                def _opt_pct(key: str) -> Optional[float]:
                    v = rt.get(key)
                    if v is None:
                        return None
                    try:
                        return max(0.0, min(100.0, float(v)))
                    except (TypeError, ValueError):
                        return None

                mechanism = rt.get("mechanism")
                if mechanism is not None and not isinstance(mechanism, str):
                    mechanism = str(mechanism)
                fine_clause = rt.get("fine_clause")
                if fine_clause is not None and not isinstance(fine_clause, str):
                    fine_clause = str(fine_clause)

                triples.append(
                    ExtractedTriple(
                        subject=rt.get("subject", ""),
                        predicate=rt.get("predicate", ""),
                        object=rt.get("object", ""),
                        triple_type=rt.get("triple_type", "relationship"),
                        source_spec=chunk.spec,
                        source_clause=chunk.clause,
                        raw_text=chunk.body[:200],
                        relation_prob=rel_prob,
                        min_pct_source=_opt_pct("min_pct_source"),
                        min_pct_target=_opt_pct("min_pct_target"),
                        mechanism=mechanism,
                        fine_clause=fine_clause,
                    )
                )

            logger.info(f"Extracted {len(triples)} triples from {chunk.spec} §{chunk.clause}")
            return triples

        except Exception as e:
            logger.error(f"Extraction failed for {chunk.spec} §{chunk.clause}: {e}")
            return []

    def _call_llm(self, system: str, prompt: str) -> str:
        """Call LLM via ReAct."""
        if not self._react_agent or not self._tool_registry:
            raise RuntimeError(
                "ExtractorAgent requires a react_agent and tool_registry; "
                "pass them to the constructor."
            )
        text, _ = self._react_agent.react_generate(
            system_prompt=system,
            user_prompt=prompt,
            tool_registry=self._tool_registry,
            max_steps=5,
            temperature=0.3,
            step_delay=_RPM_DELAY,
        )
        _parse_json_response(text)  # validate parseable
        return text
