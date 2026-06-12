"""Evaluator agent — aligned triples → validated triples (paper Sec. III-A, step 3).

Third of the three KG-construction agents. Scores every aligned triple
with the KARMA 3-signal rubric (confidence M_Con, clarity M_Cla,
relevance M_Rel); the integrate score is their mean and approval is
computed in Python against ``quality_threshold`` (q_TH = 0.9 in the
paper), keeping the Evaluator itself threshold-agnostic. Triples the
LLM flags for re-extraction come back as reprocessing flags that the
pipeline feeds into the feedback loop.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from telcoagent.agents.base import BaseTrainingFreeAgent

from ..llm.config import EnhancedLLMConfig
from ..ontology.core import ThreeGPPOntology
from .prompts import EVALUATOR_SYSTEM, EVALUATOR_USER_TEMPLATE
from .schema import AlignedEntity, ExtractedTriple, ValidatedTriple
from .shared import _RPM_DELAY, _parse_json_response
from .tools import make_evaluator_tools

logger = logging.getLogger(__name__)

_EVAL_MAX_WORKERS = int(os.environ.get("TELCOAGENT_EVAL_WORKERS", "8"))


class EvaluatorAgent:
    """Evaluates and validates aligned triples for quality."""

    def __init__(
        self,
        llm_config: EnhancedLLMConfig,
        quality_threshold: float = 0.9,
        ontology: Optional[ThreeGPPOntology] = None,
    ):
        self.llm = llm_config
        self.quality_threshold = quality_threshold
        self.ontology = ontology
        self._react_agent: Optional[BaseTrainingFreeAgent] = None

        if ontology is not None:
            model = llm_config.agent_models["kg_build"]
            self._react_agent = BaseTrainingFreeAgent(model)

    def evaluate(
        self,
        aligned_pairs: List[Tuple[ExtractedTriple, AlignedEntity]],
        batch_size: int = 75,
    ) -> Tuple[List[ValidatedTriple], List[Dict]]:
        """Evaluate aligned triple pairs for quality.

        Runs batches in parallel via ThreadPoolExecutor.
        Set TELCOAGENT_EVAL_WORKERS=N to tune concurrency (default 8).
        Set TELCOAGENT_EVAL_WORKERS=1 to force sequential mode.

        Returns:
            Tuple of (validated triples, reprocessing flags).
        """
        batches = []
        for i in range(0, len(aligned_pairs), batch_size):
            batches.append((i, aligned_pairs[i : i + batch_size]))

        # Pre-allocate result slots so we can preserve order across parallel completions.
        results_by_offset: Dict[int, List[ValidatedTriple]] = {}
        flags_by_offset: Dict[int, List[Dict]] = {}

        def _run_one(offset: int, batch_pairs):
            batch_flags: List[Dict] = []
            batch_results = self._evaluate_batch(batch_pairs, batch_flags)
            for flag in batch_flags:
                flag["triple_index"] = flag["triple_index"] + offset
            return offset, batch_results, batch_flags

        if _EVAL_MAX_WORKERS <= 1:
            for offset, batch in batches:
                if offset > 0 and _RPM_DELAY > 0:
                    time.sleep(_RPM_DELAY)
                _, br, bf = _run_one(offset, batch)
                results_by_offset[offset] = br
                flags_by_offset[offset] = bf
        else:
            logger.info(
                "Evaluator running %d batches in parallel (workers=%d, batch_size=%d)",
                len(batches),
                _EVAL_MAX_WORKERS,
                batch_size,
            )
            with ThreadPoolExecutor(max_workers=_EVAL_MAX_WORKERS) as pool:
                futures = [pool.submit(_run_one, off, b) for off, b in batches]
                for fut in as_completed(futures):
                    off, br, bf = fut.result()
                    results_by_offset[off] = br
                    flags_by_offset[off] = bf

        validated: List[ValidatedTriple] = []
        all_reprocessing_flags: List[Dict] = []
        for offset, _ in batches:
            validated.extend(results_by_offset.get(offset, []))
            all_reprocessing_flags.extend(flags_by_offset.get(offset, []))

        approved = sum(1 for v in validated if v.approved)
        logger.info(
            f"Evaluated {len(validated)} triples: "
            f"{approved} approved, {len(validated) - approved} rejected"
        )
        if all_reprocessing_flags:
            logger.info(
                f"[Feedback] {len(all_reprocessing_flags)} triples flagged for reprocessing"
            )
        return validated, all_reprocessing_flags

    def _evaluate_batch(
        self,
        pairs: List[Tuple[ExtractedTriple, AlignedEntity]],
        reprocessing_flags: List[Dict],
    ) -> List[ValidatedTriple]:
        """Evaluate a batch of aligned pairs."""
        triples_for_eval = []
        for idx, (triple, alignment) in enumerate(pairs):
            triples_for_eval.append(
                {
                    "index": idx,
                    "subject": triple.subject,
                    "predicate": triple.predicate,
                    "object": triple.object,
                    "triple_type": triple.triple_type,
                    "source_spec": triple.source_spec,
                    "source_clause": triple.source_clause,
                    "aligned_name": alignment.aligned_name,
                    "entity_type": alignment.entity_type,
                    "is_new": alignment.is_new,
                    "mapped_to": alignment.mapped_to,
                }
            )

        # NOTE: Evaluator is intentionally Θ-agnostic.  The LLM scores raw
        # KARMA 3-signal values; downstream code applies ``quality_threshold``
        # purely as a filter.  This keeps a Θ sweep methodologically clean —
        # the same triple yields the same scores regardless of Θ, so cached
        # validated triples can be reused across all Θ variants in the sweep.
        prompt = EVALUATOR_USER_TEMPLATE.format(
            triples_json=json.dumps(triples_for_eval, indent=2),
        )
        system_msg = EVALUATOR_SYSTEM

        response_text = self._call_llm(
            system_msg,
            prompt,
            reprocessing_flags,
        )
        raw_evals = _parse_json_response(response_text)

        eval_map = {}
        for re_ in raw_evals:
            eval_map[re_.get("index", -1)] = re_

        results = []
        for idx, (triple, alignment) in enumerate(pairs):
            ev = eval_map.get(idx, {})
            # KARMA 3-signal: confidence (M_Con), clarity (M_Cla), relevance (M_Rel).
            # Fall back to legacy quality_score if the LLM emitted the old schema.
            legacy = ev.get("quality_score")
            confidence = float(ev.get("confidence", legacy if legacy is not None else 0.5))
            clarity = float(ev.get("clarity", legacy if legacy is not None else 0.5))
            relevance = float(ev.get("relevance", legacy if legacy is not None else 0.5))
            confidence = max(0.0, min(1.0, confidence))
            clarity = max(0.0, min(1.0, clarity))
            relevance = max(0.0, min(1.0, relevance))

            integrate = (confidence + clarity + relevance) / 3.0
            issues = ev.get("issues", [])
            # Θ-agnostic Evaluator: approval is ALWAYS computed in Python
            # from the raw integrate_score, never taken from the LLM.  The
            # 1e-9 epsilon guards FP rounding when the score sits exactly
            # on the threshold (e.g. all signals equal Θ).
            approved = bool(integrate + 1e-9 >= self.quality_threshold)

            results.append(
                ValidatedTriple(
                    triple=triple,
                    alignment=alignment,
                    quality_score=integrate,  # mirror for kg_builder compat
                    issues=issues,
                    approved=approved,
                    confidence=confidence,
                    clarity=clarity,
                    relevance=relevance,
                    integrate_score=integrate,
                )
            )

        return results

    def _call_llm(
        self,
        system: str,
        prompt: str,
        reprocessing_flags: List[Dict],
    ) -> str:
        """Call LLM via ReAct."""
        if not self._react_agent:
            raise RuntimeError("EvaluatorAgent requires a react_agent; pass it to the constructor.")
        if self.ontology is None:
            raise RuntimeError("EvaluatorAgent requires an ontology; pass it to the constructor.")
        tool_registry = make_evaluator_tools(self.ontology, reprocessing_flags)
        text, _ = self._react_agent.react_generate(
            system_prompt=system,
            user_prompt=prompt,
            tool_registry=tool_registry,
            max_steps=5,
            temperature=0.3,
            step_delay=_RPM_DELAY,
        )
        _parse_json_response(text)  # validate parseable
        return text
