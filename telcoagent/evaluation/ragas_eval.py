"""LEGACY RAGAS-library evaluator (kept for the post-deadline ablation
runner ``scripts/run_explainer_kg_ablation.py``).

The active evaluation path used by the main pipeline is
:class:`telcoagent.evaluation.explanation_metrics.ExplanationEvaluator`,
which implements Faithfulness and the anomaly-aware Answer Relevancy
directly from the paper's §IV-A4 definitions and uses ORANSight as the
judge — decoupled from the GPT-4o-mini explainer to remove same-family
bias. This file is retained only so the existing ablation script does
not break; new code should not import :class:`TelcoAgentEvaluator`.

The two metrics computed here are exactly those defined in the RAGAS
framework (Es et al., ACL 2024 Findings):

  - Faithfulness    : claims in the report are grounded in the retrieved 3GPP context
  - Answer Relevancy: reverse-question similarity to the forecast query

Mapping to TelcoAgent components
---------------------------------
  user_input (q)  → "Forecast and explain the next 7 days of KPIs for {station_id}"
  context  c(q)   → texts in explainer_state["retrieved_contexts"]
                     (PAX-TS sensitivity context + GraphRAG + spatial + temporal)
  response a_s(q) → Explainer Agent report string (ExplanationResult.report)

Reference
---------
  Es et al. (2024). RAGAS: Automated Evaluation of Retrieval Augmented Generation.
  Findings of ACL 2024. https://aclanthology.org/2024.findings-eacl.153
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from telcoagent.config import CORE_KPI_NAMES, PREDICTION_LENGTH_H

logger = logging.getLogger(__name__)


@dataclass
class RAGASScores:
    """Official RAGAS scores (Es et al., ACL 2024) for a single TelcoAgent inference."""

    station_id: str
    faithfulness: float  # [0,1] — claims grounded in retrieved 3GPP context
    answer_relevancy: float  # [0,1] — report addresses the forecast task
    n_contexts: int = 0  # number of retrieved context chunks
    error: Optional[str] = None

    @property
    def ragas_score(self) -> float:
        """Harmonic mean of faithfulness and answer_relevancy."""
        vals = [self.faithfulness, self.answer_relevancy]
        if any(v <= 0 for v in vals):
            return 0.0
        return len(vals) / sum(1.0 / v for v in vals)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station_id": self.station_id,
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "ragas_score": round(self.ragas_score, 4),
            "n_contexts": self.n_contexts,
            "error": self.error,
        }


class TelcoAgentEvaluator:
    """RAGAS evaluator for TelcoAgent reports.

    Uses the two official reference-free RAGAS 0.4 metrics:
      - faithfulness (LLM-based NLI decomposition)
      - answer_relevancy (LLM-based question generation)

    Requires:
        pip install ragas langchain-openai
    """

    def evaluate_single(
        self,
        station_id: str,
        report: str,
        retrieved_contexts: List[str],
    ) -> RAGASScores:
        """Evaluate a single TelcoAgent inference.

        Args:
            station_id: Base station identifier.
            report: Full TelcoAgent report string.
            retrieved_contexts: Retrieved 3GPP context strings accumulated
                during the ReAct loop (domain knowledge + causal chains + spatial).

        Returns:
            RAGASScores with official faithfulness and answer_relevancy.
        """
        kpi_list = ", ".join(CORE_KPI_NAMES[:-1]) + f", and {CORE_KPI_NAMES[-1]}"
        question = (
            f"Forecast and explain the next 7 days ({PREDICTION_LENGTH_H} hours) "
            f"of 5G NR KPI behaviour for base station {station_id}. "
            f"Provide predicted trends for {kpi_list}, with 3GPP-grounded causal "
            f"analysis and actionable network engineering recommendations."
        )

        if not retrieved_contexts:
            logger.warning("No retrieved contexts for %s — faithfulness may be low", station_id)

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_openai import ChatOpenAI
            from ragas import EvaluationDataset, SingleTurnSample, evaluate
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import answer_relevancy, faithfulness

            # Legacy RAGAS judge configuration. The active explanation
            # evaluation path (`telcoagent.evaluation.explanation_metrics`)
            # uses ORANSight via vLLM and decouples the judge from the
            # GPT-4o-mini explainer; this file is preserved only for the
            # post-deadline ablation script. temperature=0 keeps NLI
            # decomposition deterministic across re-runs.
            _llm = LangchainLLMWrapper(
                ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=4096),
            )
            _emb = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
            )
            faithfulness.llm = _llm
            answer_relevancy.llm = _llm
            answer_relevancy.embeddings = _emb

            contexts = retrieved_contexts or ["No context retrieved."]
            dataset = EvaluationDataset(
                samples=[
                    SingleTurnSample(
                        user_input=question,
                        response=report,
                        retrieved_contexts=contexts,
                    )
                ]
            )

            result = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy])

            def _scalar(val) -> float:
                import numpy as np

                if isinstance(val, (list, tuple)):
                    finite = [float(v) for v in val if v is not None and not np.isnan(float(v))]
                    return float(np.mean(finite)) if finite else float("nan")
                return float(val)

            faith = _scalar(result["faithfulness"])
            ans_rel = _scalar(result["answer_relevancy"])

            logger.info(
                "[RAGAS] %s: faithfulness=%.4f, answer_relevancy=%.4f, ragas_score=%.4f",
                station_id,
                faith,
                ans_rel,
                (2 / (1 / faith + 1 / ans_rel)) if faith > 0 and ans_rel > 0 else 0.0,
            )
            return RAGASScores(
                station_id=station_id,
                faithfulness=faith,
                answer_relevancy=ans_rel,
                n_contexts=len(retrieved_contexts),
            )

        except Exception as exc:
            logger.error("RAGAS evaluation failed for %s: %s", station_id, exc)
            return RAGASScores(
                station_id=station_id,
                faithfulness=float("nan"),
                answer_relevancy=float("nan"),
                n_contexts=len(retrieved_contexts),
                error=str(exc),
            )

    def evaluate_batch(self, records: List[Dict[str, Any]]):
        """Evaluate multiple stations. Returns a pandas DataFrame."""
        import pandas as pd

        results = []
        for rec in records:
            scores = self.evaluate_single(
                station_id=rec["station_id"],
                report=rec["report"],
                retrieved_contexts=rec.get("retrieved_contexts", []),
            )
            results.append(scores.to_dict())

        df = pd.DataFrame(results)
        logger.info(
            "RAGAS batch: %d stations | mean faithfulness=%.3f | mean answer_relevancy=%.3f | mean ragas_score=%.3f",
            len(df),
            df["faithfulness"].mean(),
            df["answer_relevancy"].mean(),
            df["ragas_score"].mean(),
        )
        return df


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------


def record_from_result(
    station_id: str,
    report: str,
    retrieved_contexts: List[str],
) -> Dict[str, Any]:
    """Build a RAGAS evaluation record from Explainer Agent output."""
    return {
        "station_id": station_id,
        "report": report,
        "retrieved_contexts": retrieved_contexts,
    }
