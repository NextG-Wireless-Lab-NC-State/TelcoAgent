"""TelcoAgent predictor -- TSFM-only zero-shot forecasting (paper Sec. III-B).

Pipeline (single-pass, no ReAct loop):
    1. TSFM (Chronos-2) single forward pass on raw input KPIs. The model
       internally mean-scales each channel, so no upstream RevIN or STL
       decomposition is applied -- both would discard burst-heavy
       non-stationary signal that Chronos-2 is trained to use directly.
    2. Physical clamp: clip to KPI_NORMALIZATION_MAX, then round integer-type
       KPIs (e.g. RRC_Conn, DL_CQI) to whole numbers. The integer-KPI set is
       read from the in-memory ThreeGPPOntology already on the agent
       (no live Neo4j); no hardcoded integer table.

The runtime forward pass opens no Neo4j connection. KG-grounded reasoning lives in
the ExplainerAgent (``telcoagent.explainer``), which runs as a separate
downstream pass. Neo4j wiring on this class is optional and is used only by
the explainer / offline KG construction, never by ``predict()``.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from telcoagent.config import (
    KPI_NORMALIZATION_MAX,
    PREDICTION_LENGTH_H,
)
from telcoagent.llm.config import EnhancedLLMConfig
from telcoagent.prediction.features import KPI_NAMES

logger = logging.getLogger(__name__)


# =============================================================================
# Refinement helpers — MOVED at Stage-2 commit 2.2a.
# =============================================================================
# The 5 refinement functions (``_parse_adjustments_json``,
# ``_apply_adjustment_gate``, ``_agentic_refine``, ``_agentic_refine_v1``,
# ``_agentic_refine_v2``) now live exclusively in
# ``scripts/baselines/refinement_variants.py``. The 3 context-builder
# helpers (``_summarize_forecast``, ``_summarize_history``,
# ``_build_enriched_context``) remain as ``@staticmethod`` on TelcoAgent
# below — they have no per-instance state, are imported by the moved
# refinement module via ``TelcoAgent._summarize_forecast``, and the
# allowed-direction ``scripts.baselines -> telcoagent`` import is fine
# under the architecture guard.


# =============================================================================
# Main TelcoAgent Class
# =============================================================================


class TelcoAgent:
    """Chronos-2 multivariate TSFM-only predictor.

    The runtime forward pass (:meth:`predict`) is Chronos-2 single forward
    pass + physical clamp; it consumes no KG state. KG-grounded reasoning is
    the Explainer's job. Neo4j integration is optional and is initialised only
    when ``extracted_kg_path`` is provided or an ``ontology_store`` is
    explicitly injected; otherwise an in-memory ontology store is used and no
    live Neo4j connection is required.
    """

    PREDICTION_LENGTH = PREDICTION_LENGTH_H

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        device_map: str = "cuda",
        n_channels: int = 7,
        n_shots: int = 3,
        llm_config: Optional[EnhancedLLMConfig] = None,
        extracted_kg_path: Optional[str] = None,
        ontology_store: Optional[Any] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        enable_explainer: bool = True,
        explainer_model: Optional[str] = None,
    ):
        self.n_shots = n_shots
        self.n_channels = n_channels
        self.llm_config = llm_config or EnhancedLLMConfig()

        import torch
        from chronos import BaseChronosPipeline as Chronos2Pipeline

        logger.info("=" * 60)
        logger.info("Initializing TelcoAgent (Chronos-2)")
        logger.info(f"  Model: {model_id}")
        logger.info(f"  Device: {device_map}")
        logger.info(f"  n_channels: {n_channels}")

        self._chronos_pipeline = Chronos2Pipeline.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.float32,
        )
        self._model_id = model_id
        self._device_map = device_map

        self._explainer_model = explainer_model

        logger.info("=" * 60)

        self._enable_explainer = enable_explainer

        # Neo4j is OPTIONAL and is NEVER built on the TSFM forward-pass path.
        # The runtime predictor reads no KG state, so the default store is an
        # in-memory ontology (no live Neo4j, no connectivity check). Callers
        # that genuinely need a live Neo4j-backed store (the Explainer or
        # offline KG construction) inject one via ``ontology_store=`` — they
        # own the Neo4jConnection lifecycle. The neo4j_uri/user/password params
        # are retained only for backward-compatible call-site signatures and do
        # not trigger a connection here.
        if ontology_store is not None:
            self._ontology_store = ontology_store
        else:
            from telcoagent.stores.in_memory_ontology_store import InMemoryOntologyStore

            self._ontology_store = InMemoryOntologyStore()
            logger.info("In-memory ontology store enabled (no Neo4j)")

        # Extracted KG loading is offline-only; the TSFM predictor does not
        # consume runtime KG merges, so we warn and ignore the path here.
        if extracted_kg_path:
            logger.warning(
                "extracted_kg_path=%s ignored by the TSFM predictor; extracted-KG "
                "merging is offline-only (telcoagent.kg_construction).",
                extracted_kg_path,
            )

    def predict(
        self,
        input_kpis: np.ndarray,
        metadata: Dict[str, Any],
        output_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[str, dict], None]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Predict next 7 days (168h) via pure TSFM (Chronos-2) + physical clamp.

        2-way zero-shot: ``input_kpis`` is the single ~1944h context (days 1-81,
        ``CONTEXT_LENGTH_H``); the predictor forecasts the 168h target (days
        82-88, held out by the caller via the live 2-way split in
        ``scripts/baselines/foundation_utils.py::split_data``). There is no
        separate few-shot demonstration pool on this path.

        TSFM-only, no KG consumption:
            1. TSFM: Chronos-2 single forward pass on raw input (no internal
               normalization applied). The model does its own internal
               mean-scaling per channel and learned its noise model from raw
               5G/IoT KPI distributions, so we pass and receive data in the
               original KPI units (kbps for throughput, % for BLER, count for
               RRC). No external RevIN and no upstream STL decomposition --
               both would discard burst-heavy non-stationary signal that
               Chronos-2 is trained to use directly.
            2. Physical clamp: clip to KPI_NORMALIZATION_MAX, then round
               integer-type KPIs (RRC_Conn, DL_CQI) to whole numbers using the
               ontology-sourced integer-KPI set.

        The only KG read on this path is the integer-KPI flag, taken from the
        in-memory ``self._ontology_store.ontology`` (a ``ThreeGPPOntology``
        from ``get_default_ontology()``). No live Neo4j is opened, so this runs
        with Neo4j unreachable. No causal chains / directional rules are read.

        Pre-2.2b this method also ran an LLM 1-shot refinement step
        (``_agentic_refine``); per Stage-2 B-CRIT-1 that step has been
        moved to ``scripts/baselines/refinement_variants.py`` as an
        opt-in ablation harness. The paper-canonical predictor is
        TSFM-only.

        Does NOT generate a report -- that is the Explainer Agent's job.
        """
        t0 = time.time()
        metadata.get("station_id", "")  # documented metadata key; retained for caller contract

        if progress_callback:
            progress_callback("status", {"phase": "init", "message": "Input window loaded"})

        # --- 1. TSFM: single Chronos-2 forward pass on raw input ---
        if progress_callback:
            progress_callback("status", {"phase": "tsfm", "message": "Chronos-2 forward pass"})
        predict_fn = self._make_predict_fn(input_kpis)
        raw_pred = predict_fn(input_kpis)[:PREDICTION_LENGTH_H]
        logger.info("Chronos-2 raw forecast computed: shape=%s", raw_pred.shape)

        # --- 2. Physical clamp (KPI_NORMALIZATION_MAX + integer flags) ---
        final_pred = self._apply_physical_clamp(raw_pred)

        total_elapsed = time.time() - t0
        result_info: Dict[str, Any] = {
            "model": self._model_id,
            "device": self._device_map,
            "prediction_raw_tsfm": raw_pred,
            "total_elapsed_sec": round(total_elapsed, 2),
        }
        return final_pred, result_info

    # ------------------------------------------------------------------
    # Helpers for the TSFM + KG + OSM pipeline
    # ------------------------------------------------------------------

    def _fetch_osm_context(
        self,
        station_id: str,
        lat: Optional[float],
        lon: Optional[float],
    ) -> str:
        """Return OSM spatial-context summary."""
        osm = getattr(self, "_osm", None)
        if osm is None:
            raise RuntimeError(
                "TelcoAgent._osm is not set; assign an OpenStreetMapMCP instance "
                "to self._osm before calling predict()."
            )
        if lat is None or lon is None:
            return osm.get_spatial_context(station_id)
        return osm.get_spatial_context(station_id, lat=float(lat), lon=float(lon))

    def _format_kg_context(self, kpi_names: List[str]) -> str:
        """Render the 3GPP KG context (KPI defs + ranges + causal chains).

        For use by ExplainerAgent only; NOT called by ``predict()``. The TSFM
        forward path consumes no KG state.
        """
        ontology = getattr(self._ontology_store, "ontology", None)
        if ontology is None:
            raise RuntimeError(
                "ontology_store has no ontology; Neo4jOntologyStore must be "
                "initialised with an explicit ThreeGPPOntology instance."
            )

        parts: List[str] = ["3GPP KPI knowledge graph:"]
        for kpi in kpi_names:
            defn = ontology.get_kpi(kpi) if hasattr(ontology, "get_kpi") else None
            cap = KPI_NORMALIZATION_MAX.get(kpi)
            if defn is not None:
                rng = getattr(defn, "valid_range", None)
                desc = (getattr(defn, "description", "") or "").strip()
                unit = getattr(defn, "unit", "") or ""
                parts.append(
                    f"  - {kpi} [{unit}]: {desc} " f"(valid_range={rng}, normalization_cap={cap})"
                )
            else:
                parts.append(f"  - {kpi}: (no KG definition; cap={cap})")

        chains = getattr(ontology, "causal_chains", None) or []
        if chains:
            parts.append("Causal chains (cross-KPI):")
            for c in list(chains)[:8]:
                parts.append(f"  - {c}")
        return "\n".join(parts)

    @staticmethod
    def _summarize_forecast(pred: np.ndarray, kpi_names: List[str]) -> str:
        lines: List[str] = []
        C = pred.shape[1]
        for i in range(min(C, len(kpi_names))):
            col = pred[:, i]
            lines.append(
                f"  {kpi_names[i]:12s}: mean={col.mean():.3f} "
                f"min={col.min():.3f} max={col.max():.3f} "
                f"first24h_mean={col[:24].mean():.3f} "
                f"last24h_mean={col[-24:].mean():.3f}"
            )
        return "\n".join(lines)

    @staticmethod
    def _summarize_history(hist: np.ndarray, kpi_names: List[str]) -> str:
        last_168 = hist[-168:]
        lines: List[str] = []
        C = last_168.shape[1]
        for i in range(min(C, len(kpi_names))):
            col = last_168[:, i]
            lines.append(
                f"  {kpi_names[i]:12s}: mean={col.mean():.3f} std={col.std():.3f} "
                f"last24h_mean={col[-24:].mean():.3f}"
            )
        return "\n".join(lines)

    def _apply_physical_clamp(self, pred: np.ndarray) -> np.ndarray:
        """Clamp to [0, KPI_NORMALIZATION_MAX[kpi]] per-channel, then round
        integer-type KPIs to whole numbers.

        Delegates to ``apply_physical_clamp_strict`` for the ``[0, cap]`` clip
        (KPI_NORMALIZATION_MAX, shared by every baseline script — ADR 0001 /
        B-MED-3). The integer-KPI set (e.g. RRC_Conn, DL_CQI) is sourced from
        the in-memory ``ThreeGPPOntology`` already on this agent
        (``self._ontology_store.ontology.integer_kpis``); it is never a
        hardcoded table (principle 2). This is a pure read of in-memory KG
        metadata and requires no live Neo4j (task 07).
        """
        from telcoagent.prediction.physical_clamp import apply_physical_clamp_strict

        integer_kpi_set = self._ontology_store.ontology.integer_kpis(list(KPI_NAMES))
        return apply_physical_clamp_strict(pred, list(KPI_NAMES), integer_kpi_set=integer_kpi_set)

    def _make_predict_fn(self, input_data: np.ndarray):
        """Create a Chronos-2 predict function: (L, C) -> (168, C).

        Slices the input to the last ``CONTEXT_LENGTH_H`` hours before
        feeding Chronos-2 — matches the slicing in
        ``scripts/run_ablation.py::make_chronos2_predict_fn`` so the agent
        path (full / tsfm_agent) and the raw baseline (tsfm) call the
        model with bit-identical context windows.
        """
        import torch

        from telcoagent.config import CONTEXT_LENGTH_H

        pipeline = self._chronos_pipeline

        def predict_fn(data: np.ndarray) -> np.ndarray:
            ctx = data[-CONTEXT_LENGTH_H:] if data.shape[0] > CONTEXT_LENGTH_H else data
            input_tensor = torch.tensor(
                ctx.T[np.newaxis, :, :],
                dtype=torch.float32,
            )  # (1, C, L)
            output = pipeline.predict(input_tensor, prediction_length=self.PREDICTION_LENGTH)
            pred_samples = output[0]  # (C, n_samples, pred_len)
            pred_median = torch.median(pred_samples, dim=1).values.numpy()  # (C, pred_len)
            return pred_median.T  # (pred_len, C)

        return predict_fn

    # ------------------------------------------------------------------
    # Enriched refinement helpers (v1 / v2)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_enriched_context(raw_pred: np.ndarray, input_kpis: np.ndarray) -> str:
        """Return a rich context string for the LLM refinement prompt.

        Includes:
        - Per-KPI coefficient of variation (recent 14d) as uncertainty proxy
        - Average 24h diurnal fingerprint from the last 14d of input
        - Last 7d daily averages from input history
        - Forecast 7d daily averages
        """
        lines: List[str] = ["=== Enriched forecast context ==="]
        C = min(raw_pred.shape[1], len(KPI_NAMES))
        hist = input_kpis

        lines.append("\n[Per-KPI uncertainty (CV = std/mean, last 14d of input)]")
        recent = hist[-336:] if hist.shape[0] >= 336 else hist
        for i in range(C):
            col = recent[:, i]
            mean_v = float(np.mean(np.abs(col))) + 1e-8
            cv = float(np.std(col)) / mean_v
            lines.append(
                f"  {KPI_NAMES[i]:14s}  CV={cv:.3f}  "
                f"({'HIGH uncertainty' if cv > 0.3 else 'moderate' if cv > 0.1 else 'LOW uncertainty'})"
            )

        lines.append("\n[Diurnal fingerprint — avg 24h shape from last 14d of input]")
        n_days = min(14, hist.shape[0] // 24)
        if n_days >= 1:
            block = hist[-(n_days * 24) :]
            for i in range(C):
                mat = block[:, i].reshape(n_days, 24)
                avg = mat.mean(axis=0)
                peak_h = int(np.argmax(avg))
                trough_h = int(np.argmin(avg))
                vals = " ".join(f"{v:.2f}" for v in avg)
                lines.append(
                    f"  {KPI_NAMES[i]:14s}  peak=H{peak_h:02d}  trough=H{trough_h:02d}  [{vals}]"
                )

        lines.append("\n[Input history — last 7d daily averages]")
        for d in range(min(7, hist.shape[0] // 24)):
            day_data = hist[-(7 - d) * 24 : -(6 - d) * 24 if d < 6 else None]
            vals = "  ".join(f"{KPI_NAMES[i]}={day_data[:, i].mean():.2f}" for i in range(C))
            lines.append(f"  Day-{7 - d}: {vals}")

        lines.append("\n[Raw Chronos-2 forecast — 7d daily averages]")
        for d in range(7):
            day_data = raw_pred[d * 24 : (d + 1) * 24]
            vals = "  ".join(f"{KPI_NAMES[i]}={day_data[:, i].mean():.2f}" for i in range(C))
            lines.append(f"  FDay+{d + 1}: {vals}")

        lines.append("\n[Forecast drift vs recent 7d trend]")
        last_7d = hist[-168:] if hist.shape[0] >= 168 else hist
        for i in range(C):
            fc_mean = float(np.mean(raw_pred[:, i]))
            hist_mean = float(np.mean(last_7d[:, i]))
            ratio = fc_mean / max(hist_mean, 1e-6)
            if ratio < 0.85:
                label = "much lower"
            elif ratio < 0.95:
                label = "lower"
            elif ratio <= 1.05:
                label = "consistent"
            elif ratio <= 1.15:
                label = "higher"
            else:
                label = "much higher"
            lines.append(
                f"  {KPI_NAMES[i]:14s}  forecast_7d_mean={fc_mean:.2f}"
                f"  last_7d_mean={hist_mean:.2f}  ratio={ratio:.3f}  ({label})"
            )

        return "\n".join(lines)
