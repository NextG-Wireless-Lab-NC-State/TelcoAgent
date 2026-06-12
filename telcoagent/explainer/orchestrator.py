"""Explanation pipeline orchestrator (paper Sec. III-C).

:class:`TelcoAgentExplainer` wires the full explain pass end-to-end:

    ① deterministic anomaly detection on the forecast horizon
      (:mod:`telcoagent.explainer.anomaly_detector`),
    ② KG-grounded cause inference + OSM environmental context
      (:mod:`telcoagent.explainer.cause_inference`),
    ③ optional PAX-TS sensitivity attribution
      (:mod:`telcoagent.explainer.pax_ts`),
    ④ LLM report authoring through the ReAct
      :class:`telcoagent.explainer.react_agent.ExplainerAgent`,
    ⑤ deterministic post-processing + numerical-fidelity audit
      (:mod:`telcoagent.explainer.report`,
      :mod:`telcoagent.explainer.audit`).

:class:`ExplanationResult` is the output bundle handed back to callers
(and to RAGAS scoring in :mod:`telcoagent.evaluation`).
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from telcoagent.explainer.anomaly_detector import detect_anomaly_events
from telcoagent.explainer.anomaly_plot import plot_forecast_with_anomalies
from telcoagent.explainer.cause_inference import infer_anomaly_causes
from telcoagent.explainer.evidence import _format_structured_context, _prestuff_evidence
from telcoagent.explainer.html_report import render_html_report
from telcoagent.explainer.react_agent import ExplainerAgent
from telcoagent.explainer.report import (
    daily_means,
    dedupe_contexts,
    ensure_figures_embedded,
    format_plot_catalogue,
    inject_forecast_contexts,
    linear_slope,
    strip_preamble,
)

logger = logging.getLogger(__name__)


# ─── Result container ─────────────────────────────────────────────────────


@dataclass
class ExplanationResult:
    """Output bundle returned by :meth:`TelcoAgentExplainer.explain`.

    Attributes
    ----------
    anomaly_events
        Detected anomaly events on the forecast horizon.
    structured_context
        Compact pre-prompt anomaly snapshot fed to the ReAct agent.
    output_paths
        ``{name: path}`` mapping for serialised artefacts (CSVs, plots,
        report markdown / HTML).
    inference_time, llm_calls
        Wall-clock and LLM-call metrics for the explain pass.
    report, retrieved_contexts
        Generated markdown report and the accumulated tool-output
        contexts used for downstream Faithfulness evaluation.
    sensitivity_result
        Optional :class:`telcoagent.explainer.pax_ts.SensitivityResult`
        attached when the caller passed a non-None
        ``sensitivity_pipeline`` to :meth:`TelcoAgentExplainer.explain`.
        ``None`` indicates that PAX-TS sensitivity computation was
        deliberately deferred (typically a fast-path explain run; the
        post-deadline sensitivity batch script populates this field
        offline). The two ``[Sensitivity]`` ReAct tools degrade
        gracefully — they return an explicit error payload — when
        this field is None.
    """

    anomaly_events: List[Dict[str, Any]]
    structured_context: str
    output_paths: Dict[str, str] = field(default_factory=dict)
    inference_time: float = 0.0
    llm_calls: int = 0
    report: str = ""
    retrieved_contexts: List[str] = field(default_factory=list)
    sensitivity_result: Optional[Any] = None


# ─── TelcoAgentExplainer ──────────────────────────────────────────────────


_HISTORY_BASELINE_HOURS: int = 168


class TelcoAgentExplainer:
    """End-to-end forecast explainer (anomaly detection + LLM report).

    Usage
    -----
    >>> explainer = TelcoAgentExplainer(
    ...     ontology_store=ontology_store,
    ...     osm=osm,
    ...     model_name="openai/gpt-4o-mini",
    ...     station_id=station_id,
    ... )
    >>> result = explainer.explain(
    ...     forecast=forecast,           # (168, 7)
    ...     input_history=input_window,  # (168+, 7) for baseline
    ...     prediction=prediction_dict,  # {kpi_name: list[float]}
    ...     output_dir="./out",
    ...     lat=lat, lon=lon,
    ... )
    """

    def __init__(
        self,
        ontology_store: Any = None,
        osm: Any = None,
        model_name: str = "",
        station_id: str = "",
        kpi_names: Optional[List[str]] = None,
        relevance_model: Optional[str] = "openai/gpt-4o-mini",
        sensitivity_pipeline: Any = None,
        sensitivity_chunk_size: int = 0,
        sensitivity_result: Optional[Any] = None,
    ) -> None:
        """Construct the end-to-end explanation pipeline.

        Parameters
        ----------
        ontology_store, osm
            3GPP knowledge graph store and OSM MCP client. ``ontology_store``
            may be None during the KG-ablation arm (``ablate_kg=True``) of
            the explanation evaluation.
        model_name, station_id, kpi_names
            Identifiers for the underlying ReAct LLM, the cell, and the
            ordered KPI vocabulary.
        relevance_model
            Optional secondary LLM used by the OSM-relevance harness in
            :func:`telcoagent.explainer.cause_inference.infer_anomaly_causes`.
            Defaults to ``"openai/gpt-4o-mini"``; set to None to disable
            the harness.
        sensitivity_pipeline
            Optional Chronos-2 pipeline reference. When provided,
            :meth:`explain` runs PAX-TS perturbation analysis to populate
            the ``[Sensitivity]`` evidence channel. When None (the default,
            and the fast-path used during interactive runs), sensitivity
            is omitted from the report; the post-deadline batch script
            ``scripts/compute_sensitivity_batch.py`` can populate the
            sensitivity matrix offline and re-attach it to the saved
            :class:`ExplanationResult` for paper-grade evaluation.
        sensitivity_chunk_size
            ``chunk_size`` argument forwarded to
            :func:`telcoagent.explainer.pax_ts.compute_sensitivity`.  0
            (default) requests a single batched ``Chronos2Pipeline.predict``
            call; positive values trade latency for VRAM headroom.
        """
        from telcoagent.context.mcp_tools import OpenStreetMapMCP
        from telcoagent.prediction.features import KPI_NAMES as _DEFAULT_KPI_NAMES

        self._kpi_names = kpi_names or list(_DEFAULT_KPI_NAMES)
        self._model_name = model_name
        self._station_id = station_id
        self._ontology_store = ontology_store
        self._osm = osm or OpenStreetMapMCP()
        self._relevance_model = relevance_model
        self._sensitivity_pipeline = sensitivity_pipeline
        self._sensitivity_chunk_size = sensitivity_chunk_size
        self._precomputed_sensitivity_result = sensitivity_result
        self._cached_input_history: Optional[np.ndarray] = None
        if sensitivity_result is not None:
            sens_status = "precomputed (offline)"
        elif sensitivity_pipeline is not None:
            sens_status = "online (pipeline-driven)"
        else:
            sens_status = "deferred"
        logger.info(
            "TelcoAgentExplainer initialised (station=%s, model=%s, "
            "relevance_harness=%s, sensitivity=%s)",
            station_id,
            model_name or "<unset>",
            relevance_model or "<disabled>",
            sens_status,
        )

    # ── Main entry point ─────────────────────────────────────────────────

    def explain(
        self,
        forecast: np.ndarray,
        input_history: np.ndarray,
        prediction: Optional[Dict[str, List[float]]] = None,
        output_dir: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        forecast_start_weekday: int = 0,
        forecast_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        progress_callback: Optional[Callable[[str, dict], None]] = None,
        ablate_kg: bool = False,
        ablate_react: bool = False,
        enabled_tools: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        user_prompt_prefix: Optional[str] = None,
    ) -> ExplanationResult:
        """Run anomaly detection + report generation.

        Parameters
        ----------
        forecast
            Predicted KPI series of shape ``(168, C)``.
        input_history
            Recent observed window of shape ``(T_baseline, C)``;
            its last ``168`` rows are used as the anomaly baseline.
        prediction
            Per-KPI forecast dict; if omitted, derived from ``forecast``.
        output_dir
            If given, CSV artefacts and ``explanation.md`` are written.
        lat, lon
            Station coordinates; passed to OSM for spatial context.
        forecast_start_weekday
            Weekday index (0 = Monday) of forecast hour 0.
        forecast_metrics
            Optional per-KPI evaluation metrics computed against the
            held-out test target, e.g.
            ``{"RRC_Conn": {"sMAPE": 4.21, "MASE": 0.62, ...}, ...}``.
            These appear ONLY in the §1 Forecast Summary table — the
            single section of the report where ground-truth–derived
            numbers are permitted, for transparency about model
            reliability. The Explainer Agent is instructed never to
            propagate them into §2–§5.
        progress_callback
            Optional ``callback(phase: str, payload: dict)`` for UI hooks.
        """
        if forecast.ndim != 2:
            raise ValueError(f"forecast must be 2D (T, C); got shape {forecast.shape}")

        start_time = time.time()
        if lat is not None and lon is not None:
            self._osm.register_station(self._station_id, lat, lon)
        self._cached_input_history = input_history

        # Step 1: Anomaly events.
        if progress_callback:
            progress_callback(
                "explainer_progress",
                {
                    "phase": "anomaly_detection",
                    "percent": 0,
                },
            )
        baseline = input_history[-_HISTORY_BASELINE_HOURS:]
        events = detect_anomaly_events(
            forecast=forecast,
            baseline=baseline,
            kpi_names=self._kpi_names,
        )
        # ``ablate_kg=True`` forces every KG-derived signal off:
        # candidate_causes become [], query_graphrag_mechanism returns
        # an "error" payload, and the ReAct agent has to lean on OSM +
        # raw forecast numbers alone. OSM environmental_context is
        # preserved because OSM is independent of the 3GPP KG.
        ontology = None if ablate_kg else self._resolve_ontology()

        if ontology is None and not ablate_kg:
            raise RuntimeError(
                "TelcoAgentExplainer has no ontology; attach an ontology_store "
                "with a valid ThreeGPPOntology before calling explain(), or "
                "pass ablate_kg=True for the KG-off comparison path."
            )

        # Fetch OSM spatial context once so it can be folded into per-event
        # environmental_context evidence inside infer_anomaly_causes.
        osm_context: str = self._osm.get_spatial_context(self._station_id)

        anomaly_dicts = infer_anomaly_causes(
            events,
            ontology=ontology,
            forecast=forecast,
            kpi_names=self._kpi_names,
            baseline=baseline,
            window_hours=6,
            osm_context=osm_context,
            relevance_model=self._relevance_model,
        )

        # Step 1.5: PAX-TS sensitivity analysis.
        #
        # Three sources are supported, in priority order:
        #   1. ``sensitivity_result`` passed to __init__: a previously
        #      computed :class:`SensitivityResult` (e.g. from the offline
        #      batch runner ``scripts/run_pax_ts_batch.py``). This is the
        #      production path — explainer reuses cached PAX-TS so the
        #      ReAct agent can cite ``[Sensitivity]`` evidence in §2 and
        #      §4 without re-running the (expensive) Chronos-2 batch.
        #   2. ``sensitivity_pipeline`` passed to __init__: a live
        #      Chronos-2 pipeline. We compute sensitivity online.
        #   3. Neither: sensitivity is deferred. The two ``[Sensitivity]``
        #      ReAct tools return an explicit error payload and the LLM
        #      is instructed (via :data:`telcoagent.llm.prompts`) not to
        #      fabricate sensitivity numbers.
        sensitivity_result = self._precomputed_sensitivity_result
        if sensitivity_result is None and self._sensitivity_pipeline is not None:
            from telcoagent.config import KPI_NORMALIZATION_MAX
            from telcoagent.explainer.pax_ts import compute_sensitivity

            if progress_callback:
                progress_callback(
                    "explainer_progress",
                    {
                        "phase": "sensitivity",
                        "percent": 30,
                    },
                )
            sensitivity_result = compute_sensitivity(
                pipeline=self._sensitivity_pipeline,
                input_kpis=input_history,
                baseline_forecast=forecast,
                anomaly_events=anomaly_dicts,
                kpi_names=self._kpi_names,
                cap_table=KPI_NORMALIZATION_MAX,
                chunk_size=self._sensitivity_chunk_size,
            )

        # Step 2: Structured context for ReAct prompt.
        structured_context = _format_structured_context(
            anomaly_dicts,
            self._kpi_names,
            forecast_start_weekday,
        )

        # Step 3a: CSV artefacts.
        output_paths: Dict[str, str] = {}
        if output_dir:
            output_paths.update(
                self._write_csv_artefacts(
                    output_dir,
                    anomaly_dicts,
                )
            )

        # Step 3b: Anomaly visualisation (per-KPI + overview PNGs).
        anomaly_plot_paths: Dict[str, str] = {}
        if output_dir:
            anomaly_plot_paths = plot_forecast_with_anomalies(
                forecast=forecast,
                baseline=baseline,
                kpi_names=self._kpi_names,
                anomaly_events=anomaly_dicts,
                output_dir=str(Path(output_dir) / "plots"),
                station_id=self._station_id,
                valid_ranges=self._kpi_valid_ranges(),
            )
            for key, path in anomaly_plot_paths.items():
                output_paths[f"plot_{key}"] = path

        # Step 4: LLM report.
        if progress_callback:
            progress_callback(
                "explainer_progress",
                {
                    "phase": "report_generation",
                    "percent": 60,
                },
            )
        report = ""
        retrieved_contexts: List[str] = []
        llm_calls = 0
        if self._model_name:
            report, retrieved_contexts = self._generate_report(
                structured_context=structured_context,
                anomaly_events=anomaly_dicts,
                prediction=prediction,
                forecast_start_weekday=forecast_start_weekday,
                forecast_metrics=forecast_metrics,
                anomaly_plot_paths=anomaly_plot_paths,
                ablate_kg=ablate_kg,
                ablate_react=ablate_react,
                sensitivity_result=sensitivity_result,
                enabled_tools=enabled_tools,
                system_prompt=system_prompt,
                user_prompt_prefix=user_prompt_prefix,
            )
            llm_calls = 1

        # Step 4.5: Persist sensitivity artefacts when available.
        if output_dir and sensitivity_result is not None:
            from telcoagent.explainer.sensitivity_format import (
                write_sensitivity_artefacts,
            )

            try:
                output_paths.update(
                    write_sensitivity_artefacts(
                        out_dir=str(output_dir),
                        result=sensitivity_result,
                        kpi_names=self._kpi_names,
                        anomaly_events=anomaly_dicts,
                    )
                )
            except (ValueError, FileNotFoundError) as exc:
                # Validation errors at the artefact-writing boundary are
                # surfaced loudly rather than silently dropped (CLAUDE.md
                # "no fallback code"): the sensitivity result itself is
                # still attached to the returned ExplanationResult, but
                # we re-raise to flag a malformed write request.
                logger.error(
                    "write_sensitivity_artefacts failed: %s",
                    exc,
                )
                raise

        elapsed = time.time() - start_time
        logger.info(
            "Explanation completed in %.2fs (anomaly + %d LLM call%s)",
            elapsed,
            llm_calls,
            "" if llm_calls == 1 else "s",
        )

        if output_dir and report:
            report_path = Path(output_dir) / "explanation.md"
            report_path.write_text(report, encoding="utf-8")
            output_paths["report"] = str(report_path)
            logger.info("Saved report: %s", report_path)

            # Dark-theme HTML mirror — used as the paper-figure source.
            html_path = Path(output_dir) / "explanation.html"
            render_html_report(
                markdown_text=report,
                output_path=str(html_path),
                title=f"TelcoAgent Explanation — {self._station_id}",
            )
            output_paths["report_html"] = str(html_path)

        return ExplanationResult(
            anomaly_events=anomaly_dicts,
            structured_context=structured_context,
            output_paths=output_paths,
            inference_time=elapsed,
            llm_calls=llm_calls,
            report=report,
            retrieved_contexts=retrieved_contexts,
            sensitivity_result=sensitivity_result,
        )

    # ── CSV writers ──────────────────────────────────────────────────────

    def _write_csv_artefacts(
        self,
        output_dir: str,
        anomaly_events: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, str] = {}

        if anomaly_events:
            anomaly_path = out / "anomaly_events.csv"
            with anomaly_path.open("w", newline="") as handle:
                fieldnames = list(anomaly_events[0].keys())
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for event in anomaly_events:
                    serialised = dict(event)
                    serialised["co_occurring_kpis"] = ", ".join(
                        serialised.get("co_occurring_kpis", []) or []
                    )
                    writer.writerow(serialised)
            paths["anomaly_events"] = str(anomaly_path)

        return paths

    # ── Report generation ───────────────────────────────────────────────

    def _generate_report(
        self,
        structured_context: str,
        anomaly_events: List[Dict[str, Any]],
        prediction: Optional[Dict[str, List[float]]],
        forecast_start_weekday: int,
        forecast_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        anomaly_plot_paths: Optional[Dict[str, str]] = None,
        ablate_kg: bool = False,
        ablate_react: bool = False,
        sensitivity_result: Optional[Any] = None,
        enabled_tools: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        user_prompt_prefix: Optional[str] = None,
    ) -> tuple[str, List[str]]:
        from telcoagent.explainer.tools import (
            ExplainerToolsConfig,
            make_explainer_tools,
        )

        prediction_summary, baseline_means = self._summarise_prediction(
            prediction,
            forecast_metrics,
        )
        plot_catalogue = format_plot_catalogue(anomaly_plot_paths) if anomaly_plot_paths else ""
        input_baseline = self._build_input_baseline()

        cfg = ExplainerToolsConfig(
            anomaly_events=anomaly_events,
            prediction=prediction,
            ontology_store=None if ablate_kg else self._ontology_store,
            osm=self._osm,
            station_id=self._station_id,
            kpi_names=self._kpi_names,
            input_baseline=input_baseline,
            forecast_start_weekday=forecast_start_weekday,
            sensitivity_result=sensitivity_result,
            enabled_tools=enabled_tools,
        )
        tool_registry, agent_state = make_explainer_tools(cfg)

        # In single-shot mode every tool that the ReAct loop would have
        # called is pre-executed once so the LLM sees the same evidence
        # via the user prompt. The tool registry shares its
        # ``retrieved_contexts`` state so RAGAS auditing downstream
        # observes the same context blob in both modes.
        evidence_block = (
            _prestuff_evidence(tool_registry, anomaly_events, self._kpi_names)
            if ablate_react
            else ""
        )

        agent = ExplainerAgent(model_name=self._model_name, system_prompt=system_prompt)
        report = agent.explain_multi_turn(
            tool_registry=tool_registry,
            prediction_summary=prediction_summary,
            evidence_block=evidence_block,
            structured_context=structured_context,
            plot_catalogue=plot_catalogue,
            ablate_react=ablate_react,
            user_prompt_prefix=user_prompt_prefix,
        )
        report = strip_preamble(report)
        report = ensure_figures_embedded(report, anomaly_plot_paths)
        report = self._append_fidelity_footer(
            report=report,
            prediction=prediction,
            baseline_means=baseline_means,
            forecast_metrics=forecast_metrics,
            anomaly_events=anomaly_events,
            sensitivity_result=sensitivity_result,
        )

        contexts = agent_state.get("retrieved_contexts", [])
        inject_forecast_contexts(contexts, prediction_summary)
        self._inject_structural_contexts(contexts, prediction, baseline_means)
        return report, dedupe_contexts(contexts)

    def _append_fidelity_footer(
        self,
        *,
        report: str,
        prediction: Optional[Dict[str, List[float]]],
        baseline_means: Dict[str, float],
        forecast_metrics: Optional[Dict[str, Dict[str, float]]],
        anomaly_events: List[Dict[str, Any]],
        sensitivity_result: Optional[Any],
    ) -> str:
        """Append the deterministic §E Numerical Fidelity Check.

        The audit re-derives §1 forecast statistics from the prediction
        tensor and §4 ``|z|`` claims from the anomaly-event objects,
        then renders a paper-grade table with an ``N/M match`` footer.
        Audit failures are surfaced loudly: the footer always appends —
        if the audit cannot run (e.g. no prediction supplied) the
        section explains why instead of silently dropping.
        """
        from telcoagent.explainer.audit import (
            audit_numerical_claims,
            audit_sensitivity_consistency,
            format_fidelity_footer,
        )

        numerical = audit_numerical_claims(
            report=report,
            prediction=prediction or {},
            baseline_means=baseline_means or {},
            forecast_metrics=forecast_metrics or {},
            anomaly_events=anomaly_events or [],
        )
        sensitivity = audit_sensitivity_consistency(
            report=report,
            sensitivity_result=sensitivity_result,
            anomaly_events=anomaly_events or [],
            kpi_names=self._kpi_names,
        )
        footer = format_fidelity_footer(numerical, sensitivity)
        return report.rstrip() + "\n\n" + footer

    def _summarise_prediction(
        self,
        prediction: Optional[Dict[str, List[float]]],
        forecast_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> tuple[str, Dict[str, float]]:
        """Build the prompt-ready forecast summary lines.

        ``forecast_metrics`` may carry per-KPI sMAPE / MASE / MAE / RMSE
        computed against the held-out test target. They are appended to
        the per-KPI line so that the LLM can copy them into §1 — the
        only section permitted to surface ground-truth–derived numbers.
        """
        if not prediction:
            return "", {}

        history = self._cached_input_history
        baseline_means: Dict[str, float] = {}
        if history is not None and history.shape[0] >= _HISTORY_BASELINE_HOURS:
            baseline_window = history[-_HISTORY_BASELINE_HOURS:]
            for idx, kpi in enumerate(self._kpi_names):
                if idx < baseline_window.shape[1]:
                    baseline_means[kpi] = float(np.mean(baseline_window[:, idx]))

        metrics_by_kpi = forecast_metrics or {}

        lines: List[str] = []
        for kpi, values in prediction.items():
            if not values:
                continue
            arr = np.asarray(values, dtype=float)
            forecast_mean = float(np.mean(arr))
            forecast_min = float(np.min(arr))
            forecast_max = float(np.max(arr))
            per_day = daily_means(arr)
            slope = linear_slope(per_day)
            baseline_mean = baseline_means.get(kpi)
            increase_pct = (
                (forecast_mean - baseline_mean) / abs(baseline_mean) * 100.0
                if baseline_mean and abs(baseline_mean) > 1e-8
                else 0.0
            )
            trend_dir = "UP" if slope > 0.01 else "DOWN" if slope < -0.01 else "STABLE"
            baseline_label = f"{baseline_mean:.2f}" if baseline_mean else "N/A"
            line = (
                f"- {kpi}: min={forecast_min:.2f}, mean={forecast_mean:.2f}, "
                f"max={forecast_max:.2f}, trendSlope={slope:.4f}/day, "
                f"baselineMean={baseline_label}, "
                f"increasePercent={increase_pct:+.2f}%, trend={trend_dir}"
            )
            kpi_metrics = metrics_by_kpi.get(kpi, {})
            smape = kpi_metrics.get("sMAPE")
            mase = kpi_metrics.get("MASE")
            if smape is not None:
                line += f", sMAPE={float(smape):.2f}%"
            if mase is not None:
                line += f", MASE={float(mase):.3f}"
            lines.append(line)
        return "\n".join(lines), baseline_means

    def _build_input_baseline(self) -> Optional[Dict[str, List[float]]]:
        history = self._cached_input_history
        if history is None or history.shape[0] < _HISTORY_BASELINE_HOURS:
            return None
        window = history[-_HISTORY_BASELINE_HOURS:]
        baseline: Dict[str, List[float]] = {}
        for idx, kpi in enumerate(self._kpi_names):
            if idx < window.shape[1]:
                baseline[kpi] = window[:, idx].tolist()
        return baseline

    def _kpi_valid_ranges(self) -> Dict[str, tuple]:
        """Lookup ``valid_range`` for each KPI from the ontology, if present."""
        ontology = self._resolve_ontology()
        if ontology is None:
            return {}
        ranges: Dict[str, tuple] = {}
        for kpi in self._kpi_names:
            kpi_def = ontology.get_kpi(kpi) if hasattr(ontology, "get_kpi") else None
            valid_range = getattr(kpi_def, "valid_range", None) if kpi_def else None
            if valid_range is not None:
                ranges[kpi] = tuple(valid_range)
        return ranges

    def _resolve_ontology(self):
        """Return the :class:`ThreeGPPOntology` from the attached store."""
        store = self._ontology_store
        return getattr(store, "ontology", None) if store is not None else None

    def _inject_structural_contexts(
        self,
        contexts: List[str],
        prediction: Optional[Dict[str, List[float]]],
        baseline_means: Dict[str, float],
    ) -> None:
        contexts.append(
            f"This report analyzes the 7-day (168-hour) KPI forecast for "
            f"station {self._station_id}"
        )
        contexts.append(
            "Cross-KPI causal influences are grounded in the 3GPP "
            "knowledge graph and anomaly-event analysis on the forecast "
            "horizon."
        )
        if not prediction:
            return

        kpi_changes: Dict[str, float] = {}
        for kpi, values in prediction.items():
            if not values:
                continue
            forecast_mean = float(np.mean(np.asarray(values, dtype=float)))
            baseline_mean = baseline_means.get(kpi)
            if baseline_mean and abs(baseline_mean) > 1e-8:
                pct = (forecast_mean - baseline_mean) / abs(baseline_mean) * 100.0
                kpi_changes[kpi] = round(pct, 2)
        if not kpi_changes:
            return

        largest_kpi = max(kpi_changes, key=lambda k: abs(kpi_changes[k]))
        smallest_kpi = min(kpi_changes, key=lambda k: abs(kpi_changes[k]))
        contexts.append(
            f"{largest_kpi} shows the largest change at "
            f"{kpi_changes[largest_kpi]:+.2f}% from baseline"
        )
        contexts.append(
            f"{smallest_kpi} shows the smallest change at "
            f"{kpi_changes[smallest_kpi]:+.2f}% from baseline"
        )
        for kpi, pct in kpi_changes.items():
            direction = "increase" if pct > 0 else "decrease"
            contexts.append(f"{kpi} forecast change: {pct:+.2f}% {direction} from baseline")
