# Module map — paper Fig. 1 ↔ repository

One row per module: what it does, what it consumes, and what it
produces. Use this as the source of truth when presenting the system
decomposition (slides, reviews, onboarding).

## ① Knowledge Graph Construction (paper §III.A) — `telcoagent/kg_construction/`

Offline batch. 3GPP spec PDFs in, `data/enriched_kg.json` out. The
runtime pipelines only ever read its output.

| Module | Role | Input → Output |
|---|---|---|
| `pdf_parser.py` | Deterministic PDF → section chunking | spec PDFs → `SectionChunk` |
| `extractor.py` | **Extractor agent** (KARMA IA/RA/SA/EEA/REA) | `SectionChunk` → `ExtractedTriple` |
| `aligner.py` | **Aligner agent**: ontology mapping + conflict resolution (CRA) | triples → `(triple, AlignedEntity)` |
| `evaluator.py` | **Evaluator agent**: 3-signal confidence scoring, q_TH = 0.9 gate, reprocessing flags | pairs → `ValidatedTriple` + feedback |
| `pipeline.py` | `KGConstructionPipeline` — sequences the three agents, drives the q < q_TH reprocess loop | PDFs → validated triples |
| `kg_builder.py` | Graph assembly + (de)serialisation | validated triples → `enriched_kg.json` (NetworkX node-link) |
| `schema.py` | Pipeline dataclasses (`SectionChunk`, `ExtractedTriple`, `AlignedEntity`, `ValidatedTriple`, …) | — |
| `prompts.py` | All KG-agent prompts (extractor / aligner / evaluator / reader / summarizer / CRA) | — |
| `tools.py` | Per-agent ReAct tool factories (ontology search etc.) | — |
| `shared.py` | LLM-JSON parsing/repair, rate-limit delay | — |

## ② Prediction (paper §III.B) — `telcoagent/prediction/`

Runtime, single pass. No ReAct, no LLM, no KG consumption on the
forward path.

| Module | Role | Input → Output |
|---|---|---|
| `predictor.py` | `TelcoAgent.predict()` — Chronos-2 single forward pass over the 1944h context | (1944, 7) raw KPIs → (168, 7) forecast |
| `features.py` | Canonical `KPI_NAMES` + pure-NumPy input-window utilities (torch-free) | input window → stats / text |
| `physical_clamp.py` | `apply_physical_clamp_strict` — clip to `KPI_NORMALIZATION_MAX` + integer-KPI rounding; shared with every baseline (ADR 0001) | raw forecast → clamped forecast |

## ③ Explanation (paper §III.C) — `telcoagent/explainer/`

Runtime, separate pass over the forecast + input baseline.

| Module | Role | Input → Output |
|---|---|---|
| `orchestrator.py` | `TelcoAgentExplainer` — wires the whole explain pass; `ExplanationResult` output bundle | forecast + history → report + artefacts |
| `anomaly_detector.py` | Deterministic anomaly events (level shift / spike / trend break), z-scored vs baseline | forecast + baseline → events |
| `cause_inference.py` | KG-grounded candidate causes + OSM environmental context + optional LLM relevance harness | events + KG + OSM → enriched events |
| `pax_ts.py` | PAX-TS perturbation analysis — one batched Chronos-2 call → cross-channel sensitivity tensor `S` | input + forecast → `SensitivityResult` |
| `evidence.py` | Prompt evidence assembly: anomaly snapshot + ReAct-off prestuffer | events → prompt blocks |
| `react_agent.py` | `ExplainerAgent` — 3-turn ReAct report author + §-section stitcher | evidence + tools → 5-section markdown |
| `tools/` | The 7 ReAct tools, one module each (see below) | tool calls → grounded evidence |
| `report.py` | Deterministic post-processing: figure embeds, preamble strip, RAGAS context injection | raw report → final report |
| `audit.py` | Numerical-fidelity + sensitivity-consistency audit (the paper's Ⓔ check) | report + truths → fidelity footer |
| `sensitivity_format.py` | Sensitivity tables/CSV artefacts + markdown grids | `SensitivityResult` → tables |
| `anomaly_plot.py` / `html_report.py` | Per-KPI anomaly figures / dark-theme HTML mirror | events / markdown → PNG / HTML |

### Explainer ReAct tools — `telcoagent/explainer/tools/`

| Module | Tool(s) | Grounds |
|---|---|---|
| `factory.py` | `ExplainerToolsConfig` + `make_explainer_tools` | assembles the registry |
| `kg_mechanism.py` | `query_graphrag_mechanism` | 3GPP causal mechanism for a KPI pair |
| `spatial.py` | `query_spatial_context` | raw OSM blob + environment label |
| `temporal.py` | `query_forecast_temporal` | per-day forecast breakdown |
| `anomaly_events.py` | `query_anomaly_events` | detector events, |z|-ranked |
| `sensitivity.py` | `query_sensitivity_matrix`, `query_localized_sensitivity` | PAX-TS global / per-event |
| `verification.py` | `verify_explanation` | claim cross-check vs directional rules + OSM grounding |
| `shared.py` | — | weekday/event-time helpers |

## Shared infrastructure and support packages

| Package | Role |
|---|---|
| `agents/` | Pillar-neutral ReAct plumbing: `base.py` (`BaseTrainingFreeAgent` loop), `registry.py` (`ToolRegistry`). Used by ① and ③; ② uses none of it. |
| `ontology/` | `ThreeGPPOntology` dataclasses + KG JSON loader (`strict=True` everywhere at runtime). |
| `stores/` | Neo4j connection + ontology store (optional), in-memory store for Neo4j-free runs. |
| `context/` | OSM + Standard-KG MCP clients. |
| `llm/` | `invoke_llm` LiteLLM wrapper (single LLM entry point), `EnhancedLLMConfig`, explainer prompts. |
| `evaluation/` | RAGAS + custom Faithfulness / Answer-Relevancy scoring of explainer reports. |
| `config.py` | `CORE_KPI_NAMES`, `KPI_NORMALIZATION_MAX`, window sizes. |

## Data flow between the pipelines

```
①  kg_construction  ──writes──►  data/enriched_kg.json
                                        │ (read-only)
②  prediction       ──forecast──►  ③  explainer ──reads KG + OSM──►  report
        (no KG read)                        │
                                            ▼
                                   evaluation/ (RAGAS)
```
