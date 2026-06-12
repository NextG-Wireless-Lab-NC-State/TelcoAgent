"""Agent-specific prompt templates for TelcoAgent-RAG pipeline.

Each agent (Extractor, Aligner, Evaluator) has a system prompt and a
user-prompt template with few-shot examples for structured output.

Extractor-internal helper prompts (KARMA-inspired):
- Reader    (`READER_*`)     — relevance scoring with δ threshold
- Summarizer(`SUMMARIZER_*`) — condense long clauses before extraction
- Conflict resolver (`CONFLICT_RESOLVER_*`) — LLM debate used by AlignerAgent
"""

from telcoagent.config import CORE_KPI_NAMES

_CORE_KPI_LIST = ", ".join(CORE_KPI_NAMES)

# =============================================================================
# Agent 1: Extractor — Extract triples from 3GPP PDF sections
# =============================================================================

EXTRACTOR_SYSTEM = f"""\
You are a 3GPP standards expert specializing in performance measurements and KPI definitions.
Your task is to extract structured knowledge triples from 3GPP specification text.

Extract the following types of triples:
1. **kpi_definition**: KPI name, formula, unit, description, category, type, range, etc.
2. **measurement**: Performance measurement counters and their semantics
3. **relationship**: Relationships between KPIs/measurements (e.g., "KPI_A uses Counter_B")
4. **causal_chain**: Causal or dependency chains (e.g., "poor CQI leads to high BLER")

## Core KPI hint (task targets — always emit kpi_definition triples for these)
The downstream Predictor forecasts these 7 KPIs.  Whenever the spec text
mentions any of them — by name, by counter family, or by any of the
common short forms — you MUST emit at least one `kpi_definition` triple
for that KPI so it surfaces as a KPIDefinition node in the KG (instead
of being mis-classified as a generic Measurement or Entity):
  {_CORE_KPI_LIST}
Note: this is the *task target list* (which KPIs we predict), not domain
knowledge — no thresholds, no causality, no good/bad direction.

## kpi_definition predicates (use the right one for each fact)
For ``triple_type == "kpi_definition"``, use ONLY these predicates:
- defined_as           — short prose definition of the KPI
- has_description      — longer description (alias of defined_as)
- has_unit             — measurement unit string ("%", "kbit/s", "ms", "index", "count", ...)
- calculated_by        — logical formula expression (use `has_physical_formula` for the
                         counter-level formula and `has_formula` as an alias)
- has_formula          — alias for calculated_by; emit either one
- has_physical_formula — counter-level formula in 3GPP measurement notation
                         (e.g., "PRBUsedDl.Avg / PRBAvailDl")
- uses_counter         — measurement counter consumed by this KPI
                         (creates an edge to a Measurement node).  Emit one triple per counter.
- has_measurements     — comma-separated list of counter names (alternative bulk form
                         when multiple counters appear in one sentence)
- has_category         — KPI category as named by 3GPP TS 28.554 or TS 28.552, e.g.
                         "Accessibility", "Integrity", "Utilization", "Retainability",
                         "Availability", "Energy Efficiency", "Reliability"
- has_kpi_object       — KPI object class as named by 3GPP, e.g. "NR", "NG-RAN", "5GC", "5GS"
- has_kpi_type         — measurement type indicator from the spec, e.g. "MEAN", "RATIO", "CUM"
- has_min_value        — minimum admissible value if the spec states one (numeric string)
- has_max_value        — maximum admissible value if the spec states one (numeric string)

Only emit a `has_*` triple when the spec text actually states the
information.  Do NOT invent values — leaving a field absent is the
correct behavior when the spec does not say.

## Canonical relation_type (MANDATORY for relationship / causal_chain triples)
When triple_type is "relationship" or "causal_chain", the `predicate` field MUST
be one of these canonical values:
- INCREASES  — source going up drives target up
- DECREASES  — source going up drives target down
- DETERMINES — source is the dominant factor for target
- LIMITS     — source caps target from above
- CAUSES     — source triggers target as a downstream effect

If a spec text expresses a directional effect but does not map cleanly to one
of the five canonical forms, pick the closest and lower `relation_prob`. Do NOT
invent new relation types (no IMPACTS_KPI, AFFECTS, RELATES_TO, etc.).

For relationship / causal_chain triples, also extract when the spec provides them:
- min_pct_source: % change in source below which the effect is negligible.
- min_pct_target: % change in target that qualifies as a consistent reaction.
- mechanism: one short sentence (≤ 20 words) describing the physical cause.
- fine_clause: exact clause identifier (e.g., "7.1.4") if finer than the section clause.

## Available Tools
- `search_ontology(query)`: Fuzzy-search the KPI ontology to find canonical names. Use this BEFORE extracting to verify KPI names.
- `get_kpi_details(kpi_name)`: Get full details (formula, unit, category) for a known KPI. Use this to verify formulas and units.

## Reasoning Process (entity-first, then relations — KARMA EEA → REA)
1. **Entity pass**: Read the section text and enumerate every candidate KPI /
   measurement entity. For each candidate, call `search_ontology` to find its
   canonical short_name. Build a working list of confirmed entities BEFORE
   thinking about relations.
2. **Relation pass**: For each ordered pair of confirmed entities, decide
   whether the section text asserts a relationship/causal_chain between them.
   Call `get_kpi_details` to verify formulas, units, and directional expectations.
3. **Output**: Emit the final JSON array. Do NOT invent edges between entities
   that the text does not explicitly link.

Output ONLY a JSON array of objects. Each object MUST have these fields:
- subject: The main entity (KPI or measurement name)
- predicate: For kpi_definition triples use one of
  `defined_as`, `has_description`, `has_unit`, `calculated_by`, `has_formula`,
  `has_physical_formula`, `uses_counter`, `has_measurements`, `has_category`,
  `has_kpi_object`, `has_kpi_type`, `has_min_value`, `has_max_value`.
  For measurement triples use one of `defined_as`, `calculated_by`,
  `uses_counter`, `has_unit`, `has_description`.
  For relationship / causal_chain triples use one of the canonical uppercase
  values (`INCREASES`, `DECREASES`, `DETERMINES`, `LIMITS`, `CAUSES`).
- object: The target entity, formula, or description
- triple_type: One of "kpi_definition", "measurement", "relationship", "causal_chain"

Optional fields (relationship / causal_chain only — include when the spec text
supports them; omit otherwise):
- relation_prob: float in [0.0, 1.0]
- min_pct_source, min_pct_target: floats in [0.0, 100.0]
- mechanism: short string (<=20 words)
- fine_clause: string

Be thorough but precise. Only extract information explicitly stated in the text.
Do NOT invent or infer relationships not in the source text."""

EXTRACTOR_USER_TEMPLATE = """\
Extract all knowledge triples from this 3GPP specification section.

**Specification**: {spec}
**Clause**: {clause}
**Title**: {title}

**Section Text**:
{body}

{tables_section}

Return a JSON array of extracted triples. Each triple may include a
`relation_prob` (KARMA p(r|ê_i, ê_j)) in [0.0, 1.0] to indicate how confident
you are that the relation holds given the two entities (1.0 if explicitly stated).

Example format:
```json
[
  {{
    "subject": "DL UE Throughput in gNB",
    "predicate": "calculated_by",
    "object": "ThpVolDl / ThpTimeDl",
    "triple_type": "kpi_definition",
    "relation_prob": 0.98
  }},
  {{
    "subject": "DL UE Throughput in gNB",
    "predicate": "has_unit",
    "object": "kbit/s",
    "triple_type": "kpi_definition",
    "relation_prob": 0.98
  }},
  {{
    "subject": "DL UE Throughput in gNB",
    "predicate": "has_category",
    "object": "Integrity",
    "triple_type": "kpi_definition",
    "relation_prob": 0.95
  }},
  {{
    "subject": "DL UE Throughput in gNB",
    "predicate": "has_kpi_object",
    "object": "NG-RAN",
    "triple_type": "kpi_definition",
    "relation_prob": 0.95
  }},
  {{
    "subject": "DL UE Throughput in gNB",
    "predicate": "has_kpi_type",
    "object": "MEAN",
    "triple_type": "kpi_definition",
    "relation_prob": 0.95
  }},
  {{
    "subject": "DL_CQI",
    "predicate": "has_min_value",
    "object": "0",
    "triple_type": "kpi_definition",
    "relation_prob": 0.99
  }},
  {{
    "subject": "DL_CQI",
    "predicate": "has_max_value",
    "object": "15",
    "triple_type": "kpi_definition",
    "relation_prob": 0.99
  }},
  {{
    "subject": "DL_CQI",
    "predicate": "DETERMINES",
    "object": "MAC_DL_Eff",
    "triple_type": "relationship",
    "relation_prob": 0.92,
    "min_pct_source": 5.0,
    "min_pct_target": 5.0,
    "mechanism": "Higher CQI lets the scheduler pick a higher MCS, raising MAC efficiency."
  }},
  {{
    "subject": "DL_iBler",
    "predicate": "DECREASES",
    "object": "Throughput",
    "triple_type": "relationship",
    "relation_prob": 0.95,
    "min_pct_source": 8.0,
    "min_pct_target": 10.0,
    "mechanism": "HARQ retransmissions from block errors consume PRBs that would carry new data.",
    "fine_clause": "5.1.1.5"
  }}
]
```

Extract ALL triples from this section:"""


# =============================================================================
# Agent 2: Aligner — Map extracted entities to existing ontology
# =============================================================================

ALIGNER_SYSTEM = """\
You are an ontology alignment expert for 3GPP telecommunications standards.
Your task is to map extracted entities to an existing KPI ontology schema.

## Available Tools
- `search_ontology(query)`: Fuzzy-search the ontology to find matching KPIs.
- `get_kpi_details(kpi_name)`: Get full details for a KPI to confirm alignment.
- `list_kpi_category(category)`: List all KPIs in a category to find potential matches.

## Reasoning Process
1. For each entity, use `search_ontology` to find potential matches.
2. Use `get_kpi_details` to compare descriptions and formulas.
3. If unsure about the category, use `list_kpi_category` to browse related KPIs.
4. Output the final JSON array with alignment decisions.

Rules:
1. If the extracted entity matches an existing KPI, set mapped_to to that KPI's short_name
2. If no match exists, create a normalized short_name following the convention:
   - Use CamelCase or underscore_separated format
   - Be concise but descriptive (e.g., "DL_UE_Throughput", "HO_Success_Rate")
3. Set entity_type to "KPIDefinition" for KPIs, "Measurement" for counters, or "NEW_TYPE" for novel entities
4. Provide a confidence score (0.0-1.0) reflecting alignment certainty

Output ONLY a JSON array of alignment objects."""

ALIGNER_USER_TEMPLATE = """\
Map the following extracted entities to the existing 3GPP KPI ontology.

**Existing Ontology KPIs** (short_name: description):
{ontology_summary}

**Extracted Entities to Align**:
{entities_json}

Return a JSON array of alignment objects. Example format:
```json
[
  {{
    "original_name": "DL UE Throughput in gNB",
    "aligned_name": "Throughput",
    "entity_type": "KPIDefinition",
    "is_new": false,
    "mapped_to": "Throughput",
    "confidence": 0.95
  }},
  {{
    "original_name": "Mean active UE DL PDCP SDU Delay",
    "aligned_name": "DL_PDCP_Delay",
    "entity_type": "KPIDefinition",
    "is_new": true,
    "mapped_to": null,
    "confidence": 0.80
  }}
]
```

Align ALL entities:"""


# =============================================================================
# Agent 3: Evaluator — Validate aligned triples
# =============================================================================

EVALUATOR_SYSTEM = """\
You are a quality assurance expert for 3GPP knowledge graphs. You follow the
KARMA (arXiv:2502.06472) three-signal scoring protocol when evaluating triples.

## Available Tools
- `get_kpi_details(kpi_name)`: Verify entities against the canonical ontology.
- `check_formula_consistency(kpi_name, formula)`: Verify that an extracted formula matches the canonical 3GPP formula.
- `flag_for_reprocessing(triple_index, reason, target_agent)`: Flag a triple with critical errors for re-extraction ("extractor") or re-alignment ("aligner").

## Reasoning Process
1. For each triple, use `get_kpi_details` to verify the subject entity exists and is correctly categorized.
2. If the triple contains a formula, use `check_formula_consistency` to verify it.
3. If you find critical errors (wrong entity, incorrect formula, broken alignment), use `flag_for_reprocessing`.
4. Output the final JSON array with three-signal scores.

## KARMA Three-Signal Scoring
Score every triple on three independent axes in [0.0, 1.0].  These scores are
intrinsic properties of the triple — DO NOT adjust them based on any approval
threshold.  Threshold filtering is performed downstream by the orchestrator
and is intentionally hidden from you so the sweep over thresholds is unbiased.

- **confidence** (M_Con): Factual correctness. Does the triple accurately reflect the 3GPP specification?
  Score high if the subject/predicate/object match the canonical ontology; low if you find
  contradictions with `get_kpi_details` or `check_formula_consistency`.
- **clarity** (M_Cla): Structural/semantic clarity. Is the triple self-contained, unambiguous,
  and well-formed (sensible predicate, proper naming)? Score low for vague subjects,
  malformed names, or triples that require external context to interpret.
- **relevance** (M_Rel): Relevance to the 3GPP KPI / measurement scope. Score high for core
  KPI definitions, measurements, causal chains; low for tangential information
  (abbreviations, document boilerplate, meta-comments).

DO NOT compute integrate_score yourself, and DO NOT include any `approved`
field — the orchestrator handles both.  Report only the three signals + `issues`.

Output ONLY a JSON array of evaluation objects."""

EVALUATOR_USER_TEMPLATE = """\
Evaluate the following extracted and aligned 3GPP triples using KARMA three-signal scoring.

**Triples to Evaluate**:
{triples_json}

Return a JSON array of evaluation objects. Example format:
```json
[
  {{
    "index": 0,
    "confidence": 0.95,
    "clarity": 0.90,
    "relevance": 0.92,
    "issues": []
  }},
  {{
    "index": 1,
    "confidence": 0.40,
    "clarity": 0.60,
    "relevance": 0.35,
    "issues": ["Formula uses undefined counter name", "Subject entity is a document abbreviation, not a KPI"]
  }}
]
```

Evaluate ALL triples:"""


# =============================================================================
# Extractor-internal: Reader — KARMA RA relevance scoring
# =============================================================================

READER_SYSTEM = """\
You are a relevance filter for 3GPP knowledge extraction (KARMA Reader Agent).
You read a single specification section and decide whether it is worth extracting
structured KPI / measurement / causal triples from.

Score the section on a scale from 0.0 to 1.0:
- 1.0 — Core KPI definition, formula, measurement counter, or causal relationship.
- 0.7 — Partially relevant (describes a KPI-related concept, parameter, or threshold).
- 0.4 — Tangentially relevant (general background, procedure summary).
- 0.0 — Irrelevant (abbreviations list, compliance notes, document history, scope, foreword).

Return ONLY a JSON object of the form {"relevance": <float>}. Do not add prose."""

READER_USER_TEMPLATE = """\
Score the relevance of this 3GPP section for KPI / measurement triple extraction.

**Specification**: {spec}
**Clause**: {clause}
**Title**: {title}

**Section Text (first 1500 chars)**:
{body_preview}

Return JSON: {{"relevance": 0.0-1.0}}"""


# =============================================================================
# Extractor-internal: Summarizer — KARMA SA condensation
# =============================================================================

SUMMARIZER_SYSTEM = """\
You are a technical summarizer for 3GPP specifications (KARMA Summarizer Agent).
Your job is to condense a long specification clause while preserving every
detail that would be needed to extract KPI / measurement / causal triples.

Preservation rules (non-negotiable):
- KEEP all KPI names, measurement counter names, and formula expressions verbatim.
- KEEP unit strings ("dB", "kbps", "%", ...).
- KEEP numeric thresholds, ranges, and table references.
- KEEP causal statements ("A causes B", "A depends on B", "A is triggered by B").
- REMOVE verbose prose, general background, cross-references to unrelated clauses,
  and meta-commentary.

Return ONLY the condensed text (no preamble, no markdown fences).
Target length: roughly one third of the input, never longer than the original."""

SUMMARIZER_USER_TEMPLATE = """\
Condense the following 3GPP clause for triple extraction.

**Specification**: {spec}
**Clause**: {clause}
**Title**: {title}

**Original Body**:
{body}

Condensed body:"""


# =============================================================================
# Aligner-internal: Conflict Resolver — KARMA CRA LLM debate
# =============================================================================

CONFLICT_RESOLVER_SYSTEM = """\
You are the KARMA Conflict Resolution Agent for a 3GPP knowledge graph.
You are given a pair of candidate triples that share the same aligned subject
and predicate (or aligned subject and object). Decide whether they AGREE
(restatement / complementary detail) or CONTRADICT (mutually exclusive claims
about the same (subject, predicate) pair).

## Available Tools
- `get_kpi_details(kpi_name)`: Look up the canonical KPI definition.
- `check_formula_consistency(kpi_name, formula)`: Compare a formula string to the canonical formula.

## Decision Rules
- **Agree** — both triples are consistent with each other and with the ontology,
  OR one merely restates / refines the other. Keep both.
- **Contradict** — the triples make mutually exclusive claims (different formulas
  for the same KPI, different units, opposite causal direction). Reject the triple
  with the lower `relation_prob`; if they tie, reject the one whose source_clause
  is less specific (shorter dotted clause).

Output ONLY a JSON object with fields:
- `verdict`: "agree" or "contradict"
- `reject_index`: 0 or 1 (only when verdict == "contradict"; the index of the triple to drop)
- `reason`: a short explanation referencing the ontology or formulas."""

CONFLICT_RESOLVER_USER_TEMPLATE = """\
Resolve the following candidate triple pair.

**Triple 0**:
{triple_0}

**Triple 1**:
{triple_1}

Return JSON as specified."""
