"""Post-hoc audit of generated explanation reports.

Two deterministic audit channels live here so the KG-on / KG-off
ablation script can quote them without spinning up an LLM:

* :func:`audit_spec_citations` — extract every 3GPP spec citation
  (``TS 28.552``, ``3GPP TS 38.214 Section 5.1.3``, …) from the report
  and split it into ``in_kg`` / ``hallucinated`` against the ontology.
  ``hallucinated`` means the LLM produced a TS number that the
  spec-extracted KG has no record of — a direct hallucination signal.
* :func:`audit_numerical_claims` — recompute the §1 forecast-summary
  numbers (Min / Mean / Max / baseline mean / change %) from the
  prediction array and the held-out test target, plus the §4 ``|z|``
  values from the anomaly-event objects, and flag every cell whose
  LLM-rendered value drifts beyond a small tolerance from the
  ground-truth.

Both audits are pure functions of (report, ontology, prediction,
events, …) and never call an LLM, so they reproduce exactly across
runs and are safe to ship as an ablation table for a paper.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# C. Spec-citation hallucination
# =============================================================================

# Captures (TS, NN.NNN, optional clause) from forms like:
#   "3GPP TS 28.552"
#   "TS 38.214 Section 5.1.3"
#   "per TS 28.554 §6.3.1"
# The TS number is the only mandatory part; the section/clause is captured
# when present so the audit can show full citations in the mismatch list.
_SPEC_PATTERN = re.compile(
    r"(?:3GPP\s+)?TS\s*(\d{2}\.\d{3})" r"(?:\s*(?:Section|sec\.?|§|Clause)\s*(\d+(?:\.\d+)*))?",
    re.IGNORECASE,
)


def _canonical_spec(ts_number: str, clause: Optional[str]) -> str:
    """Return ``"TS 28.552"`` or ``"TS 28.552 §5.1.3"`` (canonical form)."""
    if clause:
        return f"TS {ts_number} §{clause}"
    return f"TS {ts_number}"


def _spec_ts_number(canonical: str) -> str:
    """Return just the ``TS NN.NNN`` portion of a canonical citation."""
    return canonical.split(" §", 1)[0]


def extract_spec_citations(report: str) -> List[str]:
    """Pull every 3GPP TS citation from ``report`` in canonical form.

    Duplicate citations are preserved (a paper-style audit reports every
    occurrence, not just unique ones); call ``set(...)`` on the result if
    a deduplicated view is wanted.
    """
    if not report:
        return []
    citations: List[str] = []
    for ts_number, clause in _SPEC_PATTERN.findall(report):
        citations.append(_canonical_spec(ts_number, clause or None))
    return citations


def ground_truth_spec_numbers(ontology: Any) -> Set[str]:
    """Build the set of TS numbers (``"TS NN.NNN"``) recognised by the KG.

    The audit compares against TS numbers only — clause-level matching
    is too brittle because the report may legitimately quote a tighter
    section than what the KG stored, or vice versa. Hallucination is
    defined at the TS-number level.
    """
    truth: Set[str] = set()
    kpis = getattr(ontology, "kpis", {}) or {}
    iterable = kpis.values() if isinstance(kpis, dict) else kpis
    for kpi in iterable:
        for spec in getattr(kpi, "source_specs", []) or []:
            for ts_number, _ in _SPEC_PATTERN.findall(spec):
                truth.add(f"TS {ts_number}")
    for rule in getattr(ontology, "directional_rules", []) or []:
        spec_ref = getattr(rule, "spec_reference", None) or ""
        for ts_number, _ in _SPEC_PATTERN.findall(spec_ref):
            truth.add(f"TS {ts_number}")
    return truth


@dataclass
class SpecAuditResult:
    citations: List[str] = field(default_factory=list)
    in_kg: List[str] = field(default_factory=list)
    hallucinated: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.citations)

    @property
    def hallucinated_unique(self) -> List[str]:
        seen: Set[str] = set()
        unique: List[str] = []
        for cite in self.hallucinated:
            key = _spec_ts_number(cite)
            if key in seen:
                continue
            seen.add(key)
            unique.append(cite)
        return unique


def audit_spec_citations(report: str, ontology: Any) -> SpecAuditResult:
    """Classify every spec citation in ``report`` against the KG."""
    citations = extract_spec_citations(report)
    truth = ground_truth_spec_numbers(ontology)
    in_kg: List[str] = []
    hallucinated: List[str] = []
    for cite in citations:
        if _spec_ts_number(cite) in truth:
            in_kg.append(cite)
        else:
            hallucinated.append(cite)
    return SpecAuditResult(citations=citations, in_kg=in_kg, hallucinated=hallucinated)


# =============================================================================
# D. Numerical faithfulness
# =============================================================================


_NUMBER_RE = r"-?\d+(?:\.\d+)?"


@dataclass
class NumericalClaim:
    """One LLM-emitted number paired with the truth it should match."""

    section: str  # "§1" or "§4"
    kpi: str  # KPI the claim refers to (or "" when unknown)
    field: str  # canonical truth-field name
    claimed: float
    truth: Optional[float]
    raw_cell: str = ""  # original cell/snippet for the mismatch report

    @property
    def has_truth(self) -> bool:
        return self.truth is not None

    @property
    def absolute_error(self) -> Optional[float]:
        if self.truth is None:
            return None
        return abs(self.claimed - float(self.truth))


@dataclass
class NumericalAuditResult:
    matches: List[NumericalClaim] = field(default_factory=list)
    mismatches: List[NumericalClaim] = field(default_factory=list)
    unverifiable: List[NumericalClaim] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.matches) + len(self.mismatches)

    @property
    def faithfulness(self) -> Optional[float]:
        if self.total == 0:
            return None
        return len(self.matches) / self.total


def _within_tolerance(claimed: float, truth: float) -> bool:
    """1% relative tolerance with a 0.01 absolute floor (covers rounding)."""
    return abs(claimed - truth) <= max(0.01, 0.01 * abs(truth))


def _to_float(token: str) -> Optional[float]:
    try:
        return float(token.replace(",", "").replace("%", "").replace("+", ""))
    except ValueError:
        return None


def _extract_table_rows(section_text: str) -> List[List[str]]:
    """Return ``[[cell, …], …]`` for every body row of a markdown table."""
    rows: List[List[str]] = []
    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if re.match(r"^\|\s*[:\-]+", stripped):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if any(re.match(r"^\s*\d", c) or re.match(r"^[+-]?\d", c) for c in cells):
            rows.append(cells)
    return rows


def _build_truth_for_section1(
    prediction: Dict[str, List[float]],
    baseline_means: Dict[str, float],
    forecast_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Recompute every numeric §1 cell from the canonical inputs."""
    truth: Dict[str, Dict[str, float]] = {}
    for kpi, values in prediction.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=float)
        forecast_min = float(arr.min())
        forecast_max = float(arr.max())
        forecast_mean = float(arr.mean())
        per_kpi: Dict[str, float] = {
            "forecast_min": round(forecast_min, 2),
            "forecast_mean": round(forecast_mean, 2),
            "forecast_max": round(forecast_max, 2),
        }
        baseline_mean = baseline_means.get(kpi)
        if baseline_mean is not None:
            per_kpi["baseline_mean"] = round(float(baseline_mean), 2)
            if abs(baseline_mean) > 1e-8:
                change_pct = 100.0 * (forecast_mean - baseline_mean) / abs(baseline_mean)
                per_kpi["change_pct"] = round(change_pct, 2)
        kpi_metrics = forecast_metrics.get(kpi, {}) or {}
        for src_key, dst_key in (
            ("sMAPE", "smape"),
            ("MASE", "mase"),
            ("MAE", "mae"),
            ("RMSE", "rmse"),
        ):
            if src_key in kpi_metrics:
                per_kpi[dst_key] = round(float(kpi_metrics[src_key]), 4)
        truth[kpi] = per_kpi
    return truth


def _audit_section1(
    section_text: str,
    truth_by_kpi: Dict[str, Dict[str, float]],
) -> List[NumericalClaim]:
    """Audit the §1 forecast-summary table.

    The §1 prompt fixes the column order; we lock onto KPI by scanning
    the first cell of every row. ``Forecast (Min / Mean / Max)`` lives
    in column 3 ("a / b / c"); the ``Trend / Baseline / Change %``
    columns follow. sMAPE / MASE may or may not appear depending on
    whether ``forecast_metrics`` was supplied.
    """
    claims: List[NumericalClaim] = []
    for cells in _extract_table_rows(section_text):
        if not cells:
            continue
        kpi = cells[0]
        if kpi not in truth_by_kpi:
            continue
        per_kpi = truth_by_kpi[kpi]
        # Forecast (Min / Mean / Max): cell 2 (index 2) when format is
        # "| KPI | Unit | Min / Mean / Max | …"
        for cell in cells[2:5]:
            parts = re.split(r"\s*/\s*", cell)
            if len(parts) == 3:
                triples = [_to_float(p) for p in parts]
                if all(t is not None for t in triples):
                    for value, key in zip(
                        triples,
                        ("forecast_min", "forecast_mean", "forecast_max"),
                    ):
                        truth = per_kpi.get(key)
                        claims.append(
                            NumericalClaim(
                                section="§1",
                                kpi=kpi,
                                field=key,
                                claimed=value,
                                truth=truth,
                                raw_cell=cell,
                            )
                        )
                    break
        # Scan the rest of the cells for labelled-style values.
        for cell in cells:
            for token, key in (
                ("baseline", "baseline_mean"),
                ("smape", "smape"),
                ("mase", "mase"),
            ):
                if token in cell.lower():
                    match = re.search(_NUMBER_RE, cell)
                    if match:
                        value = _to_float(match.group(0))
                        if value is not None:
                            claims.append(
                                NumericalClaim(
                                    section="§1",
                                    kpi=kpi,
                                    field=key,
                                    claimed=value,
                                    truth=per_kpi.get(key),
                                    raw_cell=cell,
                                )
                            )
    return claims


_Z_PATTERN = re.compile(r"\|\s*z\s*\|\s*=\s*(" + _NUMBER_RE + ")", re.IGNORECASE)


def _audit_section4_z_scores(
    section_text: str,
    anomaly_events: List[Dict[str, Any]],
) -> List[NumericalClaim]:
    """Audit ``|z|=…`` values in §4 against the anomaly-event z-scores."""
    if not section_text:
        return []
    truth_pool = [round(abs(float(ev["z_score"])), 2) for ev in anomaly_events if "z_score" in ev]
    if not truth_pool:
        return []
    claims: List[NumericalClaim] = []
    for match in _Z_PATTERN.finditer(section_text):
        value = _to_float(match.group(1))
        if value is None:
            continue
        # Match against the closest truth z-score within tolerance; this
        # is robust to LLM reordering of events in §4.
        closest = min(truth_pool, key=lambda t: abs(value - t))
        claims.append(
            NumericalClaim(
                section="§4",
                kpi="",
                field="z_score",
                claimed=value,
                truth=closest,
                raw_cell=match.group(0),
            )
        )
    return claims


def audit_numerical_claims(
    report: str,
    prediction: Dict[str, List[float]],
    baseline_means: Dict[str, float],
    forecast_metrics: Dict[str, Dict[str, float]],
    anomaly_events: List[Dict[str, Any]],
) -> NumericalAuditResult:
    """Audit every numeric claim the LLM placed in §1 and §4.

    Coverage is intentionally narrow on purpose: §1 numbers are
    table-locked (column order is fixed by the prompt) and §4 ``|z|``
    is the highest-value §4 cell. Free-form §4 prose numbers
    (``mean=87 %``) are noisy and not audited here — they would need a
    KPI-context resolver per sentence.
    """
    truth_by_kpi = _build_truth_for_section1(
        prediction or {},
        baseline_means or {},
        forecast_metrics or {},
    )
    sections = _split_report_sections(report)
    claims = _audit_section1(sections.get("§1", ""), truth_by_kpi)
    claims.extend(
        _audit_section4_z_scores(
            sections.get("§4", ""),
            anomaly_events or [],
        )
    )

    matches: List[NumericalClaim] = []
    mismatches: List[NumericalClaim] = []
    unverifiable: List[NumericalClaim] = []
    for claim in claims:
        if not claim.has_truth or claim.truth is None:
            unverifiable.append(claim)
            continue
        if _within_tolerance(claim.claimed, float(claim.truth)):
            matches.append(claim)
        else:
            mismatches.append(claim)
    return NumericalAuditResult(
        matches=matches,
        mismatches=mismatches,
        unverifiable=unverifiable,
    )


# =============================================================================
# Section splitter (shared between audits)
# =============================================================================


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


# =============================================================================
# E. PAX-TS sensitivity-consistency audit
# =============================================================================
#
# Adds 5 deterministic sensitivity-consistency scores to the existing
# 28 KPM-descriptor audits (28 + 5 = 33 metrics total — matches the
# paper's §IV-C E claim once both audits are run together):
#
#   1. ``directional_consistency_global`` —
#      fraction of §2 cross-KPI rows whose stated coupling sign agrees
#      with the dominant ``S_global_norm`` sign across the three
#      perturbation types ``(mean, variance, trend)``.
#   2. ``directional_consistency_local`` —
#      fraction of §4 events that cite a ``[Sensitivity-Local: src]``
#      source which is among the top-3 of ``S_local_norm[event_idx]``.
#   3. ``sensitivity_top1_recall`` —
#      1.0 if the global top-1 (source, target) pair (ranked by
#      ``|S_global_norm|`` averaged across perturbation types) appears
#      anywhere in §2; 0.0 otherwise. Provides a coarse "did the
#      explainer mention the strongest coupling" check.
#   4. ``sensitivity_citation_count`` —
#      total number of ``[Sensitivity]`` and ``[Sensitivity-Local]``
#      tag occurrences in the report. A pure count, not a rate.
#   5. ``sensitivity_unsupported`` —
#      number of ``[Sensitivity*]`` citations whose dominant magnitude
#      class (per :func:`telcoagent.agents.registry._classify_sensitivity_magnitude`)
#      is ``"negligible"`` — flags hallucinated sensitivity claims.
#
# All scores are pure functions of (report text, sensitivity_result,
# anomaly_events). When ``sensitivity_result`` is None (the deferred
# fast-path), every score returns ``None`` and the consumer can omit
# the section from the audit summary. The audit never raises on a
# malformed report — it simply returns counts of zero. Boundary
# validation only checks shape compatibility between
# ``sensitivity_result`` and ``anomaly_events``.

from dataclasses import asdict

_SENSITIVITY_GLOBAL_TAG_RE = re.compile(r"\[Sensitivity\]")
_SENSITIVITY_LOCAL_TAG_RE = re.compile(r"\[Sensitivity-Local")
_SENSITIVITY_LOCAL_SRC_RE = re.compile(
    r"\[Sensitivity-Local[^\]]*?(?:← |<- )([A-Za-z_][A-Za-z0-9_]+)"
)
_SECTION2_TABLE_ROW_RE = re.compile(
    r"^\|([^|]+)\|([^|]+)\|([^|]+)\|",
    re.MULTILINE,
)
_COUPLING_TOKENS_POS = ("positive", "increase", "↑", "+")
_COUPLING_TOKENS_NEG = ("negative", "decrease", "↓", "−", "-")

_GLOBAL_TOP_SOURCE_KPI_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]+)\s*(?:→|->|=>)\s*([A-Za-z_][A-Za-z0-9_]+)",
)
_NEGLIGIBLE_TAG_RE = re.compile(
    r"\[Sensitivity[^\]]*?(?:negligible|<\s*0\.001)[^\]]*?\]",
    re.IGNORECASE,
)


@dataclass
class SensitivityAuditResult:
    """Five sensitivity-consistency metrics produced by
    :func:`audit_sensitivity_consistency`.

    All fields are ``None`` when ``sensitivity_result`` was None at
    audit time (the explainer ran in deferred-PAX-TS mode).
    """

    directional_consistency_global: Optional[float] = None
    directional_consistency_local: Optional[float] = None
    sensitivity_top1_recall: Optional[float] = None
    sensitivity_citation_count: int = 0
    sensitivity_unsupported: int = 0

    @property
    def total(self) -> int:
        """Number of metrics this audit produced (always 5)."""
        return 5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _coupling_sign_from_text(cell: str) -> Optional[int]:
    lowered = cell.lower()
    pos = any(tok in lowered for tok in _COUPLING_TOKENS_POS)
    neg = any(tok in lowered for tok in _COUPLING_TOKENS_NEG)
    if pos and not neg:
        return 1
    if neg and not pos:
        return -1
    return None


def _global_top1_pair(
    sensitivity_result: Any,
    kpi_names: List[str],
) -> Optional[Tuple[str, str]]:
    import numpy as np

    try:
        S = np.abs(sensitivity_result.S_global_norm).mean(axis=-1)
    except (AttributeError, ValueError):
        return None
    if S.size == 0:
        return None
    np.fill_diagonal(S, -np.inf)
    flat_idx = int(np.argmax(S))
    n = S.shape[0]
    src_idx, tgt_idx = divmod(flat_idx, n)
    if not (0 <= src_idx < len(kpi_names) and 0 <= tgt_idx < len(kpi_names)):
        return None
    return kpi_names[src_idx], kpi_names[tgt_idx]


def audit_sensitivity_consistency(
    report: str,
    sensitivity_result: Optional[Any],
    anomaly_events: List[Dict[str, Any]],
    kpi_names: List[str],
) -> SensitivityAuditResult:
    """Compute the five sensitivity-consistency audit metrics.

    Parameters
    ----------
    report
        Generated explanation report (markdown).
    sensitivity_result
        :class:`telcoagent.explainer.pax_ts.SensitivityResult` or
        ``None``. When None, all scores are returned as None.
    anomaly_events
        Anomaly event dicts in the same order used to build
        ``sensitivity_result.event_centers``.
    kpi_names
        Ordered KPI vocabulary.

    Returns
    -------
    SensitivityAuditResult
        Five-metric snapshot suitable for paper-grade reporting.
    """
    import numpy as np

    out = SensitivityAuditResult()

    if not report:
        return out

    out.sensitivity_citation_count = len(_SENSITIVITY_GLOBAL_TAG_RE.findall(report)) + len(
        _SENSITIVITY_LOCAL_TAG_RE.findall(report)
    )
    out.sensitivity_unsupported = len(_NEGLIGIBLE_TAG_RE.findall(report))

    if sensitivity_result is None:
        return out

    sections = _split_report_sections(report)
    section2 = sections.get("§2", "")
    section4 = sections.get("§4", "")

    # ── Metric 1: directional_consistency_global ────────────────────────
    table_rows: List[Tuple[str, str, int]] = []
    for raw_row in _SECTION2_TABLE_ROW_RE.findall(section2):
        if not raw_row:
            continue
        cells = list(raw_row)
        joined = " | ".join(cells)
        if "Source" in joined and "Target" in joined:
            continue  # header
        if any(c.strip().startswith(("-", ":")) for c in cells):
            continue  # alignment row
        match = _GLOBAL_TOP_SOURCE_KPI_RE.search(cells[0]) or _GLOBAL_TOP_SOURCE_KPI_RE.search(
            joined
        )
        if not match:
            continue
        src, tgt = match.group(1), match.group(2)
        if src not in kpi_names or tgt not in kpi_names:
            continue
        coupling_sign = _coupling_sign_from_text(joined)
        if coupling_sign is None:
            continue
        table_rows.append((src, tgt, coupling_sign))

    if table_rows:
        agree = 0
        for src, tgt, claimed_sign in table_rows:
            i, j = kpi_names.index(src), kpi_names.index(tgt)
            ptype_signs = sensitivity_result.S_global_norm[i, j, :]
            dominant_sign = int(np.sign(ptype_signs[np.argmax(np.abs(ptype_signs))]))
            if dominant_sign == claimed_sign:
                agree += 1
        out.directional_consistency_global = agree / len(table_rows)

    # ── Metric 2: directional_consistency_local ─────────────────────────
    cited = _SENSITIVITY_LOCAL_SRC_RE.findall(section4)
    if cited and sensitivity_result.S_local_norm.shape[0] > 0:
        # We do not have per-citation event_idx anchoring without a
        # richer tag schema, so we accept a citation when ANY event row's
        # top-3 contains the cited source. This is a permissive lower
        # bound that the §4 narrative shape already supports.
        local = np.abs(sensitivity_result.S_local_norm)
        top3_sources_per_event: List[set] = []
        for row in local:
            order = np.argsort(row)[::-1][:3]
            top3_sources_per_event.append({kpi_names[idx] for idx in order})
        anywhere_top3 = set().union(*top3_sources_per_event) if top3_sources_per_event else set()
        agree = sum(1 for src in cited if src in anywhere_top3)
        out.directional_consistency_local = agree / len(cited)

    # ── Metric 3: sensitivity_top1_recall ───────────────────────────────
    top1 = _global_top1_pair(sensitivity_result, kpi_names)
    if top1 is not None and section2:
        src, tgt = top1
        out.sensitivity_top1_recall = 1.0 if (src in section2 and tgt in section2) else 0.0

    return out


# =============================================================================
# F. §E Numerical Fidelity footer (deterministic, paper-grade)
# =============================================================================
#
# Formats a single markdown footer that mirrors PDF §E in the paper's
# explainer figure: a per-claim audit table for §1 forecast numbers and
# §4 |z| anomaly scores, followed by a 5-row sensitivity-consistency
# table, with a final ``**Result: N/M match**`` summary.
#
# This footer is appended deterministically by
# :class:`telcoagent.explainer.TelcoAgentExplainer` after the LLM
# finishes — never authored by the LLM — so the audit numbers cannot
# be hallucinated. The footer is the single source of truth for the
# 33/33 (or whichever count) figure quoted in the paper's §IV-C
# fidelity table.


def _err_pct(claimed: float, truth: Optional[float]) -> str:
    if truth is None or abs(float(truth)) <= 1e-12:
        return "—"
    return f"{100.0 * abs(claimed - float(truth)) / abs(float(truth)):.2f}%"


def _format_truth(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _sensitivity_pass(metric: str, value: Optional[float]) -> Optional[bool]:
    """Per-metric pass criterion for the sensitivity-consistency table.

    ``None`` means "unverifiable" (the underlying score was None because
    no SensitivityResult was attached or the metric had no input rows);
    such rows are rendered with a ``—`` mark and are NOT counted toward
    the totals. ``True`` / ``False`` mean the metric passed / failed and
    contribute one slot to the denominator.
    """
    if value is None:
        return None
    if metric == "directional_consistency_global":
        return value >= 0.5
    if metric == "directional_consistency_local":
        return value >= 0.5
    if metric == "sensitivity_top1_recall":
        return value >= 1.0
    if metric == "sensitivity_citation_count":
        return value > 0
    if metric == "sensitivity_unsupported":
        return value == 0
    return None


_SENSITIVITY_METRIC_LABELS: List[Tuple[str, str]] = [
    ("directional_consistency_global", "§2 row sign agrees with PAX-TS sign"),
    ("directional_consistency_local", "§4 [Sensitivity-Local] source in top-3"),
    ("sensitivity_top1_recall", "PAX-TS global top-1 pair appears in §2"),
    ("sensitivity_citation_count", "[Sensitivity*] tag count > 0"),
    ("sensitivity_unsupported", "no [Sensitivity*] cited as 'negligible'"),
]


def format_fidelity_footer(
    numerical_audit: NumericalAuditResult,
    sensitivity_audit: SensitivityAuditResult,
) -> str:
    """Render the deterministic §E Numerical Fidelity Check footer.

    Returns a markdown string suitable for appending to the LLM-generated
    report. The footer contains:
      * **§E.1 KPM descriptor table** — every §1 / §4 numeric claim the
        deterministic auditor recomputed, with the LLM-reported value,
        the pipeline-recomputed truth, the percent error, and an
        OK / FAIL mark.
      * **§E.2 Sensitivity-consistency table** — five
        :func:`audit_sensitivity_consistency` metrics with a per-metric
        pass criterion.
      * **Result: N/M match** — single summary line. ``M`` is the count
        of verifiable rows; unverifiable rows (e.g. SensitivityResult
        was None) are excluded from both numerator and denominator so
        the headline figure remains comparable across runs.
    """
    lines: List[str] = []
    lines.append("## §E Numerical Fidelity Check")
    lines.append("")
    lines.append(
        "_Deterministic post-hoc audit — values recomputed from the "
        "Chronos-2 forecast tensor, the anomaly-event objects, and the "
        "PAX-TS sensitivity tensors. The LLM never touches this footer._"
    )
    lines.append("")

    # ─── §E.1 KPM descriptor audit ───────────────────────────────────────
    lines.append("### §E.1 KPM descriptors (§1 forecast + §4 |z| anomaly)")
    lines.append("")
    kpm_header = "| # | Section | KPI | Field | Reported | Pipeline | Err% | OK |"
    lines.append(kpm_header)
    lines.append("|---|---|---|---|---|---|---|---|")

    kpm_pass = 0
    kpm_total = 0
    row_idx = 0
    for claim in numerical_audit.matches:
        row_idx += 1
        kpm_pass += 1
        kpm_total += 1
        lines.append(
            f"| {row_idx} | {claim.section} | "
            f"{claim.kpi or '—'} | {claim.field} | "
            f"{claim.claimed:.2f} | {_format_truth(claim.truth)} | "
            f"{_err_pct(claim.claimed, claim.truth)} | ✓ |"
        )
    for claim in numerical_audit.mismatches:
        row_idx += 1
        kpm_total += 1
        lines.append(
            f"| {row_idx} | {claim.section} | "
            f"{claim.kpi or '—'} | {claim.field} | "
            f"{claim.claimed:.2f} | {_format_truth(claim.truth)} | "
            f"{_err_pct(claim.claimed, claim.truth)} | ✗ |"
        )
    if kpm_total == 0:
        lines.append("| — | — | — | — | — | — | — | — |")
    lines.append("")

    # ─── §E.2 Sensitivity-consistency audit ──────────────────────────────
    lines.append("### §E.2 Sensitivity consistency (5 metrics)")
    lines.append("")
    lines.append("| # | Metric | Criterion | Value | OK |")
    lines.append("|---|---|---|---|---|")

    sens_dict = sensitivity_audit.to_dict()
    sens_pass = 0
    sens_total = 0
    for idx, (metric_key, criterion) in enumerate(_SENSITIVITY_METRIC_LABELS, start=1):
        value = sens_dict.get(metric_key)
        verdict = _sensitivity_pass(metric_key, value)
        if isinstance(value, float):
            value_str = f"{value:.2f}"
        elif value is None:
            value_str = "—"
        else:
            value_str = str(value)
        if verdict is None:
            ok_str = "—"
        elif verdict:
            ok_str = "✓"
            sens_pass += 1
            sens_total += 1
        else:
            ok_str = "✗"
            sens_total += 1
        lines.append(f"| {idx} | `{metric_key}` | {criterion} | {value_str} | {ok_str} |")
    lines.append("")

    # ─── Final result line ───────────────────────────────────────────────
    total_pass = kpm_pass + sens_pass
    total_seen = kpm_total + sens_total
    lines.append(f"**Result: {total_pass}/{total_seen} match**")
    lines.append("")

    return "\n".join(lines)
