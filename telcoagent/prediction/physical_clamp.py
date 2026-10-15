"""Shared strict physical clamp for all forecast consumers.

This is the single implementation of KPI physical-range enforcement
(ADR 0001). All paths that emit predictions — foundation.py and every
baseline script — must route through :func:`apply_physical_clamp_strict`.

Design decisions:
- Raises ``KeyError`` for any KPI name absent from ``KPI_NORMALIZATION_MAX``
  (B-MED-3: no silent pass-through of unknown columns).
- Lower bound is always 0.0 (no KPI is physically negative).
- Upper bound is ``KPI_NORMALIZATION_MAX[kpi]`` (3GPP / TS 38.306 grounded).
- Integer-type KPIs (count-valued, e.g. RRC_Conn / DL_CQI) may be rounded to
  whole numbers AFTER the clip when the caller passes ``integer_kpi_set``.
  This module is a PURE function: it never imports the ontology and never
  decides which KPIs are integers — the caller derives that set (in production
  from the in-memory ``ThreeGPPOntology``) and passes it in.
- Returns a copy; the caller's array is never mutated.
"""

from __future__ import annotations

from typing import AbstractSet, Optional, Sequence

import numpy as np

from telcoagent.config import KPI_NORMALIZATION_MAX


def apply_physical_clamp_strict(
    pred: np.ndarray,
    kpi_names: Sequence[str],
    integer_kpi_set: Optional[AbstractSet[str]] = None,
) -> np.ndarray:
    """Clamp *pred* to [0, KPI_NORMALIZATION_MAX[kpi]] for each column.

    Parameters
    ----------
    pred:
        Float array of shape ``(T, C)`` where ``C == len(kpi_names)``.
    kpi_names:
        Ordered KPI names corresponding to columns in *pred*.
    integer_kpi_set:
        Optional set of KPI names that are integer-valued. Any column whose
        KPI name is in this set is rounded to whole numbers (via
        :func:`numpy.round`, round-half-to-even / banker's rounding) AFTER the
        ``[0, cap]`` clip. When ``None`` (default) or empty, no rounding is
        performed and the function is a pure clip — preserving backward
        compatibility with callers that do not quantize. This function does
        NOT decide which KPIs are integers; the caller derives the set (in
        production from the in-memory ``ThreeGPPOntology.integer_kpis``) and
        passes it in, keeping this module ontology-free.

    Returns
    -------
    np.ndarray
        A copy of *pred* with each column clamped to its physical range, and
        integer-KPI columns additionally rounded to whole numbers.

    Raises
    ------
    KeyError
        If any name in *kpi_names* is absent from ``KPI_NORMALIZATION_MAX``.
    ValueError
        If ``pred`` is not 2-D or ``len(kpi_names) != pred.shape[1]``.
    """
    if pred.ndim != 2:
        raise ValueError(f"pred must be 2-D, got shape {pred.shape}")
    if len(kpi_names) != pred.shape[1]:
        raise ValueError(f"kpi_names length {len(kpi_names)} != pred columns {pred.shape[1]}")

    # Validate all names up-front so we fail fast with a clear message.
    for kpi in kpi_names:
        if kpi not in KPI_NORMALIZATION_MAX:
            raise KeyError(
                f"KPI '{kpi}' is not in KPI_NORMALIZATION_MAX. "
                "Add it to telcoagent/config.py or remove it from kpi_names."
            )

    out = pred.copy()
    for i, kpi in enumerate(kpi_names):
        cap = KPI_NORMALIZATION_MAX[kpi]
        out[:, i] = np.clip(out[:, i], 0.0, cap)
        if integer_kpi_set is not None and kpi in integer_kpi_set:
            out[:, i] = np.round(out[:, i])

    return out
