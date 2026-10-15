"""Shared project constants for TelcoAgent.

Single source of truth for the 7-KPI contract, forecast window sizes, KPI
aliases, and environment-driven defaults.  Anything that was previously
duplicated across ``foundation.py``, ``rag/agents.py``, ``rag/prompts.py``,
``neo4j_store.py``, and the scripts now lives here.

Rules:
    - Do NOT reorder ``CORE_KPI_NAMES`` — downstream code depends on the
      canonical 3GPP TS 28.554 order.
    - Keep constants primitive-valued; no imports from inside ``telcoagent``
      to avoid circularity.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# 7-KPI canonical contract (3GPP TS 28.554)
# =============================================================================

CORE_KPI_NAMES: Tuple[str, ...] = (
    "RRC_Conn",
    "DL_CQI",
    "DL_iBler",
    "DL_rBler",
    "MAC_DL_Eff",
    "PRB_Util",
    "Throughput",
)
N_CHANNELS: int = len(CORE_KPI_NAMES)  # 7

# Per-KPI normalization caps for PAX-TS perturbation scaling and explainer
# input scaling. Standards-grounded:
#   - RRC_Conn: cell-level RRC connected UE capacity (TS 38.331).
#   - DL_CQI:   4-bit CQI index (TS 38.214 Sec. 5.2.2.1, range 0-15).
#   - DL_*Bler: percentage (TS 28.552 Sec. 5.1.1.x).
#   - MAC_DL_Eff & Throughput: TS 38.306 peak DL data rate for the deployed
#     PCS-band config (n2/n25, observed peak matches 10 MHz x 4x4 MIMO x
#     256-QAM ~= 222 Mbps OR 20 MHz x 2x2 MIMO x 256-QAM ~= 227 Mbps; cap
#     250 Mbps gives air-interface headroom). MAC and IP share the same
#     air-interface bound; their observed difference comes from 28.552
#     measurement-window definitions (IP excludes idle TTIs, MAC averages
#     over the full window) -- not from different physical caps.
#   - PRB_Util: percentage (TS 28.552 Sec. 5.1.1.2.x).
KPI_NORMALIZATION_MAX: Dict[str, float] = {
    "RRC_Conn": 500.0,
    "DL_CQI": 15.0,
    "DL_iBler": 100.0,
    "DL_rBler": 100.0,
    "MAC_DL_Eff": 250000.0,
    "PRB_Util": 100.0,
    "Throughput": 250000.0,
}

# Aliases used for robust matching in free-text LLM outputs.
KPI_ALIASES: Dict[str, List[str]] = {
    "RRC_Conn": ["RRC_Conn", "RRC Conn", "rrc_conn"],
    "DL_CQI": ["DL_CQI", "DL CQI", "dl_cqi", "CQI"],
    "DL_iBler": ["DL_iBler", "DL iBler", "dl_ibler", "iBLER"],
    "DL_rBler": ["DL_rBler", "DL rBler", "dl_rbler", "rBLER"],
    "MAC_DL_Eff": ["MAC_DL_Eff", "MAC DL Eff", "mac_dl_eff"],
    "PRB_Util": ["PRB_Util", "PRB Util", "prb_util"],
    "Throughput": ["Throughput", "throughput", "DL_Throughput"],
}

# =============================================================================
# Forecast window sizes (hours)
# =============================================================================

CONTEXT_LENGTH_H: int = 1944  # 81 days — Chronos-2 context window
# (sweep-best per output/amazon_chronos-2_h7d_ctx_sweep)
PREDICTION_LENGTH_H: int = 168  # 7 days  — forecast horizon
# Runtime is 2-way zero-shot (days 1-81 context -> days 82-88 target); there is
# no separate demonstration pool. See docs/data_flow.md and the live split in
# scripts/baselines/foundation_utils.py::split_data.

# =============================================================================
# Environment-driven paths
# =============================================================================

DEFAULT_KG_PATH: str = os.environ.get(
    "TELCOAGENT_KG_PATH",
    str(_REPO_ROOT / "data" / "enriched_kg.json"),
)
DEFAULT_SPECS_DIR: str = os.environ.get(
    "TELCOAGENT_SPECS_DIR",
    str(_REPO_ROOT / "data" / "specs"),
)


# =============================================================================
# Agent harness flags ("Do No Harm" + holdout-validated refinement)
# =============================================================================
# Tunable via env so the ablation script can flip individual levers.
def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default
