"""Data loading and leak-free split utilities for the 115-station benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from telcoagent.config import CORE_KPI_NAMES

KPI_NAMES = list(CORE_KPI_NAMES)
KPI_COLS_CSV = [
    "RRC_Conn_Count_Avg",
    "DL_CQI",
    "DL_iBler",
    "DL_rBler",
    "MAC DL Eff TP",
    "DL PRB Utilization",
    "DL_IP_Throughput",
]

HOURS_PER_DAY = 24
TRAIN_DAYS = 67
VAL_DAYS = 14
TEST_DAYS = 7
TRAIN_H = TRAIN_DAYS * HOURS_PER_DAY
VAL_H = VAL_DAYS * HOURS_PER_DAY
TEST_H = TEST_DAYS * HOURS_PER_DAY
TOTAL_H = (TRAIN_DAYS + VAL_DAYS + TEST_DAYS) * HOURS_PER_DAY

# Day 82 (the forecast target, days 82-88) begins at this row; day 81 ends here.
# Every fine-tune window must stay strictly below this index to be leak-free.
TEST_START_IDX = TRAIN_H + VAL_H  # = 1944

# Fine-tune inference/training context (fixed for the data-scaling experiment) and
# the forecast horizon. With context=horizon=168, Chronos2Dataset's default
# ``min_past = prediction_length`` filters out any station slice shorter than
# ``min_past + prediction_length`` = 2*168 = 336h, so the minimum regime is 2
# trailing weeks (336h); a single week (168h) cannot form one ctx+horizon window.
FINETUNE_CONTEXT_H = 168
FINETUNE_HORIZON_H = 168
FINETUNE_MIN_WEEKS = 2


@dataclass(frozen=True)
class StationSeries:
    station_id: str
    values: np.ndarray


@dataclass(frozen=True)
class SplitSeries:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def station_files(data_dir: str | Path, stations: Iterable[str] | None = None) -> List[Path]:
    data_path = Path(data_dir)
    files = sorted(data_path.glob("station_*.csv"))
    if stations:
        wanted = {s if s.startswith("station_") else f"station_{s}" for s in stations}
        files = [f for f in files if f.stem in wanted]
    return files


def load_station(csv_path: str | Path) -> StationSeries:
    path = Path(csv_path)
    df = pd.read_csv(path)
    missing = [c for c in KPI_COLS_CSV if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing KPI columns: {missing}")
    values = df[KPI_COLS_CSV].to_numpy(dtype=np.float64)
    if values.shape[0] < TOTAL_H:
        raise ValueError(f"{path}: expected at least {TOTAL_H} rows, got {values.shape[0]}")
    values = values[:TOTAL_H]
    return StationSeries(station_id=path.stem, values=values)


def split_station(values: np.ndarray) -> SplitSeries:
    train = values[:TRAIN_H]
    val = values[TRAIN_H : TRAIN_H + VAL_H]
    test = values[TRAIN_H + VAL_H : TRAIN_H + VAL_H + TEST_H]
    if train.shape[0] != TRAIN_H or val.shape[0] != VAL_H or test.shape[0] != TEST_H:
        raise ValueError("Invalid split sizes; expected 67d/14d/7d")
    return SplitSeries(train=train, val=val, test=test)


def finetune_regime(finetune_weeks: int | str) -> str:
    """Derive the regime label from the swept knob.

    0 -> "zero" (zero-shot, no fine-tune); 0 < N < full -> "few"; "full" -> "full".
    Kept model-agnostic: only the regime name depends on the knob, not the model.
    """
    if finetune_weeks == "full":
        return "full"
    weeks = int(finetune_weeks)
    if weeks == 0:
        return "zero"
    if weeks < 0:
        raise ValueError(f"finetune_weeks must be >= 0 or 'full', got {finetune_weeks!r}")
    return "few"


def finetune_slices(
    series_list: Iterable[np.ndarray | StationSeries],
    *,
    finetune_weeks: int | str,
    context_h: int = FINETUNE_CONTEXT_H,
    horizon_h: int = FINETUNE_HORIZON_H,
    stride_h: int = HOURS_PER_DAY,
) -> tuple[List[np.ndarray], int]:
    """Pooled trailing-N-week fine-tune slices ending at the day-81 boundary.

    For ``finetune_weeks=N`` each station contributes ``values[1944 - 7*N*24 : 1944]``
    (shape ``(7N*24, 7)``); ``"full"`` contributes the entire pre-test range
    ``values[0:1944]``. Slices are pooled across all stations.

    Returns ``(slices, n_train_windows)`` where ``n_train_windows`` is the TOTAL
    number of sliding (context+horizon) windows across all pooled station slices,
    counted with ``stride_h`` steps. This is what makes compute scale with the
    amount of fine-tune data: a larger N yields more windows and therefore more
    training steps when the caller converts epochs to steps via
    ``num_steps = epochs * ceil(n_train_windows / batch_size)``.

    Per station slice of length L:
        windows = max(0, (L - context_h - horizon_h) // stride_h + 1)

    Expected values with 115 stations, context_h=horizon_h=168, stride_h=24:
        2wk  (L=336h)  : 1 win/station  -> 115  total
        5wk  (L=840h)  : 22 win/station -> 2530 total
        full (L=1944h) : 68 win/station -> 7820 total

    Zero-shot (N=0) must not call this: the regime takes no fine-tune data. To keep
    a single code path (no silent fallback), ``finetune_weeks < 2`` raises.
    """
    if finetune_weeks == "full":
        start = 0
    else:
        weeks = int(finetune_weeks)
        if weeks < FINETUNE_MIN_WEEKS:
            raise ValueError(
                f"finetune_weeks must be >= {FINETUNE_MIN_WEEKS} (a 168h week cannot form a "
                f"{context_h}+{horizon_h} window); got {finetune_weeks!r}. Zero-shot must not "
                "build fine-tune slices."
            )
        start = TEST_START_IDX - weeks * 7 * HOURS_PER_DAY
    if start < 0:
        raise ValueError(f"finetune_weeks={finetune_weeks!r} exceeds available pre-test history")

    slices: List[np.ndarray] = []
    n_train_windows = 0
    for series in series_list:
        values = series.values if isinstance(series, StationSeries) else series
        sl = values[start:TEST_START_IDX]
        # Leak guard: the slice must end exactly at the day-81 boundary and never
        # reach any day-82+ index. (start:TEST_START_IDX is exclusive of 1944.)
        assert sl.shape[0] == TEST_START_IDX - start, "fine-tune slice length mismatch"
        slices.append(sl)
        # Sliding-window count for this station slice (same formula as the
        # supervised harness make_supervised_windows; 0 if slice is too short).
        n_train_windows += max(0, (sl.shape[0] - context_h - horizon_h) // stride_h + 1)
    return slices, n_train_windows


def split_finetune_slices(
    slices: List[np.ndarray],
    *,
    val_fraction: float = 0.15,
    context_h: int = FINETUNE_CONTEXT_H,
    horizon_h: int = FINETUNE_HORIZON_H,
    stride_h: int = HOURS_PER_DAY,
    seed: int = 42,
) -> tuple[List[np.ndarray], List[np.ndarray], int, int]:
    """Split pooled fine-tune slices into train and val portions (fixed-seed shuffle).

    Because all station slices within a given regime are equal length, holding out
    ``val_fraction`` of the slices is equivalent to holding out the same fraction
    of sliding windows. The split is at the SLICE level: shuffle with ``seed``,
    take the last ``ceil(val_fraction * N)`` slices as val, the rest as train.

    Returns ``(train_slices, val_slices, n_train_windows, n_val_windows)`` where
    window counts use the same ``(L - ctx - hrz) // stride + 1`` formula as
    ``finetune_slices``.

    Raises ``ValueError`` if the pool is too small to form at least one train and
    one val slice (e.g. only 1 station total).
    """
    n = len(slices)
    n_val = math.ceil(val_fraction * n)
    n_train = n - n_val
    if n_train < 1 or n_val < 1:
        raise ValueError(
            f"too few slices to split into train+val: {n} slice(s) with "
            f"val_fraction={val_fraction} gives n_train={n_train}, n_val={n_val}; "
            "need at least 2 slices (2 stations)"
        )
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    train_slices = [slices[i] for i in idx[:n_train]]
    val_slices = [slices[i] for i in idx[n_train:]]

    def _count_windows(sl: np.ndarray) -> int:
        return max(0, (sl.shape[0] - context_h - horizon_h) // stride_h + 1)

    n_train_windows = sum(_count_windows(s) for s in train_slices)
    n_val_windows = sum(_count_windows(s) for s in val_slices)
    return train_slices, val_slices, n_train_windows, n_val_windows


def make_supervised_windows(
    series_list: Iterable[np.ndarray],
    *,
    context_h: int,
    horizon_h: int,
    stride_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for series in series_list:
        max_start = series.shape[0] - context_h - horizon_h
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, stride_h):
            xs.append(series[start : start + context_h])
            ys.append(series[start + context_h : start + context_h + horizon_h])
    if not xs:
        raise ValueError(
            f"No windows created for context_h={context_h}, horizon_h={horizon_h}, stride_h={stride_h}"
        )
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)
