"""Cumulative group split for the Chronos-2 cross-cell transfer sweep.

Pure and deterministic: the only IO is listing station CSV stems. Rank ALL 13
site groups in a fixed-seed order; for G training groups use ALL stations of the
first G ranked groups as train and ALL stations of the remaining (13-G) groups as
test. The sweep x-axis is the number of training site groups.

Key properties
--------------
- train ∪ test == all stations for every G  (full coverage).
- train ∩ test == ∅ for every G             (no leakage, by construction).
- train is nested: G=1 ⊂ G=2 ⊂ ... ⊂ G=12 (single ranked order).
- G=0  → ([], all stations)  — pure zero-shot baseline.
- G=13 → raises               — nothing left to evaluate.

No silent fallback (engineering principle #1): an oversized n_train_groups raises
rather than returning a degraded result.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

from scripts.baseline_study.data import station_files


def group_of(station_id: str) -> str:
    """``station_A_10`` or bare ``A_10`` -> ``"A"`` (the site-group token)."""
    stem = station_id[len("station_") :] if station_id.startswith("station_") else station_id
    return stem.split("_", 1)[0]


def all_station_ids(data_dir: str | Path) -> List[str]:
    """Sorted ``station_*`` stems present under ``data_dir`` (the only IO)."""
    return [p.stem for p in station_files(data_dir)]


def ranked_groups(all_ids: List[str], seed: int) -> List[str]:
    """ALL site-group letters in a single fixed-seed shuffled order.

    The one global ranked order is the x-axis of the cumulative sweep: adding
    group G to the training set means taking the first G entries of this list.
    """
    groups = sorted({group_of(i) for i in all_ids})
    rng = random.Random(seed)
    rng.shuffle(groups)
    return groups


def cumulative_split(
    all_ids: List[str], n_train_groups: int, seed: int
) -> Tuple[List[str], List[str]]:
    """Return (train_ids, test_ids) for the given number of training groups.

    train_ids = sorted station ids of the first ``n_train_groups`` ranked groups.
    test_ids  = sorted station ids of the remaining groups.

    ``n_train_groups == 0``  → ([], all stations) — zero-shot baseline.
    ``n_train_groups == 13`` → raises              — nothing left to evaluate.
    """
    if n_train_groups < 0:
        raise ValueError(f"n_train_groups must be >= 0, got {n_train_groups}")
    ranked = ranked_groups(all_ids, seed)
    if n_train_groups >= len(ranked):
        raise ValueError(
            f"n_train_groups={n_train_groups} leaves no groups for the test set "
            f"(total groups: {len(ranked)})"
        )
    train_set = set(ranked[:n_train_groups])
    test_set = set(ranked[n_train_groups:])
    train_ids = sorted(i for i in all_ids if group_of(i) in train_set)
    test_ids = sorted(i for i in all_ids if group_of(i) in test_set)
    return train_ids, test_ids
