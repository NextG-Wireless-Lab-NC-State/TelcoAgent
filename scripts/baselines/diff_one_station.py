#!/usr/bin/env python3
"""Drift check between two per-station result JSONs.

Used by the paper-grade re-run workflow: after re-running a baseline with
the patched seed-aware harness, compare a single station against the
archived legacy result. Exits 0 if max per-KPI |delta| <= tolerance,
exits 1 otherwise — safe for use as a CI / shell guard.

Usage:
    python scripts/baselines/diff_one_station.py \\
        --new output/7d_prediction_tsfm/chronos2_ctx_sweep_v2/ctx_672h/station_A_10.json \\
        --old output/7d_prediction_tsfm/_archive_pre_seed/chronos2_ctx_sweep_v2/ctx_672h/station_A_10.json \\
        --metric MASE --tol 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

SUPPORTED_METRICS = ("MASE", "nRMSE", "RMSE", "MAE", "sMAPE")


def _load_per_kpi(path: Path) -> Dict[str, dict]:
    with open(path) as f:
        data = json.load(f)
    try:
        return data["metrics"]["per_kpi"]
    except KeyError as exc:
        raise ValueError(f"{path}: missing 'metrics.per_kpi' key") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diff per-KPI metrics between two per-station JSON results."
    )
    parser.add_argument("--new", required=True, type=Path, help="Path to the new station JSON")
    parser.add_argument(
        "--old", required=True, type=Path, help="Path to the archived/old station JSON"
    )
    parser.add_argument(
        "--metric",
        default="MASE",
        choices=SUPPORTED_METRICS,
        help=f"Metric to diff (default: MASE; one of {SUPPORTED_METRICS})",
    )
    parser.add_argument(
        "--tol", type=float, default=0.05, help="Max acceptable |delta| per KPI (default 0.05)"
    )
    parser.add_argument("--quiet", action="store_true", help="Print only PASS/FAIL line")
    args = parser.parse_args()

    new_per_kpi = _load_per_kpi(args.new)
    old_per_kpi = _load_per_kpi(args.old)

    common = sorted(set(new_per_kpi) & set(old_per_kpi))
    if not common:
        print(f"FAIL: no overlapping KPIs between {args.new} and {args.old}")
        return 1

    max_delta = 0.0
    over_tol = []
    if not args.quiet:
        print(f"Diff metric: {args.metric}, tolerance: {args.tol}")
        print(f"{'KPI':<14}{'new':>12}{'old':>12}{'|delta|':>12}{'status':>8}")
        print("-" * 58)
    for kpi in common:
        new_v = new_per_kpi[kpi].get(args.metric)
        old_v = old_per_kpi[kpi].get(args.metric)
        if new_v is None or old_v is None:
            print(f"FAIL: {kpi} missing {args.metric} in one side " f"(new={new_v}, old={old_v})")
            return 1
        delta = abs(float(new_v) - float(old_v))
        max_delta = max(max_delta, delta)
        ok = delta <= args.tol
        if not ok:
            over_tol.append(kpi)
        if not args.quiet:
            mark = "OK" if ok else "DRIFT"
            print(f"{kpi:<14}{new_v:>12.4f}{old_v:>12.4f}{delta:>12.4f}{mark:>8}")

    if over_tol:
        print(
            f"\nFAIL: {len(over_tol)} KPI(s) over tolerance "
            f"(max |delta| = {max_delta:.4f} > {args.tol}): "
            f"{', '.join(over_tol)}"
        )
        return 1

    print(f"\nPASS: max |delta| = {max_delta:.4f} <= {args.tol}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
