#!/usr/bin/env python3
"""Component-ablation figures — explanation arms + prediction comparison.

Reads the two component-ablation artefacts and renders a 2-panel figure
(PNG + PDF):

* Panel A — explanation RAGAS: Faithfulness + Answer Relevancy mean per
  arm (Full / −KG / −Predictor / −Explainer) with bootstrap-style
  error bars (station stdev / sqrt(n)).
* Panel B — prediction: Chronos-2 vs seasonal-naive 7-KPI-mean nRMSE
  and MASE with the same error-bar convention.

Inputs (defaults assume the standard component-ablation layout):
* ``output/ablation/component/judge/judge_per_station.csv``
* ``output/ablation/component/prediction_per_station.csv``

Output:
* ``output/ablation/component/component_ablation.png`` + ``.pdf``

Idempotent — safe to re-run.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ARMS = ["kg_on", "kg_off", "minus_predictor", "minus_explainer"]
_ARM_LABELS = {
    "kg_on": "Full",
    "kg_off": "-KG",
    "minus_predictor": "-Predictor",
    "minus_explainer": "-Explainer",
}
_DEFAULT_DIR = Path("/home/gkim26/Desktop/workplace/telcoagent/output/ablation/component")


def _norm(name: str) -> str:
    return name[len("station_") :] if name.startswith("station_") else name


def _load_judge(judge_csv: Path) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    out: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    with judge_csv.open() as fh:
        for r in csv.DictReader(fh):
            sid = _norm(r["station_id"])

            def _f(key: str) -> Optional[float]:
                raw = r.get(key, "")
                try:
                    return float(raw) if raw not in (None, "") else None
                except ValueError:
                    return None

            out.setdefault(sid, {})[r["arm"]] = {
                "faithfulness": _f("faithfulness"),
                "answer_relevancy": _f("answer_relevancy"),
                "n_events": _f("n_events"),
            }
    return out


def _mean_sem(vals: List[float]) -> tuple[float, float]:
    arr = np.array([v for v in vals if v is not None], dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, sem


def _load_prediction(pred_csv: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    with pred_csv.open() as fh:
        for r in csv.DictReader(fh):
            sid = _norm(r["station_id"])
            out.setdefault(sid, {})[r["forecaster"]] = {
                "mean_nRMSE": float(r["mean_nRMSE"]),
                "mean_MASE": float(r["mean_MASE"]),
            }
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Component-ablation figure.")
    ap.add_argument(
        "--judge-csv", type=Path, default=_DEFAULT_DIR / "judge" / "judge_per_station.csv"
    )
    ap.add_argument(
        "--prediction-csv", type=Path, default=_DEFAULT_DIR / "prediction_per_station.csv"
    )
    ap.add_argument("--out-prefix", type=Path, default=_DEFAULT_DIR / "component_ablation")
    args = ap.parse_args(argv)

    judge = _load_judge(args.judge_csv)
    pred = _load_prediction(args.prediction_csv)
    stations = sorted(judge.keys())

    # ── Panel A data — explanation RAGAS per arm ─────────────────────────
    faith_mean, faith_sem, ar_mean, ar_sem = [], [], [], []
    for arm in _ARMS:
        f_vals = [judge[s].get(arm, {}).get("faithfulness") for s in stations]
        # AR only on event-bearing stations for this arm
        a_vals = [
            judge[s].get(arm, {}).get("answer_relevancy")
            for s in stations
            if (judge[s].get(arm, {}).get("n_events") or 0) >= 1
        ]
        fm, fs = _mean_sem(f_vals)
        am, as_ = _mean_sem(a_vals)
        faith_mean.append(fm)
        faith_sem.append(fs)
        ar_mean.append(am)
        ar_sem.append(as_)

    # ── Panel B data — prediction ────────────────────────────────────────
    pstations = sorted(pred.keys())
    chr_nrmse = _mean_sem([pred[s]["chronos2"]["mean_nRMSE"] for s in pstations])
    nai_nrmse = _mean_sem([pred[s]["seasonal_naive"]["mean_nRMSE"] for s in pstations])
    chr_mase = _mean_sem([pred[s]["chronos2"]["mean_MASE"] for s in pstations])
    nai_mase = _mean_sem([pred[s]["seasonal_naive"]["mean_MASE"] for s in pstations])

    # ── render ───────────────────────────────────────────────────────────
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    labels = [_ARM_LABELS[a] for a in _ARMS]
    x = np.arange(len(labels))
    w = 0.36
    axA.bar(
        x - w / 2, faith_mean, w, yerr=faith_sem, capsize=3, label="Faithfulness", color="#3b6fb0"
    )
    axA.bar(
        x + w / 2, ar_mean, w, yerr=ar_sem, capsize=3, label="Answer Relevancy", color="#c0563b"
    )
    axA.set_xticks(x)
    axA.set_xticklabels(labels)
    axA.set_ylabel("RAGAS score (0-1)")
    axA.set_title("(A) Explanation quality per ablation arm")
    axA.set_ylim(0, 1)
    axA.legend(fontsize=8)

    bx = np.arange(2)
    axB.bar(
        bx - w / 2,
        [chr_nrmse[0], chr_mase[0]],
        w,
        yerr=[chr_nrmse[1], chr_mase[1]],
        capsize=3,
        label="Chronos-2 (Full)",
        color="#3b6fb0",
    )
    axB.bar(
        bx + w / 2,
        [nai_nrmse[0], nai_mase[0]],
        w,
        yerr=[nai_nrmse[1], nai_mase[1]],
        capsize=3,
        label="seasonal-naive (-Predictor)",
        color="#c0563b",
    )
    axB.set_xticks(bx)
    axB.set_xticklabels(["7-KPI mean nRMSE", "7-KPI mean MASE"])
    axB.set_ylabel("error (lower is better)")
    axB.set_title("(B) Prediction accuracy")
    axB.legend(fontsize=8)

    fig.tight_layout()
    png = args.out_prefix.with_suffix(".png")
    pdf = args.out_prefix.with_suffix(".pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
