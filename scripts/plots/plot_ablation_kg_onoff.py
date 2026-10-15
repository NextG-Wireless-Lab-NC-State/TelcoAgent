#!/usr/bin/env python3
"""AX-2: Plot explainer KG on/off over n=5 stations across 2 independent runs.

Reads RAGAS judge per-station CSV(s), computes per-arm Faithfulness +
Answer-Relevancy mean, paired gap, σ_per_row across the 2 runs, writes:
  - results.csv (station x arm x faithfulness_r1, faithfulness_r2,
                 answer_relevancy_r1, answer_relevancy_r2, mean_F, sigma_F)
  - results.xlsx
  - kg_onoff_main.png + .pdf  (faithfulness bars per arm with error bars)

Idempotent.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

DEFAULT_DIR = Path("/home/gkim26/Desktop/workplace/telcoagent/output/ablation/kg_onoff")
ARMS = ["kg_on", "kg_off"]


def _load_judge(judge_csv: Path) -> dict:
    """Returns {(station, arm): {'faithfulness': f, 'answer_relevancy': a}}."""
    out = {}
    if not judge_csv.exists():
        return out
    with open(judge_csv) as f:
        for r in csv.DictReader(f):
            sid, arm = r["station_id"], r["arm"]
            try:
                f_val = float(r["faithfulness"]) if r["faithfulness"] else None
            except ValueError:
                f_val = None
            try:
                a_val = float(r["answer_relevancy"]) if r["answer_relevancy"] else None
            except ValueError:
                a_val = None
            out[(sid, arm)] = {"faithfulness": f_val, "answer_relevancy": a_val}
    return out


def aggregate(run_dir: Path) -> Path:
    r1 = _load_judge(run_dir / "run1" / "judge" / "judge_per_station.csv")
    r2 = _load_judge(run_dir / "run2" / "judge" / "judge_per_station.csv")
    # Stations present in both runs (n=5 subset)
    stations = sorted({s for (s, _) in r1.keys()} & {s for (s, _) in r2.keys()})
    rows = []
    for sid in stations:
        for arm in ARMS:
            v1 = r1.get((sid, arm), {})
            v2 = r2.get((sid, arm), {})
            f1, f2 = v1.get("faithfulness"), v2.get("faithfulness")
            a1, a2 = v1.get("answer_relevancy"), v2.get("answer_relevancy")
            sigma_f = statistics.stdev([f1, f2]) if f1 is not None and f2 is not None else None
            mean_f = statistics.mean([f1, f2]) if f1 is not None and f2 is not None else None
            rows.append(
                {
                    "station_id": sid,
                    "arm": arm,
                    "faithfulness_r1": f1,
                    "faithfulness_r2": f2,
                    "answer_relevancy_r1": a1,
                    "answer_relevancy_r2": a2,
                    "mean_faithfulness": round(mean_f, 4) if mean_f else None,
                    "sigma_per_row_faithfulness": round(sigma_f, 4) if sigma_f else None,
                }
            )
    out_csv = run_dir / "results.csv"
    if not rows:
        # Stub for downstream: write header-only
        with open(out_csv, "w", newline="") as f:
            f.write(
                "station_id,arm,faithfulness_r1,faithfulness_r2,answer_relevancy_r1,"
                "answer_relevancy_r2,mean_faithfulness,sigma_per_row_faithfulness\n"
            )
        return out_csv
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    try:
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "kg_onoff"
        ws.append(list(rows[0].keys()))
        for r in rows:
            ws.append(list(r.values()))
        wb.save(run_dir / "results.xlsx")
    except ImportError:
        pass
    return out_csv


def plot(csv_path: Path, out_prefix: Path) -> None:
    import json

    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return
    by_arm = {a: [] for a in ARMS}
    sigma_arm: dict[str, list[float]] = {a: [] for a in ARMS}
    for r in rows:
        if not r["mean_faithfulness"]:
            continue
        by_arm[r["arm"]].append(float(r["mean_faithfulness"]))
        if r.get("sigma_per_row_faithfulness"):
            try:
                sigma_arm[r["arm"]].append(float(r["sigma_per_row_faithfulness"]))
            except ValueError:
                pass  # non-numeric note (e.g. "deterministic_judge") — skip
    n_stations = max(len(by_arm[a]) for a in ARMS)

    # If per-row sigma is absent (deterministic judge), fall back to paired stdev
    # from results_aggregates.json so the error bar reflects the real noise floor.
    err_label = "σ_per_row"
    agg_json = csv_path.parent / "results_aggregates.json"
    if not any(sigma_arm[a] for a in ARMS) and agg_json.exists():
        agg = json.loads(agg_json.read_text())
        paired_stdev = agg.get("faithfulness_paired_stdev")
        if paired_stdev is not None:
            for a in ARMS:
                sigma_arm[a] = [float(paired_stdev)]
            err_label = "paired σ"

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        }
    )
    means = [statistics.mean(by_arm[a]) if by_arm[a] else 0 for a in ARMS]
    errs = [statistics.mean(sigma_arm[a]) if sigma_arm[a] else 0 for a in ARMS]
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.bar(
        ARMS,
        means,
        yerr=errs,
        capsize=4,
        color=["#1f77b4", "#ff7f0e"],
        alpha=0.85,
    )
    ax.set_ylabel(f"RAGAS Faithfulness (mean ± {err_label}, n={n_stations})")
    ax.set_title("AX-2: Explainer KG on/off")
    fig.tight_layout(pad=0.5)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_prefix}_main.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_prefix}_main.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help="Prefix path for output files; "
        "writes {prefix}_main.png and {prefix}_main.pdf. "
        "Defaults to '{--dir}/kg_onoff' for back-compat.",
    )
    args = ap.parse_args()
    csv_path = args.input or aggregate(args.dir)
    out_prefix = args.out_prefix or (args.dir / "kg_onoff")
    plot(csv_path, out_prefix)
    print(f"wrote {csv_path}, {out_prefix}_main.{{png,pdf}}")


if __name__ == "__main__":
    main()
