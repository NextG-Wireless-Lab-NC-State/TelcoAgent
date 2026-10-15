"""Q6 cross-judge aggregation: KG-null robustness across 3 local judges.

Consumes the three judge CSV files produced by running
``run_explainer_ablation_judge.py`` with each local Ollama model against
the Q1 kg_on / kg_off explanations:

    q6_crossjudge/oransight/judge_per_station.csv
    q6_crossjudge/gemma3/judge_per_station.csv
    q6_crossjudge/qwen/judge_per_station.csv

For each judge the script computes the station-paired kg_on − kg_off diff
(Faithfulness and Answer Relevancy) using the same paired_comparison()
function as the main component-ablation analysis.  It then reports:

  1. Per-judge paired statistics table (mean_diff, CI, p, d, n).
  2. Per-station Faithfulness score matrix (one column per judge) and the
     Pearson correlation + ICC(2,1) between judge pairs.
  3. KG-null sign-consistency table: for each judge, does the kg_on − kg_off
     Faithfulness diff have the same sign (or is it zero / indeterminate)?
     The robustness verdict is "consistent-null" when all three judges
     produce a CI that straddles 0 (i.e. cannot reject null), "consistent-
     detect" when all three CIs exclude 0 on the same side, and
     "inconsistent" when judges disagree on direction or detectability.

Outputs (under ``--out-dir``)
-----------------------------
  cross_judge_stats.csv      per-judge paired statistics (Faithfulness + AR)
  cross_judge_stats.json     same, machine-readable
  cross_judge_correlation.csv judge-pair Pearson r + ICC for Faithfulness
  cross_judge_robustness.csv  KG-null sign-consistency verdict
  cross_judge_matrix.csv     station × judge Faithfulness (wide)

Usage
-----
    PYTHONPATH=. python scripts/ablation/analyze_cross_judge.py \\
        --judge-dir output/ablation/six_axis/q6_crossjudge \\
        --out-dir output/ablation/six_axis/q6_crossjudge/analysis

Notes
-----
- Answer Relevancy is restricted to event-bearing stations where BOTH arms
  have a scored (non-NaN) AR value — same treatment as the main analysis.
- ICC is the two-way mixed-effects ICC(2,1) (consistency) implemented here
  directly via the standard one-way ANOVA decomposition so scipy.stats is
  the only dependency.
- Stations are matched by their normalised name (strip ``station_`` prefix
  for matching; keep original form in outputs).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ablation.stats import paired_comparison  # noqa: E402

# Judge names must match the sub-directory names under --judge-dir.
_JUDGE_NAMES: Tuple[str, ...] = ("oransight", "gemma3", "qwen")
_ARMS: Tuple[str, str] = ("kg_on", "kg_off")
_METRICS: Tuple[str, str] = ("faithfulness", "answer_relevancy")


# ─── IO ───────────────────────────────────────────────────────────────────


def _norm_station(name: str) -> str:
    """Strip ``station_`` prefix for cross-file matching."""
    return name[len("station_") :] if name.startswith("station_") else name


def _load_judge_csv(
    path: Path,
) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """Load one judge_per_station.csv.

    Returns ``{station_norm -> {arm -> {faithfulness, answer_relevancy, n_events}}}``.
    """
    out: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            sid = _norm_station(row["station_id"])
            arm = row["arm"]

            def _f(key: str) -> Optional[float]:
                raw = row.get(key, "")
                if raw in ("", "nan", "None", "null"):
                    return None
                try:
                    return float(raw)
                except ValueError:
                    return None

            out.setdefault(sid, {})[arm] = {
                "faithfulness": _f("faithfulness"),
                "answer_relevancy": _f("answer_relevancy"),
                "n_events": _f("n_events"),
            }
    return out


# ─── Statistics helpers ───────────────────────────────────────────────────


def _extract_paired_arrays(
    data: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    metric: str,
    arm_a: str = "kg_on",
    arm_b: str = "kg_off",
    require_both_scored: bool = False,
) -> Tuple[List[str], List[float], List[float]]:
    """Return parallel lists (station_ids, scores_a, scores_b).

    When ``require_both_scored=True``, stations where either arm's metric
    is None are excluded (used for AR which is only valid when events exist).
    """
    sids: List[str] = []
    a_vals: List[float] = []
    b_vals: List[float] = []
    for sid in sorted(data):
        arms = data[sid]
        val_a = arms.get(arm_a, {}).get(metric)
        val_b = arms.get(arm_b, {}).get(metric)
        if val_a is None or val_b is None:
            if require_both_scored:
                continue
            # For faithfulness treat missing as 0.0 (scored absent = worst).
            val_a = val_a if val_a is not None else 0.0
            val_b = val_b if val_b is not None else 0.0
        sids.append(sid)
        a_vals.append(float(val_a))
        b_vals.append(float(val_b))
    return sids, a_vals, b_vals


def _icc_two_way_mixed(
    judge_matrix: np.ndarray,
) -> float:
    """ICC(2,1) consistency — two-way mixed-effects model.

    Parameters
    ----------
    judge_matrix : ndarray of shape (n_stations, n_judges)

    Returns
    -------
    ICC(2,1) value in [-1, 1].

    Formula reference: Shrout & Fleiss (1979), equation for ICC(2,1).
    """
    n, k = judge_matrix.shape
    if n < 2 or k < 2:
        return float("nan")

    grand_mean = judge_matrix.mean()
    row_means = judge_matrix.mean(axis=1)
    col_means = judge_matrix.mean(axis=0)

    ss_total = ((judge_matrix - grand_mean) ** 2).sum()
    ss_rows = k * ((row_means - grand_mean) ** 2).sum()
    ss_cols = n * ((col_means - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_error
    if denom == 0.0:
        return float("nan")
    return float((ms_rows - ms_error) / denom)


def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    with np.errstate(invalid="ignore"):
        r = float(np.corrcoef(a, b)[0, 1])
    return r


# ─── Robustness verdict ───────────────────────────────────────────────────


def _robustness_verdict(stats_rows: List[dict]) -> str:
    """Classify KG-null sign consistency across judges.

    Returns one of:
        "consistent-null"    all judges: 95% CI straddles 0 (cannot reject null)
        "consistent-detect"  all judges: CI excludes 0 on the same side
        "inconsistent"       judges disagree on direction or detectability
    """
    verdicts: List[str] = []
    for row in stats_rows:
        ci_low = row["ci_low"]
        ci_high = row["ci_high"]
        if ci_low <= 0.0 <= ci_high:
            verdicts.append("null")
        elif ci_low > 0.0:
            verdicts.append("positive")
        else:
            verdicts.append("negative")

    if all(v == "null" for v in verdicts):
        return "consistent-null"
    if len(set(v for v in verdicts if v != "null")) == 1 and "null" not in verdicts:
        return "consistent-detect"
    return "inconsistent"


# ─── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Q6 cross-judge aggregation: paired kg_on−kg_off statistics "
            "for 3 local Ollama judges + inter-judge correlation/ICC + "
            "KG-null robustness verdict."
        ),
    )
    parser.add_argument(
        "--judge-dir",
        required=True,
        help=(
            "Directory containing oransight/, gemma3/, qwen/ sub-dirs, "
            "each with a judge_per_station.csv."
        ),
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for cross-judge analysis artefacts.",
    )
    parser.add_argument(
        "--judges",
        default=",".join(_JUDGE_NAMES),
        help="Comma-separated judge names matching sub-dir names.",
    )
    args = parser.parse_args()

    judge_dir = Path(args.judge_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    judge_names: List[str] = [j.strip() for j in args.judges.split(",") if j.strip()]

    # ── Load all judge CSVs ─────────────────────────────────────────────
    judge_data: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]] = {}
    for jname in judge_names:
        csv_path = judge_dir / jname / "judge_per_station.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Judge CSV not found: {csv_path}\n"
                "Run run_explainer_ablation_judge.py for each local model first."
            )
        judge_data[jname] = _load_judge_csv(csv_path)
        n_stations = len(judge_data[jname])
        print(f"Loaded {jname}: {n_stations} stations")

    # ── Per-judge paired statistics ─────────────────────────────────────
    stats_rows: List[dict] = []
    for jname in judge_names:
        data = judge_data[jname]
        for metric in _METRICS:
            require_both = metric == "answer_relevancy"
            sids, a_vals, b_vals = _extract_paired_arrays(
                data, metric, require_both_scored=require_both
            )
            if len(sids) < 2:
                print(f"  [{jname}/{metric}] skipping — only {len(sids)} paired stations")
                row = {
                    "judge": jname,
                    "metric": metric,
                    "mean_diff": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "p_value": float("nan"),
                    "cohens_d": float("nan"),
                    "n": len(sids),
                    "comparison": "kg_on_minus_kg_off",
                }
            else:
                cmp = paired_comparison(a_vals, b_vals, n_boot=10_000, seed=42)
                row = {
                    "judge": jname,
                    "metric": metric,
                    "comparison": "kg_on_minus_kg_off",
                    **cmp,
                }
            stats_rows.append(row)
            print(
                f"  [{jname}/{metric}] n={row['n']}, "
                f"mean_diff={row['mean_diff']:.4f}, "
                f"CI=[{row['ci_low']:.4f}, {row['ci_high']:.4f}], "
                f"p={row['p_value']:.4f}, d={row['cohens_d']:.3f}"
            )

    # ── Write paired-stats CSV + JSON ───────────────────────────────────
    stats_csv_path = out_dir / "cross_judge_stats.csv"
    stats_json_path = out_dir / "cross_judge_stats.json"
    fieldnames = [
        "judge",
        "metric",
        "comparison",
        "n",
        "mean_diff",
        "ci_low",
        "ci_high",
        "p_value",
        "cohens_d",
    ]
    with stats_csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)
    with stats_json_path.open("w") as fh:
        json.dump(stats_rows, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\nWrote {stats_csv_path}")
    print(f"Wrote {stats_json_path}")

    # ── Build station × judge Faithfulness matrix for correlation / ICC ─
    # Intersection of stations across all judges.
    all_station_sets = [set(judge_data[j].keys()) for j in judge_names]
    common_stations = sorted(all_station_sets[0].intersection(*all_station_sets[1:]))
    print(f"\n{len(common_stations)} stations common across all {len(judge_names)} judges")

    # kg_on faithfulness per station per judge (NaN if missing).
    def _kg_on_faithfulness(data, sid):
        return data.get(sid, {}).get("kg_on", {}).get("faithfulness")

    matrix_rows: List[dict] = []
    faithfulness_matrix = np.full((len(common_stations), len(judge_names)), np.nan)
    for i, sid in enumerate(common_stations):
        row = {"station_id": sid}
        for j, jname in enumerate(judge_names):
            val = _kg_on_faithfulness(judge_data[jname], sid)
            row[f"faithfulness_{jname}"] = val if val is not None else ""
            if val is not None:
                faithfulness_matrix[i, j] = val
        matrix_rows.append(row)

    matrix_csv_path = out_dir / "cross_judge_matrix.csv"
    matrix_fieldnames = ["station_id"] + [f"faithfulness_{j}" for j in judge_names]
    with matrix_csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=matrix_fieldnames)
        writer.writeheader()
        writer.writerows(matrix_rows)
    print(f"Wrote {matrix_csv_path}")

    # ── Inter-judge correlation + ICC ───────────────────────────────────
    corr_rows: List[dict] = []
    # Pairwise Pearson r between judges.
    for ji in range(len(judge_names)):
        for jj in range(ji + 1, len(judge_names)):
            jname_i = judge_names[ji]
            jname_j = judge_names[jj]
            col_i = faithfulness_matrix[:, ji]
            col_j = faithfulness_matrix[:, jj]
            # Drop rows where either is NaN.
            valid = ~(np.isnan(col_i) | np.isnan(col_j))
            r = _pearson_r(col_i[valid], col_j[valid])
            n_valid = int(valid.sum())
            corr_rows.append(
                {
                    "judge_a": jname_i,
                    "judge_b": jname_j,
                    "metric": "faithfulness_kg_on",
                    "pearson_r": round(r, 4) if not np.isnan(r) else "",
                    "n": n_valid,
                }
            )
            print(f"  Pearson r ({jname_i} vs {jname_j}): " f"r={r:.4f} (n={n_valid})")

    # ICC(2,1) across all judges — drop stations with any NaN column.
    valid_rows = ~np.isnan(faithfulness_matrix).any(axis=1)
    icc_matrix = faithfulness_matrix[valid_rows]
    icc_val = _icc_two_way_mixed(icc_matrix)
    corr_rows.append(
        {
            "judge_a": ",".join(judge_names),
            "judge_b": "(all)",
            "metric": "faithfulness_kg_on",
            "pearson_r": "",
            "icc_2_1": round(icc_val, 4) if not np.isnan(icc_val) else "",
            "n": int(valid_rows.sum()),
        }
    )
    print(f"  ICC(2,1) across all judges: {icc_val:.4f} (n={int(valid_rows.sum())})")

    corr_csv_path = out_dir / "cross_judge_correlation.csv"
    corr_fieldnames = ["judge_a", "judge_b", "metric", "pearson_r", "icc_2_1", "n"]
    with corr_csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=corr_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(corr_rows)
    print(f"Wrote {corr_csv_path}")

    # ── KG-null robustness verdict (Faithfulness) ───────────────────────
    faith_stats = [r for r in stats_rows if r["metric"] == "faithfulness"]
    verdict = _robustness_verdict(faith_stats)

    robustness_rows: List[dict] = []
    for row in faith_stats:
        ci_low = row["ci_low"]
        ci_high = row["ci_high"]
        if np.isnan(ci_low) or np.isnan(ci_high):
            judge_verdict = "indeterminate"
        elif ci_low <= 0.0 <= ci_high:
            judge_verdict = "null"
        elif ci_low > 0.0:
            judge_verdict = "positive"
        else:
            judge_verdict = "negative"
        robustness_rows.append(
            {
                "judge": row["judge"],
                "metric": row["metric"],
                "mean_diff": row["mean_diff"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "p_value": row["p_value"],
                "n": row["n"],
                "judge_verdict": judge_verdict,
                "overall_robustness": verdict,
            }
        )

    robustness_csv_path = out_dir / "cross_judge_robustness.csv"
    robustness_fieldnames = [
        "judge",
        "metric",
        "n",
        "mean_diff",
        "ci_low",
        "ci_high",
        "p_value",
        "judge_verdict",
        "overall_robustness",
    ]
    with robustness_csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=robustness_fieldnames)
        writer.writeheader()
        writer.writerows(robustness_rows)
    print(f"Wrote {robustness_csv_path}")

    print(f"\n=== KG-null robustness verdict (Faithfulness): {verdict} ===")
    for r in robustness_rows:
        print(
            f"  {r['judge']:20s} judge_verdict={r['judge_verdict']:10s} "
            f"CI=[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        )


if __name__ == "__main__":
    main()
