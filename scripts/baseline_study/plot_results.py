#!/usr/bin/env python3
"""Generate paper-style baseline-study plots from aggregated CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-dir", default="output/baseline_study/full/paper_tables")
    parser.add_argument("--out-dir", default="output/baseline_study/full/figures")
    args = parser.parse_args()
    tables = Path(args.tables_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    leaderboard = pd.read_csv(tables / "leaderboard.csv")
    completed = leaderboard[leaderboard["status"] == "completed"].copy()
    if not completed.empty:
        score_col = "test_mean_MAE" if "test_mean_MAE" in completed.columns else "test_mean_sMAPE"
        x_label = "Test mean MAE" if score_col == "test_mean_MAE" else "Test mean sMAPE (%)"
        completed = completed.sort_values(score_col).head(30)
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(completed))))
        labels = completed["model_family"].astype(str) + "/" + completed["model_name"].astype(str)
        ax.barh(labels, completed[score_col])
        ax.invert_yaxis()
        ax.set_xlabel(x_label)
        ax.set_title("Baseline Leaderboard")
        fig.tight_layout()
        fig.savefig(out_dir / "leaderboard_primary_metric.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(completed.get("params", 0), completed[score_col])
        ax.set_xscale("symlog")
        ax.set_xlabel("Trainable parameters")
        ax.set_ylabel(x_label)
        ax.set_title("Model Size vs Forecast Error")
        fig.tight_layout()
        fig.savefig(out_dir / "params_vs_primary_metric.png", dpi=200)
        plt.close(fig)

    per_kpi_path = tables / "metrics_per_kpi.csv"
    if per_kpi_path.exists():
        per_kpi = pd.read_csv(per_kpi_path)
        if not per_kpi.empty:
            pivot = per_kpi.pivot_table(
                index=["model_family", "model_name"],
                columns="kpi",
                values="sMAPE",
                aggfunc="mean",
            )
            fig, ax = plt.subplots(figsize=(12, max(4, 0.28 * len(pivot))))
            im = ax.imshow(pivot.values, aspect="auto")
            ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
            ax.set_yticks(
                range(len(pivot.index)),
                [f"{fam}/{name}" for fam, name in pivot.index],
            )
            ax.set_title("Per-KPI sMAPE Heatmap")
            fig.colorbar(im, ax=ax, label="sMAPE (%)")
            fig.tight_layout()
            fig.savefig(out_dir / "per_kpi_heatmap.png", dpi=200)
            plt.close(fig)

    print(f"Wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
