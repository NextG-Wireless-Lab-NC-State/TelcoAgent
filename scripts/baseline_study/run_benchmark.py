#!/usr/bin/env python3
"""Unified, leak-free benchmark driver (no W&B sweep controller).

Enumerates every model config (trained baselines + zero-shot TSFMs), runs each
as a direct ``train.py`` subprocess in the right conda env, with a deterministic
unique run_name and ``output.overwrite=true`` so re-runs are idempotent and
collision-free. Resumable: a config whose ``result.json`` already exists is
skipped. TSFM families run in the isolated ``telco-chronos-test`` env (chronos
2.0.0 + uni2ts + momentfm); everything else in ``telcoagent``.

Usage:
    python scripts/baseline_study/run_benchmark.py --dry-run
    python scripts/baseline_study/run_benchmark.py --group statistical classical darts thuml
    /path/telco-chronos-test/bin/python scripts/baseline_study/run_benchmark.py --group tsfm

(The TSFM group must be launched with the chronos env python; the baseline
groups with the telcoagent env python. --group lets you split by env.)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.baseline_study.spatial_split import (  # noqa: E402
    all_station_ids,
    cumulative_split,
)

OUT_ROOT = ROOT / "output" / "baseline_study_v2" / "full"
CFG = "scripts/baseline_study/configs/base"

# Cumulative group sweep (cross-cell transfer). Rank ALL 13 site groups with a
# fixed seed; for G training groups use ALL stations of the first G ranked groups
# as train and the remaining (13-G) groups as test. The test set shrinks as G
# grows (full 115 at G=0, single group at G=12). G=0 is the pure zero-shot
# baseline; G=13 is excluded (nothing left to evaluate).
DATA_DIR = "data/station"
CUM_SEED = 42
CUM_GROUP_GRID = list(range(13))  # 0..12 training site groups

TELCO_PY = "/home/gkim26/miniconda3/envs/telcoagent/bin/python"
CHRONOS_PY = "/home/gkim26/miniconda3/envs/telco-chronos-test/bin/python"
# Chronos-2 FINE-TUNING needs transformers>=4.49 (Chronos2Pipeline.fit -> HF Trainer's
# save_only_model / eval_strategy / adamw_torch_fused). The shared chronos env is pinned
# to transformers==4.33.3 by momentfm, so fine-tune runs use a dedicated env (clone +
# transformers>=4.49). Zero-shot Moirai/MOMENT keep CHRONOS_PY.
CHRONOS_FT_PY = "/home/gkim26/miniconda3/envs/telco-chronos-ft/bin/python"


def _job(group, env, base, name, family, sets):
    return {
        "group": group,
        "env": env,
        "base": f"{CFG}/{base}",
        "name": name,
        "family": family,
        "sets": sets,
    }


def build_jobs() -> list[dict]:
    jobs: list[dict] = []

    # --- statistical (CPU) ---
    for kind in ["naive", "seasonal_naive", "historic_average"]:
        jobs.append(
            _job(
                "statistical",
                TELCO_PY,
                "statistical.yaml",
                f"stat_{kind}",
                "statistical",
                {"model.kind": kind},
            )
        )
    for win in [24, 168]:
        jobs.append(
            _job(
                "statistical",
                TELCO_PY,
                "statistical.yaml",
                f"stat_window_{win}",
                "statistical",
                {"model.kind": "window_average", "model.window_h": win},
            )
        )

    # --- classical / ML (CPU): TREE-ONLY, default HP, single config each ---
    # Linear models (ridge/lasso/elasticnet) dropped. context_h=168 lookback to
    # match the DL models. multivariate_direct = one model per horizon step.
    for kind in ["random_forest", "extra_trees", "xgboost", "lightgbm"]:
        jobs.append(
            _job(
                "classical",
                TELCO_PY,
                "classical.yaml",
                f"cls_{kind}_c168",
                "classical",
                {"model.kind": kind, "task.context_h": 168},
            )
        )

    # Darts dropped entirely (per-KPI 7x overhead + val-336 context constraint).
    # NLinear/NHITS abandoned; DLinear remains available via THUML; recurrent +
    # MLP + N-BEATS are covered by the custom joint DL family below.

    # DL is layers-only: no separate thuml context sweep. Every DL model is
    # evaluated across num_layers 1..8 at the unified fixed training config.

    # --- layers ablation (GPU): per-KPI vs num_layers, layers 1..8 @ ctx168 ---
    # Unified DL training: context=168, horizon=168, epochs=100, patience=5,
    # batch=32, lr=1e-4 (cosine). Only num_layers varies.
    LAYERS = [1, 2, 3, 4, 5, 6, 7, 8]
    THUML_DL = ["patchtst", "itransformer", "timexer", "fedformer", "autoformer", "informer"]
    for kind, L in itertools.product(THUML_DL, LAYERS):
        jobs.append(
            _job(
                "layers",
                TELCO_PY,
                "thuml.yaml",
                f"thuml_{kind}_L{L}_c168",
                "thuml",
                {
                    "model.kind": kind,
                    "model.name": f"thuml_{kind}",
                    "model.num_layers": L,
                    "task.context_h": 168,
                },
            )
        )

    # --- custom joint DL (GPU): replaces Darts. Recurrent (lstm/gru/rnn) + mlp +
    # nbeats, all multivariate-joint + epoch-based, layers 1..8 @ ctx168. These
    # double as the leaderboard entry (best layer selected on val). ---
    for kind, L in itertools.product(["lstm", "gru", "rnn", "nbeats", "mlp"], LAYERS):
        jobs.append(
            _job(
                "customdl",
                TELCO_PY,
                "customdl.yaml",
                f"custom_{kind}_L{L}_c168",
                "customdl",
                {
                    "model.kind": kind,
                    "model.name": f"custom_{kind}",
                    "model.num_layers": L,
                    "task.context_h": 168,
                },
            )
        )

    # --- TSFM (GPU, zero-shot, chronos env) ---
    tsfm = [("chronos", "chronos2"), ("moirai", "moirai_large"), ("moment", "moment")]
    for (family, name), ctx in itertools.product(tsfm, [168, 336, 672, 1008, 1344, 1608]):
        jobs.append(
            _job(
                "tsfm",
                CHRONOS_PY,
                "foundation.yaml",
                f"{name}_c{ctx}",
                family,
                {"model.family": family, "model.name": name, "model.context_h": ctx},
            )
        )

    # --- Chronos-2 fine-tune regime sweep (GPU, chronos env) ---
    # Data-scaling curve: x-axis = trailing fine-tune weeks ending at day 81; the
    # forecast target (day 82-88) is fixed. zero(0) + weeks 2..11 + full. context
    # is pinned to 168 (ctx == horizon) so inference is identical across regimes
    # and the 2-week minimum window holds. run_name encodes the regime point so the
    # per-run output dirs never collide; aggregate_regime_sweep.py collapses them.
    REGIME_GRID: list[int | str] = [0, *range(2, 12), "full"]
    for weeks in REGIME_GRID:
        jobs.append(
            _job(
                "chronos_regime",
                CHRONOS_FT_PY,
                "foundation.yaml",
                f"chronos2_ftw{weeks}_c168",
                "chronos",
                {
                    "model.family": "chronos",
                    "model.name": "chronos2",
                    "model.context_h": 168,
                    "task.finetune_weeks": weeks,
                },
            )
        )

    # --- Chronos-2 cumulative group sweep (GPU, chronos-ft env) ---
    # Cross-cell transfer: rank ALL 13 site groups with a fixed seed; for G
    # training groups use ALL stations of the first G ranked groups as train and
    # the remaining (13-G) groups as test. The test set shrinks as G grows.
    # Training history fixed at "full" (day1-81) to isolate the spatial axis;
    # G=0 is the pure zero-shot baseline (train=[], test=all 115 stations).
    # Concrete seeded id lists computed ONCE here (determinism centralized) and
    # passed via --set so each run's config.yaml records the exact split.
    all_ids = all_station_ids(DATA_DIR)
    for g in CUM_GROUP_GRID:
        train_ids, test_ids = cumulative_split(all_ids, g, CUM_SEED)
        jobs.append(
            _job(
                "chronos_cumulative",
                CHRONOS_FT_PY,
                "foundation.yaml",
                f"chronos2_cum{g}_c168",
                "chronos",
                {
                    "model.family": "chronos",
                    "model.name": "chronos2",
                    "model.context_h": 168,
                    "task.finetune_weeks": 0 if g == 0 else "full",
                    "data.train_stations": train_ids,
                    "data.test_stations": test_ids,
                },
            )
        )

    return jobs


def result_path(job: dict) -> Path:
    return OUT_ROOT / job["family"] / _model_name(job) / job["name"] / "result.json"


def _model_name(job: dict) -> str:
    # train.py writes to root/<family>/<model.name>/<run_name>; model.name may be
    # overridden in sets, else equals the base-config default.
    return str(job["sets"].get("model.name", _BASE_NAME.get(job["family"], job["family"])))


_BASE_NAME = {
    "statistical": "statistical",
    "classical": "classical",
    "darts": "darts_lstm",
    "thuml": "thuml_patchtst",
    "chronos": "chronos2",
    "moirai": "moirai_large",
    "moment": "moment",
}


VAL_H, TRAIN_H, HORIZON = 336, 1608, 168


def validate_job(job: dict) -> tuple[str, str]:
    """Pre-flight feasibility check. Returns (level, message); level in PASS/WARN/FAIL."""
    s = job["sets"]
    fam = job["family"]
    ctx = int(s.get("task.context_h", 168))
    if fam == "darts":
        # darts windows the standalone 336h val split: needs ctx + horizon <= VAL_H.
        if ctx + HORIZON > VAL_H:
            return "FAIL", f"darts ctx={ctx}+{HORIZON} > val {VAL_H} (would skip: series too short)"
    if fam in ("thuml", "customdl"):
        # both build training windows from the 1608h train region
        if ctx + HORIZON > TRAIN_H:
            return "FAIL", f"{fam} ctx={ctx}+{HORIZON} > train {TRAIN_H}"
    if fam == "classical" and s.get("model.kind") in ("lasso", "elasticnet") and ctx > 24:
        return "WARN", f"lasso/elasticnet ctx={ctx}: 168 horizon models may exceed 2h timeout"
    return "PASS", ""


def run_job(job: dict, idx: int, total: int) -> str:
    lvl, msg = validate_job(job)
    if lvl == "FAIL":
        print(f"[{idx}/{total}] PREFLIGHT-FAIL {job['name']}: {msg}")
        return "preflight-fail"
    rp = result_path(job)
    if rp.exists():
        print(f"[{idx}/{total}] SKIP (done) {job['name']}")
        return "skip"
    cmd = [
        job["env"],
        "scripts/baseline_study/train.py",
        "--config",
        job["base"],
        "--set",
        f"run_name={job['name']}",
        "--set",
        "output.root=output/baseline_study_v2/full",
    ]
    for k, v in job["sets"].items():
        # parse_override round-trips the value through yaml.safe_load, so non-scalar
        # values (station-id lists/dicts) must be JSON-serialized to survive intact;
        # JSON is a YAML subset, so safe_load reads it back to the same Python object.
        val = json.dumps(v) if isinstance(v, (list, dict)) else v
        cmd += ["--set", f"{k}={val}"]
    print(f"[{idx}/{total}] RUN  {job['group']:11s} {job['name']}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    dt = time.time() - t0
    # train.py writes skipped.json + returns rc=0 on fit-failure; treat that as
    # a FAILURE, not success, so silent skips can never masquerade as "ok".
    skip_file = rp.parent / "skipped.json"
    if proc.returncode != 0:
        status = f"FAIL rc={proc.returncode}"
    elif rp.exists():
        status = "ok"
    elif skip_file.exists():
        reason = ""
        try:
            reason = json.loads(skip_file.read_text()).get("reason", "")[:90]
        except Exception:
            pass
        status = f"FAIL skipped: {reason}"
    else:
        status = "FAIL no-result"
    print(f"[{idx}/{total}] {status} {job['name']} ({dt:.0f}s)")
    return status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--group",
        nargs="*",
        default=None,
        help="subset of groups: statistical classical darts thuml tsfm",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--preflight", action="store_true", help="validate feasibility of every job without running"
    )
    args = ap.parse_args()

    jobs = build_jobs()
    if args.group:
        jobs = [j for j in jobs if j["group"] in set(args.group)]
    # BASELINE_EXCLUDE_KINDS=fedformer,autoformer -> drop jobs whose run_name
    # contains any listed kind. Lets a sweep skip a too-slow model without
    # fabricating its result.json (skipped.json never gates resume).
    exclude = {k for k in os.environ.get("BASELINE_EXCLUDE_KINDS", "").split(",") if k}
    if exclude:
        jobs = [j for j in jobs if not any(k in j["name"] for k in exclude)]

    if args.preflight:
        from collections import Counter

        tally: Counter = Counter()
        for j in jobs:
            lvl, msg = validate_job(j)
            tally[lvl] += 1
            if lvl != "PASS":
                print(f"  {lvl:5s} {j['group']:11s} {j['name']:28s} {msg}")
        print(f"\npreflight: {dict(tally)} over {len(jobs)} jobs")
        return 1 if tally.get("FAIL") else 0

    if args.dry_run:
        from collections import Counter

        c = Counter(j["group"] for j in jobs)
        for j in jobs:
            done = "DONE" if result_path(j).exists() else "    "
            print(
                f"  {done} {j['group']:11s} {j['name']:28s} env={ {CHRONOS_PY: 'chronos', CHRONOS_FT_PY: 'chronos-ft'}.get(j['env'], 'telco') }"
            )
        print(f"\ntotal {len(jobs)} jobs: {dict(c)}")
        return 0

    total = len(jobs)
    summary: dict[str, int] = {}
    for i, job in enumerate(jobs, 1):
        st = run_job(job, i, total)
        summary[st] = summary.get(st, 0) + 1
    print(f"\nDONE. {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
