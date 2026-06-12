#!/usr/bin/env python3
"""Train/evaluate one baseline-study config and optionally log to W&B."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from scripts.baseline_study.config import load_config, parse_override, set_by_dotted_key
from scripts.baseline_study.data import (
    TEST_H,
    TRAIN_H,
    StationSeries,
    load_station,
    split_station,
    station_files,
)
from scripts.baseline_study.metrics import (
    aggregate_station_metrics,
    apply_clamp,
    compute_metrics,
    write_station_artifacts,
    write_summary,
)
from scripts.baseline_study.models import build_model


def _wandb_init(cfg: Dict, run_dir: Path):
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", True):
        return None
    if wandb_cfg.get("mode"):
        os.environ["WANDB_MODE"] = str(wandb_cfg["mode"])
    try:
        import wandb
    except ImportError:
        return None
    model = cfg.get("model", {})
    return wandb.init(
        project=wandb_cfg.get("project", "telcoagent-baselines"),
        entity=wandb_cfg.get("entity"),
        group=model.get("family", "unknown"),
        name=model.get("name"),
        config=cfg,
        dir=str(run_dir),
    )


def _log_wandb(run, split_name: str, summary: Dict, station_rows: List[Dict]) -> None:
    if run is None:
        return
    aggregate = summary.get("aggregate", {})
    prefix = split_name
    payload = {
        f"{prefix}/mean_sMAPE": aggregate.get("mean_sMAPE"),
        f"{prefix}/mean_MASE": aggregate.get("mean_MASE"),
        f"{prefix}/mean_nRMSE": aggregate.get("mean_nRMSE"),
        f"{prefix}/mean_MAE": aggregate.get("mean_MAE"),
        f"{prefix}/mean_MSE": aggregate.get("mean_MSE"),
    }
    # Per-KPI breakdown (so KPIs with very different scales are not flattened
    # into a single mean — e.g. RRC_Conn ~500 dominates DL_CQI ~15).
    for kpi, kpi_metrics in (aggregate.get("per_kpi") or {}).items():
        for metric_name, metric_value in kpi_metrics.items():
            payload[f"{prefix}/per_kpi/{kpi}/{metric_name}"] = metric_value
    run.log({k: v for k, v in payload.items() if v is not None})
    try:
        import wandb

        table_rows = [
            [
                r["station_id"],
                r["status"],
                r.get("metrics", {}).get("mean_sMAPE"),
                r.get("metrics", {}).get("mean_MASE"),
                r.get("metrics", {}).get("mean_nRMSE"),
                r.get("metrics", {}).get("mean_MAE"),
                r.get("metrics", {}).get("mean_MSE"),
            ]
            for r in station_rows
        ]
        run.log(
            {
                f"{prefix}/station_metrics": wandb.Table(
                    columns=[
                        "station_id",
                        "status",
                        "mean_sMAPE",
                        "mean_MASE",
                        "mean_nRMSE",
                        "mean_MAE",
                        "mean_MSE",
                    ],
                    data=table_rows,
                )
            }
        )
    except Exception:
        return


def build_resources_row(
    *,
    rt_metrics: Dict,
    meta: Dict,
    n_params: int,
    n_stations: int,
    n_samples: int,
) -> Dict:
    """Assemble the per-run resources row from tracker primitives + model metadata.

    ``regime``, ``finetune_weeks`` and ``n_train_windows`` are taken straight from
    the model's ``metadata()`` (a single derived source — no family-based hardcode,
    no fallback) so the regime-sweep aggregator can plot accuracy/resource vs the
    fine-tune week count.
    """
    return {
        **rt_metrics,
        "n_params": n_params,
        "n_stations": n_stations,
        "n_samples": n_samples,  # config-stride train windows (supervised baselines)
        "regime": meta.get("regime"),
        "finetune_weeks": meta.get("finetune_weeks"),
        "n_train_windows": meta.get("n_train_windows"),
        "best_val_loss": meta.get("best_val_loss"),
        "epochs_trained": meta.get("epochs_trained"),
    }


def _evaluate_split(
    *,
    model,
    stations: List[StationSeries],
    output_dir: Path,
    split_name: str,
    horizon_h: int,
    eval_horizon_h: int,
) -> List[Dict]:
    rows = []
    for station in stations:
        split = split_station(station.values)
        if split_name == "val":
            history = split.train
            truth = split.val[:horizon_h]
        elif split_name == "test":
            history = pd.concat(
                [pd.DataFrame(split.train), pd.DataFrame(split.val)], ignore_index=True
            ).to_numpy()
            truth = split.test[:horizon_h]
        else:
            raise ValueError(f"Unknown split_name={split_name}")
        try:
            pred = apply_clamp(model.predict(history, horizon_h, station_id=station.station_id))
            pred = pred[:eval_horizon_h]
            truth = truth[:eval_horizon_h]
            metrics = compute_metrics(pred, truth, input_window=history)
            rows.append(
                write_station_artifacts(
                    output_dir=output_dir,
                    station_id=station.station_id,
                    split_name=split_name,
                    pred=pred,
                    truth=truth,
                    metrics=metrics,
                )
            )
        except Exception as exc:
            rows.append({"station_id": station.station_id, "status": "error", "error": str(exc)})
    return rows


def _run(cfg: Dict, stations_override: List[str] | None = None) -> int:
    task = cfg.setdefault("task", {})
    horizon_h = int(task.get("horizon_h", TEST_H))
    eval_horizon_h = int(task.get("eval_horizon_h", horizon_h))
    if eval_horizon_h > TEST_H:
        raise ValueError(f"eval_horizon_h={eval_horizon_h} exceeds test horizon {TEST_H}")
    data = cfg.get("data", {})
    data_dir = data.get("data_dir", "data/station")
    # Decouple fit-stations from eval-stations (spatial station-holdout). When the
    # spatial keys are absent, train_ids == test_ids == the single configured set,
    # so the temporal sweep and every other model is byte-identical to before.
    # ``is not None`` (not truthiness) so an EXPLICIT empty train_stations list — the
    # K=0 spatial / zero-shot point — is honored as "no fine-tune stations" instead of
    # silently falling through to the full station set.
    has_train_key = data.get("train_stations") is not None
    has_test_key = data.get("test_stations") is not None
    if stations_override is not None:
        train_ids = stations_override
    elif has_train_key:
        train_ids = data["train_stations"]
    else:
        train_ids = data.get("stations")
    test_ids = data["test_stations"] if has_test_key else train_ids
    # An explicit empty id list means "no stations" (K=0 zero-shot); station_files
    # treats a falsy ``stations`` as "no filter -> all files", so short-circuit here.
    train_paths = [] if train_ids == [] else station_files(data_dir, train_ids)
    test_paths = station_files(data_dir, test_ids)
    if not test_paths:
        raise FileNotFoundError(f"No station CSV files found in {data_dir}")

    train_stations = [load_station(p) for p in train_paths]
    test_stations = [load_station(p) for p in test_paths]
    # Leak guard: only when BOTH new keys are explicitly set (the spatial path). The
    # K=0 case has an empty train_stations list (zero-shot) and is exempt — an empty
    # train set cannot overlap the test set.
    if has_train_key and has_test_key:
        overlap = {s.station_id for s in train_stations} & {s.station_id for s in test_stations}
        if overlap:
            raise ValueError(f"train/test station overlap (leak): {sorted(overlap)}")
    # ``stations`` drives fit + val (fit-side); the test split evaluates the held-out
    # ``test_stations``. Without the spatial keys these are the same object/set.
    stations = train_stations
    model = build_model(cfg)
    output_root = Path(cfg.get("output", {}).get("root", "output/baseline_study/full"))
    run_name = cfg.get("run_name") or f"{model.family}_{model.name}_{int(time.time())}"
    wandb_run_id = os.environ.get("WANDB_RUN_ID")
    if wandb_run_id and not str(run_name).endswith(wandb_run_id):
        run_name = f"{run_name}_{wandb_run_id}"
    run_dir = output_root / model.family / model.name / run_name
    if (
        run_dir.exists()
        and any(run_dir.iterdir())
        and not cfg.get("output", {}).get("overwrite", False)
    ):
        run_dir = output_root / model.family / model.name / f"{run_name}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Fine-tune regimes write their HF Trainer checkpoints under the run dir so a
    # regime sweep never collides on the chronos default timestamped dir. The model
    # reads this back through config["output"]["finetune_dir"] in fit(); zero-shot
    # families ignore it.
    cfg.setdefault("output", {})["finetune_dir"] = str(run_dir / "finetune")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    wandb_run = _wandb_init(cfg, run_dir)
    t0 = time.time()
    if wandb_run is not None:
        wandb_run.log(
            {
                "status/started": 1,
                "data/n_stations": len(stations),
                "task/horizon_h": horizon_h,
            }
        )
    from scripts.baseline_study.resource_tracker import (  # noqa: PLC0415
        ResourceTracker,
        append_resource_csv,
    )

    rt = ResourceTracker()
    rt.__enter__()  # track fit + val + test as one compute block
    try:
        model.fit(stations)
    except Exception as exc:
        rt.__exit__(None, None, None)
        skipped = {
            "status": "skipped",
            "model_family": model.family,
            "model_name": model.name,
            "forecast_mode": model.forecast_mode,
            "reason": str(exc),
            "timestamp": time.time(),
        }
        (run_dir / "skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
        if wandb_run is not None:
            wandb_run.log({"status/skipped": 1, "skip/reason": str(exc)})
            wandb_run.finish(exit_code=0)
        print(f"SKIPPED {model.family}/{model.name}: {exc}")
        return 0

    try:
        extra = {
            **model.metadata(),
            "horizon_h": horizon_h,
            "n_stations": len(stations),
            "elapsed_fit_sec": round(time.time() - t0, 2),
        }
        if wandb_run is not None:
            wandb_run.log(
                {
                    "fit/elapsed_sec": extra["elapsed_fit_sec"],
                    "model/params": model.parameter_count,
                    "status/fit_completed": 1,
                    "fit/epochs_trained": model.metadata().get("epochs_trained"),
                    "fit/best_val_loss": model.metadata().get("best_val_loss"),
                    "fit/early_stopped": int(bool(model.metadata().get("early_stopped"))),
                }
            )
        val_rows = _evaluate_split(
            model=model,
            stations=stations,
            output_dir=run_dir,
            split_name="val",
            horizon_h=horizon_h,
            eval_horizon_h=eval_horizon_h,
        )
        val_summary = write_summary(run_dir, "val", val_rows, extra)
        _log_wandb(wandb_run, "val", val_summary, val_rows)

        test_t0 = time.time()
        test_rows = _evaluate_split(
            model=model,
            stations=test_stations,
            output_dir=run_dir,
            split_name="test",
            horizon_h=horizon_h,
            eval_horizon_h=eval_horizon_h,
        )
        sec_per_station = (time.time() - test_t0) / max(1, len(test_stations))
        test_summary = write_summary(
            run_dir,
            "test",
            test_rows,
            {**extra, "runtime_sec_per_station": round(sec_per_station, 4)},
        )
        _log_wandb(wandb_run, "test", test_summary, test_rows)
    finally:
        # Always tear down the tracker so codecarbon/zeus/pynvml threads are released
        # and the partial resource row is still emitted even if val/test/summary raised.
        rt.__exit__(None, None, None)

    # Guard: fit() may succeed while EVERY station prediction errors (e.g. a predict-side
    # shape/device bug). Such a run must NOT be reported "ok" with empty metrics.
    if test_summary.get("aggregate", {}).get("successful", 0) == 0:
        failed = {
            "status": "failed",
            "model_family": model.family,
            "model_name": model.name,
            "reason": f"0/{len(test_stations)} test-station predictions succeeded",
            "timestamp": time.time(),
        }
        (run_dir / "failed.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        print(
            f"FAILED {model.family}/{model.name}: "
            f"0/{len(test_stations)} test predictions succeeded"
        )
        return 1

    meta = model.metadata()
    ctx_h = int(cfg.get("task", {}).get("context_h", horizon_h))
    stride_h = int(cfg.get("training", {}).get("stride_h", 24))
    wins_per_station = max(0, (TRAIN_H - ctx_h - horizon_h) // stride_h + 1)
    resources = build_resources_row(
        rt_metrics=rt.metrics,
        meta=meta,
        n_params=model.parameter_count,
        n_stations=len(stations),
        n_samples=len(stations) * wins_per_station,
    )
    final = {
        "run_dir": str(run_dir),
        "model": meta,
        "val": aggregate_station_metrics(val_rows),
        "test": aggregate_station_metrics(test_rows),
        "runtime": {
            "total_elapsed_sec": round(time.time() - t0, 2),
            "sec_per_station": round(sec_per_station, 4),
        },
        "resources": resources,
    }
    (run_dir / "result.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    append_resource_csv(
        Path(output_root) / "resources.csv",
        {
            "run_name": run_name,
            "family": model.family,
            "name": model.name,
            "kind": cfg.get("model", {}).get("kind"),
            "num_layers": cfg.get("model", {}).get("num_layers"),
            "context_h": ctx_h,
            "horizon_h": horizon_h,
            "test_mean_sMAPE": final["test"].get("mean_sMAPE"),
            "test_mean_MAE": final["test"].get("mean_MAE"),
            **resources,
        },
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "runtime/sec_per_station": sec_per_station,
                "model/params": model.parameter_count,
                "status/completed": 1,
            }
        )
        wandb_run.finish()
    print(json.dumps(final, indent=2))
    return 0


def main(argv: List[str] | None = None) -> int:
    import signal

    timeout_s = int(os.environ.get("BASELINE_RUN_TIMEOUT_SEC", "7200"))
    if timeout_s > 0:

        def _on_alarm(*_):
            raise TimeoutError(f"run exceeded {timeout_s}s")

        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(timeout_s)
    parser = argparse.ArgumentParser(description="Run one TelcoAgent baseline-study config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stations", nargs="*", help="Optional station stems")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], help="Dotted key override"
    )
    args, unknown = parser.parse_known_args(argv)

    cfg = load_config(args.config)
    wandb_style = []
    for item in unknown:
        if item.startswith("--") and "=" in item:
            wandb_style.append(item[2:])
        elif item.startswith("--"):
            raise ValueError(f"Unsupported argument from sweep: {item}; use --key=value")
        else:
            raise ValueError(f"Unsupported positional argument from sweep: {item}")
    for raw in [*args.overrides, *wandb_style]:
        key, value = parse_override(raw)
        set_by_dotted_key(cfg, key, value)
    return _run(cfg, stations_override=args.stations)


if __name__ == "__main__":
    raise SystemExit(main())
