#!/usr/bin/env python3
"""Create W&B sweeps or run local sweep agents for baseline-study configs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="W&B sweep helper for TelcoAgent baselines")
    parser.add_argument("--sweep-config", required=True)
    parser.add_argument("--project", default="telcoagent-baselines")
    parser.add_argument("--entity")
    parser.add_argument("--count", type=int, help="Number of runs for wandb agent")
    parser.add_argument("--create-only", action="store_true")
    args = parser.parse_args()

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is not installed; pip install wandb or run train.py directly"
        ) from exc

    with Path(args.sweep_config).open("r", encoding="utf-8") as fh:
        sweep_cfg = yaml.safe_load(fh)
    sweep_id = wandb.sweep(sweep=sweep_cfg, project=args.project, entity=args.entity)
    print(f"Sweep ID: {sweep_id}")
    if args.create_only:
        return 0
    cmd = ["wandb", "agent"]
    if args.count:
        cmd.extend(["--count", str(args.count)])
    cmd.append(sweep_id)
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
