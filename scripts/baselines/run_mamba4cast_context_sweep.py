#!/usr/bin/env python3
"""Mamba4Cast context-length sweep (channel-independent, all 7 KPIs).

Mamba4Cast (https://github.com/automl/Mamba4Cast) is a Mamba/SSM-based zero-
shot TSFM. It is **channel-independent** — channels are stacked along the
batch axis for one forward pass. This runner reuses the inference primitives
from ``run_mamba4cast.py`` (which already calls ``.eval()`` and wraps the
forward in ``torch.no_grad()``).

Prereqs (see scripts/baselines/run_mamba4cast.py docstring for details):
    - Mamba4Cast cloned at ``<repo_root>/Mamba4Cast``
    - Pretrained weights at ``models/mamba4cast_2l_1024_conv_i5e5.pth``
    - ``mamba-ssm==2.2.2`` and ``causal-conv1d==1.4.0`` installed

Usage:
    conda run -n telcoagent python scripts/baselines/run_mamba4cast_context_sweep.py \\
        --context-days 1 3 7 14 21 28 42 56 70 81

    # smoke test
    conda run -n telcoagent python scripts/baselines/run_mamba4cast_context_sweep.py \\
        --stations station_A_10 --context-days 28
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
# Mamba4Cast's upstream code imports its submodules under top-level names like
# ``training.utils``, so ``Mamba4Cast/src_torch`` must also be on sys.path.
sys.path.insert(0, str(project_root / "Mamba4Cast"))
sys.path.insert(0, str(project_root / "Mamba4Cast" / "src_torch"))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

from scripts.baselines.foundation_utils import (
    BaseTSFMPredictor,
    make_sweep_argparser,
    resolve_sweep_grid,
    run_sweep,
    safe_output_name,
)
from scripts.baselines.run_mamba4cast import (
    DEFAULT_SCALER,
    DEFAULT_WEIGHTS,
    load_model,
    predict,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class Mamba4CastPredictor(BaseTSFMPredictor):
    """Mamba4Cast zero-shot SSM predictor (channel-independent).

    Inference contract:
        - ``run_mamba4cast.load_model`` returns a model already in eval mode.
        - ``run_mamba4cast.predict`` wraps the forward in ``torch.no_grad()``
          internally.
        - Channels are stacked along the batch axis; cross-channel attention
          is NOT used (Mamba4Cast does not have a cross-variate mechanism).
    """

    family = "mamba4cast"

    def __init__(
        self,
        weights_path: Path = DEFAULT_WEIGHTS,
        scaler: str = DEFAULT_SCALER,
    ):
        super().__init__()
        self.weights_path = Path(weights_path)
        self.scaler = scaler
        self.model_id = f"Mamba4Cast/{self.weights_path.stem}"
        self._model = None
        self._device: torch.device | None = None
        self.extra_info = {
            "scaler": scaler,
            "multivariate_mode": "channel_independent",
        }

    def load(self, device: str) -> None:
        torch_device = torch.device(
            device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        logger.info("Loading Mamba4Cast: %s (device=%s)", self.weights_path, torch_device)
        # run_mamba4cast.load_model already calls .eval() on the SSMModelMulti.
        self._model = load_model(self.weights_path, torch_device, self.scaler)
        self._device = torch_device

    def predict(
        self,
        input_window: np.ndarray,
        context_h: int,
        prediction_length: int,
    ) -> np.ndarray:
        # run_mamba4cast.predict already wraps the forward in torch.no_grad().
        # Returns shape (prediction_length, n_channels).
        return predict(
            self._model,
            input_window,
            context_h,
            prediction_length,
            self._device,
            self.scaler,
        )


def main():
    parser = make_sweep_argparser(description="Mamba4Cast context-length sweep")
    parser.add_argument(
        "--weights", default=str(DEFAULT_WEIGHTS), help="Path to mamba4cast_2l_1024_conv_i5e5.pth"
    )
    parser.add_argument(
        "--scaler", default=DEFAULT_SCALER, choices=["min_max", "custom_robust", "identity"]
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    context_days, context_hours, max_ctx_d, prediction_length = resolve_sweep_grid(args, parser)

    weights_path = Path(args.weights)
    data_dir = Path(args.data_dir)
    output_root = Path(
        args.output_dir
        or f"output/{safe_output_name('Mamba4Cast/' + weights_path.stem)}_h{args.horizon_days}d_ctx_sweep"
    )

    csv_files = sorted(data_dir.glob("station_*.csv"))
    if args.stations:
        csv_files = [f for f in csv_files if f.stem in args.stations]
    if not csv_files:
        parser.error(f"No station CSVs found under {data_dir}")

    predictor = Mamba4CastPredictor(weights_path=weights_path, scaler=args.scaler)
    predictor.load(args.device)

    print("\nMamba4Cast Sweep (channel-independent across 7 KPIs)")
    print(f"  Weights:      {weights_path}")
    print(f"  Scaler:       {args.scaler}")
    print(f"  Horizon:      {prediction_length}h ({args.horizon_days}d)")
    print(
        f"  Context grid: {len(context_days)} values "
        f"({context_days[0]}..{context_days[-1]} days, max valid = {max_ctx_d}d)"
    )
    print(f"  Stations:     {len(csv_files)}")
    print(f"  Device:       {args.device}")
    print(f"  Output:       {output_root}")

    run_sweep(
        predictor,
        csv_files=csv_files,
        context_hours=context_hours,
        output_root=output_root,
        prediction_length=prediction_length,
        skip_existing=args.skip_existing,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
