#!/usr/bin/env python3
"""Chronos-2 context-length sweep (cross-channel multivariate).

Amazon Chronos-2 (``amazon/chronos-2``) is a 120M-parameter encoder-only
TSFM with **group attention** that natively shares information across
variates of a multivariate series. We feed all 7 KPIs as a single
``(1, 7, context_h)`` tensor so group-attention activates across them.

Inference protocol (per official chronos-forecasting docs):
    - ``pipeline.model.eval()`` is set after load.
    - ``with torch.no_grad():`` wraps every prediction.
    - ``cross_learning=True`` enables group attention across all batch
      items, the canonical multivariate setting.
    - ``num_samples=100`` for stable median forecast.

Library version pin:
    chronos-forecasting **v2.0.0** is required. v2.2.2 has a regression
    that returns the input mean as a flat prediction. Keep this runner
    on a separate conda env (``telco-chronos-test``) where v2.0.0 lives.

Usage:
    conda run -n telco-chronos-test python scripts/baselines/run_chronos2_context_sweep.py

    # smoke test
    conda run -n telco-chronos-test python scripts/baselines/run_chronos2_context_sweep.py \\
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

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

from scripts.baselines.foundation_utils import (
    BaseTSFMPredictor,
    make_sweep_argparser,
    resolve_sweep_grid,
    run_sweep,
    safe_output_name,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_MODEL_ID = "amazon/chronos-2"


class Chronos2Predictor(BaseTSFMPredictor):
    """Chronos-2 cross-channel multivariate predictor.

    All 7 KPIs are passed as a single ``(1, 7, context_h)`` tensor, which
    invokes the model's multivariate path (group attention shares
    information across the 7 variates).

    Note: chronos-forecasting v2.0.0 does NOT expose ``num_samples`` —
    sample count is fixed at 21 quantile points internally. We pass
    ``predict_batches_jointly=True`` (this version's name for
    ``cross_learning``) to keep group attention active when more than one
    series is in the batch.
    """

    family = "chronos2"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        predict_batches_jointly: bool = True,
    ):
        super().__init__()
        self.model_id = model_id
        self.predict_batches_jointly = predict_batches_jointly
        self._pipeline = None
        self.extra_info = {
            "multivariate_mode": "group_attention",
            "predict_batches_jointly": predict_batches_jointly,
        }
        # ``run_sweep`` will fill ``extra_info['lib_version']`` automatically
        # via ``_resolve_lib_version`` so the value reflects the actually-
        # installed wheel rather than a docstring claim.

    def load(self, device: str) -> None:
        try:
            from chronos import Chronos2Pipeline
        except ImportError as e:
            raise ImportError(
                "chronos-forecasting not available. This runner requires v2.0.0 "
                "(v2.2.2 has a flat-prediction regression). Use the "
                "`telco-chronos-test` conda env, or install with: "
                "pip install chronos-forecasting==2.0.0"
            ) from e

        torch_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        logger.info("Loading Chronos-2: %s (device=%s)", self.model_id, torch_device)
        pipeline = Chronos2Pipeline.from_pretrained(
            self.model_id,
            device_map=torch_device,
            dtype=torch.float32,
        )
        # Explicit eval — Chronos2Pipeline's underlying nn.Module is exposed
        # as `pipeline.model`, and we want dropout/batchnorm in eval mode.
        if hasattr(pipeline, "model"):
            pipeline.model.eval()
        self._pipeline = pipeline

    def predict(
        self,
        input_window: np.ndarray,
        context_h: int,
        prediction_length: int,
    ) -> np.ndarray:
        if input_window.shape[0] < context_h:
            raise ValueError(f"input_window has only {input_window.shape[0]}h, need {context_h}h")
        context = input_window[-context_h:]  # (context_h, C)
        # Chronos-2 expects (batch, n_variates, history_length) for multivariate.
        input_tensor = torch.tensor(
            context.T[np.newaxis, :, :],
            dtype=torch.float32,
        )  # (1, C, context_h)

        # BaseTSFMPredictor.__call__ already wraps this in torch.inference_mode(),
        # but the explicit no_grad here is harmless and matches the official
        # chronos-forecasting recipe.
        with torch.no_grad():
            output = self._pipeline.predict(
                input_tensor,
                prediction_length=prediction_length,
                predict_batches_jointly=self.predict_batches_jointly,
            )
        # Output shape: list[Tensor], each (n_variates, n_samples, prediction_length)
        pred_samples = output[0]  # (C, n_samples, H)
        pred = torch.median(pred_samples, dim=1).values.cpu().numpy()  # (C, H)
        return pred.T  # (H, C)


def main():
    parser = make_sweep_argparser(
        description="Chronos-2 context-length sweep",
        default_model_id=DEFAULT_MODEL_ID,
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    context_days, context_hours, max_ctx_d, prediction_length = resolve_sweep_grid(args, parser)

    data_dir = Path(args.data_dir)
    output_root = Path(
        args.output_dir or f"output/{safe_output_name(args.model)}_h{args.horizon_days}d_ctx_sweep"
    )

    csv_files = sorted(data_dir.glob("station_*.csv"))
    if args.stations:
        csv_files = [f for f in csv_files if f.stem in args.stations]
    if not csv_files:
        parser.error(f"No station CSVs found under {data_dir}")

    predictor = Chronos2Predictor(model_id=args.model)
    predictor.load(args.device)

    print("\nChronos-2 Sweep (cross-channel, group attention)")
    print(f"  Model:                    {args.model}")
    print(f"  Horizon:                  {prediction_length}h ({args.horizon_days}d)")
    print(
        f"  Context grid:             {len(context_days)} values "
        f"({context_days[0]}..{context_days[-1]} days, max valid = {max_ctx_d}d)"
    )
    print("  predict_batches_jointly:  True")
    print(f"  Stations:                 {len(csv_files)}")
    print(f"  Device:                   {args.device}")
    print(f"  Output:                   {output_root}")

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
