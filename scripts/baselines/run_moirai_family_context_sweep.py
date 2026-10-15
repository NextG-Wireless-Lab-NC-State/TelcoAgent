#!/usr/bin/env python3
"""Moirai-family context-length sweep (cross-channel / Any-Variate Attention).

Supports the multivariate-capable Moirai variants:

    Salesforce/moirai-1.x-R-*       -> uni2ts.model.moirai
    Salesforce/moirai-moe-*-R-*     -> uni2ts.model.moirai_moe

Moirai-2.0 dropped multivariate / Any-Variate Attention support, so it is
NOT compatible with this project's PAX-TS cross-channel requirement and is
explicitly rejected.

All variants run with ``target_dim = 7`` (the seven KPIs share the same
Any-Variate group), ``num_samples = 100`` (raised from uni2ts's default 20
for a more stable median forecast), and an explicit ``module.eval()`` +
``torch.no_grad()`` around the forward pass.

Usage:
    conda run -n telcoagent python scripts/baselines/run_moirai_family_context_sweep.py \\
        --model Salesforce/moirai-1.1-R-small

    # smoke test
    conda run -n telcoagent python scripts/baselines/run_moirai_family_context_sweep.py \\
        --model Salesforce/moirai-moe-1.0-R-base \\
        --stations station_A_10 --context-days 28
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

from scripts.baselines.foundation_utils import (
    N_CHANNELS,
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


NUM_SAMPLES = 100  # raised from uni2ts default 20 — more stable median


def _resolve_classes(model_id: str) -> Tuple[type, type, str]:
    """Pick (ModuleCls, ForecastCls, family_label) from the HF model id.

    Order matters: ``moirai-moe-`` must be checked before the generic
    ``moirai-1.`` prefix matches.

    Moirai-2.x is rejected because the v2.0 release dropped multivariate
    forecasting; using it would silently fall back to univariate and
    contradict the cross-channel attention story PAX-TS depends on.
    """
    name = model_id.lower().rsplit("/", 1)[-1]
    if "moirai-moe" in name:
        from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

        return MoiraiMoEModule, MoiraiMoEForecast, "moirai_moe"
    if name.startswith("moirai-1."):
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

        return MoiraiModule, MoiraiForecast, "moirai"
    if name.startswith("moirai-2."):
        raise ValueError(
            f"Moirai-2.x ({model_id}) dropped multivariate / Any-Variate "
            f"Attention support; this project requires cross-channel "
            f"forecasting. Use Salesforce/moirai-1.1-R-* or "
            f"Salesforce/moirai-moe-1.0-R-* instead."
        )
    raise ValueError(
        f"Unrecognised Moirai-family model id: {model_id!r}. "
        f"Expected a name starting with 'moirai-1.' or 'moirai-moe-'."
    )


class MoiraiPredictor(BaseTSFMPredictor):
    """Cross-channel multivariate predictor for Moirai-1.x and Moirai-MoE.

    Inference protocol:
        - ``target_dim = 7`` activates Any-Variate Attention so the seven
          KPIs influence each other.
        - ``module.eval()`` is called explicitly after load.
        - ``predict()`` is wrapped in ``torch.no_grad()`` (in addition to
          the inference_mode wrapper enforced by BaseTSFMPredictor).
        - ``num_samples = 100`` for stable median forecast.
    """

    def __init__(
        self,
        model_id: str,
        num_samples: int = NUM_SAMPLES,
    ):
        super().__init__()
        self.model_id = model_id
        ModuleCls, ForecastCls, family = _resolve_classes(model_id)
        self.family = family
        self._ModuleCls = ModuleCls
        self._ForecastCls = ForecastCls
        self.num_samples = num_samples
        self._module = None
        self._forecast_wrapper = None
        self._wrapper_ctx_h: int | None = None
        self.extra_info = {
            "multivariate_mode": "any_variate_attention",
            "num_samples": num_samples,
        }

    def load(self, device: str) -> None:
        torch_device = torch.device(
            device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        logger.info("Loading %s module: %s (device=%s)", self.family, self.model_id, torch_device)
        module = self._ModuleCls.from_pretrained(self.model_id)
        module = module.to(torch_device)
        module.eval()  # explicit — disables dropout / freezes batchnorm stats
        self._module = module

    def _build_wrapper(self, context_h: int, prediction_length: int):
        """Re-build the *Forecast wrapper if context_h or horizon changed.

        Each Moirai-family Forecast class has a slightly different
        signature, so we only pass the kwargs that exist on the chosen
        class:
            - MoiraiForecast (1.x):    accepts patch_size="auto" and num_samples
            - MoiraiMoEForecast (MoE): patch_size must be an int (default 16);
                                        accepts num_samples
        """
        kwargs = dict(
            module=self._module,
            prediction_length=prediction_length,
            context_length=context_h,
            target_dim=N_CHANNELS,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            num_samples=self.num_samples,
        )
        if self.family == "moirai":
            kwargs["patch_size"] = "auto"
        elif self.family == "moirai_moe":
            kwargs["patch_size"] = 16
        return self._ForecastCls(**kwargs)

    def predict(
        self,
        input_window: np.ndarray,
        context_h: int,
        prediction_length: int,
    ) -> np.ndarray:
        import pandas as pd
        from gluonts.dataset.common import ListDataset

        if input_window.shape[0] < context_h:
            raise ValueError(f"input_window has only {input_window.shape[0]}h, need {context_h}h")

        # Cache the wrapper across stations within the same context length;
        # rebuild only when context_h changes.
        cache_key = (context_h, prediction_length)
        if self._wrapper_ctx_h != cache_key:
            self._forecast_wrapper = self._build_wrapper(context_h, prediction_length)
            self._wrapper_ctx_h = cache_key

        context = input_window[-context_h:]  # (context_h, C)
        dataset = ListDataset(
            [
                {
                    "start": pd.Timestamp("2024-01-01"),
                    "target": context.T,  # (C, context_h)
                }
            ],
            freq="h",
            one_dim_target=False,
        )

        with torch.no_grad():
            predictor = self._forecast_wrapper.create_predictor(batch_size=1)
            forecasts = list(predictor.predict(dataset))

        forecast = forecasts[0]
        pred = np.median(forecast.samples, axis=0)  # (prediction_length, C)
        if pred.shape[0] < prediction_length:
            pad = prediction_length - pred.shape[0]
            pred = np.pad(pred, ((0, pad), (0, 0)), mode="edge")
        return pred[:prediction_length]


def main():
    parser = make_sweep_argparser(
        description="Moirai-family (1.x / MoE) context-length sweep",
        require_model_id=True,
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

    predictor = MoiraiPredictor(model_id=args.model)
    predictor.load(args.device)

    print(f"\nMoirai-family Sweep (cross-channel, target_dim={N_CHANNELS})")
    print(f"  Model:         {args.model}")
    print(f"  Family:        {predictor.family}")
    print(f"  Horizon:       {prediction_length}h ({args.horizon_days}d)")
    print(
        f"  Context grid:  {len(context_days)} values "
        f"({context_days[0]}..{context_days[-1]} days, max valid = {max_ctx_d}d)"
    )
    print(f"  num_samples:   {NUM_SAMPLES}")
    print(f"  Stations:      {len(csv_files)}")
    print(f"  Device:        {args.device}")
    print(f"  Output:        {output_root}")

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
