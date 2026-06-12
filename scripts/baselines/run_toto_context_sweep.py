#!/usr/bin/env python3
"""Toto context-length sweep (cross-channel, factorized space-time attention).

Toto (Datadog, ``Datadog/Toto-Open-Base-1.0``) is a 151M-parameter decoder-
only TSFM trained on observability metrics. Its 11:1 (time:variate) factorized
attention provides native cross-variate modelling, which is exactly the
attention surface PAX-TS needs.

We feed all 7 KPIs as a single ``id_mask=zeros`` group so the variate-block
attention runs across the full KPI set per forward pass.

Prereqs:
    pip install --no-deps toto-ts==0.2.0 rotary-embedding-torch
    # (we install --no-deps to keep our torch / numpy versions intact;
    #  toto-ts pins are too tight)

Usage:
    conda run -n telcoagent python scripts/baselines/run_toto_context_sweep.py

    # smoke test
    conda run -n telcoagent python scripts/baselines/run_toto_context_sweep.py \\
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


DEFAULT_MODEL_ID = "Datadog/Toto-Open-Base-1.0"
HOURLY_INTERVAL_SEC = 3600
NUM_SAMPLES = 256  # paper-recommended sample budget
SAMPLES_PER_BATCH = 64  # split 256 samples into 4 forwards to keep peak GPU
# memory bounded at long contexts (RTX 4090 24GB OOMs
# at ctx >= 1032h with samples_per_batch=256)


class TotoPredictor(BaseTSFMPredictor):
    """Toto cross-channel multivariate predictor.

    Inference protocol:
        - ``id_mask=zeros`` puts all 7 KPIs in the same group, enabling the
          variate-block attention across them.
        - ``timestamp_seconds=zeros`` is supported in v0.2.0 (Toto handles
          variable resolutions implicitly when timestamps are zeroed).
        - point forecast = median over ``num_samples=256`` quantile samples
          (matches the paper-recommended setting).
    """

    family = "toto"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        num_samples: int = NUM_SAMPLES,
        samples_per_batch: int = SAMPLES_PER_BATCH,
    ):
        super().__init__()
        self.model_id = model_id
        self.num_samples = num_samples
        self.samples_per_batch = samples_per_batch
        self._device: torch.device | None = None
        self._forecaster = None
        self.extra_info = {
            "multivariate_mode": "factorized_space_time_attention",
            "num_samples": num_samples,
        }

    def load(self, device: str) -> None:
        """Load Toto via local snapshot + ``Toto.load_from_checkpoint``.

        We bypass ``Toto.from_pretrained`` because toto-ts 0.2.0 calls into
        a deprecated ``huggingface_hub`` signature; ``snapshot_download``
        gives us a directory containing both the safetensors and config.json
        which ``load_from_checkpoint`` reads directly.
        """
        from huggingface_hub import snapshot_download
        from toto.inference.forecaster import TotoForecaster
        from toto.model.toto import Toto

        torch_device = torch.device(
            device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        logger.info("Loading Toto: %s (device=%s)", self.model_id, torch_device)
        ckpt_dir = snapshot_download(
            repo_id=self.model_id,
            allow_patterns=["model.safetensors", "config.json"],
        )
        toto = Toto.load_from_checkpoint(ckpt_dir, map_location="cpu").to(torch_device)
        toto.eval()  # explicit — disables dropout/batchnorm-train statistics
        self._device = torch_device
        self._forecaster = TotoForecaster(toto.model)

    def predict(
        self,
        input_window: np.ndarray,
        context_h: int,
        prediction_length: int,
    ) -> np.ndarray:
        from toto.data.util.dataset import MaskedTimeseries

        if input_window.shape[0] < context_h:
            raise ValueError(f"input_window has only {input_window.shape[0]}h, need {context_h}h")
        context = input_window[-context_h:]  # (context_h, C)
        n_hours, n_channels = context.shape
        assert (
            n_channels == self.n_channels
        ), f"expected {self.n_channels} channels, got {n_channels}"

        series = torch.tensor(context.T, dtype=torch.float32, device=self._device)  # (C, ctx)
        inputs = MaskedTimeseries(
            series=series,
            padding_mask=torch.full_like(series, True, dtype=torch.bool),
            id_mask=torch.zeros_like(series),  # all variates share one group
            timestamp_seconds=torch.zeros_like(series),
            time_interval_seconds=torch.full(
                (n_channels,),
                HOURLY_INTERVAL_SEC,
                dtype=torch.long,
                device=self._device,
            ),
        )

        forecast = self._forecaster.forecast(
            inputs,
            prediction_length=prediction_length,
            num_samples=self.num_samples,
            samples_per_batch=self.samples_per_batch,
        )
        median = forecast.median.squeeze(0).detach().cpu().numpy()  # (C, H)
        return median.T  # (H, C)


def main():
    parser = make_sweep_argparser(
        description="Toto context-length sweep",
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

    predictor = TotoPredictor(model_id=args.model)
    predictor.load(args.device)

    print("\nToto Sweep (cross-channel, factorized space-time attention)")
    print(f"  Model:        {args.model}")
    print(f"  Horizon:      {prediction_length}h ({args.horizon_days}d)")
    print(
        f"  Context grid: {len(context_days)} values "
        f"({context_days[0]}..{context_days[-1]} days, max valid = {max_ctx_d}d)"
    )
    print(f"  Stations:     {len(csv_files)}")
    print(f"  Device:       {args.device}")
    print(f"  num_samples:  {NUM_SAMPLES}")
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
