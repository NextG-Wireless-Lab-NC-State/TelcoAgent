#!/usr/bin/env python3
"""MOMENT context-length sweep (channel-independent, batched 7 KPIs).

MOMENT (CMU, ``AutonLab/MOMENT-1-large``) is a fixed-input-length encoder-only
TSFM (seq_len = 512). Multivariate forecasts are produced by stacking the 7
KPIs along the batch axis in a single forward pass — the model itself does
not share information across channels (channel-independent).

Inference protocol:
    - ``model.init()`` then ``model.eval()`` (the original code skipped
      .eval(), leaving dropout enabled in MOMENT's forecast head).
    - All 7 channels forwarded in a single batch ``(7, 1, 512)`` tensor
      instead of 7 sequential forwards (~7x speed-up, identical output).
    - ``with torch.no_grad():`` wraps the forward.

Note on context length:
    The model's physical input is always 512h regardless of ``--context-days``.
    Shorter contexts are left-padded with edge-replication; longer ones are
    truncated to the trailing 512h. This means the effective sweep cap is
    ~21d (=504h) — beyond that, results plateau.

Usage:
    conda run -n telcoagent python scripts/baselines/run_moment_context_sweep.py

    # smoke test
    conda run -n telcoagent python scripts/baselines/run_moment_context_sweep.py \\
        --stations station_A_10 --context-days 21
"""

from __future__ import annotations

import os

# Block transformers' optional tensorflow import path. Recent transformers
# versions transitively import ``image_transforms`` -> ``tensorflow`` from
# ``video_utils``; on a torch-only env with NumPy 2.x the TF wheel fails
# to load with a binary-compat error. We don't need TF at all.
os.environ.setdefault("USE_TF", "NO")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

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


DEFAULT_MODEL_ID = "AutonLab/MOMENT-1-large"
MOMENT_SEQ_LEN = 512  # fixed input length for MOMENT-1-large


class MomentPredictor(BaseTSFMPredictor):
    """MOMENT zero-shot predictor (channel-independent, batched).

    The official MOMENT API forecasts each channel independently. This
    predictor stacks all 7 KPIs along the batch axis so a single forward
    pass produces the full ``(168, 7)`` forecast.
    """

    family = "moment"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        super().__init__()
        self.model_id = model_id
        self._model = None
        self._device: torch.device | None = None
        self.extra_info = {
            "multivariate_mode": "channel_independent",
            "model_seq_len": MOMENT_SEQ_LEN,
        }

    def load(self, device: str) -> None:
        from momentfm import MOMENTPipeline

        torch_device = torch.device(
            device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        logger.info("Loading MOMENT: %s (device=%s)", self.model_id, torch_device)
        # Important: forecast_horizon is baked into the model at init time,
        # so we must pass it before calling .init().
        model = MOMENTPipeline.from_pretrained(
            self.model_id,
            model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": 168,  # base horizon; we reshape per-call
            },
        )
        model.init()
        # Move to device. Some MOMENTPipeline builds expose .to() at the
        # wrapper level, others only on the inner .model — try both.
        try:
            model = model.to(torch_device)
        except AttributeError:
            if hasattr(model, "model"):
                model.model = model.model.to(torch_device)
            elif hasattr(model, "backbone"):
                model.backbone = model.backbone.to(torch_device)
        # Explicit eval — disables dropout in forecast head.
        try:
            model.eval()
        except AttributeError:
            if hasattr(model, "model"):
                model.model.eval()
        self._model = model
        self._device = torch_device

    def predict(
        self,
        input_window: np.ndarray,
        context_h: int,
        prediction_length: int,
    ) -> np.ndarray:
        if input_window.shape[0] < context_h:
            raise ValueError(f"input_window has only {input_window.shape[0]}h, need {context_h}h")
        context = input_window[-context_h:]  # (context_h, C)
        n_hours, n_channels = context.shape

        # Per-channel: pad/truncate to MOMENT's fixed 512h input.
        per_channel = np.empty((n_channels, MOMENT_SEQ_LEN), dtype=np.float32)
        for c in range(n_channels):
            chan = context[:, c].astype(np.float32)
            if len(chan) < MOMENT_SEQ_LEN:
                chan = np.pad(chan, (MOMENT_SEQ_LEN - len(chan), 0), mode="edge")
            elif len(chan) > MOMENT_SEQ_LEN:
                chan = chan[-MOMENT_SEQ_LEN:]
            per_channel[c] = chan

        # Batched forward: shape (batch=n_channels, n_var=1, seq_len)
        x = torch.from_numpy(per_channel).unsqueeze(1).to(self._device)

        with torch.no_grad():
            output = self._model(x_enc=x)

        # output.forecast: shape (batch=n_channels, n_var=1, horizon) for
        # this task; we squeeze the singleton variate axis.
        forecast = output.forecast.squeeze(1).detach().cpu().numpy()  # (n_channels, H_model)
        H_model = forecast.shape[-1]

        # Pad / truncate per-channel to the requested prediction_length.
        if H_model >= prediction_length:
            forecast = forecast[:, :prediction_length]
        else:
            pad_size = prediction_length - H_model
            forecast = np.pad(forecast, ((0, 0), (0, pad_size)), mode="edge")

        return forecast.T  # (prediction_length, n_channels)


def main():
    parser = make_sweep_argparser(
        description="MOMENT context-length sweep",
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

    predictor = MomentPredictor(model_id=args.model)
    predictor.load(args.device)

    print("\nMOMENT Sweep (channel-independent, batched 7 KPIs)")
    print(f"  Model:          {args.model}")
    print(f"  Horizon:        {prediction_length}h ({args.horizon_days}d)")
    print(
        f"  Context grid:   {len(context_days)} values "
        f"({context_days[0]}..{context_days[-1]} days, max valid = {max_ctx_d}d)"
    )
    print(
        f"  MOMENT seq_len: {MOMENT_SEQ_LEN} "
        f"(contexts > {MOMENT_SEQ_LEN}h are truncated; <{MOMENT_SEQ_LEN}h are edge-padded)"
    )
    print(f"  Stations:       {len(csv_files)}")
    print(f"  Device:         {args.device}")
    print(f"  Output:         {output_root}")

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
