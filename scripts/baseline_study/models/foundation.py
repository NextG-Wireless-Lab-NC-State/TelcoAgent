"""Time-series foundation models (TSFM) behind the BaselineModel interface.

Chronos-2, Moirai, and MOMENT are zero-shot: ``fit()`` only loads weights, and
``predict(history, horizon_h)`` forecasts from the last ``context_h`` rows of the
history the harness supplies. Because the harness feeds ``train`` (1608h) at the
val split and ``train+val`` (1944h) at the test split, ``context_h`` behaves like
any other swept hyperparameter and is selected on val — closing the test-set
context-selection leak of the legacy runners.

Keep ``context_h`` <= 1608 (67d) in sweeps so both splits can supply it and the
selected context is identical across val/test.

Inference logic is ported faithfully from the legacy runners:
``scripts/baselines/run_{chronos2,moirai,moment}_context_sweep.py``.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from scripts.baseline_study.data import (
    StationSeries,
    finetune_regime,
    finetune_slices,
    split_finetune_slices,
)
from scripts.baseline_study.models.base import BaselineModel

N_CHANNELS = 7
MOMENT_SEQ_LEN = 512


def _resolve_device(config: dict) -> str:
    requested = str(config.get("runtime", {}).get("device", "cuda"))
    if requested == "cuda":
        import torch

        if not torch.cuda.is_available():
            return "cpu"
    return requested


class _FoundationModel(BaselineModel):
    """Shared scaffolding for zero-shot TSFM runners."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.context_h = int(self.model_cfg.get("context_h", 1608))
        self.model_id = str(self.model_cfg.get("model_id", self._default_model_id()))
        self.device = _resolve_device(config)
        self._pipeline = None
        # Fine-tune regime knob (model-agnostic; only ChronosModel wires it for now).
        # 0 -> zero-shot; int>=2 -> few-shot N trailing weeks; "full" -> entire
        # pre-test range. Resolved here so metadata() can report it even for the
        # zero-shot families that ignore it.
        task = config.get("task", {})
        self.finetune_weeks: int | str = task.get("finetune_weeks", 0)
        self.horizon_h = int(task.get("horizon_h", 168))
        self.regime = finetune_regime(self.finetune_weeks)
        self._n_train_windows = 0
        # Cumulative-group-sweep provenance (absent for the temporal sweep -> None).
        # n_train_stations / n_train_groups: count of fine-tune stations / distinct
        # site groups among them. n_test_groups: distinct site groups in test_stations.
        # site group = token at split("_")[1] e.g. "station_C_3" -> "C".
        # All three are None when the corresponding key is absent from config.
        data = config.get("data", {})
        train_stations = data.get("train_stations")
        test_stations = data.get("test_stations")
        self.n_train_stations: int | None = (
            len(train_stations) if train_stations is not None else None
        )
        self.n_train_groups: int | None = (
            len({s.split("_")[1] for s in train_stations}) if train_stations is not None else None
        )
        self.n_test_groups: int | None = (
            len({s.split("_")[1] for s in test_stations}) if test_stations is not None else None
        )
        # Test-station count: the shrinking-test-set denominator for the cumulative
        # sweep (115 at G=0 down to one group at G=12). Derived alongside n_test_groups.
        self.n_test_stations: int | None = len(test_stations) if test_stations is not None else None

    def _default_model_id(self) -> str:
        raise NotImplementedError

    def _context(self, history: np.ndarray) -> np.ndarray:
        if history.shape[0] < self.context_h:
            raise ValueError(
                f"{self.name}: history has {history.shape[0]}h, need context_h={self.context_h}h"
            )
        return history[-self.context_h :]

    def metadata(self) -> dict:
        meta = super().metadata()
        meta.update(
            {
                "model_id": self.model_id,
                "context_h": self.context_h,
                "device": self.device,
                "finetune_weeks": self.finetune_weeks,
                "regime": self.regime,
                "n_train_windows": self._n_train_windows,
                "n_train_stations": self.n_train_stations,
                "n_train_groups": self.n_train_groups,
                "n_test_groups": self.n_test_groups,
                "n_test_stations": self.n_test_stations,
            }
        )
        return meta


class ChronosModel(_FoundationModel):
    """Chronos-2 cross-channel (group attention over the 7 KPIs).

    Requires chronos-forecasting==2.0.0 (2.2.2 regresses to flat predictions).
    """

    def _default_model_id(self) -> str:
        return "amazon/chronos-2"

    def fit(self, stations: Iterable[StationSeries]) -> None:
        import torch
        from chronos import Chronos2Pipeline

        device = self.device if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
        # chronos-forecasting==2.0.0's from_pretrained forwards kwargs to the
        # model __init__, which does not accept ``dtype``; the model already
        # loads as float32 by default.
        pipeline = Chronos2Pipeline.from_pretrained(self.model_id, device_map=device)

        if self.regime == "zero":
            # Zero-shot: load weights, no fine-tuning (single path, no fallback).
            if hasattr(pipeline, "model"):
                pipeline.model.eval()
            self._pipeline = pipeline
            return

        # Few-shot / full: fine-tune on the pooled trailing-N-week slices ending at
        # day 81 (leak-free), using ONLY the official Chronos-2 fit() API. Passing
        # validation_inputs enables best-model selection (load_best_model_at_end on
        # eval_loss). Chronos-2 v2.0.0 has no patience/early-stopping knob, so training
        # runs the full num_steps and fit() returns the best-by-val-loss checkpoint.
        ft = self.config.get("finetune", {})
        epochs = int(ft.get("epochs", 100))
        batch_size = int(ft.get("batch_size", 32))
        lr = float(ft.get("lr", 1e-4))
        lr_scheduler = str(ft.get("lr_scheduler", "cosine"))
        # stride_h defaults to 24h, matching the supervised harness.
        stride_h = int(ft.get("stride_h", 24))
        val_fraction = float(ft.get("val_fraction", 0.15))

        all_slices, _ = finetune_slices(
            stations,
            finetune_weeks=self.finetune_weeks,
            context_h=self.context_h,
            horizon_h=self.horizon_h,
            stride_h=stride_h,
        )
        # 15% val holdout at the slice level (fixed seed). All slices within a regime
        # are equal length, so slice-level fraction == window-level fraction.
        train_slices, val_slices, n_train_windows, _ = split_finetune_slices(
            all_slices,
            val_fraction=val_fraction,
            context_h=self.context_h,
            horizon_h=self.horizon_h,
            stride_h=stride_h,
        )
        self._n_train_windows = n_train_windows

        steps_per_epoch = max(1, math.ceil(n_train_windows / batch_size))
        num_steps = epochs * steps_per_epoch  # epochs -> documented num_steps budget

        # chronos2 consumes channel-first (n_variates, history) series, matching predict().
        train_inputs = [s.T for s in train_slices]
        val_inputs = [s.T for s in val_slices]

        finetune_dir = self.config.get("output", {}).get("finetune_dir")
        self._pipeline = pipeline.fit(
            train_inputs,
            prediction_length=self.horizon_h,
            validation_inputs=val_inputs,
            context_length=self.context_h,
            learning_rate=lr,
            num_steps=num_steps,
            batch_size=batch_size,
            output_dir=finetune_dir,
            lr_scheduler_type=lr_scheduler,
        )
        if hasattr(self._pipeline, "model"):
            self._pipeline.model.eval()

    def predict(
        self, history: np.ndarray, horizon_h: int, station_id: str | None = None
    ) -> np.ndarray:
        import torch

        context = self._context(history)  # (context_h, C)
        input_tensor = torch.tensor(context.T[np.newaxis, :, :], dtype=torch.float32)  # (1, C, ctx)
        with torch.no_grad():
            output = self._pipeline.predict(
                input_tensor, prediction_length=horizon_h, predict_batches_jointly=True
            )
        pred_samples = output[0]  # (C, n_samples, H)
        pred = torch.median(pred_samples, dim=1).values.cpu().numpy()  # (C, H)
        return pred.T.astype(np.float32)  # (H, C)


class MoiraiModel(_FoundationModel):
    """Moirai Any-Variate Attention (target_dim=7), median of 20 samples."""

    def _default_model_id(self) -> str:
        return "Salesforce/moirai-1.1-R-large"

    def fit(self, stations: Iterable[StationSeries]) -> None:
        import torch
        from uni2ts.model.moirai import MoiraiModule

        module = MoiraiModule.from_pretrained(self.model_id)
        if self.device == "cuda" and torch.cuda.is_available():
            module = module.to(self.device)
        self._module = module
        self._horizon = None  # forecaster built lazily once horizon is known

    def _build_forecaster(self, horizon_h: int):
        from uni2ts.model.moirai import MoiraiForecast

        return MoiraiForecast(
            module=self._module,
            prediction_length=horizon_h,
            context_length=self.context_h,
            patch_size="auto",
            num_samples=20,
            target_dim=N_CHANNELS,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

    def predict(
        self, history: np.ndarray, horizon_h: int, station_id: str | None = None
    ) -> np.ndarray:
        import pandas as pd
        from gluonts.dataset.common import ListDataset

        context = self._context(history)  # (context_h, C)
        forecaster = self._build_forecaster(horizon_h)
        dataset = ListDataset(
            [{"start": pd.Timestamp("2024-01-01"), "target": context.T}],  # (C, ctx)
            freq="h",
            one_dim_target=False,
        )
        predictor = forecaster.create_predictor(batch_size=1)
        forecast = list(predictor.predict(dataset))[0]
        pred = np.median(forecast.samples, axis=0)  # (H, C)
        if pred.shape[0] < horizon_h:
            pred = np.pad(pred, ((0, horizon_h - pred.shape[0]), (0, 0)), mode="edge")
        return pred[:horizon_h].astype(np.float32)


class MomentModel(_FoundationModel):
    """MOMENT channel-independent zero-shot forecast (fixed 512h input)."""

    def _default_model_id(self) -> str:
        return "AutonLab/MOMENT-1-large"

    def fit(self, stations: Iterable[StationSeries]) -> None:
        import torch
        from momentfm import MOMENTPipeline

        device = torch.device(
            self.device if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        model = MOMENTPipeline.from_pretrained(
            self.model_id,
            model_kwargs={"task_name": "forecasting", "forecast_horizon": 168},
        )
        model.init()
        try:
            model = model.to(device)
        except AttributeError:
            if hasattr(model, "model"):
                model.model = model.model.to(device)
        try:
            model.eval()
        except AttributeError:
            if hasattr(model, "model"):
                model.model.eval()
        self._pipeline = model
        self._device = device

    def predict(
        self, history: np.ndarray, horizon_h: int, station_id: str | None = None
    ) -> np.ndarray:
        import torch

        context = self._context(history)  # (context_h, C)
        per_channel = np.empty((N_CHANNELS, MOMENT_SEQ_LEN), dtype=np.float32)
        for c in range(N_CHANNELS):
            chan = context[:, c].astype(np.float32)
            if len(chan) < MOMENT_SEQ_LEN:
                chan = np.pad(chan, (MOMENT_SEQ_LEN - len(chan), 0), mode="edge")
            elif len(chan) > MOMENT_SEQ_LEN:
                chan = chan[-MOMENT_SEQ_LEN:]
            per_channel[c] = chan
        x = torch.from_numpy(per_channel).unsqueeze(1).to(self._device)
        with torch.no_grad():
            output = self._pipeline(x_enc=x)
        forecast = output.forecast.squeeze(1).detach().cpu().numpy()  # (C, H_model)
        h_model = forecast.shape[-1]
        if h_model >= horizon_h:
            forecast = forecast[:, :horizon_h]
        else:
            forecast = np.pad(forecast, ((0, 0), (0, horizon_h - h_model)), mode="edge")
        return forecast.T.astype(np.float32)  # (H, C)
