"""Darts-backed baseline models (epoch-based PyTorch Lightning)."""

from __future__ import annotations

import inspect
import os
from typing import Iterable, List

import numpy as np

from scripts.baseline_study.data import KPI_NAMES, StationSeries, split_station
from scripts.baseline_study.models.base import BaselineModel

# ---------------------------------------------------------------------------
# Kind dispatch: string -> (darts model class, constructor name kwarg)
# ---------------------------------------------------------------------------

_RNN_KINDS = {"lstm", "gru"}
_BLOCK_RNN_KINDS = {"block_lstm", "block_gru"}
_PLAIN_KINDS = {"tcn", "nbeats", "nhits", "nlinear", "dlinear"}


def _build_darts_model(
    kind: str,
    input_chunk_length: int,
    output_chunk_length: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    pl_trainer_kwargs: dict,
    model_kwargs: dict,
):
    """Instantiate one Darts model, filtering kwargs to only those accepted."""
    from darts.models import (
        BlockRNNModel,
        DLinearModel,
        NBEATSModel,
        NHiTSModel,
        NLinearModel,
        RNNModel,
        TCNModel,
    )

    kind_lower = kind.lower()

    # Map kind -> (class, extra fixed kwargs)
    if kind_lower == "lstm":
        cls = RNNModel
        fixed = {"model": "LSTM"}
    elif kind_lower == "gru":
        cls = RNNModel
        fixed = {"model": "GRU"}
    elif kind_lower == "tcn":
        cls = TCNModel
        fixed = {}
    elif kind_lower == "block_lstm":
        cls = BlockRNNModel
        fixed = {"model": "LSTM"}
    elif kind_lower == "block_gru":
        cls = BlockRNNModel
        fixed = {"model": "GRU"}
    elif kind_lower == "nbeats":
        cls = NBEATSModel
        fixed = {}
    elif kind_lower == "nhits":
        cls = NHiTSModel
        fixed = {}
    elif kind_lower == "nlinear":
        cls = NLinearModel
        fixed = {}
    elif kind_lower == "dlinear":
        cls = DLinearModel
        fixed = {}
    else:
        raise KeyError(
            f"DartsForecastModel: unsupported kind={kind!r}. "
            f"Available: lstm, gru, tcn, block_lstm, block_gru, "
            f"nbeats, nhits, nlinear, dlinear"
        )

    # Filter architecture kwargs to only those accepted by this class
    filtered_model_kwargs = _filter_kwargs(cls, model_kwargs)

    candidate = {
        "input_chunk_length": input_chunk_length,
        "output_chunk_length": output_chunk_length,
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "optimizer_kwargs": {"lr": lr},
        "pl_trainer_kwargs": pl_trainer_kwargs,
        **filtered_model_kwargs,
    }

    filtered = _filter_kwargs(cls, candidate)
    # Always keep fixed kwargs (model="LSTM" etc.) even if not in filter
    filtered.update(fixed)

    return cls(**filtered)


def _filter_kwargs(cls, kwargs: dict) -> dict:
    """Return only kwargs accepted by cls.__init__."""
    sig = inspect.signature(cls.__init__)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


class DartsForecastModel(BaselineModel):
    """Univariate-x7 wrapper around the Darts library (PyTorch Lightning)."""

    family = "darts"

    def __init__(self, config: dict):
        super().__init__(config)
        self.task_cfg = config.get("task", {})
        self.train_cfg = config.get("training", {})
        self.runtime_cfg = config.get("runtime", {})

        self.context_h = int(self.task_cfg.get("context_h", 168))
        self.horizon_h = int(self.task_cfg.get("horizon_h", 168))
        self.eval_horizon_h = int(self.task_cfg.get("eval_horizon_h", self.horizon_h))

        # Training hyperparameters
        n_epochs_raw = self.train_cfg.get("n_epochs", self.train_cfg.get("train_epochs", 10))
        self.n_epochs = int(n_epochs_raw)
        self.patience = int(self.train_cfg.get("patience", 5))
        self.batch_size = int(self.train_cfg.get("batch_size", 32))
        self.lr = float(self.train_cfg.get("lr", 1e-3))

        # Architecture
        self.kind = str(self.model_cfg.get("kind", "lstm")).lower()
        self.hidden_size = int(self.model_cfg.get("hidden_size", 128))
        self.num_layers = int(self.model_cfg.get("num_layers", 2))
        self.dropout = float(self.model_cfg.get("dropout", 0.1))

        # Device
        self.device = str(self.runtime_cfg.get("device", "cuda"))

        # Derived name/forecast_mode if not set in config
        if not self.model_cfg.get("name"):
            self.name = f"darts_{self.kind}"
        if not self.model_cfg.get("forecast_mode"):
            self.forecast_mode = "univariate_x7"

        # State after fit()
        self.models: List = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_timeseries(self, array_1d: np.ndarray):
        """Convert 1-D numpy array to darts TimeSeries."""
        from darts import TimeSeries

        return TimeSeries.from_values(array_1d.astype(np.float32))

    def _pl_trainer_kwargs(self, use_cuda: bool) -> dict:
        """Build pl_trainer_kwargs with EarlyStopping callback."""
        try:
            from pytorch_lightning.callbacks import EarlyStopping
        except ImportError:
            from lightning.pytorch.callbacks import EarlyStopping

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                mode="min",
            )
        ]
        return {
            "accelerator": "gpu" if use_cuda else "cpu",
            "devices": 1,
            "enable_progress_bar": False,
            "callbacks": callbacks,
        }

    # ------------------------------------------------------------------
    # BaselineModel interface
    # ------------------------------------------------------------------

    @property
    def parameter_count(self) -> int:
        if not self.models:
            return 0
        total = 0
        for model in self.models:
            try:
                for p in model.model.parameters():
                    if p.requires_grad:
                        total += p.numel()
            except Exception:
                pass
        return int(total)

    def fit(self, stations: Iterable[StationSeries]) -> None:
        import torch

        # Optional GPU memory cap
        frac_str = os.environ.get("BASELINE_GPU_MEM_FRAC", "1.0")
        frac = float(frac_str)
        use_cuda = self.device == "cuda" and torch.cuda.is_available()
        if use_cuda and 0.0 < frac < 1.0:
            torch.cuda.set_per_process_memory_fraction(frac)

        pl_trainer_kwargs = self._pl_trainer_kwargs(use_cuda)

        from torch.optim.lr_scheduler import CosineAnnealingLR

        model_kwargs = {
            "hidden_dim": self.hidden_size,
            "hidden_size": self.hidden_size,
            "n_rnn_layers": self.num_layers,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            # Darts RNNModel constraint: training_length >= input_chunk_length
            # AND val_series.length >= training_length. We honor the first
            # (must) and let Darts skip the run if val_series is too short
            # (it'll surface as SKIPPED in W&B).
            "training_length": max(self.context_h, self.horizon_h),
            # Cosine annealing LR — best-practice. Sweeps then only need a
            # single lr value (max_lr) instead of bayes over a log-uniform range.
            "lr_scheduler_cls": CosineAnnealingLR,
            "lr_scheduler_kwargs": {"T_max": self.n_epochs},
        }

        station_list = list(stations)
        splits = [split_station(s.values) for s in station_list]

        self.models = []
        for kpi_idx in range(len(KPI_NAMES)):
            # Build per-station train and val TimeSeries for this KPI
            train_series_list = [self._to_timeseries(split.train[:, kpi_idx]) for split in splits]
            # val_series feeds Darts' internal early-stopping; use full
            # split.val (not eval_horizon_h truncation, which only applies
            # to the final metric evaluation in train.py).
            val_series_list = [self._to_timeseries(split.val[:, kpi_idx]) for split in splits]

            model = _build_darts_model(
                kind=self.kind,
                input_chunk_length=self.context_h,
                output_chunk_length=self.horizon_h,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                pl_trainer_kwargs=pl_trainer_kwargs,
                model_kwargs=model_kwargs,
            )

            model.fit(
                train_series_list,
                val_series=val_series_list,
            )
            self.models.append(model)

    def predict(
        self, history: np.ndarray, horizon_h: int, station_id: str | None = None
    ) -> np.ndarray:
        if not self.models:
            raise RuntimeError("DartsForecastModel.predict() called before fit()")

        preds = []
        for kpi_idx, model in enumerate(self.models):
            kpi_series = self._to_timeseries(history[:, kpi_idx])
            forecast = model.predict(n=horizon_h, series=kpi_series)
            values = forecast.values().flatten().astype(np.float32)
            preds.append(values[:horizon_h])

        return np.stack(preds, axis=1)

    def metadata(self) -> dict:
        meta = super().metadata()
        meta.update(
            {
                "forecast_mode": self.forecast_mode,
                "n_kpis": len(KPI_NAMES),
                "kind": self.kind,
                "n_epochs": self.n_epochs,
                "patience": self.patience,
                "context_h": self.context_h,
                "horizon_h": self.horizon_h,
            }
        )
        return meta
