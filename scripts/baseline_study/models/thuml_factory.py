"""THUML Time-Series-Library wrapper for the dense baseline study.

Wraps Exp_Long_Term_Forecast from archive/thuml_tsl for multivariate joint
forecasting of all 7 KPIs simultaneously.
"""

from __future__ import annotations

import os
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts.baseline_study.data import KPI_NAMES, StationSeries, split_station
from scripts.baseline_study.models.base import BaselineModel
from scripts.baseline_study.models.thuml_multistation import STATION_LEN as _STATION_LEN

# ---------------------------------------------------------------------------
# Absolute path to the vendored THUML repo
# ---------------------------------------------------------------------------

_THUML_ROOT = str(Path(__file__).resolve().parents[3] / "archive" / "thuml_tsl")

# ---------------------------------------------------------------------------
# Kind → THUML CamelCase model name
# ---------------------------------------------------------------------------

_KIND_MAP: dict[str, str] = {
    "patchtst": "PatchTST",
    "itransformer": "iTransformer",
    "timesnet": "TimesNet",
    "timemixer": "TimeMixer",
    "tsmixer": "TSMixer",
    "timexer": "TimeXer",
    "fedformer": "FEDformer",
    "informer": "Informer",
    "autoformer": "Autoformer",
    "dlinear": "DLinear",
    "tide": "TiDE",
}


def _resolve_kind(kind: str) -> str:
    key = kind.lower()
    if key not in _KIND_MAP:
        raise KeyError(
            f"THUMLForecastModel: unsupported kind={kind!r}. " f"Available: {sorted(_KIND_MAP)}"
        )
    return _KIND_MAP[key]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

# THUML's Dataset_Custom re-orders columns: date + all-but-target + target.
# We use the last KPI column as the "target" placeholder (it is a multivariate
# run so target is never actually used as a single-channel filter).
_TARGET_COL = KPI_NAMES[-1]  # "Throughput" — last KPI stays last


def _stations_to_csv(stations: list[StationSeries], *, use_split: str) -> str:
    """Concatenate station data into a single long CSV understood by Dataset_Custom.

    Returns path to a temporary CSV file. Caller is responsible for cleanup.
    Dataset_Custom auto-splits 70/20/10 from this combined CSV — we write
    train+val+test rows so THUML sees the full timeline.
    """
    frames = []
    # Use a fixed anchor date; only relative time matters for THUML's timeenc.
    anchor = pd.Timestamp("2000-01-01")
    for st in stations:
        split = split_station(st.values)
        if use_split == "train":
            data = split.train
        elif use_split == "val":
            data = split.val
        elif use_split == "test":
            data = split.test
        else:  # "all"
            data = st.values
        n = data.shape[0]
        dates = pd.date_range(start=anchor, periods=n, freq="h")
        df = pd.DataFrame(data, columns=KPI_NAMES)
        df.insert(0, "date", dates.strftime("%Y-%m-%d %H:%M:%S"))
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    # Ensure target column is last (Dataset_Custom requirement)
    cols = list(combined.columns)
    cols.remove(_TARGET_COL)
    cols.remove("date")
    combined = combined[["date"] + cols + [_TARGET_COL]]
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    combined.to_csv(tmp.name, index=False)
    return tmp.name


def _history_to_csv(history: np.ndarray, *, horizon_h: int = 168) -> str:
    """Write a single-station history array to a temp CSV for inference.

    THUML's Dataset_Custom does an internal 70/20/10 split, so even when we
    only want the test slice we must supply a CSV long enough that the test
    portion (~20%) contains at least seq_len + pred_len rows. Pad by
    prepending repeats of the first row; the inference window is taken from
    the END of the test slice, so the original history is preserved there.
    """
    n = history.shape[0]
    # Need test_portion >= seq_len + pred_len. With 20% test: total >= 5*(n + horizon).
    min_total = max(5 * (n + horizon_h), n + 100)
    pad_rows = max(0, min_total - n)
    if pad_rows > 0:
        pad = np.tile(history[:1], (pad_rows, 1))
        padded = np.concatenate([pad, history], axis=0)
    else:
        padded = history
    total = padded.shape[0]
    anchor = pd.Timestamp("2000-01-01")
    dates = pd.date_range(start=anchor, periods=total, freq="h")
    df = pd.DataFrame(padded, columns=KPI_NAMES)
    df.insert(0, "date", dates.strftime("%Y-%m-%d %H:%M:%S"))
    cols = list(df.columns)
    cols.remove(_TARGET_COL)
    cols.remove("date")
    df = df[["date"] + cols + [_TARGET_COL]]
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp.name, index=False)
    return tmp.name


# ---------------------------------------------------------------------------
# THUMLForecastModel
# ---------------------------------------------------------------------------


class THUMLForecastModel(BaselineModel):
    """Multivariate-joint wrapper around THUML's Exp_Long_Term_Forecast."""

    family = "thuml"

    def __init__(self, config: dict):
        super().__init__(config)
        task_cfg = config.get("task", {})
        train_cfg = config.get("training", {})
        runtime_cfg = config.get("runtime", {})

        self.context_h = int(task_cfg.get("context_h", 168))
        self.horizon_h = int(task_cfg.get("horizon_h", 168))
        self.eval_horizon_h = int(task_cfg.get("eval_horizon_h", self.horizon_h))

        self.n_epochs = int(train_cfg.get("n_epochs", 10))
        self.patience = int(train_cfg.get("patience", 5))
        self.batch_size = int(train_cfg.get("batch_size", 32))
        self.lr = float(train_cfg.get("lr", 1e-4))

        self.kind = str(self.model_cfg.get("kind", "patchtst")).lower()
        self.hidden_size = int(self.model_cfg.get("hidden_size", 128))
        self.num_layers = int(self.model_cfg.get("num_layers", 2))
        self.dropout = float(self.model_cfg.get("dropout", 0.1))

        self.device = str(runtime_cfg.get("device", "cuda"))

        if not self.model_cfg.get("name"):
            self.name = f"thuml_{self.kind}"
        # per-station: one THUML Exp model fitted on each station's own history
        self.forecast_mode = "multivariate_local"

        self._stations: dict = {}
        self._args = None
        self._ckpt_dir: str | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_args(self, root_path: str, data_path: str) -> Namespace:
        kind_cc = _resolve_kind(self.kind)
        use_gpu = self.device == "cuda"
        ckpt_dir = tempfile.mkdtemp(prefix="thuml_ckpt_")
        self._ckpt_dir = ckpt_dir

        return Namespace(
            task_name="long_term_forecast",
            model=kind_cc,
            data="multistation",
            station_len=_STATION_LEN,
            features="M",
            seq_len=self.context_h,
            label_len=self.context_h // 2,
            pred_len=self.horizon_h,
            enc_in=7,
            dec_in=7,
            c_out=7,
            d_model=self.hidden_size,
            e_layers=self.num_layers,
            d_layers=1,
            d_ff=4 * self.hidden_size,
            n_heads=8,
            dropout=self.dropout,
            embed="timeF",
            factor=1,
            distil=True,
            batch_size=self.batch_size,
            train_epochs=self.n_epochs,
            patience=self.patience,
            learning_rate=self.lr,
            des="Exp",
            loss="MSE",
            lradj="cosine",
            use_amp=False,
            use_gpu=use_gpu,
            gpu=0,
            gpu_type="cuda",
            use_multi_gpu=False,
            devices="0",
            device_ids=[0],
            checkpoints=ckpt_dir,
            num_workers=0,
            itr=1,
            freq="h",
            target=_TARGET_COL,
            activation="gelu",
            do_predict=False,
            # Dataset_Custom extra args
            root_path=root_path,
            data_path=data_path,
            # Augmentation / DTW flags expected by some THUML internals
            augmentation_ratio=0,
            seed=42,
            jitter=False,
            scaling=False,
            permutation=False,
            randompermutation=False,
            magwarp=False,
            timewarp=False,
            windowslice=False,
            windowwarp=False,
            rotation=False,
            spawner=False,
            dtwwarp=False,
            shapedtwwarp=False,
            wdba=False,
            discdtw=False,
            discsdtw=False,
            extra_tag="",
            # Misc args referenced by various THUML models / data providers
            seasonal_patterns="Monthly",
            inverse=False,
            top_k=5,
            num_kernels=6,
            moving_avg=25,
            channel_independence=1,
            decomp_method="moving_avg",
            use_norm=1,
            down_sampling_layers=0,
            down_sampling_window=1,
            down_sampling_method=None,
            p_hidden_dims=[128, 128],
            p_hidden_layers=2,
            patch_len=16,
            stride=8,
            mask_rate=0.25,
            anomaly_ratio=0.25,
            expand=2,
            d_conv=4,
            n_topk=5,
        )

    def _push_thuml_path(self) -> None:
        if _THUML_ROOT not in sys.path:
            sys.path.insert(0, _THUML_ROOT)

    # ------------------------------------------------------------------
    # BaselineModel interface
    # ------------------------------------------------------------------

    @property
    def parameter_count(self) -> int:
        if not self._stations:
            return 0
        model = next(iter(self._stations.values()))["exp"].model
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    def fit(self, stations: Iterable[StationSeries]) -> None:
        import torch

        # Optional GPU memory cap
        frac_str = os.environ.get("BASELINE_GPU_MEM_FRAC", "1.0")
        frac = float(frac_str)
        if self.device == "cuda" and torch.cuda.is_available() and 0.0 < frac < 1.0:
            torch.cuda.set_per_process_memory_fraction(frac)

        # Fall back to CPU if CUDA not available
        effective_device = self.device
        if effective_device == "cuda" and not torch.cuda.is_available():
            effective_device = "cpu"

        station_list = list(stations)
        self._push_thuml_path()

        # THUML scans relative 'models/' dir — must cd to its root
        orig_dir = os.getcwd()
        self._stations = {}
        try:
            os.chdir(_THUML_ROOT)
            # Register the station-boundary-aware dataset so THUML splits a
            # station's own timeline 67/14/7 (single-station CSV here).
            from data_provider import data_factory  # noqa: PLC0415
            from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast  # noqa: PLC0415

            from scripts.baseline_study.models.thuml_multistation import (  # noqa: PLC0415
                Dataset_MultiStation,
            )

            data_factory.data_dict["multistation"] = Dataset_MultiStation

            for st in station_list:
                # one CSV holding only this station -> its own 67/14/7 split
                csv_path = _stations_to_csv([st], use_split="all")
                try:
                    args = self._build_args(
                        root_path=str(Path(csv_path).parent), data_path=Path(csv_path).name
                    )
                    args.use_gpu = effective_device == "cuda"
                    exp = Exp_Long_Term_Forecast(args)
                    exp.train(setting=f"thuml_fit_{st.station_id}")
                    # Capture the train-time scaler so predict() can scale/inverse
                    try:
                        train_data, _ = exp._get_data(flag="train")
                        scaler = getattr(train_data, "scaler", None)
                    except Exception:
                        scaler = None
                    # store on CPU so 115 per-station models never co-reside on GPU
                    exp.model.to("cpu")
                    if effective_device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self._stations[st.station_id] = {"exp": exp, "args": args, "scaler": scaler}
                finally:
                    try:
                        os.unlink(csv_path)
                    except OSError:
                        pass
        finally:
            os.chdir(orig_dir)

        if not self._stations:
            raise ValueError("THUMLForecastModel.fit received no stations")
        self._args = next(iter(self._stations.values()))["args"]

    def predict(
        self, history: np.ndarray, horizon_h: int, station_id: str | None = None
    ) -> np.ndarray:
        if station_id is None:
            raise ValueError(f"{self.name} is a per-station model; predict() requires station_id")
        if not self._stations:
            raise RuntimeError("THUMLForecastModel.predict() called before fit()")
        if station_id not in self._stations:
            raise KeyError(f"{self.name}: no model fitted for station_id={station_id!r}")

        import torch

        bundle = self._stations[station_id]
        args = bundle["args"]
        scaler = bundle["scaler"]
        model = bundle["exp"].model

        orig_dir = os.getcwd()
        try:
            os.chdir(_THUML_ROOT)

            seq_len = args.seq_len
            label_len = args.label_len

            model.eval()
            device = torch.device("cpu")  # per-station models are stored on CPU

            # Take the last seq_len rows of history. If history shorter than
            # seq_len, left-pad with the first available row.
            if history.shape[0] < seq_len:
                pad = np.tile(history[:1], (seq_len - history.shape[0], 1))
                ctx = np.concatenate([pad, history], axis=0)
            else:
                ctx = history[-seq_len:]

            # Standardize using this station's fit-time scaler (identity if absent).
            ctx_scaled = scaler.transform(ctx) if scaler is not None else ctx

            batch_x = torch.tensor(ctx_scaled, dtype=torch.float32).unsqueeze(0).to(device)

            # Time marks: timeF encoding yields 4 features for hourly freq.
            anchor = pd.Timestamp("2000-01-01")
            ctx_dates = pd.date_range(anchor, periods=seq_len, freq="h")
            future_dates = pd.date_range(
                ctx_dates[-1] + pd.Timedelta(hours=1), periods=horizon_h, freq="h"
            )
            full_dates = ctx_dates.append(future_dates)

            from utils.timefeatures import time_features  # noqa: PLC0415

            tf = time_features(full_dates, freq="h").transpose(1, 0).astype(np.float32)
            x_mark_np = tf[:seq_len]
            y_mark_np = tf[seq_len - label_len :]

            batch_x_mark = torch.tensor(x_mark_np, dtype=torch.float32).unsqueeze(0).to(device)
            batch_y_mark = torch.tensor(y_mark_np, dtype=torch.float32).unsqueeze(0).to(device)

            # Decoder input: label prefix from history + zeros for forecast
            label_prefix = batch_x[:, -label_len:, :].clone()
            forecast_zeros = torch.zeros(1, horizon_h, 7, dtype=torch.float32, device=device)
            dec_inp_full = torch.cat([label_prefix, forecast_zeros], dim=1)

            with torch.no_grad():
                output = model(batch_x, batch_x_mark, dec_inp_full, batch_y_mark)

            # output shape: (1, pred_len, 7)
            pred = output[:, -horizon_h:, :].squeeze(0).cpu().numpy()

            # Inverse-transform back to original KPI scale.
            if scaler is not None and hasattr(scaler, "inverse_transform"):
                pred = scaler.inverse_transform(pred)

            return pred.astype(np.float32)
        finally:
            os.chdir(orig_dir)

    def metadata(self) -> dict:
        meta = super().metadata()
        meta.update(
            {
                "forecast_mode": self.forecast_mode,
                "kind": self.kind,
                "n_epochs": self.n_epochs,
                "patience": self.patience,
                "thuml_root": _THUML_ROOT,
            }
        )
        return meta
