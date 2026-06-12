"""Tabular lag-feature baselines using scikit-learn when available."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from scripts.baseline_study.data import KPI_NAMES, StationSeries, split_station
from scripts.baseline_study.models.base import BaselineModel


class ClassicalRegressorModel(BaselineModel):
    """Per-station tabular regressors: one model set fitted on each station's own
    history. LightGBM forecasts per-KPI (its regressor rejects 2D targets); every
    other kind forecasts all 7 KPIs jointly via a direct per-horizon regressor."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.context_h = int(config.get("task", {}).get("context_h", 168))
        self.horizon_h = int(config.get("task", {}).get("horizon_h", 168))
        self.stride_h = int(config.get("training", {}).get("stride_h", 24))
        self.models: dict = {}
        kind = str(self.model_cfg.get("kind", "ridge"))
        self.forecast_mode = "univariate_local" if kind == "lightgbm" else "multivariate_local"

    def fit(self, stations: Iterable[StationSeries]) -> None:
        kind = str(self.model_cfg.get("kind", "ridge"))
        if kind in {"ridge", "lasso", "elasticnet"} and self.context_h > 72:
            raise ValueError(
                f"linear classical kind={kind} with context_h={self.context_h} is intractable; "
                "sweep config should cap context_h<=72"
            )
        self.models = {}
        for station in stations:
            train = split_station(station.values).train
            max_start = train.shape[0] - self.context_h - self.horizon_h
            xs, ys = [], []
            for start in range(0, max_start + 1, self.stride_h):
                xs.append(self._features(train[start : start + self.context_h]))
                ys.append(train[start + self.context_h : start + self.context_h + self.horizon_h])
            if not xs:
                raise ValueError(f"No classical training windows for station {station.station_id}")
            self.models[station.station_id] = self._fit_station(np.stack(xs), np.stack(ys))

    def _fit_station(self, x: np.ndarray, y: np.ndarray) -> list:
        """Fit one station's model set. LightGBM -> one MultiOutputRegressor per KPI
        (its regressor rejects 2D y); others -> one multi-output regressor per
        horizon step forecasting all KPIs jointly."""
        model_cls = self._estimator_class()
        params = self._estimator_params()
        if str(self.model_cfg.get("kind", "ridge")) == "lightgbm":
            from sklearn.multioutput import MultiOutputRegressor

            return [
                MultiOutputRegressor(model_cls(**params)).fit(x, y[:, :, k])
                for k in range(len(KPI_NAMES))
            ]
        return [model_cls(**params).fit(x, y[:, h, :]) for h in range(self.horizon_h)]

    def predict(
        self, history: np.ndarray, horizon_h: int, station_id: str | None = None
    ) -> np.ndarray:
        if horizon_h != self.horizon_h:
            raise ValueError(f"{self.name}: configured horizon={self.horizon_h}, got {horizon_h}")
        if station_id is None:
            raise ValueError(f"{self.name} is a per-station model; predict() requires station_id")
        if station_id not in self.models:
            raise KeyError(f"{self.name}: no model fitted for station_id={station_id!r}")
        models = self.models[station_id]
        x = self._features(history[-self.context_h :])[None, :]
        if str(self.model_cfg.get("kind", "ridge")) == "lightgbm":
            # 7 MultiOutputRegressors, each yielding (1, horizon) for one KPI
            return np.stack([m.predict(x)[0] for m in models], axis=1)
        # horizon estimators, each yielding (1, 7) for one step
        return np.stack([m.predict(x)[0] for m in models], axis=0)

    def _features(self, window: np.ndarray) -> np.ndarray:
        lags = window.reshape(-1)
        means = window.mean(axis=0)
        stds = window.std(axis=0)
        last = window[-1]
        return np.concatenate([lags, means, stds, last])

    def _estimator_class(self):
        kind = str(self.model_cfg.get("kind", "ridge"))
        try:
            from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
            from sklearn.linear_model import ElasticNet, Lasso, Ridge
        except ImportError as exc:
            raise RuntimeError("classical baselines require scikit-learn") from exc
        if kind == "ridge":
            return Ridge
        if kind == "lasso":
            return Lasso
        if kind == "elasticnet":
            return ElasticNet
        if kind == "random_forest":
            return RandomForestRegressor
        if kind == "extra_trees":
            return ExtraTreesRegressor
        if kind == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise RuntimeError("xgboost baseline requires xgboost") from exc
            return XGBRegressor
        if kind == "lightgbm":
            try:
                from lightgbm import LGBMRegressor
            except ImportError as exc:
                raise RuntimeError("lightgbm baseline requires lightgbm") from exc
            return LGBMRegressor
        raise KeyError(f"Unknown classical kind={kind!r}; KPIs={KPI_NAMES}")

    def _estimator_params(self) -> dict:
        kind = str(self.model_cfg.get("kind", "ridge"))
        raw = dict(self.model_cfg.get("params", {}))
        allow = {
            "ridge": {"alpha", "fit_intercept", "solver", "random_state"},
            "lasso": {"alpha", "fit_intercept", "max_iter", "random_state", "selection"},
            "elasticnet": {
                "alpha",
                "l1_ratio",
                "fit_intercept",
                "max_iter",
                "random_state",
                "selection",
            },
            "random_forest": {
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "random_state",
                "n_jobs",
            },
            "extra_trees": {
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "random_state",
                "n_jobs",
            },
            "xgboost": {
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "random_state",
                "n_jobs",
            },
            "lightgbm": {
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "random_state",
                "n_jobs",
            },
        }
        params = {k: v for k, v in raw.items() if k in allow.get(kind, set()) and v is not None}
        if kind in {"random_forest", "extra_trees", "xgboost", "lightgbm"}:
            params.setdefault("n_jobs", -1)
        if kind in {"lasso", "elasticnet"}:
            params.setdefault("max_iter", 5000)
        return params
