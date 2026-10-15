"""statsforecast-backed non-parametric baselines."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import HistoricAverage, Naive, SeasonalNaive, WindowAverage

from scripts.baseline_study.data import KPI_NAMES, StationSeries, split_station
from scripts.baseline_study.models.base import BaselineModel

_KIND_MAP = {
    "naive": lambda cfg: Naive(),
    "historic_average": lambda cfg: HistoricAverage(),
    "window_average": lambda cfg: WindowAverage(window_size=int(cfg.get("window_h", 24))),
    "seasonal_naive": lambda cfg: SeasonalNaive(season_length=int(cfg.get("season_h", 168))),
}


class StatsForecastModel(BaselineModel):
    """statsforecast wrapper for Naive, HistoricAverage, WindowAverage, SeasonalNaive."""

    def __init__(self, config: dict):
        super().__init__(config)
        if not self.model_cfg.get("forecast_mode"):
            self.forecast_mode = "univariate_x7"
        if not self.model_cfg.get("family"):
            self.family = "statsforecast"
        kind = str(self.model_cfg.get("kind", "naive"))
        if not self.model_cfg.get("name"):
            self.name = f"sf_{kind}"
        self._kind = kind
        self._horizon_h = int(config.get("task", {}).get("horizon_h", 168))
        self._window_h = int(self.model_cfg.get("window_h", 24))
        self._season_h = int(self.model_cfg.get("season_h", 168))

        if kind not in _KIND_MAP:
            raise KeyError(f"Unknown statsforecast kind={kind!r}. Valid: {list(_KIND_MAP)}")

        # Per-KPI fitted StatsForecast objects (set after fit())
        self._sf_models: List[StatsForecast] = []

    # ------------------------------------------------------------------
    # BaselineModel interface
    # ------------------------------------------------------------------

    @property
    def parameter_count(self) -> int:
        return 0

    def fit(self, stations: Iterable[StationSeries]) -> None:
        stations = list(stations)
        self._sf_models = []

        for kpi_idx in range(len(KPI_NAMES)):
            rows = []
            for s in stations:
                train_arr = split_station(s.values).train  # shape (TRAIN_H,)
                kpi_series = train_arr[:, kpi_idx]
                n = len(kpi_series)
                rows.append(
                    pd.DataFrame(
                        {
                            "unique_id": s.station_id,
                            "ds": pd.date_range("2000-01-01", periods=n, freq="h"),
                            "y": kpi_series,
                        }
                    )
                )
            long_df = pd.concat(rows, ignore_index=True)
            model_instance = _KIND_MAP[self._kind](self.model_cfg)
            sf = StatsForecast(models=[model_instance], freq="h", n_jobs=1)
            sf.fit(df=long_df)
            self._sf_models.append(sf)

    def predict(
        self, history: np.ndarray, horizon_h: int, station_id: str | None = None
    ) -> np.ndarray:
        if not self._sf_models:
            raise RuntimeError("Call fit() before predict().")

        preds = []
        for kpi_idx, sf in enumerate(self._sf_models):
            kpi_series = history[:, kpi_idx]
            n = len(kpi_series)
            pred_df = pd.DataFrame(
                {
                    "unique_id": "predict",
                    "ds": pd.date_range("2000-01-01", periods=n, freq="h"),
                    "y": kpi_series,
                }
            )
            fcast = sf.forecast(df=pred_df, h=horizon_h)
            # Column name matches the model class name (e.g. "SeasonalNaive", "Naive", ...)
            value_col = [c for c in fcast.columns if c not in ("unique_id", "ds")][0]
            preds.append(fcast[value_col].to_numpy(dtype=np.float64))

        return np.stack(preds, axis=1)  # (horizon_h, 7)

    def metadata(self) -> Dict:
        base = super().metadata()
        base.update(
            {
                "forecast_mode": self.forecast_mode,
                "n_kpis": len(KPI_NAMES),
                "kind": self._kind,
                "window_h": self._window_h,
                "season_h": self._season_h,
            }
        )
        return base
