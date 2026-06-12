"""Baseline model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable

import numpy as np

from scripts.baseline_study.data import StationSeries


class BaselineModel(ABC):
    def __init__(self, config: dict):
        self.config = config
        self.model_cfg = config.get("model", {})
        self.name = str(self.model_cfg.get("name", self.__class__.__name__))
        self.family = str(self.model_cfg.get("family", "unknown"))
        self.forecast_mode = str(self.model_cfg.get("forecast_mode", "unspecified"))

    @property
    def parameter_count(self) -> int:
        return 0

    @abstractmethod
    def fit(self, stations: Iterable[StationSeries]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self, history: np.ndarray, horizon_h: int, station_id: str | None = None
    ) -> np.ndarray:
        raise NotImplementedError

    def metadata(self) -> Dict:
        return {
            "model_family": self.family,
            "model_name": self.name,
            "forecast_mode": self.forecast_mode,
            "model_params": self.parameter_count,
        }
