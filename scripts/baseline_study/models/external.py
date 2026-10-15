"""Environment-gated placeholders for heavyweight pretrained/external runners."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from scripts.baseline_study.data import StationSeries
from scripts.baseline_study.models.base import BaselineModel


class EnvironmentGatedModel(BaselineModel):
    """Record external models in the unified suite without silent omission.

    This wrapper intentionally fails with an explicit message. Heavyweight
    foundation models already have bespoke runners in archived legacy scripts;
    integrating each one into this framework should be done behind this same
    interface so skipped dependencies appear in the leaderboard.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.reason = str(self.model_cfg.get("skip_reason", "external runner not integrated"))

    def fit(self, stations: Iterable[StationSeries]) -> None:
        raise RuntimeError(f"{self.name} skipped: {self.reason}")

    def predict(
        self, history: np.ndarray, horizon_h: int, station_id: str | None = None
    ) -> np.ndarray:
        raise RuntimeError(f"{self.name} skipped: {self.reason}")
