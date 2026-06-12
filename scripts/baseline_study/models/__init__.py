"""Model registry for the dense baseline study.

Factory classes are imported lazily so that a minimal environment only needs
the dependencies of the family it actually runs. This lets the foundation-model
families (chronos/moirai/moment), which pin heavyweight conflicting versions,
run the same ``train.py`` in an isolated env without pulling in
darts/thuml/sklearn, and vice versa.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

from scripts.baseline_study.models.base import BaselineModel

# family -> (module path, class name); imported on first build() of that family
MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    "statistical": ("scripts.baseline_study.models.statsforecast_factory", "StatsForecastModel"),
    "statsforecast": ("scripts.baseline_study.models.statsforecast_factory", "StatsForecastModel"),
    "classical": ("scripts.baseline_study.models.classical", "ClassicalRegressorModel"),
    "darts": ("scripts.baseline_study.models.darts_factory", "DartsForecastModel"),
    "torch": ("scripts.baseline_study.models.darts_factory", "DartsForecastModel"),
    "thuml": ("scripts.baseline_study.models.thuml_factory", "THUMLForecastModel"),
    "external": ("scripts.baseline_study.models.external", "EnvironmentGatedModel"),
    "customdl": ("scripts.baseline_study.models.custom_dl", "CustomDLModel"),
    "chronos": ("scripts.baseline_study.models.foundation", "ChronosModel"),
    "moirai": ("scripts.baseline_study.models.foundation", "MoiraiModel"),
    "moment": ("scripts.baseline_study.models.foundation", "MomentModel"),
}


def build_model(config: dict) -> BaselineModel:
    family = config.get("model", {}).get("family")
    if family not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model.family={family!r}; choices={sorted(MODEL_REGISTRY)}")
    module_path, class_name = MODEL_REGISTRY[family]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)(config)
