"""Configuration loading helpers for baseline-study entrypoints."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"{path}: top-level YAML must be a mapping")
    return cfg


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def set_by_dotted_key(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        nxt = cursor.setdefault(part, {})
        if not isinstance(nxt, dict):
            raise TypeError(f"Cannot set {dotted_key}: {part} is not a mapping")
        cursor = nxt
    cursor[parts[-1]] = value


def parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Override must be key=value, got: {raw}")
    key, value = raw.split("=", 1)
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        parsed = value
    return key, parsed
