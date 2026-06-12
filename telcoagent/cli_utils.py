"""CLI helpers shared by every ``scripts/run_*.py`` runner.

Folds three copy-pasted boilerplate blocks into one call:

    1. Prepend the project root to ``sys.path`` (lets scripts import
       ``telcoagent.*`` without installing the package).
    2. Load ``.env`` from the project root if present.
    3. Configure the root logger with the house format.

Optionally suppress the TensorFlow / TensorRT C++ log spam that leaks
from Chronos-2 / Moirai pipelines.

Usage (drop-in replacement for the previous 15-line header)::

    from telcoagent.cli_utils import bootstrap, ensure_output_dir
    logger = bootstrap(suppress_tf_warnings=True)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# The project root is this file's grandparent: .../telcoagent/cli_utils.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def project_root() -> Path:
    """Return the repo root (``.../telcoagent``)."""
    return _PROJECT_ROOT


def ensure_on_path(root: Optional[Path] = None) -> None:
    """Insert the repo root at the head of ``sys.path`` if missing."""
    root = root or _PROJECT_ROOT
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def load_env(root: Optional[Path] = None) -> None:
    """Load ``.env`` from the repo root (best-effort; no-op if missing)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional in minimal envs
        return
    env_path = (root or _PROJECT_ROOT) / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def setup_logging(
    level: int = logging.INFO,
    name: Optional[str] = None,
    format_: str = _LOG_FORMAT,
) -> logging.Logger:
    """Configure the root logger with the house format and return a child logger."""
    logging.basicConfig(level=level, format=format_)
    return logging.getLogger(name or __name__)


def bootstrap(
    *,
    log_level: int = logging.INFO,
    suppress_tf_warnings: bool = False,
    logger_name: Optional[str] = None,
) -> logging.Logger:
    """One-call setup for a ``scripts/run_*.py`` entry point.

    Equivalent of the old 15-line preamble:
        sys.path insert + load_dotenv + logging.basicConfig + TF env guard.
    """
    if suppress_tf_warnings:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    ensure_on_path()
    load_env()
    return setup_logging(level=log_level, name=logger_name)


def ensure_output_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if missing, return as ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
