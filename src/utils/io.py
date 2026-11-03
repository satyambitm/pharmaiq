"""Small helpers for saving/loading artifacts with clear errors when they are missing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class MissingArtifactError(FileNotFoundError):
    """Raised when an expected generated artifact does not exist yet."""


def require(path: Path, hint: str = "python scripts/train_all.py") -> Path:
    """Return `path` or raise with the command that produces it."""
    if not path.exists():
        raise MissingArtifactError(f"Missing {path.name}. Run `{hint}` first.")
    return path


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def load_json(path: Path, hint: str = "python scripts/train_all.py") -> Any:
    return json.loads(require(path, hint).read_text(encoding="utf-8"))


def save_model(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_model(path: Path, hint: str = "python scripts/train_all.py") -> Any:
    return joblib.load(require(path, hint))


def load_table(path: Path, hint: str = "python scripts/train_all.py") -> pd.DataFrame:
    require(path, hint)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
