"""Paths and helpers for the deployable model bundle in `models/` (tracked in Git)."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from src import config

MODELS = config.PROJECT_ROOT / "models"
METRICS = MODELS / "metrics"

XGB_MODEL = MODELS / "xgboost.joblib"
XGB_META = MODELS / "xgboost_metadata.json"
XGB_RESIDUALS = MODELS / "xgboost_residual_quantiles.json"
PROPHET_FORECASTS = MODELS / "prophet_forecasts.csv"
PROPHET_META = MODELS / "prophet_metadata.json"
KMEANS_MODEL = MODELS / "kmeans.joblib"
SEGMENTS = MODELS / "segments.csv"
SEGMENT_PROFILES = MODELS / "segment_profiles.csv"
SEGMENT_CENTERS = MODELS / "segment_centers_standardized.csv"
K_SELECTION = MODELS / "k_selection.csv"
HORIZON_METRICS = METRICS / "horizon_metrics.csv"
HORIZON_REGION_METRICS = METRICS / "horizon_metrics_by_region.csv"
INTERVAL_COVERAGE = METRICS / "interval_coverage.csv"
ONE_STEP_SCORECARD = METRICS / "one_step_scorecard.csv"
SHAP_GLOBAL = METRICS / "shap_global.json"
SHAP_LOCAL = METRICS / "shap_local_examples.json"
MANIFEST = MODELS / "manifest.json"

RUNTIME_PACKAGES = ["numpy", "pandas", "scikit-learn", "xgboost", "scipy", "joblib"]


def package_versions(names: list[str] | None = None) -> dict[str, str]:
    """Installed versions; `xgboost-cpu` is reported as xgboost when that is what is installed."""
    out = {"python": platform.python_version()}
    for name in names or RUNTIME_PACKAGES + ["prophet", "shap"]:
        for dist in ([name, "xgboost-cpu"] if name == "xgboost" else [name]):
            try:
                out[name] = importlib_metadata.version(dist)
                break
            except importlib_metadata.PackageNotFoundError:
                out[name] = "not installed"
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_manifest() -> dict:
    """Record every bundle file with size and hash, plus the package versions that produced it."""
    files = sorted(p for p in MODELS.rglob("*") if p.is_file() and p.name not in {"manifest.json", "README.md"})
    manifest = {
        "created_at_utc": now_utc(),
        "data_version": config.DATA_SOURCE,
        "seed": config.RANDOM_SEED,
        "packages": package_versions(),
        "files": [{"path": p.relative_to(MODELS).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files],
        "total_bytes": sum(p.stat().st_size for p in files),
    }
    write_json(manifest, MANIFEST)
    return manifest
