"""Load the deployable model bundle and fail loudly on version mismatches.

Used by the FastAPI backend at startup. Imports only inference dependencies (no Prophet/SHAP).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)
PINNED = ("scikit-learn", "xgboost")


class BundleError(RuntimeError):
    """The model bundle is missing, incomplete or incompatible with installed packages."""


@dataclass
class ModelBundle:
    """Everything the API serves, loaded once at startup."""

    root: Path
    xgb_pipeline: Any
    xgb_meta: dict
    xgb_residuals: dict
    prophet_forecasts: pd.DataFrame
    prophet_meta: dict
    kmeans: dict
    segments: pd.DataFrame
    segment_profiles: pd.DataFrame
    segment_centers: pd.DataFrame
    k_selection: pd.DataFrame
    horizon_metrics: pd.DataFrame
    horizon_region_metrics: pd.DataFrame
    interval_coverage: pd.DataFrame
    one_step_scorecard: pd.DataFrame | None
    shap_global: dict
    shap_local: list
    manifest: dict = field(default_factory=dict)


def _installed(dist: str) -> str | None:
    for name in ([dist, "xgboost-cpu"] if dist == "xgboost" else [dist]):
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


def check_versions(meta: dict) -> None:
    """Saved scikit-learn/XGBoost objects must be loaded by the versions that created them."""
    for pkg in PINNED:
        trained, installed = meta.get("packages", {}).get(pkg), _installed(pkg)
        if trained and installed != trained:
            raise BundleError(f"{pkg} version mismatch: model trained with {trained}, installed {installed}")


def _json(path: Path) -> Any:
    if not path.exists():
        raise BundleError(f"missing bundle file: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path, dates: tuple[str, ...] = ()) -> pd.DataFrame:
    if not path.exists():
        raise BundleError(f"missing bundle file: {path.name}")
    return pd.read_csv(path, parse_dates=list(dates))


def load_bundle(root: Path) -> ModelBundle:
    """Load and validate all artifacts under `root` (the `models/` directory)."""
    if not root.exists():
        raise BundleError(f"models directory not found: {root}")
    xgb_meta = _json(root / "xgboost_metadata.json")
    check_versions(xgb_meta)
    model_path = root / "xgboost.joblib"
    if not model_path.exists():
        raise BundleError("missing bundle file: xgboost.joblib")
    metrics = root / "metrics"
    scorecard = metrics / "one_step_scorecard.csv"
    b = ModelBundle(
        root=root,
        xgb_pipeline=joblib.load(model_path),
        xgb_meta=xgb_meta,
        xgb_residuals=_json(root / "xgboost_residual_quantiles.json"),
        prophet_forecasts=_csv(root / "prophet_forecasts.csv", ("origin_month", "target_month")),
        prophet_meta=_json(root / "prophet_metadata.json"),
        kmeans=joblib.load(root / "kmeans.joblib"),
        segments=_csv(root / "segments.csv"),
        segment_profiles=_csv(root / "segment_profiles.csv"),
        segment_centers=_csv(root / "segment_centers_standardized.csv"),
        k_selection=_csv(root / "k_selection.csv"),
        horizon_metrics=_csv(metrics / "horizon_metrics.csv"),
        horizon_region_metrics=_csv(metrics / "horizon_metrics_by_region.csv"),
        interval_coverage=_csv(metrics / "interval_coverage.csv"),
        one_step_scorecard=pd.read_csv(scorecard) if scorecard.exists() else None,
        shap_global=_json(metrics / "shap_global.json"),
        shap_local=_json(metrics / "shap_local_examples.json"),
        manifest=json.loads((root / "manifest.json").read_text(encoding="utf-8")) if (root / "manifest.json").exists() else {},
    )
    logger.info("Loaded model bundle from %s (xgboost %s, prophet %s)", root, xgb_meta["model_version"], b.prophet_meta["model_version"])
    return b
