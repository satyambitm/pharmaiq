"""Central configuration: paths, seed, dataset dimensions and evaluation cutoffs.

Values can be overridden with environment variables (see `.env.example`).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://pharmaiq_user:pharmaiq_local@localhost:5432/pharmaiq"
)

DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = PROJECT_ROOT / os.getenv("ARTIFACT_DIR", "artifacts")
MODELS_DIR = ARTIFACT_DIR / "models"
METRICS_DIR = ARTIFACT_DIR / "metrics"
FORECASTS_DIR = ARTIFACT_DIR / "forecasts"
CLUSTERS_DIR = ARTIFACT_DIR / "clusters"
FIGURES_DIR = ARTIFACT_DIR / "figures"
REPORTS_DIR = ARTIFACT_DIR / "reports"
SQL_DIR = PROJECT_ROOT / "sql"

CLEAN_CSV = RAW_DIR / "synthetic_sales_clean.csv"
DIRTY_CSV = RAW_DIR / "synthetic_sales_dirty.csv"
METADATA_JSON = RAW_DIR / "generation_metadata.json"
PANEL_PARQUET = PROCESSED_DIR / "monthly_panel.parquet"
REPAIRED_DIRTY_CSV = PROCESSED_DIR / "dirty_repaired.csv"
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"

DATA_SOURCE = "synthetic_v1"
EQUATION_VERSION = "gen-eq-1.0"

START_MONTH = pd.Timestamp("2021-01-01")
N_MONTHS = 60

REGIONS: dict[str, str] = {
    "R01": "North",
    "R02": "South",
    "R03": "East",
    "R04": "West",
    "R05": "Central",
    "R06": "Northeast",
}
PRODUCTS: dict[str, str] = {
    "P01": "Therapy A",
    "P02": "Therapy B",
    "P03": "Therapy C",
}
THERAPEUTIC_GROUP: dict[str, str] = {
    "P01": "Group 1 (non-specific)",
    "P02": "Group 2 (non-specific)",
    "P03": "Group 3 (non-specific)",
}

# Forecast evaluation contract (target-month boundaries, inclusive).
TRAIN_END = pd.Timestamp("2023-12-01")
VALID_START = pd.Timestamp("2024-01-01")
VALID_END = pd.Timestamp("2024-12-01")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-12-01")

# Expanding-window backtest folds inside 2022-2024 (never touch 2025).
BACKTEST_FOLDS: list[dict[str, pd.Timestamp]] = [
    {"train_end": pd.Timestamp("2023-06-01"), "val_start": pd.Timestamp("2023-07-01"), "val_end": pd.Timestamp("2023-12-01")},
    {"train_end": pd.Timestamp("2023-12-01"), "val_start": pd.Timestamp("2024-01-01"), "val_end": pd.Timestamp("2024-06-01")},
    {"train_end": pd.Timestamp("2024-06-01"), "val_start": pd.Timestamp("2024-07-01"), "val_end": pd.Timestamp("2024-12-01")},
]

# K-Means descriptors use history up to and including this month only.
SEGMENTATION_CUTOFF = TRAIN_END


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a single consistent log format for CLI entry points."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_dirs() -> None:
    """Create generated-output directories if missing."""
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, METRICS_DIR, FORECASTS_DIR,
              CLUSTERS_DIR, FIGURES_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
