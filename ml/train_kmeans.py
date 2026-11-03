"""Fit K-Means market-cell segmentation on training-period history and save it for the API.

Usage: python -m ml.train_kmeans
"""

from __future__ import annotations

import logging

import joblib
import pandas as pd

from ml import bundle
from src import config
from src.segmentation.kmeans import SEGMENT_FEATURES, run_segmentation

logger = logging.getLogger("ml.train_kmeans")


def main() -> None:
    config.setup_logging()
    panel = pd.read_parquet(config.PANEL_PARQUET)
    res = run_segmentation(panel)
    joblib.dump({"model": res["model"], "scaler": res["scaler"], "features": SEGMENT_FEATURES,
                 "labels": res["labels"], "cutoff": str(config.SEGMENTATION_CUTOFF.date())}, bundle.KMEANS_MODEL)
    res["segments"].drop(columns=["history_end"]).to_csv(bundle.SEGMENTS, index=False)
    res["profiles"].to_csv(bundle.SEGMENT_PROFILES, index=False)
    res["centers_z"].to_csv(bundle.SEGMENT_CENTERS, index=False)
    res["k_table"].to_csv(bundle.K_SELECTION, index=False)
    k = int(res["k_table"].loc[res["k_table"]["chosen"], "k"].iloc[0])
    logger.info("Saved K-Means with k=%d, labels=%s", k, res["labels"])


if __name__ == "__main__":
    main()
