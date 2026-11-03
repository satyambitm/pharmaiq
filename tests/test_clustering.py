import pandas as pd

from src import config
from src.segmentation.kmeans import SEGMENT_FEATURES, market_cell_features, run_segmentation


def test_kmeans_only_train_period(panel):
    base = market_cell_features(panel)
    shocked = panel.copy()
    shocked.loc[shocked["month_start"] > config.SEGMENTATION_CUTOFF, "units_sold"] *= 50
    after = market_cell_features(shocked)
    pd.testing.assert_frame_equal(base, after)
    assert (base["history_end"] <= config.SEGMENTATION_CUTOFF).all()


def test_kmeans_clusters_valid(panel):
    res = run_segmentation(panel)
    seg = res["segments"]
    k = int(res["k_table"].loc[res["k_table"]["chosen"], "k"].iloc[0])
    assert len(seg) == 18
    assert seg["cluster"].nunique() == k and 2 <= k <= 4
    assert seg.groupby("cluster").size().min() >= 1
    assert not seg[SEGMENT_FEATURES].isna().any().any()
    assert not seg["segment_label"].str.contains("ROI", case=False).any()
