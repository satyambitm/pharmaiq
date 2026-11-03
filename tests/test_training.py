"""Training core on uploaded data: models the data cannot support are skipped with a reason, never faked."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest.fields import suggest_mapping
from src.ingest.files import read_upload
from src.ingest.pipeline import MappingConfig, transform
from src.training.forecast import choose_spec, score
from src.training.run import train_dataset

ROOT = Path(__file__).resolve().parents[1]


def _prepare(name: str, **cfg):
    content = (ROOT / "data" / "byod_examples" / name).read_bytes()
    raw = read_upload(content, name, 5 * 1024 * 1024, 100_000).frame
    res = transform(raw, MappingConfig(columns=suggest_mapping(list(raw.columns)), **cfg),
                    today=pd.Timestamp("2026-11-03"))
    assert res.ok
    return res.monthly, res.report


def test_short_units_only_data_gets_naive_baseline_only():
    rows, report = _prepare("region_units_only.csv", single_product_label="Brand X")
    out = train_dataset(rows, report, with_prophet=False)
    fc = out.results["forecasting"]
    assert fc["recommended_model"] == "naive_last_month"
    assert "xgboost" not in fc["evaluation"]["metrics"]
    assert {f["model"] for f in fc["forecasts"]} == {"naive_last_month"}
    assert out.results["explainability"]["status"] == "skipped"
    assert out.results["segmentation"]["status"] == "skipped" and out.results["segmentation"]["reason"]
    assert fc["features"]["drivers"] == []  # no price or marketing columns, so none are invented


def test_training_is_deterministic_for_a_seed():
    rows, report = _prepare("territory_monthly_sales.csv")
    a = train_dataset(rows, report, seed=7, with_prophet=False).results
    b = train_dataset(rows, report, seed=7, with_prophet=False).results
    assert a["forecasting"]["evaluation"] == b["forecasting"]["evaluation"]
    assert a["forecasting"]["forecasts"] == b["forecasting"]["forecasts"]
    assert a["segmentation"]["k"] == b["segmentation"]["k"]
    assert all(f["y_pred"] >= 0 for f in a["forecasting"]["forecasts"])


def test_spec_drops_long_lags_for_short_history():
    assert 12 not in choose_spec(18, []).lags
    assert 12 in choose_spec(30, []).lags


def test_score_matches_definitions():
    y, p = pd.Series([100.0, 200.0]), pd.Series([110.0, 180.0])
    s = score(y, p)
    assert s["MAE"] == 15.0 and np.isclose(s["RMSE"], np.sqrt((100 + 400) / 2), atol=1e-4)
    assert np.isclose(s["WAPE_pct"], 30 / 300 * 100, atol=1e-3)
