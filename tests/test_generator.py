from decimal import Decimal

import pandas as pd

from src.data.generate import generate_clean


def test_generator_seed_repeatability(clean_raw):
    again = generate_clean(42)
    pd.testing.assert_frame_equal(clean_raw, again)
    assert clean_raw.to_csv(index=False) == again.to_csv(index=False)
    assert not clean_raw.equals(generate_clean(7))


def test_unique_panel_keys_and_expected_rows(clean_raw):
    assert len(clean_raw) == 60 * 6 * 3 == 1080
    assert not clean_raw.duplicated(["month_start", "region_id", "product_id"]).any()
    assert clean_raw["month_start"].nunique() == 60
    assert (clean_raw["units_sold"] >= 0).all()
    assert set(clean_raw["data_source"]) == {"synthetic_v1"}


def test_revenue_reconciliation(clean_raw):
    for row in clean_raw.itertuples(index=False):
        gross = Decimal(row.gross_revenue_inr)
        assert gross == row.units_sold * Decimal(row.unit_price_inr)
        assert Decimal(row.net_revenue_inr) == gross - Decimal(row.discount_inr)
        assert Decimal(row.discount_inr) <= gross
