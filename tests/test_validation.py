from collections import Counter

from src.data.validate import repair, run_checks


def test_clean_data_passes_all_checks(clean_raw):
    report = run_checks(clean_raw)
    assert report["passed"], report


def test_dirty_data_failures_detected(dirty_raw):
    dirty, log = dirty_raw
    injected = Counter(a["type"] for a in log)
    report = run_checks(dirty)
    assert not report["passed"]
    assert report["duplicate_keys"] == injected["duplicate_key"]
    assert report["missing_panel_keys"] == injected["missing_panel_row"]
    assert report["non_month_start_dates"] == injected["mid_month_date"]
    assert report["missing_values"]["competitor_index"] == injected["missing_competitor_index"]
    assert report["negative_values"]["marketing_spend_inr"] == injected["negative_marketing_spend"]
    assert report["net_mismatch"] >= injected["net_revenue_mismatch"]


def test_repair_fixes_everything_except_missing_rows(dirty_raw):
    dirty, log = dirty_raw
    repaired, _ = repair(dirty)
    report = run_checks(repaired)
    assert report["duplicate_keys"] == 0
    assert report["net_mismatch"] == 0
    assert report["negative_values"] == {}
    assert report["non_month_start_dates"] == 0
    assert report["missing_panel_keys"] == sum(a["type"] == "missing_panel_row" for a in log)
