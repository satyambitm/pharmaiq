"""Ingestion core: safe file reading, mapping checks, cleaning rules, aggregation and the quality report."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.ingest.fields import suggest_mapping
from src.ingest.files import FileRejected, read_upload
from src.ingest.pipeline import MappingConfig, MappingError, parse_number, parse_dates, transform

MB = 1024 * 1024
TODAY = pd.Timestamp("2026-11-03")


def _read(text: str, name: str = "data.csv", max_rows: int = 1000):
    return read_upload(text.encode(), name, 5 * MB, max_rows)


def _run(text: str, today=TODAY, **cfg):
    columns = cfg.pop("columns", None)
    raw = _read(text).frame
    return transform(raw, MappingConfig(columns=columns or suggest_mapping(list(raw.columns)), **cfg), today=today)


def _xlsx(frame: pd.DataFrame, extra: dict[str, bytes] | None = None) -> bytes:
    buf = io.BytesIO()
    frame.to_excel(buf, index=False)
    if not extra:
        return buf.getvalue()
    out = io.BytesIO()
    with zipfile.ZipFile(buf) as src, zipfile.ZipFile(out, "w") as dst:
        for item in src.infolist():
            dst.writestr(item, src.read(item.filename))
        for name, data in extra.items():
            dst.writestr(name, data)
    return out.getvalue()


def _monthly_csv(months=24, regions=("North", "South", "East"), products=("A", "B")) -> str:
    lines = ["month,region,product,units,price"]
    for m in pd.period_range("2023-01", periods=months, freq="M"):
        for r in regions:
            for p in products:
                lines.append(f"{m},{r},{p},100,10")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- file safety

class TestFiles:
    def test_csv_with_semicolons_and_bom(self):
        parsed = read_upload("\ufeffmonth;region;units\n2024-01;North;5\n".encode(), "x.csv", MB, 10)
        assert list(parsed.frame.columns) == ["month", "region", "units"] and parsed.delimiter == ";"

    def test_cp1252_fallback_warns(self):
        parsed = read_upload("month,region,units\n2024-01,Côte,5\n".encode("cp1252"), "x.csv", MB, 10)
        assert parsed.frame.loc[0, "region"] == "Côte" and parsed.warnings

    @pytest.mark.parametrize("name", ["data.txt", "data.json", "data", "data.csv.exe"])
    def test_unsupported_extension(self, name):
        with pytest.raises(FileRejected) as exc:
            read_upload(b"a,b\n1,2\n", name, MB, 10)
        assert exc.value.unsupported_type

    def test_binary_content_renamed_to_csv(self):
        with pytest.raises(FileRejected):
            read_upload(b"MZ\x90\x00\x03\x00\x00\x00\x04\x00", "sales.csv", MB, 10)

    def test_zip_renamed_to_csv(self):
        with pytest.raises(FileRejected, match="content is XLSX"):
            read_upload(_xlsx(pd.DataFrame({"a": [1], "b": [2]})), "sales.csv", MB, 10)

    def test_csv_renamed_to_xlsx(self):
        with pytest.raises(FileRejected, match="content is CSV"):
            read_upload(b"a,b\n1,2\n", "sales.xlsx", MB, 10)

    def test_legacy_xls_rejected(self):
        with pytest.raises(FileRejected) as exc:
            read_upload(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 100, "sales.xls", MB, 10)
        assert exc.value.unsupported_type

    def test_xlsx_roundtrip(self):
        frame = pd.DataFrame({"month": ["2024-01", "2024-02"], "region": ["N", "S"], "units": [1, 2]})
        parsed = read_upload(_xlsx(frame), "sales.xlsx", MB, 10)
        assert parsed.file_type == "xlsx" and parsed.frame["units"].tolist() == ["1", "2"]

    def test_xlsx_with_macros_rejected(self):
        content = _xlsx(pd.DataFrame({"a": [1], "b": [2]}), {"xl/vbaProject.bin": b"\x00" * 10})
        with pytest.raises(FileRejected, match="[Mm]acro"):
            read_upload(content, "sales.xlsx", MB, 10)

    def test_arbitrary_zip_named_xlsx_rejected(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("readme.txt", "hello")
        with pytest.raises(FileRejected):
            read_upload(buf.getvalue(), "sales.xlsx", MB, 10)

    def test_zip_bomb_ratio_rejected(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", "<Types/>")
            z.writestr("xl/workbook.xml", "<workbook/>")
            z.writestr("xl/worksheets/sheet1.xml", "0" * (20 * MB))
        with pytest.raises(FileRejected):
            read_upload(buf.getvalue(), "sales.xlsx", 5 * MB, 10)

    def test_size_and_row_limits(self):
        with pytest.raises(FileRejected, match="MB"):
            read_upload(b"a,b\n" + b"1,2\n" * 1000, "x.csv", 100, 10_000)
        with pytest.raises(FileRejected, match="rows"):
            read_upload(b"a,b\n" + b"1,2\n" * 11, "x.csv", MB, 10)
        with pytest.raises(FileRejected):
            read_upload(b"", "x.csv", MB, 10)

    @pytest.mark.parametrize("text", ["a,a\n1,2\n", "a,,c\n1,2,3\n", "onlyone\n1\n", "a,b\n1,2,3,4\n"])
    def test_malformed_headers_or_rows(self, text):
        with pytest.raises(FileRejected):
            read_upload(text.encode(), "x.csv", MB, 10)

    def test_blank_rows_dropped_but_row_numbers_kept(self):
        parsed = _read("month,region,units\n2024-01,N,1\n,,\n2024-02,N,x\n")
        assert len(parsed.frame) == 2 and list(parsed.frame.index) == [0, 2]


# --------------------------------------------------------------------------- parsing helpers

@pytest.mark.parametrize("raw, value, currency", [
    ("1,234.50", 1234.5, None), ("₹ 2,000", 2000, "INR"), ("$12", 12, "USD"), ("-3", -3, None), ("1e3", 1000, None),
])
def test_parse_number(raw, value, currency):
    got, cur, invalid = parse_number(raw)
    assert not invalid and got == pytest.approx(value) and cur == currency


@pytest.mark.parametrize("raw", ["1.234,5", "12abc", "1,23", "--1", "N/A"])
def test_parse_number_rejects_ambiguous_or_text(raw):
    assert parse_number(raw)[2] is True


def test_day_first_settled_by_evidence():
    parsed = parse_dates(pd.Series(["03/02/2024", "25/02/2024"]), dayfirst=True)
    assert parsed.dates.tolist() == [pd.Timestamp("2024-02-03"), pd.Timestamp("2024-02-25")]
    assert parse_dates(pd.Series(["2024-13", "Jan 2024"])).dates.isna().tolist() == [True, False]


# --------------------------------------------------------------------------- mapping

class TestMapping:
    def test_missing_required_fields(self):
        with pytest.raises(MappingError) as exc:
            _run("month,region,product\n2024-01,N,A\n", columns={"date": "month", "region": "region"})
        assert any("Units sold" in e for e in exc.value.errors)

    def test_column_not_in_file_and_duplicate_use(self):
        with pytest.raises(MappingError) as exc:
            _run("month,region,units\n2024-01,N,1\n",
                 columns={"date": "month", "region": "units", "units_sold": "units", "product": "missing"})
        joined = " ".join(exc.value.errors)
        assert "not in the file" in joined and "one field only" in joined

    def test_product_required_unless_single_product_label(self):
        with pytest.raises(MappingError, match="single-product"):
            _run("month,region,units\n2024-01,N,1\n", columns={"date": "month", "region": "region", "units_sold": "units"})
        res = _run("month,region,units\n2024-01,N,1\n", columns={"date": "month", "region": "region",
                                                                  "units_sold": "units"}, single_product_label="Brand X")
        assert res.monthly["product"].tolist() == ["Brand X"] and res.report["capabilities"]["single_product"]

    def test_suggest_mapping_synonyms(self):
        m = suggest_mapping(["Transaction Month", "Territory", "Drug Name", "Qty", "Net Amount", "Promotion Cost"])
        assert m == {"date": "Transaction Month", "region": "Territory", "product": "Drug Name", "units_sold": "Qty",
                     "revenue": "Net Amount", "marketing_spend": "Promotion Cost"}


# --------------------------------------------------------------------------- cleaning and report

class TestTransform:
    def test_rejections_are_counted_with_row_numbers(self):
        text = ("month,region,product,units,price\n2024-01,N,A,10,5\n2024-13,N,A,10,5\n2024-02,N,A,-1,5\n"
                "2024-03,N,A,abc,5\n2024-04,,A,1,5\n2024-05,N,A,1,0\n2031-01,N,A,1,5\n1980-01,N,A,1,5\n")
        rep = _run(text).report
        assert rep["valid_rows"] == 1 and rep["status"] == "invalid"  # > 50 % rejected is blocking
        assert rep["rejection_reasons"] == {"invalid_date": 1, "negative_units_sold": 1, "non_numeric_units_sold": 1,
                                            "missing_region": 1, "zero_unit_price": 1, "date_in_future": 1,
                                            "date_before_1990": 1}
        assert {"row": 3, "reasons": ["invalid_date"]} in rep["rejected_examples"]  # header is row 1

    def test_exact_duplicates_counted_and_optionally_dropped(self):
        text = _monthly_csv(months=3, regions=("N",), products=("A",)) + "2023-01,N,A,100,10\n"
        kept = _run(text)
        assert kept.report["exact_duplicate_rows"] == 1 and kept.monthly.loc[0, "units_sold"] == 200
        dropped = _run(text, drop_exact_duplicates=True)
        assert dropped.report["dropped_exact_duplicates"] == 1 and dropped.monthly.loc[0, "units_sold"] == 100

    def test_daily_rows_aggregate_to_weighted_monthly_price(self):
        text = "date,region,product,units,price\n2024-01-05,N,A,10,2\n2024-01-06,N,A,30,4\n2024-02-03,N,A,5,3\n"
        res = _run(text)
        jan = res.monthly.iloc[0]
        assert res.report["detected_frequency"] != "monthly"
        assert jan["units_sold"] == 40 and jan["revenue"] == 140 and jan["unit_price"] == pytest.approx(3.5)
        assert jan["n_source_rows"] == 2

    def test_daily_detection(self):
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        text = "date,region,product,units\n" + "".join(f"{d.date()},N,A,1\n" for d in dates)
        res = _run(text)
        assert res.report["detected_frequency"] == "daily" and res.monthly["units_sold"].tolist() == [31, 29]

    def test_weekly_detection(self):
        dates = pd.date_range("2024-01-01", periods=12, freq="W-MON")
        text = "date,region,product,units\n" + "".join(f"{d.date()},N,A,1\n" for d in dates)
        assert _run(text).report["detected_frequency"] == "weekly"

    def test_missing_price_disables_revenue_analysis_and_is_never_invented(self):
        res = _run("month,region,product,units\n2024-01,N,A,10\n2024-02,N,A,12\n")
        assert res.monthly["unit_price"].isna().all() and res.monthly["revenue"].isna().all()
        caps = res.report["capabilities"]
        assert not caps["revenue_analysis"]["enabled"] and "never invented" in caps["revenue_analysis"]["reason"]
        assert not caps["marketing_analysis"]["enabled"]

    def test_partial_marketing_months_are_left_missing(self):
        text = "date,region,product,units,promo\n2024-01-05,N,A,1,100\n2024-01-20,N,A,1,\n2024-02-05,N,A,1,50\n"
        res = _run(text, columns={"date": "date", "region": "region", "product": "product", "units_sold": "units",
                                  "marketing_spend": "promo"})
        assert np.isnan(res.monthly.loc[0, "marketing_spend"]) and res.monthly.loc[1, "marketing_spend"] == 50

    def test_mixed_currency_blocks(self):
        res = _run("month,region,product,units,price\n2024-01,N,A,1,₹5\n2024-02,N,A,1,$5\n")
        assert res.report["status"] == "invalid" and any("mix currencies" in e for e in res.report["errors"])

    def test_declared_currency_must_match(self):
        res = _run("month,region,product,units,price\n2024-01,N,A,1,$5\n", currency="INR")
        assert res.report["status"] == "invalid"

    def test_gaps_reported_not_filled(self):
        text = "month,region,product,units\n2024-01,N,A,1\n2024-04,N,A,1\n"
        res = _run(text)
        assert len(res.monthly) == 2 and res.report["gaps"]["missing_months_total"] == 2
        assert any("not filled" in w for w in res.report["warnings"])

    def test_readiness_levels(self):
        assert _run(_monthly_csv(months=24)).report["forecast_readiness"]["level"] == "full"
        assert _run(_monthly_csv(months=13)).report["forecast_readiness"]["level"] == "short_history"
        assert _run(_monthly_csv(months=4)).report["forecast_readiness"]["level"] == "insufficient"
        assert _run(_monthly_csv(months=24)).report["segmentation_readiness"]["status"] == "ready"
        small = _run(_monthly_csv(months=24, regions=("N",)))
        assert small.report["segmentation_readiness"]["status"] == "insufficient"

    def test_mapping_does_not_modify_source_frame(self):
        raw = _read("month,region,product,units\n2024-01, N ,A,1\n").frame
        before = raw.copy()
        transform(raw, MappingConfig(columns=suggest_mapping(list(raw.columns))), today=TODAY)
        pd.testing.assert_frame_equal(raw, before)
