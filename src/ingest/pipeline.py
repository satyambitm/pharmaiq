"""Map, validate and aggregate an uploaded table into a monthly region x product panel.

Rules that keep the output honest:
* Rows with an invalid required or supplied value are rejected (and counted by reason), never repaired.
* Unmapped business variables stay empty; nothing is imputed or invented.
* Within a month, flows (units, revenue, spend, interactions) are summed, state variables
  (market size, competitor count, ...) are averaged, and price is units-weighted revenue / units.
* A monthly sum is left empty if any contributing record lacks the value, so totals are never understated.
* Mixed currencies or units of measure block ingestion: there is no conversion to apply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.ingest.fields import (FIELDS, MEAN_FIELDS, MODEL_FEATURE_FIELDS, MONEY, MONTHLY_COLUMNS, NUMERIC, REQUIRED,
                               SCHEMA_VERSION)

KEY = ["month_start", "region", "product"]
SERIES = ["region", "product"]
MIN_DATE = pd.Timestamp("1990-01-01")
MAX_LABEL_LENGTH = 80
MAX_REJECTION_SHARE = 0.5
FEATURE_MIN_COVERAGE = 0.9
CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
NUMBER = re.compile(r"^[+-]?(\d{1,3}(,\d{3})+(\.\d+)?|\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?$")
ISO_DATE = re.compile(r"^\d{4}-\d{1,2}(-\d{1,2})?([ T].*)?$")
NUMERIC_DATE = re.compile(r"^(\d{1,2})([/.\-])(\d{1,2})\2(\d{4}|\d{2})$")
DATE_FORMAT = re.compile(r"^[%a-zA-Z0-9 \-/:.,_]{2,40}$")
CURRENCY_TOKENS = sorted({"₹": "INR", "RS.": "INR", "RS": "INR", "INR": "INR", "US$": "USD", "$": "USD", "USD": "USD",
                          "€": "EUR", "EUR": "EUR", "£": "GBP", "GBP": "GBP"}.items(), key=lambda kv: -len(kv[0]))
READINESS_LEVELS = [(24, "full"), (12, "short_history"), (6, "baseline_only"), (0, "insufficient")]


class MappingError(ValueError):
    """The column mapping cannot be applied to this file."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class MappingConfig:
    columns: dict[str, str]
    single_product_label: str | None = None
    dayfirst: bool = False
    date_format: str | None = None
    drop_exact_duplicates: bool = False
    currency: str | None = None


@dataclass
class TransformResult:
    monthly: pd.DataFrame
    report: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.report.get("status") == "ready_to_ingest"


# ---------------------------------------------------------------- parsing helpers

def parse_number(value: object) -> tuple[float, str | None, bool]:
    """Return (number, currency code, invalid). Accepts '1,234.5', '₹ 1200', '12 USD'; rejects '1.234,5'."""
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return np.nan, None, False
    text = str(value).replace("\u00a0", " ").strip()
    currency = None
    upper = text.upper()
    for token, code in CURRENCY_TOKENS:
        if upper.startswith(token):
            currency, text = code, text[len(token):].strip()
            break
        if upper.endswith(token):
            currency, text = code, text[: -len(token)].strip()
            break
    if not NUMBER.match(text):
        return np.nan, currency, True
    return float(text.replace(",", "")), currency, False


def parse_numbers(values: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    uniq = {v: parse_number(v) for v in values.dropna().unique()}
    get = lambda i: values.map(lambda v: uniq[v][i] if v in uniq else (np.nan, None, False)[i])  # noqa: E731
    return get(0).astype(float), get(1), get(2).astype(bool)


@dataclass
class DateParse:
    dates: pd.Series
    method: str
    ambiguous: int = 0       # distinct values that are valid both ways but give different dates
    only_this_way: int = 0   # distinct values valid only under the chosen reading (evidence for it)


def _to_datetime(uniq: pd.Series, **kwargs) -> pd.Series:
    parsed = pd.to_datetime(uniq, errors="coerce", **kwargs)
    return parsed.dt.tz_localize(None) if getattr(parsed.dt, "tz", None) is not None else parsed


def parse_dates(values: pd.Series, dayfirst: bool = False, fmt: str | None = None) -> DateParse:
    """Parse each distinct value once. ISO dates (YYYY-MM[-DD]) never depend on the day-first setting."""
    uniq = pd.Series(values.dropna().unique())
    if uniq.empty:
        return DateParse(pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]"), "none")
    if fmt:
        out = DateParse(_to_datetime(uniq, format=fmt), f"format {fmt}")
    elif uniq.astype(str).str.match(ISO_DATE).all():
        out = DateParse(_to_datetime(uniq, format="ISO8601"), "ISO 8601")
    elif (parts := uniq.astype(str).str.extract(NUMERIC_DATE)).notna().all().all() \
            and parts[1].nunique() == 1 and parts[3].str.len().nunique() == 1:
        # dateutil silently swaps day and month when the first number exceeds 12, so numeric dates use
        # strict formats: a value like 25/03/2024 is then evidence for day-first.
        sep, year = parts[1].iloc[0], "%Y" if len(parts[3].iloc[0]) == 4 else "%y"
        day_fmt, month_fmt = f"%d{sep}%m{sep}{year}", f"%m{sep}%d{sep}{year}"
        chosen, other = (day_fmt, month_fmt) if dayfirst else (month_fmt, day_fmt)
        parsed, alt = _to_datetime(uniq, format=chosen), _to_datetime(uniq, format=other)
        out = DateParse(parsed, f"{'day' if dayfirst else 'month'}-first ({chosen})",
                        ambiguous=int((parsed.notna() & alt.notna() & (parsed != alt)).sum()),
                        only_this_way=int((parsed.notna() & alt.isna()).sum()))
    else:
        parsed = _to_datetime(uniq, format="mixed", dayfirst=dayfirst)
        alt = _to_datetime(uniq, format="mixed", dayfirst=not dayfirst)
        out = DateParse(parsed, "day-first" if dayfirst else "month-first",
                        ambiguous=int((parsed.notna() & alt.notna() & (parsed != alt)).sum()),
                        only_this_way=int((parsed.notna() & alt.isna()).sum()))
    out.dates = values.map(dict(zip(uniq, out.dates))).astype("datetime64[ns]")
    return out


def suggest_dayfirst(values: pd.Series) -> bool:
    """Prefer the reading that leaves fewer values unparseable (e.g. '25/03/2024' only works day-first)."""
    sample = values.dropna().drop_duplicates().head(5000)
    return bool(parse_dates(sample, dayfirst=True).dates.notna().sum()
                > parse_dates(sample, dayfirst=False).dates.notna().sum())


def clean_labels(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    text = values.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    invalid = text.notna() & ((text.str.len() > MAX_LABEL_LENGTH) | text.str.contains(CONTROL_CHARS))
    return text.astype(object).where(text.notna(), None), invalid.fillna(False).astype(bool)


def _detect_type(values: pd.Series) -> str:
    sample = values.dropna().head(2000)
    if sample.empty:
        return "empty"
    _, _, bad = parse_numbers(sample)
    if (~bad).mean() >= 0.9:
        return "number"
    dates = parse_dates(sample, dayfirst=suggest_dayfirst(sample)).dates
    if dates.notna().mean() >= 0.9:
        return "date"
    return "text"


# ---------------------------------------------------------------- upload profile

def profile(raw: pd.DataFrame, suggested: dict[str, str], preview_rows: int = 20) -> dict:
    """Column types, a preview and a quick summary based on the *suggested* mapping (before confirmation)."""
    columns = [{
        "name": c,
        "detected_type": _detect_type(raw[c]),
        "missing_pct": round(float(raw[c].isna().mean() * 100), 2),
        "n_unique": int(raw[c].nunique()),
        "examples": [str(v)[:40] for v in raw[c].dropna().unique()[:3]],
    } for c in raw.columns]
    quick: dict = {"based_on": "suggested mapping (confirm it in the next step)", "row_count": int(len(raw))}
    options = {"dayfirst": False}
    if "date" in suggested:
        options["dayfirst"] = suggest_dayfirst(raw[suggested["date"]])
        d = parse_dates(raw[suggested["date"]], dayfirst=options["dayfirst"]).dates
        quick["date_start"] = None if d.isna().all() else str(d.min().date())
        quick["date_end"] = None if d.isna().all() else str(d.max().date())
    for f in ("region", "product"):
        if f in suggested:
            vals = raw[suggested[f]].dropna().astype(str)
            quick[f"n_{f}s"] = int(vals.nunique())
            quick[f"{f}_examples"] = sorted(vals.unique())[:10]
    preview = raw.head(preview_rows).astype(object).where(raw.head(preview_rows).notna(), None)
    return {"columns": columns, "preview": preview.to_dict("records"), "quick_summary": quick,
            "suggested_mapping": suggested, "suggested_options": options}


# ---------------------------------------------------------------- mapping

def check_mapping(cfg: MappingConfig, available: list[str]) -> list[str]:
    errors = []
    unknown = sorted(set(cfg.columns) - set(FIELDS))
    if unknown:
        errors.append(f"Unknown PharmaIQ fields: {', '.join(unknown)}.")
    for f in REQUIRED:
        if not cfg.columns.get(f):
            errors.append(f"'{FIELDS[f].label}' is required; map it to a column.")
    missing = sorted({c for c in cfg.columns.values() if c not in available})
    if missing:
        errors.append(f"Mapped columns not in the file: {', '.join(missing)}.")
    used = [c for c in cfg.columns.values() if c]
    dupes = sorted({c for c in used if used.count(c) > 1})
    if dupes:
        errors.append(f"Each column can be mapped to one field only: {', '.join(dupes)}.")
    if "product" not in cfg.columns:
        label = (cfg.single_product_label or "").strip()
        if not label:
            errors.append("No product column is mapped. Map one, or give an explicit single-product label "
                          "(for example 'All products') to confirm this is a single-product dataset.")
        elif len(label) > MAX_LABEL_LENGTH or CONTROL_CHARS.search(label):
            errors.append("The single-product label must be at most 80 printable characters.")
    if cfg.date_format and not DATE_FORMAT.match(cfg.date_format):
        errors.append("The date format may only contain strftime codes (like %d/%m/%Y) and separators.")
    if cfg.currency and not re.fullmatch(r"[A-Z]{3}", cfg.currency):
        errors.append("The currency must be a 3-letter ISO code such as INR or USD.")
    return errors


# ---------------------------------------------------------------- transform

def _months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    return (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month) + 1


def _detect_frequency(valid: pd.DataFrame) -> str:
    """'monthly' when each series has at most one distinct date per month (repeated rows on that date are
    duplicate keys, reported separately); otherwise daily / weekly / irregular from the median date gap."""
    dates = valid[SERIES + ["date"]].drop_duplicates().sort_values(SERIES + ["date"])
    if not dates.assign(_m=dates["date"].dt.to_period("M")).duplicated(SERIES + ["_m"]).any():
        return "monthly"
    med = float(dates.groupby(SERIES)["date"].diff().dt.days.dropna().median())
    if med <= 1.5:
        return "daily"
    if 6 <= med <= 8:
        return "weekly"
    return "irregular"


def _level(months: float) -> str:
    return next(name for threshold, name in READINESS_LEVELS if months >= threshold)


def transform(raw: pd.DataFrame, cfg: MappingConfig, today: pd.Timestamp | None = None) -> TransformResult:
    """Apply a confirmed mapping. Raises MappingError for an unusable mapping; data problems go in the report."""
    errors = check_mapping(cfg, list(raw.columns))
    if errors:
        raise MappingError(errors)
    today = (today or pd.Timestamp.today()).normalize()
    mapped = {f: c for f, c in cfg.columns.items() if c}
    src = list(mapped.values())
    warnings: list[str] = []
    errors = []

    exact_dupes = raw.duplicated(subset=src, keep="first")
    work = raw.loc[~exact_dupes, src] if cfg.drop_exact_duplicates else raw[src]
    rows = pd.DataFrame(index=work.index)
    reasons: dict[str, pd.Series] = {}

    date_col = work[mapped["date"]]
    parsed_dates = parse_dates(date_col, cfg.dayfirst, cfg.date_format)
    rows["date"], date_method = parsed_dates.dates, parsed_dates.method
    reasons["missing_date"] = date_col.isna()
    reasons["invalid_date"] = date_col.notna() & rows["date"].isna()
    reasons["date_before_1990"] = rows["date"] < MIN_DATE
    reasons["date_in_future"] = rows["date"] > today + pd.Timedelta(days=1)
    if parsed_dates.ambiguous and not parsed_dates.only_this_way:
        warnings.append(f"{parsed_dates.ambiguous} distinct dates read differently day-first vs month-first and nothing "
                        f"in the file settles it; they were read {date_method}. Confirm the day-first setting or give "
                        "an explicit date format.")

    for f in ("region", "product"):
        if f in mapped:
            rows[f], bad = clean_labels(work[mapped[f]])
            reasons[f"missing_{f}"] = rows[f].isna()
            reasons[f"invalid_{f}_label"] = bad
    if "product" not in mapped:
        rows["product"] = " ".join((cfg.single_product_label or "").split())

    detected_currency: list[pd.Series] = []
    for f in [n for n in NUMERIC if n in mapped]:
        values, currency, invalid = parse_numbers(work[mapped[f]])
        rows[f] = values
        if f == "units_sold":
            reasons["missing_units_sold"] = work[mapped[f]].isna()
        reasons[f"non_numeric_{f}"] = invalid
        reasons[f"negative_{f}"] = values < 0
        if f == "unit_price":
            reasons["zero_unit_price"] = values == 0
        if f in MONEY:
            detected_currency.append(currency)
    for f in ("currency", "unit_of_measure"):
        if f in mapped:
            rows[f] = work[mapped[f]].astype("string").str.strip().str.upper()

    reason_frame = pd.DataFrame(reasons).fillna(False).astype(bool)
    rejected = reason_frame.any(axis=1)
    valid = rows[~rejected].copy()
    rejection_counts = {k: int(v.sum()) for k, v in reason_frame.items() if v.any()}
    examples = [{"row": int(i) + 2, "reasons": [k for k, v in reason_frame.loc[i].items() if v]}
                for i in reason_frame.index[rejected][:10]]

    currencies = {c for s in detected_currency for c in s[~rejected].dropna()}
    if "currency" in valid:
        currencies |= set(valid["currency"].dropna())
    if len(currencies) > 1:
        errors.append(f"Money columns mix currencies ({', '.join(sorted(currencies))}). Convert to one currency "
                      "before uploading; PharmaIQ does not invent exchange rates.")
    if cfg.currency and currencies and currencies != {cfg.currency}:
        errors.append(f"The declared currency {cfg.currency} does not match the file ({', '.join(sorted(currencies))}).")
    currency = cfg.currency or (next(iter(currencies)) if len(currencies) == 1 else None)
    if currency is None and any(f in mapped for f in MONEY):
        warnings.append("No currency was declared or detected; money values are stored without a currency code.")
    uom = sorted(set(valid["unit_of_measure"].dropna())) if "unit_of_measure" in valid else []
    if len(uom) > 1:
        errors.append(f"Units of measure are mixed ({', '.join(uom[:5])}). Convert quantities to one unit first.")

    if "revenue" in mapped:
        valid["_revenue"] = valid["revenue"]
        if "unit_price" in mapped:
            implied = valid["units_sold"] * valid["unit_price"]
            mismatch = (implied - valid["revenue"]).abs() > 0.01 * valid["revenue"].abs().clip(lower=1)
            if int(mismatch.sum()):
                warnings.append(f"{int(mismatch.sum())} rows have revenue that differs from units x unit price by more "
                                "than 1%. Revenue is taken from the revenue column; monthly price is revenue / units.")
    elif "unit_price" in mapped:
        valid["_revenue"] = valid["units_sold"] * valid["unit_price"]

    frequency = _detect_frequency(valid) if not valid.empty else "unknown"
    monthly = _aggregate(valid, mapped)
    duplicate_key_rows = int(valid.assign(month_start=valid["date"].dt.to_period("M"))
                             .duplicated(["month_start", *SERIES], keep=False).sum()) if frequency == "monthly" else 0
    if duplicate_key_rows:
        warnings.append(f"{duplicate_key_rows} monthly rows share a region/product/month and were summed. If they are "
                        "repeated entries rather than separate sales, enable 'drop exact duplicates' or fix the source.")
    if exact_dupes.any() and not cfg.drop_exact_duplicates:
        warnings.append(f"{int(exact_dupes.sum())} rows are exact duplicates of an earlier row. They were kept "
                        "(transactional data can repeat); enable 'drop exact duplicates' if they are double entries.")
    if frequency in ("daily", "weekly", "irregular") and not valid.empty:
        first, last = valid["date"].min(), valid["date"].max()
        if first.day > 7:
            warnings.append(f"Data starts on {first.date()}, so {first:%b %Y} may be a partial month.")
        if last < last + pd.offsets.MonthEnd(0) - pd.Timedelta(days=7):
            warnings.append(f"Data ends on {last.date()}, so {last:%b %Y} may be a partial month.")

    report = _report(raw, cfg, mapped, rows, valid, monthly, rejected, rejection_counts, examples, exact_dupes,
                     duplicate_key_rows, frequency, date_method, currency, uom, warnings, errors)
    return TransformResult(monthly=monthly, report=report)


def _aggregate(valid: pd.DataFrame, mapped: dict[str, str]) -> pd.DataFrame:
    if valid.empty:
        return pd.DataFrame(columns=MONTHLY_COLUMNS)
    v = valid.assign(month_start=valid["date"].dt.to_period("M").dt.to_timestamp())
    g = v.groupby(KEY, sort=True)
    out = g.agg(units_sold=("units_sold", "sum"), n_source_rows=("units_sold", "size"))

    def complete_sum(col: str) -> pd.Series:
        total = g[col].sum(min_count=1)
        return total.where(g[col].count() == out["n_source_rows"])

    out["revenue"] = complete_sum("_revenue") if "_revenue" in v else np.nan
    for f in ("marketing_spend", "doctor_interactions"):
        out[f] = complete_sum(f) if f in mapped else np.nan
    for f in MEAN_FIELDS:
        out[f] = g[f].mean() if f in mapped else np.nan
    out["unit_price"] = (out["revenue"] / out["units_sold"]).where(out["units_sold"] > 0)
    return out.reset_index()[MONTHLY_COLUMNS]


def _report(raw, cfg, mapped, rows, valid, monthly, rejected, rejection_counts, examples, exact_dupes,
            duplicate_key_rows, frequency, date_method, currency, uom, warnings, errors) -> dict:
    total = int(len(raw))
    n_rejected = int(rejected.sum())
    if valid.empty:
        errors.append("No valid rows remain after validation; check the column mapping and date settings.")
    elif n_rejected / max(len(rows), 1) > MAX_REJECTION_SHARE:
        errors.append(f"{n_rejected} of {len(rows)} rows were rejected (more than half). The mapping or date "
                      "settings are probably wrong; fix them before ingesting.")

    coverage: dict = {"date_start": None, "date_end": None, "months_spanned": 0, "n_regions": 0, "n_products": 0,
                      "n_series": 0, "regions": [], "products": []}
    readiness: dict = {"level": "insufficient", "series_by_level": {}, "min_months": 0, "median_months": 0,
                       "missing_month_share_pct": None, "notes": []}
    gaps: dict = {"missing_months_total": 0, "series_with_gaps": 0, "examples": []}
    segmentation = {"status": "insufficient", "reason": "No valid monthly data."}
    features: dict = {"available": [], "too_sparse": [], "not_mapped": [f for f in MODEL_FEATURE_FIELDS if f not in mapped]}

    if not monthly.empty:
        spans = monthly.groupby(SERIES)["month_start"].agg(["min", "max", "count"])
        expected = _months_between(spans["min"], spans["max"])
        missing = expected - spans["count"]
        coverage = {
            "date_start": str(valid["date"].min().date()), "date_end": str(valid["date"].max().date()),
            "months_spanned": int(_months_between(pd.Series([monthly["month_start"].min()]),
                                                  pd.Series([monthly["month_start"].max()])).iloc[0]),
            "n_regions": int(monthly["region"].nunique()), "n_products": int(monthly["product"].nunique()),
            "n_series": int(len(spans)),
            "regions": sorted(monthly["region"].unique())[:100], "products": sorted(monthly["product"].unique())[:100],
        }
        gaps = {"missing_months_total": int(missing.sum()), "series_with_gaps": int((missing > 0).sum()),
                "examples": [{"region": r, "product": p, "missing_months": int(m)}
                             for (r, p), m in missing[missing > 0].sort_values(ascending=False).head(10).items()]}
        if gaps["missing_months_total"]:
            warnings.append(f"{gaps['missing_months_total']} region/product months between each series' first and last "
                            "month have no records. They are left missing, not filled with zeros or estimates.")
        levels = spans["count"].map(_level)
        median_months = float(spans["count"].median())
        readiness = {
            "level": _level(median_months),
            "series_by_level": {k: int(v) for k, v in levels.value_counts().items()},
            "min_months": int(spans["count"].min()), "median_months": median_months,
            "missing_month_share_pct": round(float(missing.sum() / expected.sum() * 100), 2),
            "notes": _readiness_notes(levels, mapped),
        }
        if coverage["n_series"] >= 6 and coverage["n_regions"] >= 3:
            segmentation = {"status": "ready", "reason": f"{coverage['n_series']} region/product cells across "
                                                         f"{coverage['n_regions']} regions."}
        else:
            segmentation = {"status": "insufficient",
                            "reason": f"Segmentation needs at least 3 regions and 6 region/product cells; this dataset has "
                                      f"{coverage['n_regions']} regions and {coverage['n_series']} cells."}
        for f in MODEL_FEATURE_FIELDS:
            if f in mapped or (f == "unit_price" and "revenue" in mapped):
                share = float(monthly[f].notna().mean())
                (features["available"] if share >= FEATURE_MIN_COVERAGE else features["too_sparse"]).append(f)
                if f in features["not_mapped"]:
                    features["not_mapped"].remove(f)

    capabilities = {
        "revenue_analysis": {"enabled": "unit_price" in features["available"],
                             "reason": None if "unit_price" in features["available"] else
                             "Needs a unit price or revenue column with at least 90% monthly coverage; prices are never invented."},
        "marketing_analysis": {"enabled": "marketing_spend" in features["available"],
                               "reason": None if "marketing_spend" in features["available"] else
                               "Needs a marketing spend column with at least 90% monthly coverage."},
        "competitor_analysis": {"enabled": bool({"competitor_price", "competitor_count"} & set(features["available"])),
                                "reason": None if {"competitor_price", "competitor_count"} & set(features["available"])
                                else "Needs competitor price or competitor count."},
        "single_product": "product" not in mapped,
    }
    if capabilities["single_product"]:
        warnings.append(f"No product column: every row is labelled '{rows['product'].iloc[0] if len(rows) else ''}' "
                        "(single-product dataset).")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "invalid" if errors else "ready_to_ingest",
        "total_rows": total,
        "exact_duplicate_rows": int(exact_dupes.sum()),
        "dropped_exact_duplicates": int(exact_dupes.sum()) if cfg.drop_exact_duplicates else 0,
        "valid_rows": int(len(valid)),
        "rejected_rows": n_rejected,
        "rejection_reasons": rejection_counts,
        "rejected_examples": examples,
        "missing_value_pct": {f: round(float(raw[c].isna().mean() * 100), 2) for f, c in mapped.items()},
        "duplicate_key_rows": duplicate_key_rows,
        "detected_frequency": frequency,
        "date_parsing": date_method,
        "aggregation": {"source_rows": int(len(valid)), "monthly_rows": int(len(monthly)),
                        "rules": {"sum": ["units_sold", "revenue", "marketing_spend", "doctor_interactions"],
                                  "mean": MEAN_FIELDS, "units_weighted": ["unit_price"]}},
        "currency": currency,
        "unit_of_measure": uom[0] if len(uom) == 1 else None,
        "coverage": coverage,
        "gaps": gaps,
        "forecast_readiness": readiness,
        "segmentation_readiness": segmentation,
        "model_features": features,
        "capabilities": capabilities,
        "warnings": warnings,
        "errors": errors,
    }


def _readiness_notes(levels: pd.Series, mapped: dict[str, str]) -> list[str]:
    notes = []
    if (levels == "insufficient").any():
        notes.append(f"{int((levels == 'insufficient').sum())} series have fewer than 6 months and will be excluded "
                     "from forecasting.")
    if (levels == "baseline_only").any():
        notes.append("Series with 6-11 months support only naive baselines and non-seasonal models.")
    if (levels == "short_history").any():
        notes.append("Series with 12-23 months cannot learn yearly seasonality; seasonal components will be disabled.")
    if "marketing_spend" not in mapped:
        notes.append("No marketing spend: forecasts will be sales-history-only.")
    return notes
