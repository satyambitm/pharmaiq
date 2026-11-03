"""Canonical PharmaIQ fields an uploaded file can be mapped onto, and how each is aggregated to a month."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

SCHEMA_VERSION = "byod-1"

Kind = Literal["date", "label", "number", "code"]
# sum:   flows that add up within a month (units, revenue, spend, interactions)
# mean:  state variables repeated on every record (market size, competitor count)
# price: derived as revenue / units (a units-weighted price), never summed
Aggregation = Literal["key", "sum", "mean", "price", "check"]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    kind: Kind
    aggregation: Aggregation
    required: bool = False
    money: bool = False
    description: str = ""


FIELDS: dict[str, FieldSpec] = {f.name: f for f in [
    FieldSpec("date", "Date", "date", "key", required=True,
              description="Transaction or period date. Daily and weekly records are aggregated to calendar months."),
    FieldSpec("region", "Region", "label", "key", required=True, description="Region, territory, zone or city."),
    FieldSpec("units_sold", "Units sold", "number", "sum", required=True, description="Quantity sold (non-negative)."),
    FieldSpec("product", "Product", "label", "key",
              description="Product, brand or SKU. If absent, give an explicit single-product label."),
    FieldSpec("unit_price", "Unit price", "number", "price", money=True,
              description="Selling price per unit. Monthly price is units-weighted (revenue / units)."),
    FieldSpec("revenue", "Revenue", "number", "sum", money=True,
              description="Sales value. If absent but unit price is given, revenue = units x price per record."),
    FieldSpec("marketing_spend", "Marketing spend", "number", "sum", money=True,
              description="Promotion or marketing cost (non-negative)."),
    FieldSpec("doctor_interactions", "Doctor interactions", "number", "sum",
              description="Detailing calls, rep visits or other HCP interactions (count)."),
    FieldSpec("competitor_price", "Competitor price", "number", "mean", money=True,
              description="Competitor selling price; averaged within a month."),
    FieldSpec("competitor_count", "Competitor count", "number", "mean",
              description="Number of competing products; averaged within a month."),
    FieldSpec("market_size", "Market size", "number", "mean",
              description="Market size or potential; averaged within a month."),
    FieldSpec("hcp_count", "Healthcare professional count", "number", "mean",
              description="Number of HCPs in the region; averaged within a month."),
    FieldSpec("currency", "Currency code", "code", "check",
              description="Currency of money columns. Mixed currencies are rejected (no conversion is invented)."),
    FieldSpec("unit_of_measure", "Unit of measure", "code", "check",
              description="Pack or unit type. Mixed units are rejected."),
]}

REQUIRED = [f.name for f in FIELDS.values() if f.required]
NUMERIC = [f.name for f in FIELDS.values() if f.kind == "number"]
MONEY = [f.name for f in FIELDS.values() if f.money]
SUM_FIELDS = [f.name for f in FIELDS.values() if f.aggregation == "sum" and f.name != "units_sold"]
MEAN_FIELDS = [f.name for f in FIELDS.values() if f.aggregation == "mean"]
MODEL_FEATURE_FIELDS = ["unit_price", "marketing_spend", "doctor_interactions", "competitor_price",
                        "competitor_count", "market_size", "hcp_count"]
MONTHLY_COLUMNS = ["month_start", "region", "product", "units_sold", "revenue", "unit_price", *MODEL_FEATURE_FIELDS[1:],
                   "n_source_rows"]

SYNONYMS: dict[str, list[str]] = {
    "date": ["date", "month", "month_start", "period", "transaction_month", "transaction_date", "sale_date",
             "invoice_date", "order_date", "sales_month", "week", "day"],
    "region": ["region", "territory", "zone", "area", "region_name", "state", "city", "district", "market"],
    "product": ["product", "product_name", "drug", "drug_name", "brand", "sku", "molecule", "item"],
    "units_sold": ["units_sold", "units", "quantity", "qty", "sales_quantity", "volume", "units_sold_qty", "packs"],
    "unit_price": ["unit_price", "price", "selling_price", "avg_price", "asp", "price_per_unit"],
    "revenue": ["revenue", "sales_value", "net_sales", "net_amount", "sales_amount", "gross_sales", "amount", "value",
                "turnover"],
    "marketing_spend": ["marketing_spend", "promotion_cost", "promo_spend", "marketing_cost", "promotion_spend",
                        "marketing", "ad_spend"],
    "doctor_interactions": ["doctor_interactions", "hcp_calls", "detailing_calls", "rep_visits", "calls", "visits"],
    "competitor_price": ["competitor_price", "rival_price", "competition_price"],
    "competitor_count": ["competitor_count", "n_competitors", "competitors", "rival_count"],
    "market_size": ["market_size", "market_potential", "tam", "market_value"],
    "hcp_count": ["hcp_count", "doctor_count", "physicians", "prescribers", "n_doctors", "doctors"],
    "currency": ["currency", "currency_code", "ccy"],
    "unit_of_measure": ["unit_of_measure", "uom", "pack_type", "unit_type"],
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def suggest_mapping(columns: list[str]) -> dict[str, str]:
    """Best-effort mapping by column name. Only a suggestion: the user confirms every field."""
    normalized = {_norm(c): c for c in columns}
    used: set[str] = set()
    out: dict[str, str] = {}
    for field, names in SYNONYMS.items():
        for n in names:
            col = normalized.get(n)
            if col is not None and col not in used:
                out[field] = col
                used.add(col)
                break
    return out
