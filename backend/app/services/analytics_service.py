"""Descriptive KPIs over the stored monthly sales (no modeling)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from backend.app.core.errors import InvalidRequest
from backend.app.db.connection import SalesRepository
from src.config import PRODUCTS, REGIONS

CR = 1e7


def _growth_last12(monthly_rev: pd.Series) -> float | None:
    """Revenue growth of the last 12 months vs the 12 before; None if fewer than 24 months."""
    if len(monthly_rev) < 24:
        return None
    last, prior = monthly_rev.iloc[-12:].sum(), monthly_rev.iloc[-24:-12].sum()
    return round(float((last / prior - 1) * 100), 2) if prior > 0 else None


def _load(repo: SalesRepository, start: date | None, end: date | None, product_ids: list[str] | None) -> pd.DataFrame:
    if start and end and start > end:
        raise InvalidRequest("start_month must be on or before end_month")
    df = repo.fetch_sales(start=start, end=end, product_ids=product_ids)
    if df.empty:
        raise InvalidRequest("No data in the selected window")
    return df


def overview(repo: SalesRepository, start: date | None = None, end: date | None = None) -> dict:
    df = _load(repo, start, end, None)
    m = df.groupby("month_start").agg(net=("net_revenue_inr", "sum"), units=("units_sold", "sum"),
                                      spend=("marketing_spend_inr", "sum")).sort_index()
    mix = df.groupby("product_id")["net_revenue_inr"].sum()
    total_net = float(df["net_revenue_inr"].sum())
    return {
        "window_start": m.index.min().date(), "window_end": m.index.max().date(), "months": len(m),
        "kpis": {
            "net_revenue_cr": round(total_net / CR, 2),
            "units_sold": int(df["units_sold"].sum()),
            "marketing_spend_cr": round(float(df["marketing_spend_inr"].sum()) / CR, 2),
            "avg_discount_pct": round(float(df["discount_inr"].sum() / df["gross_revenue_inr"].sum() * 100), 2),
            "revenue_growth_pct_last12_vs_prior12": _growth_last12(m["net"]),
        },
        "monthly": [{"month": i.date(), "net_revenue_cr": round(r.net / CR, 4), "units_sold": int(r.units),
                     "marketing_spend_cr": round(r.spend / CR, 4)} for i, r in m.iterrows()],
        "product_mix": [{"product_id": p, "product_name": PRODUCTS[p], "net_revenue_cr": round(float(v) / CR, 2),
                         "share_pct": round(float(v) / total_net * 100, 2)} for p, v in mix.items()],
    }


def regions(repo: SalesRepository, start: date | None = None, end: date | None = None,
            product_ids: list[str] | None = None) -> dict:
    df = _load(repo, start, end, product_ids)
    total_net = float(df["net_revenue_inr"].sum())
    rows = []
    for rid, g in df.groupby("region_id"):
        net, spend = float(g["net_revenue_inr"].sum()), float(g["marketing_spend_inr"].sum())
        rows.append({
            "region_id": rid, "region_name": REGIONS[rid],
            "net_revenue_cr": round(net / CR, 2), "units_sold": int(g["units_sold"].sum()),
            "marketing_spend_cr": round(spend / CR, 2),
            "spend_to_revenue_pct": round(spend / net * 100, 2) if net > 0 else 0.0,
            "avg_discount_pct": round(float(g["discount_inr"].sum() / g["gross_revenue_inr"].sum() * 100), 2),
            "revenue_share_pct": round(net / total_net * 100, 2),
            "growth_pct_last12_vs_prior12": _growth_last12(g.groupby("month_start")["net_revenue_inr"].sum().sort_index()),
        })
    monthly = df.groupby(["region_id", "month_start"]).agg(net=("net_revenue_inr", "sum"), units=("units_sold", "sum")).reset_index()
    return {
        "window_start": df["month_start"].min().date(), "window_end": df["month_start"].max().date(),
        "product_ids": sorted(df["product_id"].unique()),
        "regions": sorted(rows, key=lambda r: -r["net_revenue_cr"]),
        "monthly": [{"region_id": r.region_id, "month": r.month_start.date(), "net_revenue_cr": round(r.net / CR, 4),
                     "units_sold": int(r.units)} for r in monthly.itertuples()],
    }
