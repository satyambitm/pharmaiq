from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from backend.app.schemas.common import Meta


class Kpis(BaseModel):
    net_revenue_cr: float = Field(description="Net revenue in INR crore over the selected window")
    units_sold: int
    marketing_spend_cr: float
    avg_discount_pct: float
    revenue_growth_pct_last12_vs_prior12: float | None = Field(
        None, description="Null when the window has fewer than 24 months")


class MonthlyPoint(BaseModel):
    month: date
    net_revenue_cr: float
    units_sold: int
    marketing_spend_cr: float


class ProductShare(BaseModel):
    product_id: str
    product_name: str
    net_revenue_cr: float
    share_pct: float


class OverviewResponse(BaseModel):
    meta: Meta
    window_start: date
    window_end: date
    months: int
    kpis: Kpis
    monthly: list[MonthlyPoint]
    product_mix: list[ProductShare]


class RegionSummary(BaseModel):
    region_id: str
    region_name: str
    net_revenue_cr: float
    units_sold: int
    marketing_spend_cr: float
    spend_to_revenue_pct: float
    avg_discount_pct: float
    revenue_share_pct: float
    growth_pct_last12_vs_prior12: float | None = None


class RegionMonthly(BaseModel):
    region_id: str
    month: date
    net_revenue_cr: float
    units_sold: int


class RegionsResponse(BaseModel):
    meta: Meta
    window_start: date
    window_end: date
    product_ids: list[str]
    regions: list[RegionSummary]
    monthly: list[RegionMonthly]
