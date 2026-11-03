from __future__ import annotations

from pydantic import BaseModel

from backend.app.schemas.common import Meta


class SegmentCell(BaseModel):
    region_id: str
    region_name: str
    product_id: str
    product_name: str
    cluster: int
    segment_label: str
    avg_monthly_units: float
    cv_units: float
    yoy_growth_pct: float
    avg_discount_pct: float
    spend_per_unit_inr: float
    pc1: float
    pc2: float


class SegmentProfile(BaseModel):
    cluster: int
    segment_label: str
    n_cells: int
    avg_monthly_units: float
    cv_units: float
    yoy_growth_pct: float
    avg_discount_pct: float
    spend_per_unit_inr: float
    seasonality_amplitude: float
    campaign_frequency: float


class KSelectionRow(BaseModel):
    k: int
    silhouette: float
    inertia: float
    min_cluster_size: int
    chosen: bool


class SegmentsResponse(BaseModel):
    meta: Meta
    k: int
    features: list[str]
    history_cutoff: str
    notes: list[str]
    cells: list[SegmentCell]
    profiles: list[SegmentProfile]
    k_selection: list[KSelectionRow]
