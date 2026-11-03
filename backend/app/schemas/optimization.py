from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from backend.app.schemas.common import Meta
from src.config import REGIONS

REGION_NAMES = set(REGIONS.values())


class OptimizeRequest(BaseModel):
    total_budget_cr: float = Field(..., gt=0, le=1000, description="Total marketing budget in INR crore")
    scenario: Literal["low", "base", "high"] = Field("base", description="Scales every assumed response curve")
    competitor_pressure: dict[str, float] = Field(
        default_factory=dict,
        description="Region name -> fraction in [0, 1) by which that region's assumed response shrinks")

    @field_validator("competitor_pressure")
    @classmethod
    def _check_pressure(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = set(v) - REGION_NAMES
        if unknown:
            raise ValueError(f"unknown regions: {sorted(unknown)}; expected names from {sorted(REGION_NAMES)}")
        if any(not 0.0 <= p < 1.0 for p in v.values()):
            raise ValueError("competitor pressure values must be in [0, 1)")
        return v


class AllocationRow(BaseModel):
    region: str
    spend_cr: float
    assumed_gain_cr: float
    marginal_gain_per_cr: float
    min_cr: float
    max_cr: float
    within_bounds: bool


class Strategy(BaseModel):
    strategy: str
    total_assumed_gain_cr: float
    feasible: bool
    allocation: list[AllocationRow]


class OptimizeResponse(BaseModel):
    meta: Meta
    label: str
    assumptions: list[str]
    total_budget_cr: float
    scenario: str
    feasible_budget_range_cr: tuple[float, float]
    strategies: list[Strategy]
    optimized_vs_equal_split_cr: float
    optimized_vs_status_quo_cr: float
