from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.schemas.common import Meta
from src.config import REGIONS

REGION_NAMES = set(REGIONS.values())


class OptimizeRequest(BaseModel):
    total_budget_cr: float = Field(..., gt=0, le=1000, description="Total marketing budget in INR crore")
    scenario: Literal["low", "base", "high"] = Field("base", description="Scales every assumed response curve")
    competitor_pressure: dict[str, float] = Field(
        default_factory=dict,
        description="Region name -> fraction in [0, 1) by which that region's assumed response shrinks")
    min_spend_cr: float = Field(5.0, ge=0, le=200, description="Minimum spend per region (INR crore)")
    max_spend_cr: float = Field(30.0, gt=0, le=500, description="Maximum spend per region (INR crore)")

    @model_validator(mode="after")
    def _check_bounds(self) -> "OptimizeRequest":
        if self.min_spend_cr > self.max_spend_cr:
            raise ValueError("min_spend_cr must not exceed max_spend_cr")
        return self

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


class SensitivityPoint(BaseModel):
    budget_cr: float
    optimized_gain_cr: float
    equal_split_gain_cr: float
    equal_split_feasible: bool


class ResponseCurve(BaseModel):
    region: str
    alpha: float
    scale: float
    min_spend: float
    max_spend: float
    status_quo_share: float


class OptimizeResponse(BaseModel):
    meta: Meta
    label: str
    assumptions: list[str]
    total_budget_cr: float
    scenario: str
    feasible_budget_range_cr: tuple[float, float]
    response_curves: list[ResponseCurve]
    strategies: list[Strategy]
    optimized_vs_equal_split_cr: float
    optimized_vs_status_quo_cr: float
    sensitivity: list[SensitivityPoint]
