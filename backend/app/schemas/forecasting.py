from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from backend.app.schemas.common import Meta


class ForecastPoint(BaseModel):
    product_id: str
    product_name: str
    target_month: date
    step: int
    predicted_units: float
    lower_units: float | None = None
    upper_units: float | None = None


class HistoryPoint(BaseModel):
    product_id: str
    month: date
    units_sold: int


class Interval(BaseModel):
    level: float = Field(description="Nominal coverage, e.g. 0.8 for an 80% interval")
    method: str
    empirical_coverage_2025: dict[int, float] = Field(
        default_factory=dict, description="Share of 2025 backtest actuals inside the interval, by horizon")


class HorizonAccuracy(BaseModel):
    horizon_months: int
    mae_units: float
    rmse_units: float
    wape_pct: float
    n_predictions: int


class ForecastResponse(BaseModel):
    meta: Meta
    region_id: str
    region_name: str
    model: str
    model_version: str
    origin_month: date = Field(description="Last observed month; forecasts start the month after")
    horizon: int
    generated: str = Field(description="'live' (computed on request from the stored model) or 'precomputed'")
    assumptions: list[str]
    interval: Interval
    forecast: list[ForecastPoint]
    history: list[HistoryPoint]
    backtest_accuracy_2025: list[HorizonAccuracy]


class ScoreRow(BaseModel):
    model: str
    horizon_months: int
    n_predictions: int
    mae_units: float
    rmse_units: float
    wape_pct: float


class CoverageRow(BaseModel):
    model: str
    horizon_months: int
    nominal: float
    empirical_coverage: float
    n: int
    mean_width_units: float


class FeatureImportance(BaseModel):
    feature: str
    mean_abs_shap: float


class Contribution(BaseModel):
    feature: str
    feature_value: float
    shap: float


class LocalExplanation(BaseModel):
    region_id: str
    product_id: str
    target_month: str
    actual_units: float
    base_value: float
    prediction: float
    contributions: list[Contribution]
    other_features_shap: float


class ModelInsightsResponse(BaseModel):
    meta: Meta
    evaluation: str
    horizon_scorecard: list[ScoreRow]
    one_step_scorecard: list[dict] | None
    interval_coverage: list[CoverageRow]
    shap_caveat: str
    global_importance: list[FeatureImportance]
    local_examples: list[LocalExplanation]
    model_versions: dict[str, str]
