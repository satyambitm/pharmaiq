"""The six read-only tools. None takes a dataset argument: each is bound to the server-chosen context."""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from backend.app.agent.context import DemoContext, UploadContext

Month = Field(None, pattern=r"^\d{4}-\d{2}$", description="YYYY-MM, inclusive")


class SummaryArgs(BaseModel):
    start_month: str | None = Month
    end_month: str | None = Month
    region: str | None = Field(None, description="One region name; omit for all regions")


class ForecastArgs(BaseModel):
    region: str = Field(..., description="Region name")
    horizon: int = Field(3, ge=1, le=3, description="Months ahead, 1-3")
    model: str | None = Field(None, description="Model name; omit for the default/recommended model")
    product: str | None = Field(None, description="One product name; omit for all products")


class NoArgs(BaseModel):
    pass


class BudgetArgs(BaseModel):
    total_budget_cr: float = Field(..., gt=0, le=1000, description="Total marketing budget in INR crore")
    scenarios: list[Literal["low", "base", "high"]] | None = Field(None, description="Default: all three")
    competitor_pressure: dict[str, float] | None = Field(
        None, description="Region name -> fraction in [0, 1) by which that region's assumed response shrinks")


def build_tools(ctx: DemoContext | UploadContext) -> list[BaseTool]:
    models = ("'xgboost' (default) or 'prophet'" if ctx.kind == "demo"
              else "'xgboost' or 'naive_last_month' (default: the model recommended by the holdout evaluation)")

    def tool(fn, name: str, description: str, schema: type[BaseModel]) -> BaseTool:
        return StructuredTool.from_function(func=fn, name=name, description=description, args_schema=schema)

    return [
        tool(lambda start_month=None, end_month=None, region=None: ctx.sales_summary(start_month, end_month, region),
             "get_sales_summary",
             "Totals and breakdowns of historical sales (units; revenue and marketing spend only when the data has "
             "them) for a month window, optionally one region.", SummaryArgs),
        tool(lambda region, horizon=3, model=None, product=None: ctx.sales_forecast(region, horizon, model, product),
             "get_sales_forecast",
             f"Stored model forecast of monthly units for one region, 1-3 months ahead, with the model's evaluated "
             f"holdout accuracy. model: {models}.", ForecastArgs),
        tool(lambda: ctx.market_segments(), "get_market_segments",
             "K-Means segments of region x product cells with segment profiles.", NoArgs),
        tool(lambda: ctx.explain_forecast(), "explain_forecast",
             "Global SHAP feature importance of the forecasting model, with its caveat.", NoArgs),
        tool(lambda total_budget_cr, scenarios=None, competitor_pressure=None:
             ctx.compare_budget_scenarios(total_budget_cr, scenarios, competitor_pressure),
             "compare_budget_scenarios",
             "Compare budget allocations (optimized, equal split, status quo) across low/base/high scenarios "
             "under ASSUMED response curves. Results are simulations, not measured ROI.", BudgetArgs),
        tool(lambda: ctx.data_quality(), "get_data_quality",
             "Data source, coverage, missing values, validation results and which analyses the data supports.",
             NoArgs),
    ]
