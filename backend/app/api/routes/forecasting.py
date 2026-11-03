from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.app.api.deps import get_bundle, get_meta, get_repository
from backend.app.db.connection import SalesRepository
from backend.app.schemas.common import ForecastModel, Meta, RegionId
from backend.app.schemas.forecasting import ForecastResponse, ModelInsightsResponse
from backend.app.services import forecast_service, insights_service
from ml.model_loader import ModelBundle
from src.modeling.recursive import MAX_HORIZON

router = APIRouter(prefix="/api/v1", tags=["forecasting"])


@router.get("/forecasts", response_model=ForecastResponse,
            summary="1-3 month unit forecasts for every product in a region, with 80% intervals")
def forecasts(region_id: RegionId = Query(..., description="Region code R01-R06"),
              horizon: int = Query(3, ge=1, le=MAX_HORIZON, description="Months ahead"),
              model: ForecastModel = Query(ForecastModel.xgboost),
              repo: SalesRepository = Depends(get_repository), bundle: ModelBundle = Depends(get_bundle),
              meta: Meta = Depends(get_meta)) -> dict:
    return {"meta": meta, **forecast_service.forecast(repo, bundle, region_id.value, horizon, model.value)}


@router.get("/model-insights", response_model=ModelInsightsResponse,
            summary="Backtest scorecards, interval coverage and SHAP explanations (associations, not causes)")
def model_insights(bundle: ModelBundle = Depends(get_bundle), meta: Meta = Depends(get_meta)) -> dict:
    return {"meta": meta, **insights_service.model_insights(bundle)}
