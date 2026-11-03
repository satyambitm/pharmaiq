from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, Query

from backend.app.api.deps import get_meta, get_repository
from backend.app.db.connection import SalesRepository
from backend.app.schemas.analytics import OverviewResponse, RegionsResponse
from backend.app.schemas.common import Meta, ProductId
from backend.app.services import analytics_service

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/overview", response_model=OverviewResponse, summary="Portfolio KPIs, monthly trend and product mix")
def overview(start_month: date | None = Query(None, description="Inclusive, first day of month"),
             end_month: date | None = Query(None, description="Inclusive, first day of month"),
             repo: SalesRepository = Depends(get_repository), meta: Meta = Depends(get_meta)) -> dict:
    return {"meta": meta, **analytics_service.overview(repo, start_month, end_month)}


@router.get("/regions", response_model=RegionsResponse, summary="Per-region performance with optional filters")
def regions(start_month: date | None = Query(None), end_month: date | None = Query(None),
            product_id: list[ProductId] | None = Query(None, description="Repeat to select several products"),
            repo: SalesRepository = Depends(get_repository), meta: Meta = Depends(get_meta)) -> dict:
    products = [p.value for p in product_id] if product_id else None
    return {"meta": meta, **analytics_service.regions(repo, start_month, end_month, products)}
