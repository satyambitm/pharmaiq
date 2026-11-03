from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.deps import get_meta
from backend.app.schemas.common import Meta
from backend.app.schemas.optimization import OptimizeRequest, OptimizeResponse
from backend.app.services import optimization_service

router = APIRouter(prefix="/api/v1", tags=["optimization"])


@router.post("/optimize-budget", response_model=OptimizeResponse,
             summary="Scenario-based regional budget allocation under assumed response curves")
def optimize_budget(req: OptimizeRequest, meta: Meta = Depends(get_meta)) -> dict:
    return {"meta": meta, **optimization_service.optimize_budget(
        req.total_budget_cr, req.scenario, req.competitor_pressure, req.min_spend_cr, req.max_spend_cr)}
