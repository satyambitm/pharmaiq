from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.api.deps import get_bundle, get_meta
from backend.app.schemas.common import Meta
from backend.app.schemas.segmentation import SegmentsResponse
from backend.app.services import segmentation_service
from ml.model_loader import ModelBundle

router = APIRouter(prefix="/api/v1", tags=["segmentation"])


@router.get("/segments", response_model=SegmentsResponse, summary="K-Means segments of region x product market cells")
def segments(bundle: ModelBundle = Depends(get_bundle), meta: Meta = Depends(get_meta)) -> dict:
    return {"meta": meta, **segmentation_service.segments(bundle)}
