"""Shared response pieces and the fixed vocabularies the API accepts."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from src.config import DATA_SOURCE, PRODUCTS, REGIONS

SYNTHETIC_NOTE = ("All data is synthetic (generated with a fixed seed) and illustrative. "
                  "It does not describe any real company, product or market.")

RegionId = Enum("RegionId", {k: k for k in REGIONS}, type=str)
ProductId = Enum("ProductId", {k: k for k in PRODUCTS}, type=str)


class ForecastModel(str, Enum):
    xgboost = "xgboost"
    prophet = "prophet"


class Meta(BaseModel):
    data_source: str = DATA_SOURCE
    data_backend: str
    disclaimer: str = SYNTHETIC_NOTE


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
