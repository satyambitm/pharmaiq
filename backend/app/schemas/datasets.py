"""Request and response models for bring-your-own-data endpoints."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from src.ingest.fields import FIELDS

CanonicalField = Enum("CanonicalField", {k: k for k in FIELDS}, type=str)
SourceColumn = Annotated[str, StringConstraints(min_length=1, max_length=120)]


class DatasetStatus(str, Enum):
    uploaded = "uploaded"     # file stored, not yet mapped
    invalid = "invalid"       # last validation found blocking errors
    validated = "validated"   # report is clean; waiting for the user to confirm ingestion
    ready = "ready"           # monthly rows ingested and usable


class StoredMapping(BaseModel):
    """The mapping and parsing options applied by the last validation."""
    columns: dict[str, str]
    single_product_label: str | None = None
    dayfirst: bool | None = None
    date_format: str | None = None
    drop_exact_duplicates: bool = False
    currency: str | None = None


class DatasetMeta(BaseModel):
    dataset_id: str
    dataset_name: str
    source_type: Literal["csv", "xlsx"]
    status: DatasetStatus
    created_at: datetime
    updated_at: datetime
    file_name: str
    file_size_bytes: int
    file_sha256: str
    source_row_count: int
    row_count: int | None = None
    date_start: date | None = None
    date_end: date | None = None
    schema_mapping: StoredMapping | None = None
    schema_version: str | None = None


class ColumnProfile(BaseModel):
    name: str
    detected_type: Literal["number", "date", "text", "empty"]
    missing_pct: float
    n_unique: int
    examples: list[str]


class UploadProfile(BaseModel):
    columns: list[ColumnProfile]
    preview: list[dict[str, str | None]]
    quick_summary: dict
    suggested_mapping: dict[str, str]
    suggested_options: dict[str, bool]
    file_warnings: list[str] = []
    sheet_names: list[str] = []
    encoding: str | None = None
    delimiter: str | None = None


class UploadResponse(BaseModel):
    dataset: DatasetMeta
    profile: UploadProfile


class ValidateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mapping: dict[CanonicalField, SourceColumn] = Field(..., description="PharmaIQ field -> uploaded column name")
    single_product_label: str | None = Field(None, max_length=80,
                                             description="Required when no product column is mapped")
    dayfirst: bool = Field(False, description="Read ambiguous numeric dates as DD/MM")
    date_format: str | None = Field(None, max_length=40, description="Optional strftime format, e.g. %d-%m-%Y")
    drop_exact_duplicates: bool = Field(False, description="Drop rows identical to an earlier row before aggregating")
    currency: str | None = Field(None, pattern=r"^[A-Z]{3}$", description="Declared ISO currency of money columns")


class Coverage(BaseModel):
    date_start: date | None
    date_end: date | None
    months_spanned: int
    n_regions: int
    n_products: int
    n_series: int
    regions: list[str]
    products: list[str]


class GapExample(BaseModel):
    region: str
    product: str
    missing_months: int


class Gaps(BaseModel):
    missing_months_total: int
    series_with_gaps: int
    examples: list[GapExample]


class ForecastReadiness(BaseModel):
    level: Literal["full", "short_history", "baseline_only", "insufficient"]
    series_by_level: dict[str, int]
    min_months: int
    median_months: float
    missing_month_share_pct: float | None
    notes: list[str]


class SegmentationReadiness(BaseModel):
    status: Literal["ready", "insufficient"]
    reason: str


class ModelFeatures(BaseModel):
    available: list[str]
    too_sparse: list[str]
    not_mapped: list[str]


class Capability(BaseModel):
    enabled: bool
    reason: str | None


class Capabilities(BaseModel):
    revenue_analysis: Capability
    marketing_analysis: Capability
    competitor_analysis: Capability
    single_product: bool


class RejectedExample(BaseModel):
    row: int = Field(..., description="Row number as shown in a spreadsheet (header is row 1)")
    reasons: list[str]


class Aggregation(BaseModel):
    source_rows: int
    monthly_rows: int
    rules: dict[str, list[str]]


class QualityReport(BaseModel):
    schema_version: str
    status: Literal["ready_to_ingest", "invalid"]
    total_rows: int
    exact_duplicate_rows: int
    dropped_exact_duplicates: int
    valid_rows: int
    rejected_rows: int
    rejection_reasons: dict[str, int]
    rejected_examples: list[RejectedExample]
    missing_value_pct: dict[str, float]
    duplicate_key_rows: int
    detected_frequency: Literal["monthly", "weekly", "daily", "irregular", "unknown"]
    date_parsing: str
    aggregation: Aggregation
    currency: str | None
    unit_of_measure: str | None
    coverage: Coverage
    gaps: Gaps
    forecast_readiness: ForecastReadiness
    segmentation_readiness: SegmentationReadiness
    model_features: ModelFeatures
    capabilities: Capabilities
    warnings: list[str]
    errors: list[str]


class MonthlyRow(BaseModel):
    month_start: date
    region: str
    product: str
    units_sold: float
    revenue: float | None
    unit_price: float | None
    marketing_spend: float | None
    doctor_interactions: float | None
    competitor_price: float | None
    competitor_count: float | None
    market_size: float | None
    hcp_count: float | None
    n_source_rows: int


class ValidationResponse(BaseModel):
    dataset: DatasetMeta
    report: QualityReport
    monthly_preview: list[MonthlyRow]


class DatasetDetail(BaseModel):
    dataset: DatasetMeta
    report: QualityReport | None


class DatasetList(BaseModel):
    items: list[DatasetMeta]
    total: int
    limit: int
    offset: int


class DeleteResponse(BaseModel):
    dataset_id: str
    deleted: bool


class FieldInfo(BaseModel):
    name: str
    label: str
    required: bool
    description: str


class UploadSettings(BaseModel):
    uploads_enabled: bool
    mode: Literal["disabled", "local", "api_key"]
    auth_header: str | None
    max_upload_mb: float
    max_upload_rows: int
    max_datasets_per_owner: int
    supported_types: list[str]
    fields: list[FieldInfo]
    store_available: bool
    note: str
