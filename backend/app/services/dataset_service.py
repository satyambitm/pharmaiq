"""Upload -> map/validate -> confirm workflow for user datasets.

Validation never writes monthly rows; only `activate` does, after the user has reviewed the report.
`activate` re-reads the stored original file and re-applies the stored mapping, so what is ingested is
exactly what the user approved (the pipeline is deterministic) and the source file is never modified.
"""

from __future__ import annotations

import logging

from backend.app.core.auth import Owner
from backend.app.core.config import Settings
from backend.app.core.errors import AppError, Conflict, InvalidRequest, NotFound
from backend.app.db.datasets import DatasetStore
from src.ingest.fields import SCHEMA_VERSION, suggest_mapping
from src.ingest.files import FileRejected, ParsedFile, read_upload
from src.ingest.pipeline import MappingConfig, MappingError, TransformResult, profile, transform

logger = logging.getLogger(__name__)
PREVIEW_ROWS = 20


def _parse(settings: Settings, content: bytes, file_name: str) -> ParsedFile:
    try:
        return read_upload(content, file_name, settings.max_upload_bytes, settings.max_upload_rows)
    except FileRejected as exc:
        raise AppError(415 if exc.unsupported_type else 422, "file_rejected", str(exc)) from exc


def _owned(store: DatasetStore, owner: Owner, dataset_id: str, with_report: bool = False) -> dict:
    meta = store.get(owner.owner_id, dataset_id, with_report=with_report)
    if meta is None:
        raise NotFound("Dataset not found")
    return meta


def upload(store: DatasetStore, settings: Settings, owner: Owner, file_name: str, content: bytes,
           dataset_name: str | None) -> dict:
    if store.count(owner.owner_id) >= settings.max_datasets_per_owner:
        raise Conflict(f"You already have {settings.max_datasets_per_owner} datasets; delete one before uploading another.")
    parsed = _parse(settings, content, file_name)
    name = " ".join((dataset_name or file_name).split())[:120] or "Untitled dataset"
    meta = store.create(owner.owner_id, name, parsed.file_type, file_name, content, parsed.sha256, len(parsed.frame))
    logger.info("dataset_uploaded owner=%s dataset=%s type=%s rows=%d bytes=%d", owner.owner_id, meta["dataset_id"],
                parsed.file_type, len(parsed.frame), parsed.size_bytes)
    return {"dataset": meta, "profile": _profile(parsed)}


def _profile(parsed: ParsedFile) -> dict:
    prof = profile(parsed.frame, suggest_mapping(list(parsed.frame.columns)), PREVIEW_ROWS)
    return {**prof, "file_warnings": parsed.warnings, "sheet_names": parsed.sheet_names, "encoding": parsed.encoding,
            "delimiter": parsed.delimiter}


def file_profile(store: DatasetStore, settings: Settings, owner: Owner, dataset_id: str) -> dict:
    meta = _owned(store, owner, dataset_id)
    content = store.file_content(owner.owner_id, dataset_id)
    if content is None:
        raise NotFound("Dataset file not found")
    return {"dataset": meta, "profile": _profile(_parse(settings, content, meta["file_name"]))}


def _run(store: DatasetStore, settings: Settings, owner: Owner, meta: dict, cfg: MappingConfig) -> TransformResult:
    content = store.file_content(owner.owner_id, meta["dataset_id"])
    if content is None:
        raise NotFound("Dataset file not found")
    parsed = _parse(settings, content, meta["file_name"])
    try:
        return transform(parsed.frame, cfg)
    except MappingError as exc:
        raise InvalidRequest(" ".join(exc.errors)) from exc


def _config(mapping: dict) -> MappingConfig:
    return MappingConfig(columns=dict(mapping["columns"]), **{k: v for k, v in mapping.items() if k != "columns"})


def validate(store: DatasetStore, settings: Settings, owner: Owner, dataset_id: str, request: dict) -> dict:
    meta = _owned(store, owner, dataset_id)
    if meta["status"] == "ready":
        raise Conflict("This dataset is already ingested. Upload the file again to apply a different mapping.")
    mapping = {"columns": {k.value if hasattr(k, "value") else k: v for k, v in request["mapping"].items()},
               **{k: request[k] for k in ("single_product_label", "dayfirst", "date_format", "drop_exact_duplicates",
                                         "currency")}}
    result = _run(store, settings, owner, meta, _config(mapping))
    store.save_validation(owner.owner_id, dataset_id, mapping, SCHEMA_VERSION, result.report)
    logger.info("dataset_validated owner=%s dataset=%s status=%s valid=%d rejected=%d", owner.owner_id, dataset_id,
                result.report["status"], result.report["valid_rows"], result.report["rejected_rows"])
    preview = result.monthly.head(PREVIEW_ROWS)
    return {"dataset": _owned(store, owner, dataset_id), "report": result.report,
            "monthly_preview": preview.astype(object).where(preview.notna(), None).to_dict("records")}


def activate(store: DatasetStore, settings: Settings, owner: Owner, dataset_id: str) -> dict:
    meta = _owned(store, owner, dataset_id, with_report=True)
    if meta["status"] == "ready":
        return {"dataset": _owned(store, owner, dataset_id), "report": meta["quality_report"]}
    if meta["status"] != "validated" or not meta["schema_mapping"]:
        raise Conflict("Validate the dataset and resolve every blocking error before confirming ingestion.")
    result = _run(store, settings, owner, meta, _config(meta["schema_mapping"]))
    if not result.ok:
        raise Conflict("Validation no longer passes for this file; validate it again.")
    store.ingest(owner.owner_id, dataset_id, result.monthly)
    logger.info("dataset_activated owner=%s dataset=%s monthly_rows=%d", owner.owner_id, dataset_id, len(result.monthly))
    return {"dataset": _owned(store, owner, dataset_id), "report": result.report}


def detail(store: DatasetStore, owner: Owner, dataset_id: str) -> dict:
    meta = _owned(store, owner, dataset_id, with_report=True)
    return {"dataset": meta, "report": meta.pop("quality_report")}


def list_datasets(store: DatasetStore, owner: Owner, limit: int, offset: int) -> dict:
    items, total = store.list(owner.owner_id, limit, offset)
    return {"items": items, "total": total, "limit": limit, "offset": offset}


def remove(store: DatasetStore, owner: Owner, dataset_id: str) -> dict:
    if not store.delete(owner.owner_id, dataset_id):
        raise NotFound("Dataset not found")
    logger.info("dataset_deleted owner=%s dataset=%s", owner.owner_id, dataset_id)
    return {"dataset_id": dataset_id, "deleted": True}
