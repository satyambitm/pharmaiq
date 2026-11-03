"""AI Business Analyst chat. The dataset is resolved and authorized here, before the model sees anything."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, Request

from backend.app.agent.context import DemoContext, UploadContext
from backend.app.agent.limits import AgentLimiter
from backend.app.agent.runner import run_turn
from backend.app.api.deps import get_bundle, get_dataset_store, get_repository
from backend.app.core.auth import get_owner, get_settings_from_app, owner_from_key
from backend.app.core.config import Settings
from backend.app.core.errors import AppError, Conflict, NotFound
from backend.app.db.training import TrainingStore
from backend.app.schemas.agent import AgentStatus, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/agent", tags=["agent"])
TOOLS = ["get_sales_summary", "get_sales_forecast", "get_market_segments", "explain_forecast",
         "compare_budget_scenarios", "get_data_quality"]


def default_llm(settings: Settings):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.agent_model, api_key=settings.openai_api_key.get_secret_value(),
                      temperature=0, timeout=settings.agent_timeout_seconds, max_retries=1)


def _limiter(request: Request, settings: Settings) -> AgentLimiter:
    lim = getattr(request.app.state, "agent_limiter", None)
    if lim is None:
        lim = request.app.state.agent_limiter = AgentLimiter(settings.agent_requests_per_hour,
                                                             settings.agent_requests_per_day)
    return lim


def _client(request: Request) -> str:
    # Render's proxy appends the address it saw as the last X-Forwarded-For entry; earlier entries are client-set.
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[-1].strip() if fwd else (request.client.host if request.client else "unknown")


@router.get("/status", response_model=AgentStatus, summary="Whether the AI analyst is available")
def status(settings: Settings = Depends(get_settings_from_app)) -> dict:
    note = ("Answers come only from the six read-only tools over one dataset; every answer lists the tools used."
            if settings.agent_enabled else "The AI analyst is not enabled on this deployment.")
    return {"enabled": settings.agent_enabled, "mode": settings.agent_mode,
            "model": settings.agent_model if settings.agent_enabled else None, "tools": TOOLS,
            "requests_per_hour": settings.agent_requests_per_hour, "note": note}


@router.post("/chat", response_model=ChatResponse, summary="Ask the AI analyst about the demo or your dataset")
def chat(body: ChatRequest, request: Request, settings: Settings = Depends(get_settings_from_app),
         x_api_key: str | None = Header(None, alias="X-API-Key")) -> dict:
    if not settings.agent_enabled:
        raise AppError(503, "agent_disabled", "The AI analyst is not enabled on this deployment.")
    caller = owner_from_key(settings, x_api_key).owner_id if settings.agent_mode == "api_key" else None

    if body.dataset_id is None:
        ctx = DemoContext(get_repository(request), get_bundle(request))
        ref = {"kind": "demo", "label": ctx.label}
    else:
        owner = get_owner(request, x_api_key)
        store = get_dataset_store(request)
        did = str(body.dataset_id)
        meta = store.get(owner.owner_id, did, with_report=True)
        if meta is None:
            raise NotFound("Dataset not found")
        if meta["status"] != "ready":
            raise Conflict("Confirm ingestion of this dataset before asking questions about it.")
        version = TrainingStore(store.engine).latest_version(owner.owner_id, did)
        ctx = UploadContext(meta, store.rows(owner.owner_id, did), version)
        caller = owner.owner_id
        ref = {"kind": "upload", "label": ctx.label, "dataset_id": did,
               "model_version": version["version_id"] if version else None}

    refusal = _limiter(request, settings).check(caller or _client(request))
    if refusal:
        raise AppError(429, "rate_limited", refusal)

    factory = getattr(request.app.state, "agent_llm_factory", None) or default_llm
    try:
        out = run_turn(factory(settings), ctx, body.message, [t.model_dump() for t in body.history],
                       settings.agent_max_tool_calls, settings.agent_timeout_seconds)
    except AppError:
        raise
    except Exception as exc:  # provider errors: never echo them, they can include request details
        logger.warning("agent_llm_failed error=%s", type(exc).__name__)
        raise AppError(502, "llm_unavailable", "The language model could not be reached. Try again shortly.") from exc
    logger.info("agent_turn dataset=%s tools=%s unverified=%d duration_ms=%d", ref["kind"],
                ",".join(c["tool"] for c in out["tool_calls"]) or "-", len(out["grounding"]["unverified"]),
                out["duration_ms"])
    return {**out, "dataset": ref, "model": settings.agent_model}
