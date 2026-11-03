"""AI Business Analyst request and response models."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=12, description="Earlier turns, oldest first")
    dataset_id: UUID | None = Field(None, description="An uploaded dataset you own; omit for the synthetic demo")


class ToolCall(BaseModel):
    tool: str
    args: dict[str, Any]
    ok: bool


class Grounding(BaseModel):
    numbers_checked: int
    unverified: list[str] = Field(description="Numbers in the answer that no tool result or user message contained")


class DatasetRef(BaseModel):
    kind: Literal["demo", "upload"]
    label: str
    dataset_id: str | None = None
    model_version: str | None = None


class ChatResponse(BaseModel):
    answer: str
    dataset: DatasetRef
    tool_calls: list[ToolCall]
    caveats: list[str]
    grounding: Grounding
    model: str
    duration_ms: int


class AgentStatus(BaseModel):
    enabled: bool
    mode: str
    model: str | None
    tools: list[str]
    requests_per_hour: int
    note: str
