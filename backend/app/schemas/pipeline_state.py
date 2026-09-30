from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PipelineStage(BaseModel):
    name: str
    status: Literal["SUCCESS", "PENDING", "FAILED"] = "PENDING"
    detail: str = ""


class PipelineState(BaseModel):
    dataset_id: str
    dataset_name: str
    source_type: str = "CSV"
    source_path: str = ""
    rows: int = 0
    columns: int = 0
    profile: dict[str, Any] = Field(default_factory=dict)
    plan: list[dict[str, Any]] = Field(default_factory=list)
    before_quality: dict[str, Any] = Field(default_factory=dict)
    after_quality: dict[str, Any] = Field(default_factory=dict)
    processed_path: str | None = None
    status: Literal["PENDING", "SUCCESS", "FAILED"] = "PENDING"
    dataframe: Any = None
    stages: list[PipelineStage] = Field(default_factory=list)
    logs: list[dict[str, Any]] = Field(default_factory=list)
