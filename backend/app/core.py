from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from app.agents.data_cleaning import DataCleaningAgent
from app.agents.data_profiler import DataProfilerAgent
from app.agents.pipeline_planner import PipelinePlannerAgent
from app.agents.source_analyzer import SourceAnalyzerAgent
from app.agents.transformation import TransformationAgent
from app.graph.agent_graph import run_langgraph_pipeline
from app.schemas.pipeline_state import PipelineState
from app.tools.data_tools import PROCESSED_DIR as TOOL_PROCESSED_DIR
from app.tools.data_tools import build_plan, load_frame as _load_frame, profile_frame as _profile_frame, quality as _quality

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "data" / "uploads"
PROCESSED_DIR = TOOL_PROCESSED_DIR
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DATASETS: dict[str, dict[str, Any]] = {}
RUNS: dict[str, dict[str, Any]] = {}
LOGS: list[dict[str, Any]] = []


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def quality(df: pd.DataFrame) -> dict[str, Any]:
    return _quality(df)


def load_frame(path: Path) -> pd.DataFrame:
    return _load_frame(path)


def profile_frame(df: pd.DataFrame) -> dict[str, Any]:
    return _profile_frame(df)


def log(agent: str, action: str, reason: str, tool: str, status: str = "SUCCESS", output: str = "") -> dict[str, Any]:
    record = {"id": str(uuid4()), "agent": agent, "action": action, "reason": reason, "tool": tool, "status": status, "output": output, "timestamp": now()}
    LOGS.append(record)
    return record


def plan_for(profile: dict[str, Any]) -> list[dict[str, str]]:
    return build_plan(profile)


def execute_run(dataset_id: str) -> dict[str, Any]:
    dataset = DATASETS[dataset_id]
    run_id = str(uuid4())
    run = run_langgraph_pipeline(dataset_id, dataset)
    run["id"] = run_id
    for record in run.get("logs", []):
        record.update({
            "id": str(uuid4()),
            "run_id": run_id,
            "dataset_id": dataset_id,
        })
        LOGS.append(record)
    RUNS[run_id] = run
    return run
