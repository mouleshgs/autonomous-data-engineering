from __future__ import annotations

import threading
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
BACKGROUND_TASKS: dict[str, threading.Thread] = {}


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


def execute_run(dataset_id: str, model_type: str = "generic", target_column: str | None = None) -> dict[str, Any]:
    print(f"--> [core.execute_run] fetching dataset {dataset_id}", flush=True)
    dataset = DATASETS[dataset_id]
    run_id = str(uuid4())
    print(f"--> [core.execute_run] dataset={dataset.get('name')}, path={dataset.get('path')}", flush=True)
    try:
        run = run_langgraph_pipeline(dataset_id, dataset, model_type=model_type, target_column=target_column)
    except TypeError as te:
        print(f"--> [core.execute_run] TypeError fallback: {te}", flush=True)
        run = run_langgraph_pipeline(dataset_id, dataset)
    print(f"--> [core.execute_run] pipeline finished! status={run.get('status')}", flush=True)
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


def schedule_run(dataset_id: str, model_type: str = "generic", target_column: str | None = None) -> dict[str, Any]:
    dataset = DATASETS[dataset_id]
    run_id = str(uuid4())
    initial_run = {
        "id": run_id,
        "dataset_id": dataset_id,
        "dataset_name": dataset.get("name", "unknown"),
        "status": "STARTED",
        "started_at": now(),
        "finished_at": None,
        "before": {},
        "after": {},
        "plan": [],
        "stages": [],
        "logs": [],
        "model_type": model_type,
        "target_column": target_column,
        "model_benchmark": None,
    }
    RUNS[run_id] = initial_run

    def worker() -> None:
        try:
            final_run = execute_run(dataset_id, model_type=model_type, target_column=target_column)
            final_run["id"] = run_id
            if final_run.get("status") == "SUCCESS":
                final_run["status"] = "SUCCESS"
            RUNS[run_id] = final_run
        except Exception as exc:  # pragma: no cover - background worker safeguard
            RUNS[run_id] = {
                **initial_run,
                "status": "FAILED",
                "error": str(exc),
                "finished_at": now(),
            }

    thread = threading.Thread(target=worker, daemon=True)
    BACKGROUND_TASKS[run_id] = thread
    thread.start()
    return initial_run
