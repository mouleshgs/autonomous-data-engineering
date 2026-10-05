import re
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


def init_datasets_from_disk():
    if not UPLOAD_DIR.exists():
        return
    for file_path in UPLOAD_DIR.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in {".csv", ".xlsx", ".xls", ".json"}:
            dataset_id = file_path.stem
            if dataset_id in DATASETS:
                continue
            try:
                df = load_frame(file_path)
                processed_candidate = PROCESSED_DIR / f"{dataset_id}.csv"
                is_processed = processed_candidate.exists()
                clean_name = file_path.name
                if re.match(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}", file_path.stem):
                    cols_str = " ".join(str(c).lower() for c in df.columns)
                    if "order" in cols_str or "customer" in cols_str:
                        clean_name = "customer_orders.csv"
                    elif "sale" in cols_str or "revenue" in cols_str:
                        clean_name = "sales_data.csv"
                    elif "patient" in cols_str or "admission" in cols_str:
                        clean_name = "hospital_admissions.csv"
                    elif "laptop" in cols_str:
                        clean_name = "laptop_specs.csv"
                    else:
                        clean_name = "ecommerce_orders.csv"
                item = {
                    "id": dataset_id,
                    "name": clean_name,
                    "source_type": file_path.suffix[1:].upper(),
                    "path": str(file_path),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "profile": profile_frame(df),
                    "status": "READY" if is_processed else "PROFILED",
                    "processed_path": str(processed_candidate) if is_processed else None,
                    "quality_before": quality(df) if is_processed else None,
                    "quality_after": quality(load_frame(processed_candidate)) if is_processed else None,
                    "preview": df.head(8).fillna("").to_dict(orient="records"),
                    "uploaded_at": now(),
                }
                DATASETS[dataset_id] = item
            except Exception:
                pass


init_datasets_from_disk()


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
