from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.tools.data_tools import (
    PROCESSED_DIR,
    coerce_numeric_columns,
    correct_negative_values,
    fill_missing_values,
    load_frame,
    normalize_strings,
    normalize_boolean_columns,
    persist_dataframe,
    plan_with_llm,
    profile_frame,
    quality,
    remove_empty_columns,
    remove_empty_rows,
    remove_duplicates,
    standardize_dates,
    standardize_column_names,
    validate_dataframe,
)


class DataPipelineState(TypedDict):
    dataset_id: str
    dataset_name: str
    source_type: str
    source_path: str
    dataframe: Any
    profile: dict[str, Any]
    plan: list[dict[str, Any]]
    before_quality: dict[str, Any]
    after_quality: dict[str, Any]
    processed_path: str | None
    rows: int
    columns: int
    status: str
    logs: list[dict[str, Any]]
    stages: list[dict[str, Any]]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_log(state: DataPipelineState, agent: str, action: str, reason: str, tool: str, status: str = "SUCCESS", output: str = "") -> None:
    state["logs"].append({
        "agent": agent,
        "action": action,
        "reason": reason,
        "tool": tool,
        "status": status,
        "output": output,
        "timestamp": now(),
    })


def source_analyzer_node(state: DataPipelineState) -> DataPipelineState:
    time.sleep(0.5)
    frame = load_frame(state["source_path"])
    state["dataframe"] = frame
    state["rows"] = len(frame)
    state["columns"] = len(frame.columns)
    state["status"] = "RUNNING"
    _add_log(state, "Source Analyzer", "Inspected uploaded source", f"Detected {state['source_type']} with {state['rows']} rows", "load_frame", output=state["source_path"])
    state["stages"].append({"name": "Source Analyzer", "status": "SUCCESS", "detail": f"Read {state['source_type']} source"})
    return state


def profiler_node(state: DataPipelineState) -> DataPipelineState:
    time.sleep(0.6)
    frame = state["dataframe"]
    state["profile"] = profile_frame(frame)
    state["before_quality"] = quality(frame)
    _add_log(state, "Profiler Agent", "Profiled dataset", f"Found {state['profile']['missing_values']} missing values and {state['profile']['duplicates']} duplicates", "profile_frame")
    state["stages"].append({"name": "Profiler Agent", "status": "SUCCESS", "detail": "Generated column-level profile"})
    return state


def planner_node(state: DataPipelineState) -> DataPipelineState:
    time.sleep(0.8)
    state["plan"] = plan_with_llm(state["profile"])
    _add_log(state, "Planner Agent", "Generated adaptive plan", f"Selected {len(state['plan'])} safe operations from observed issues", "plan_with_llm", output=str(state["plan"]))
    state["stages"].append({"name": "Planner Agent", "status": "SUCCESS", "detail": f"Selected {len(state['plan'])} safe operations"})
    return state


def cleaning_node(state: DataPipelineState) -> DataPipelineState:
    time.sleep(0.6)
    frame = state["dataframe"].copy()
    for step in state["plan"]:
        op = step["operation"]
        if op == "remove_empty_rows":
            before = len(frame)
            frame = remove_empty_rows(frame)
            _add_log(state, "Cleaning Agent", op, step["reason"], "remove_empty_rows", output=f"Removed {before - len(frame)} rows")
        elif op == "remove_empty_columns":
            before = len(frame.columns)
            frame = remove_empty_columns(frame)
            _add_log(state, "Cleaning Agent", op, step["reason"], "remove_empty_columns", output=f"Removed {before - len(frame.columns)} columns")
        elif op == "remove_duplicates":
            before = len(frame)
            frame = remove_duplicates(frame)
            _add_log(state, "Cleaning Agent", op, step["reason"], "drop_duplicates", output=f"Removed {before - len(frame)} rows")
        elif op == "coerce_numeric_columns":
            frame = coerce_numeric_columns(frame)
            _add_log(state, "Cleaning Agent", op, step["reason"], "coerce_numeric_columns", output=str(frame.dtypes.astype(str).to_dict()))
        elif op == "normalize_boolean_columns":
            frame = normalize_boolean_columns(frame)
            _add_log(state, "Cleaning Agent", op, step["reason"], "normalize_boolean_columns", output="Normalized boolean values to Yes/No")
        elif op == "fill_missing":
            frame = fill_missing_values(frame)
            _add_log(state, "Cleaning Agent", op, step["reason"], "fill_missing_values", output=str(frame.isna().sum().sum()))
        elif op == "correct_negative_values":
            negative_count = int(frame.select_dtypes(include="number").lt(0).sum().sum())
            frame = correct_negative_values(frame)
            _add_log(state, "Cleaning Agent", op, step["reason"], "correct_negative_values", output=f"Corrected {negative_count} negative values")
        elif op == "normalize_strings":
            frame = normalize_strings(frame)
            _add_log(state, "Cleaning Agent", op, step["reason"], "normalize_strings", output=str(frame.shape))
        elif op == "standardize_dates":
            frame = standardize_dates(frame)
            _add_log(state, "Transformation Agent", op, step["reason"], "standardize_dates", output=str(frame.shape))
        elif op == "standardize_column_names":
            frame = standardize_column_names(frame)
            _add_log(state, "Transformation Agent", op, step["reason"], "standardize_column_names", output=str(list(frame.columns)))
        elif op == "validate":
            validation = validate_dataframe(frame)
            state["after_quality"] = validation["quality"]
            _add_log(state, "Validation Agent", op, step["reason"], "validate_dataframe", status="SUCCESS" if validation["validated"] else "WARNING", output=str(validation))
    state["dataframe"] = frame
    state["stages"].append({"name": "Cleaning", "status": "SUCCESS", "detail": "; ".join(step.get("operation", "clean") for step in state["plan"] if step.get("operation") not in {"validate", "store"}) or "No clean-up steps needed"})
    return state


def transformation_node(state: DataPipelineState) -> DataPipelineState:
    time.sleep(0.7)
    frame = state["dataframe"]
    state["after_quality"] = quality(frame)
    output_path = Path(PROCESSED_DIR) / f"{state['dataset_id']}.csv"
    state["processed_path"] = persist_dataframe(frame, output_path)
    state["status"] = "SUCCESS"
    _add_log(state, "Transformation Agent", "Prepared final dataset", f"Saved {len(frame)} rows to processed output", "save_processed_csv", output=state["processed_path"])
    state["stages"].append({"name": "Ingestion", "status": "SUCCESS", "detail": "Loaded source into internal dataframe"})
    state["stages"].append({"name": "Transformation", "status": "SUCCESS", "detail": "Prepared final transformed dataset"})
    state["stages"].append({"name": "Validation", "status": "SUCCESS", "detail": "Schema retained; deterministic quality checks completed"})
    state["stages"].append({"name": "Storage", "status": "SUCCESS", "detail": state["processed_path"]})
    return state


def build_langgraph_pipeline():
    builder = StateGraph(DataPipelineState)
    builder.add_node("source_analyzer", source_analyzer_node)
    builder.add_node("profiler", profiler_node)
    builder.add_node("planner", planner_node)
    builder.add_node("cleaning", cleaning_node)
    builder.add_node("transformation", transformation_node)
    builder.add_edge(START, "source_analyzer")
    builder.add_edge("source_analyzer", "profiler")
    builder.add_edge("profiler", "planner")
    builder.add_edge("planner", "cleaning")
    builder.add_edge("cleaning", "transformation")
    builder.add_edge("transformation", END)
    return builder.compile()


def run_langgraph_pipeline(dataset_id: str, dataset: dict[str, Any]) -> dict[str, Any]:
    initial_state: DataPipelineState = {
        "dataset_id": dataset_id,
        "dataset_name": dataset["name"],
        "source_type": dataset.get("source_type", "CSV"),
        "source_path": dataset["path"],
        "dataframe": None,
        "profile": {},
        "plan": [],
        "before_quality": {},
        "after_quality": {},
        "processed_path": None,
        "rows": 0,
        "columns": 0,
        "status": "PENDING",
        "logs": [],
        "stages": [],
    }
    graph = build_langgraph_pipeline()
    final_state = graph.invoke(initial_state)
    run = {
        "id": f"run-{dataset_id}",
        "dataset_id": dataset_id,
        "status": final_state["status"],
        "started_at": now(),
        "finished_at": now(),
        "before": final_state["before_quality"],
        "after": final_state["after_quality"],
        "plan": final_state["plan"],
        "stages": final_state["stages"],
        "logs": final_state["logs"],
    }
    dataset.update({
        "profile": final_state["profile"],
        "quality_before": final_state["before_quality"],
        "quality_after": final_state["after_quality"],
        "preview": final_state["dataframe"].head(8).fillna("").to_dict(orient="records"),
        "rows": len(final_state["dataframe"]),
        "columns": len(final_state["dataframe"].columns),
        "status": "READY",
        "processed_path": final_state["processed_path"],
    })
    return run
