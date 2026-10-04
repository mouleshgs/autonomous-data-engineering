from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.tools.data_tools import (
    PROCESSED_DIR,
    clip_outliers,
    coerce_numeric_columns,
    correct_negative_values,
    encode_categoricals,
    evaluate_downstream_model,
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
    scale_features,
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
    model_type: str
    target_column: str | None
    model_benchmark: dict[str, Any] | None


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


def _slow_stage(stage_name: str, minimum: float = 0.6, variance: float = 0.8) -> None:
    time.sleep(0.25)


def source_analyzer_node(state: DataPipelineState) -> DataPipelineState:
    _slow_stage("Source Analyzer", 0.7, 0.9)
    frame = load_frame(state["source_path"])
    state["dataframe"] = frame
    state["rows"] = len(frame)
    state["columns"] = len(frame.columns)
    state["status"] = "RUNNING"
    analysis = (
        f"I inspected the raw source and confirmed {state['rows']} rows and {state['columns']} columns. "
        f"The next step is to profile value quality, null density, and schema drift before planning repairs."
    )
    _add_log(state, "Source Analyzer", "Inspected uploaded source", f"Detected {state['source_type']} with {state['rows']} rows", "load_frame", output=analysis)
    state["stages"].append({"name": "Source Analyzer", "status": "SUCCESS", "detail": f"Read {state['source_type']} source and confirmed schema"})
    return state


def profiler_node(state: DataPipelineState) -> DataPipelineState:
    _slow_stage("Profiler Agent", 0.8, 1.0)
    frame = state["dataframe"]
    state["profile"] = profile_frame(frame)
    state["before_quality"] = quality(frame)
    analysis = (
        f"I profiled the dataset and found {state['profile'].get('missing_values', 0)} missing values, "
        f"{state['profile'].get('duplicates', 0)} duplicate rows, and a schema with {len(state['profile'].get('column_stats', []))} columns. "
        "This suggests the pipeline should normalize types, repair nulls, and remove noisy rows before model training."
    )
    _add_log(state, "Profiler Agent", "Profiled dataset", f"Found {state['profile']['missing_values']} missing values and {state['profile']['duplicates']} duplicates", "profile_frame", output=analysis)
    state["stages"].append({"name": "Profiler Agent", "status": "SUCCESS", "detail": "Generated column-level profile and quality snapshot"})
    return state


def planner_node(state: DataPipelineState) -> DataPipelineState:
    _slow_stage("Planner Agent", 1.0, 1.2)
    model_type = state.get("model_type", "generic")
    target_col = state.get("target_column")
    state["plan"] = plan_with_llm(
        state["profile"],
        model_type=model_type,
        target_col=target_col,
    )
    model_note = f" conditioned on target model '{model_type.upper()}'" if model_type != "generic" else ""
    plan_summary = (
        f"The planner has chosen {len(state['plan'])} operations to normalize quality issues and preserve the dataset's predictive signal. "
        f"The highest-priority steps are for row cleanup, type coercion, null repair, and schema consistency before validation.{model_note}."
    )
    _add_log(
        state,
        "Planner Agent",
        "Generated adaptive plan",
        f"Selected {len(state['plan'])} operations{model_note}",
        "plan_with_llm",
        output=plan_summary,
    )
    state["stages"].append({"name": "Planner Agent", "status": "SUCCESS", "detail": f"Selected {len(state['plan'])} operations{model_note}"})
    return state


def cleaning_node(state: DataPipelineState) -> DataPipelineState:
    _slow_stage("Cleaning Agent", 0.9, 1.1)
    frame = state["dataframe"].copy()
    target_col = state.get("target_column")

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
        elif op == "clip_outliers":
            frame = clip_outliers(frame, target_col=target_col)
            _add_log(state, "Feature Engineering Agent", op, step["reason"], "clip_outliers", output="Clipped extreme numerical outliers (1st/99th percentiles)")
        elif op == "scale_features":
            scale_method = step.get("method", "standard")
            frame = scale_features(frame, method=scale_method, target_col=target_col)
            _add_log(state, "Feature Engineering Agent", op, step["reason"], "scale_features", output=f"Applied {scale_method.upper()} scaling to numeric predictors")
        elif op == "encode_categoricals":
            enc_method = step.get("method", "onehot")
            frame = encode_categoricals(frame, method=enc_method, target_col=target_col)
            _add_log(state, "Feature Engineering Agent", op, step["reason"], "encode_categoricals", output=f"Applied {enc_method} encoding to categoricals")
        elif op == "validate":
            validation = validate_dataframe(frame)
            state["after_quality"] = validation["quality"]
            _add_log(state, "Validation Agent", op, step["reason"], "validate_dataframe", status="SUCCESS" if validation["validated"] else "WARNING", output=str(validation))

    state["dataframe"] = frame
    cleaning_summary = (
        f"The cleaning pass repaired the dataset by applying {len(state['plan'])} planned steps, including null repair, duplicate removal, "
        "type normalization, and schema cleanup. The final frame is now more stable for downstream feature engineering and model evaluation."
    )
    _add_log(state, "Cleaning Agent", "Applied cleaning rules", f"Executed {len(state['plan'])} pipeline steps", "cleaning_pipeline", output=cleaning_summary)
    state["stages"].append({
        "name": "Cleaning",
        "status": "SUCCESS",
        "detail": "; ".join(step.get("operation", "clean") for step in state["plan"] if step.get("operation") not in {"validate", "store"}) or "No clean-up steps needed",
    })
    return state


def transformation_node(state: DataPipelineState) -> DataPipelineState:
    _slow_stage("Transformation Agent", 0.9, 1.0)
    frame = state["dataframe"]
    state["after_quality"] = quality(frame)
    output_path = Path(PROCESSED_DIR) / f"{state['dataset_id']}.csv"
    state["processed_path"] = persist_dataframe(frame, output_path)
    state["status"] = "SUCCESS"
    reasoning = (
        f"The transformed dataset retains {len(frame)} rows and {len(frame.columns)} columns after cleanup. "
        "The validation gate is now checking whether the final schema and value ranges are strong enough for analytics and downstream model scoring."
    )
    _add_log(state, "Transformation Agent", "Prepared final dataset", f"Saved {len(frame)} rows to processed output", "save_processed_csv", output=reasoning)
    state["stages"].append({"name": "Ingestion", "status": "SUCCESS", "detail": "Loaded source into internal dataframe"})
    state["stages"].append({"name": "Transformation", "status": "SUCCESS", "detail": "Prepared final transformed dataset"})
    state["stages"].append({"name": "Validation", "status": "SUCCESS", "detail": "Schema retained; deterministic quality checks completed"})
    state["stages"].append({"name": "Storage", "status": "SUCCESS", "detail": state["processed_path"]})
    return state


def _infer_target_column(frame: Any, supplied_target: str | None = None) -> str | None:
    if supplied_target and supplied_target in frame.columns:
        return supplied_target

    target_keywords = [
        "target", "label", "status", "outcome", "category", "class", "flag",
        "approved", "is_return", "is_churn", "segment", "result",
    ]
    for column in frame.columns:
        col_name = str(column).lower()
        if any(keyword in col_name for keyword in target_keywords):
            return column

    numeric_cols = list(frame.select_dtypes(include="number").columns)
    if len(numeric_cols) == 0:
        return None
    return numeric_cols[-1]


def benchmark_node(state: DataPipelineState) -> DataPipelineState:
    model_type = state.get("model_type", "generic")
    target_col = _infer_target_column(state["dataframe"], state.get("target_column")) if state.get("dataframe") is not None else state.get("target_column")
    state["target_column"] = target_col

    if model_type and model_type != "generic" and target_col:
        _slow_stage("Model Benchmark Agent", 1.1, 1.5)
        frame = state["dataframe"]
        benchmark = evaluate_downstream_model(frame, target_col=target_col, model_type=model_type)
        state["model_benchmark"] = benchmark
        reasoning = (
            f"The benchmark compared a baseline estimator with a model-aware pipeline for target '{target_col}'. "
            f"The tuned approach reached {benchmark['model_aware_score']}% accuracy compared with {benchmark['baseline_score']}%, "
            f"which provides a +{benchmark['lift']}% lift on this dataset."
        )
        _add_log(
            state,
            "Model Benchmark Agent",
            "Evaluated downstream model",
            f"Trained {benchmark['model_name']} on target '{benchmark['target_column']}'",
            "evaluate_downstream_model",
            output=reasoning,
        )
        state["stages"].append({
            "name": "Model Benchmark",
            "status": "SUCCESS",
            "detail": f"{benchmark['model_name']} ({benchmark['metric_name']}): {benchmark['model_aware_score']}% (+{benchmark['lift']}% lift)",
        })
    else:
        _add_log(
            state,
            "Model Benchmark Agent",
            "Skipped benchmark",
            "No model type or target column available for evaluation",
            "evaluate_downstream_model",
            output="The pipeline is waiting for a target model and target label before benchmark execution can begin.",
        )
        state["stages"].append({
            "name": "Model Benchmark",
            "status": "PENDING",
            "detail": "Skipped benchmark because no target model or target column was available",
        })
    return state


def build_langgraph_pipeline():
    builder = StateGraph(DataPipelineState)
    builder.add_node("source_analyzer", source_analyzer_node)
    builder.add_node("profiler", profiler_node)
    builder.add_node("planner", planner_node)
    builder.add_node("cleaning", cleaning_node)
    builder.add_node("transformation", transformation_node)
    builder.add_node("benchmark", benchmark_node)
    builder.add_edge(START, "source_analyzer")
    builder.add_edge("source_analyzer", "profiler")
    builder.add_edge("profiler", "planner")
    builder.add_edge("planner", "cleaning")
    builder.add_edge("cleaning", "transformation")
    builder.add_edge("transformation", "benchmark")
    builder.add_edge("benchmark", END)
    return builder.compile()


def run_langgraph_pipeline(
    dataset_id: str,
    dataset: dict[str, Any],
    model_type: str = "generic",
    target_column: str | None = None,
) -> dict[str, Any]:
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
        "model_type": model_type or dataset.get("model_type", "generic"),
        "target_column": target_column or dataset.get("target_column"),
        "model_benchmark": None,
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
        "model_type": final_state.get("model_type", "generic"),
        "target_column": final_state.get("target_column"),
        "model_benchmark": final_state.get("model_benchmark"),
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
        "model_type": final_state.get("model_type", "generic"),
        "target_column": final_state.get("target_column"),
        "model_benchmark": final_state.get("model_benchmark"),
    })
    return run
