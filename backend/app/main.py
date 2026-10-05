from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import re
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .core import DATASETS, LOGS, PROCESSED_DIR, RUNS, UPLOAD_DIR, execute_run, load_frame, now, profile_frame, schedule_run

app = FastAPI(title="Autonomous Data Engineering Platform", version="0.1.0")
cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins != ["*"] else ["*"],
    allow_credentials=True if cors_origins != ["*"] else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

class RunPipelineRequest(BaseModel):
    model_type: str = "generic"
    target_column: str | None = None

@app.get("/")
@app.head("/")
def root():
    return {"status": "ok", "service": "Autonomous Data Engineering Platform"}

@app.get("/api/health")
def health(): return {"status": "ok", "mode": "DEMO_MODE"}

@app.post("/api/datasets/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls", ".json", ".pdf"}: raise HTTPException(400, "Supported files: CSV, Excel, JSON, PDF")
    dataset_id = str(uuid4()); path = UPLOAD_DIR / f"{dataset_id}{suffix}"; path.write_bytes(await file.read())
    if suffix == ".pdf":
        item = {"id": dataset_id, "name": file.filename, "source_type": "PDF", "path": str(path), "rows": 0, "columns": 0, "status": "INDEXED", "quality_after": None, "uploaded_at": now()}
    else:
        try: df = load_frame(path)
        except Exception as exc: raise HTTPException(422, f"Could not read file: {exc}") from exc
        if df.empty:
            path.unlink(missing_ok=True)
            raise HTTPException(422, "Dataset contains no data rows")
        item = {"id": dataset_id, "name": file.filename, "source_type": suffix[1:].upper(), "path": str(path), "rows": len(df), "columns": len(df.columns), "profile": profile_frame(df), "status": "PROFILED", "quality_before": None, "quality_after": None, "preview": df.head(8).fillna("").to_dict(orient="records"), "uploaded_at": now()}
    DATASETS[dataset_id] = item
    return item

@app.get("/api/datasets")
def datasets(): return list(DATASETS.values())

@app.get("/api/datasets/{dataset_id}")
def dataset(dataset_id: str):
    if dataset_id not in DATASETS: raise HTTPException(404, "Dataset not found")
    return DATASETS[dataset_id]

@app.get("/api/datasets/{dataset_id}/profile")
def profile(dataset_id: str):
    item = DATASETS.get(dataset_id)
    if not item: raise HTTPException(404, "Dataset not found")
    return item.get("profile", {})

@app.get("/api/datasets/{dataset_id}/columns")
def dataset_columns(dataset_id: str):
    item = DATASETS.get(dataset_id)
    if not item: raise HTTPException(404, "Dataset not found")
    profile = item.get("profile", {})
    stats = profile.get("column_stats", [])
    cols = [c["name"] for c in stats]
    return {"columns": cols}

@app.get("/api/datasets/{dataset_id}/download")
def download_processed_dataset(dataset_id: str):
    item = DATASETS.get(dataset_id)
    if not item: raise HTTPException(404, "Dataset not found")
    processed_path = item.get("processed_path")
    if not processed_path: raise HTTPException(409, "Run the pipeline before downloading the processed dataset")
    path = Path(processed_path).resolve()
    if path.parent != PROCESSED_DIR.resolve() or not path.is_file(): raise HTTPException(404, "Processed dataset is not available")
    return FileResponse(path, media_type="text/csv", filename=f"{Path(item['name']).stem}_processed.csv")

@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str):
    if dataset_id not in DATASETS:
        raise HTTPException(404, "Dataset not found")
    item = DATASETS.pop(dataset_id)
    if item.get("path"):
        try:
            Path(item["path"]).unlink(missing_ok=True)
        except Exception:
            pass
    if item.get("processed_path"):
        try:
            Path(item["processed_path"]).unlink(missing_ok=True)
        except Exception:
            pass
    runs_to_remove = [rid for rid, r in RUNS.items() if r.get("dataset_id") == dataset_id]
    for rid in runs_to_remove:
        RUNS.pop(rid, None)
    return {"status": "success", "deleted": dataset_id}

@app.post("/api/pipelines/{dataset_id}/run")
def run_pipeline(dataset_id: str, payload: RunPipelineRequest = None):
    print(f"--> [PIPELINE RUN] dataset_id={dataset_id}, payload={payload}", flush=True)
    if dataset_id not in DATASETS:
        print(f"--> [PIPELINE 404] {dataset_id} not in DATASETS (available: {list(DATASETS.keys())})", flush=True)
        raise HTTPException(404, "Dataset not found")
    model_type = payload.model_type if payload else "generic"
    target_column = payload.target_column if payload else None
    print(f"--> [STARTING EXECUTE_RUN] model_type={model_type}, target_column={target_column}", flush=True)
    try:
        run = execute_run(dataset_id, model_type=model_type, target_column=target_column)
        print(f"--> [PIPELINE SUCCESS] run_id={run.get('id')}", flush=True)
        return run
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"--> [PIPELINE ERROR] {exc}", flush=True)
        raise HTTPException(422, f"Pipeline paused: {exc}") from exc

@app.get("/api/pipelines/{run_id}")
def pipeline(run_id: str):
    if run_id not in RUNS: raise HTTPException(404, "Pipeline run not found")
    return RUNS[run_id]

@app.get("/api/pipelines/{run_id}/logs")
def run_logs(run_id: str):
    run = RUNS.get(run_id)
    if not run: raise HTTPException(404, "Pipeline run not found")
    return run.get("logs", [])

@app.get("/api/agents/logs")
def logs(): return list(reversed(LOGS))

@app.get("/api/dashboard/summary")
def summary():
    completed = [r for r in RUNS.values() if r["status"] == "SUCCESS"]
    scores = [r["after"]["overall"] for r in completed]
    return {"total_datasets": len(DATASETS), "total_records": sum(d.get("rows", 0) for d in DATASETS.values()), "successful_pipelines": len(completed), "failed_pipelines": 0, "average_quality": round(sum(scores) / len(scores), 1) if scores else 0, "recent_runs": list(reversed(completed))[:5]}

@app.post("/api/analytics/query")
def analytics(query: Query):
    question = (query.question or "").strip()
    normalized = question.lower()
    frames = []
    dataset_name = "dataset"

    for item in DATASETS.values():
        path_str = item.get("processed_path") or item.get("path")
        if not path_str:
            continue
        path = Path(path_str)
        if path.exists():
            try:
                frames.append(load_frame(path))
                dataset_name = item.get("name", "dataset")
            except Exception:
                continue

    if not frames:
        return {
            "route": "HYBRID_SQL_RAG",
            "answer": "Run a pipeline or upload a dataset first so there is data to query.",
            "sql": "SELECT * FROM processed_dataset LIMIT 0;",
            "evidence": [],
            "sources": [{"kind": "system", "title": "No dataset available", "snippet": "No processed dataset is available yet."}],
        }

    df = frames[-1].copy()
    numeric_cols = list(df.select_dtypes(include="number").columns)
    dim_cols = [col for col in df.columns if col not in numeric_cols]
    table_name = re.sub(r"[^\w]", "_", dataset_name.split(".")[0]) or "processed_dataset"
    column_map = {str(col).lower(): col for col in df.columns}
    group_col = next((column_map[name] for name in ["region", "country", "category", "segment", "product", "status", "channel", "department", "customer", "company"] if name in column_map), dim_cols[0] if dim_cols else None)

    def to_jsonable(value):
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, (list, tuple)):
            return [to_jsonable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): to_jsonable(v) for k, v in value.items()}
        return str(value)

    rag_context = [
        {
            "kind": "knowledge",
            "title": "Autonomous data engineering workflow",
            "snippet": "The platform profiles raw sources, creates a plan, cleans the dataset, validates the result, stores the processed output, and can benchmark a downstream model against a target column.",
        },
        {
            "kind": "dataset",
            "title": "Processed-data summary",
            "snippet": f"The latest processed dataset has {len(df):,} rows and {len(df.columns)} columns. Numeric columns: {', '.join(map(str, numeric_cols[:5])) if numeric_cols else 'none'}.",
        },
    ]

    def find_column_search(question_text: str):
        q_norm = question_text.lower().replace("_", " ")
        for col in df.columns:
            col_lower = str(col).lower().replace("_", " ")
            patterns = [
                rf"in\s+{re.escape(col_lower)}\s+column\s+for\s+['\"]?([^'\"]+?)['\"]?(?:\s+column)?$",
                rf"in\s+{re.escape(col_lower)}\s+for\s+['\"]?([^'\"]+?)['\"]?$",
                rf"for\s+['\"]?([^'\"]+?)['\"]?\s+in\s+{re.escape(col_lower)}(?:\s+column)?",
                rf"where\s+{re.escape(col_lower)}\s+(?:is|equals|=)\s+['\"]?([^'\"]+?)['\"]?",
                rf"filter\s+by\s+{re.escape(col_lower)}\s+(?:is|to|=)?\s*['\"]?([^'\"]+?)['\"]?",
            ]
            for pattern in patterns:
                match = re.search(pattern, q_norm)
                if match:
                    value = match.group(1).strip().strip("'\"")
                    if value and value not in {"column", "field", "attribute"}:
                        return col, value
            if any(token in q_norm for token in ["search", "find", "lookup", "filter", "show", "where"]):
                for token in ["for", "about", "with", "on"]:
                    if f"{token} " in q_norm:
                        remainder = q_norm.split(f"{token} ", 1)[1]
                        if f" {col_lower}" in remainder:
                            value = remainder.split(f" {col_lower}", 1)[0].strip()
                            if value:
                                return col, value
        return None, None

    # 1. Search by column value
    search_column, search_value = find_column_search(normalized)
    if search_column and search_value:
        search_series = df[str(search_column)].astype(str).str.contains(re.escape(search_value), case=False, na=False)
        filtered = df.loc[search_series].copy()
        if filtered.empty:
            filtered = df.loc[df[str(search_column)].astype(str).str.contains(re.escape(search_value.split()[0]), case=False, na=False)].copy()
        limit = min(len(filtered), 10)
        evidence = filtered.head(limit).fillna("").to_dict(orient="records")
        answer = f"I found {len(filtered):,} matching rows where {search_column} contains \"{search_value}\"."
        sql = f"SELECT * FROM {table_name} WHERE {search_column} LIKE '%{search_value}%' LIMIT {limit};"
        return {
            "route": "HYBRID_SQL_RAG",
            "answer": f"{answer} This is a direct column-level search using the dataset's structured values and the related RAG context.",
            "sql": sql,
            "evidence": [to_jsonable(item) for item in evidence],
            "columns": list(df.columns),
            "sources": rag_context,
        }

    # 2. Grouped comparisons / aggregations (e.g. Which region had the highest total revenue?)
    if (any(token in normalized for token in ["highest", "largest", "max", "maximum", "peak", "most"]) or any(token in normalized for token in ["lowest", "smallest", "minimum", "least"])) and (group_col and (str(group_col).lower() in normalized or any(term in normalized for term in ["which", "by", "per", "grouped", "each"]))):
        metric = next((candidate for candidate in numeric_cols if str(candidate).lower().replace("_", " ") in normalized or any(token in normalized for token in [str(candidate).lower(), str(candidate).lower().replace("_", " ")])), numeric_cols[0] if numeric_cols else None)
        if group_col and metric:
            grouped = df.groupby(group_col, dropna=False)[metric].sum(numeric_only=True).reset_index()
            if any(token in normalized for token in ["lowest", "smallest", "minimum", "least"]):
                row = grouped.nsmallest(1, metric).iloc[0]
                answer = f"{row[group_col]} had the lowest total {metric} at {float(row[metric]):,.2f}."
            else:
                row = grouped.nlargest(1, metric).iloc[0]
                answer = f"{row[group_col]} had the highest total {metric} at {float(row[metric]):,.2f}."
            sql = f"SELECT {group_col}, SUM({metric}) AS total_{metric} FROM {table_name} GROUP BY {group_col} ORDER BY total_{metric} DESC LIMIT 5;"
            evidence = grouped.head(5).to_dict(orient="records")
            return {
                "route": "HYBRID_SQL_RAG",
                "answer": f"{answer} This fits the platform's SQL analytics pattern for grouped comparisons and is consistent with the project's autonomous pipeline style.",
                "sql": sql,
                "evidence": [to_jsonable(item) for item in evidence],
                "columns": list(grouped.columns),
                "sources": rag_context,
            }

    # 3. Bottom N records / Tail / Last rows
    if any(token in normalized for token in ["bottom", "last", "tail"]):
        match = re.search(r"\b(\d+)\b", normalized)
        limit = int(match.group(1)) if match else 5
        limit = min(max(limit, 1), 50)
        rows = df.tail(limit).fillna("").to_dict(orient="records")
        return {
            "route": "HYBRID_SQL_RAG",
            "answer": f"Retrieved bottom {limit} records from '{dataset_name}'.",
            "sql": f"SELECT * FROM {table_name} ORDER BY _rowid_ DESC LIMIT {limit};",
            "evidence": [to_jsonable(item) for item in rows],
            "columns": list(df.columns),
            "sources": rag_context,
        }

    # 4. Top N records / Preview / Head
    if ("record" in normalized and any(t in normalized for t in ["top", "latest", "first", "show", "head"])) or ("show" in normalized and any(t in normalized for t in ["row", "record", "preview"])) or any(t in normalized for t in ["top 5", "top 10", "head rows", "first 5"]):
        match = re.search(r"\b(\d+)\b", normalized)
        limit = int(match.group(1)) if match else 5
        limit = min(max(limit, 1), 50)
        rows = df.head(limit).fillna("").to_dict(orient="records")
        return {
            "route": "HYBRID_SQL_RAG",
            "answer": f"Here are the top {limit} records in the dataset '{dataset_name}'.",
            "sql": f"SELECT * FROM {table_name} LIMIT {limit};",
            "evidence": [to_jsonable(item) for item in rows],
            "columns": list(df.columns),
            "sources": rag_context,
        }

    # 5. Highest / Maximum / Lowest / Minimum single value
    if any(token in normalized for token in ["highest", "largest", "max", "maximum", "peak", "most"]) or any(token in normalized for token in ["lowest", "smallest", "minimum", "least"]):
        metric = next((candidate for candidate in numeric_cols if str(candidate).lower().replace("_", " ") in normalized or any(token in normalized for token in [str(candidate).lower(), str(candidate).lower().replace("_", " ")])), numeric_cols[0] if numeric_cols else None)
        if metric:
            is_min = any(token in normalized for token in ["lowest", "smallest", "minimum", "least"])
            sorted_df = df.sort_values(by=metric, ascending=is_min).head(5)
            best = float(sorted_df[metric].iloc[0])
            answer = f"The {'minimum' if is_min else 'maximum'} {metric} is {best:,.2f}."
            sql = f"SELECT * FROM {table_name} ORDER BY {metric} {'ASC' if is_min else 'DESC'} LIMIT 5;"
            evidence = [to_jsonable(sorted_df.iloc[0].to_dict())]
            return {
                "route": "HYBRID_SQL_RAG",
                "answer": f"{answer} This answer is backed by the structured dataset and also matches the RAG interpretation of the data pipeline workflow.",
                "sql": sql,
                "evidence": evidence,
                "columns": list(df.columns),
                "sources": rag_context,
            }

    # 6. Average of numeric columns / Summary statistics
    if any(token in normalized for token in ["average", "avg", "mean", "numeric", "summary", "statistic"]):
        metric = next((candidate for candidate in numeric_cols if str(candidate).lower().replace("_", " ") in normalized), None)
        if metric:
            avg = float(df[metric].mean())
            answer = f"The average {metric} is {avg:,.2f}."
            sql = f"SELECT AVG({metric}) AS avg_{metric} FROM {table_name};"
            evidence = [{"metric": metric, "average_value": avg}]
            return {
                "route": "HYBRID_SQL_RAG",
                "answer": f"{answer} The project's RAG context confirms this is the standard aggregate metric produced by the autonomous profiling pipeline.",
                "sql": sql,
                "evidence": [to_jsonable(item) for item in evidence],
                "columns": list(evidence[0].keys()) if evidence else [],
                "sources": rag_context,
            }
        elif numeric_cols:
            summary_rows = []
            for col in numeric_cols:
                series = df[col].dropna()
                if not series.empty:
                    summary_rows.append({
                        "metric_column": col,
                        "average_mean": round(float(series.mean()), 2),
                        "minimum_value": round(float(series.min()), 2),
                        "maximum_value": round(float(series.max()), 2),
                        "median_value": round(float(series.median()), 2),
                    })
            cols_sql = ", ".join([f"AVG({col}) AS avg_{col}" for col in numeric_cols[:4]])
            return {
                "route": "HYBRID_SQL_RAG",
                "answer": f"Computed average and distribution metrics across {len(numeric_cols)} numeric fields in '{dataset_name}'.",
                "sql": f"SELECT {cols_sql} FROM {table_name};",
                "evidence": [to_jsonable(item) for item in summary_rows],
                "columns": ["metric_column", "average_mean", "minimum_value", "maximum_value", "median_value"],
                "sources": rag_context,
            }

    # 7. Total count / Total records and columns / Size
    if any(token in normalized for token in ["count", "how many", "number of rows", "total record", "total row", "size", "rows and column"]):
        evidence = [{
            "dataset_name": dataset_name,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns_count": len(numeric_cols),
            "dimension_columns_count": len(dim_cols),
        }]
        return {
            "route": "HYBRID_SQL_RAG",
            "answer": f"The processed dataset contains {len(df):,} rows across {len(df.columns)} columns.",
            "sql": f"SELECT COUNT(*) AS total_rows, {len(df.columns)} AS total_columns FROM {table_name};",
            "evidence": [to_jsonable(item) for item in evidence],
            "columns": list(evidence[0].keys()),
            "sources": rag_context,
        }

    # 8. Fallback default: show top 5 records with dataset overview
    numeric_summary = ", ".join(str(col) for col in numeric_cols[:5]) if numeric_cols else "no numeric metrics"
    answer = f"I found {len(df):,} processed records across {len(df.columns)} columns in '{dataset_name}'. Showing top 5 rows below:"
    evidence = df.head(5).fillna("").to_dict(orient="records")
    return {
        "route": "HYBRID_SQL_RAG",
        "answer": f"{answer} Numeric fields available include: {numeric_summary}.",
        "sql": f"SELECT * FROM {table_name} LIMIT 5;",
        "evidence": [to_jsonable(item) for item in evidence],
        "columns": list(df.columns),
        "sources": rag_context,
    }

@app.post("/api/rag/query")
def rag(query: Query):
    return {"route": "DOCUMENT_RAG", "answer": "Demo mode can index PDF metadata; configure an embedding provider for grounded document answers.", "sources": []}
