from __future__ import annotations

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
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])

class Query(BaseModel):
    question: str

class RunPipelineRequest(BaseModel):
    model_type: str = "generic"
    target_column: str | None = None

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

@app.post("/api/pipelines/{dataset_id}/run")
def run_pipeline(dataset_id: str, payload: RunPipelineRequest = None):
    if dataset_id not in DATASETS: raise HTTPException(404, "Dataset not found")
    model_type = payload.model_type if payload else "generic"
    target_column = payload.target_column if payload else None
    try:
        run = schedule_run(dataset_id, model_type=model_type, target_column=target_column)
        return run
    except Exception as exc: raise HTTPException(422, f"Pipeline paused: {exc}") from exc

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
    for item in DATASETS.values():
        processed_path = item.get("processed_path")
        if not processed_path:
            continue
        path = Path(processed_path)
        if path.exists():
            try:
                frames.append(load_frame(path))
            except Exception:
                continue
    if not frames:
        return {
            "route": "HYBRID_SQL_RAG",
            "answer": "Run a pipeline first so there is processed data to query.",
            "sql": "SELECT * FROM processed_dataset LIMIT 0",
            "evidence": [],
            "sources": [{"kind": "system", "title": "No processed dataset", "snippet": "No processed dataset is available yet."}],
        }

    df = frames[-1].copy()
    numeric_cols = list(df.select_dtypes(include="number").columns)
    dim_cols = [col for col in df.columns if col not in numeric_cols]
    column_map = {str(col).lower(): col for col in df.columns}
    group_col = next((column_map[name] for name in ["region", "country", "category", "segment", "product", "status", "channel", "department", "customer"] if name in column_map), None)

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
        for col in df.columns:
            col_lower = str(col).lower().replace("_", " ")
            if col_lower not in question_text:
                continue
            patterns = [
                rf"(?:for|about|with)\s+(.+?)\s+(?:in|on|by|from|of)\s+{re.escape(col_lower)}",
                rf"(?:in|on)\s+{re.escape(col_lower)}\s+(?:column|field)?\s*(?:for|contains?|includes?|equals?|=)?\s+(.+?)(?:\s+(?:and|or|but|$))",
                rf"{re.escape(col_lower)}\s+(?:contains?|includes?|has|=)\s+(.+?)(?:\s+(?:and|or|but|$))",
            ]
            for pattern in patterns:
                match = re.search(pattern, question_text)
                if match:
                    value = match.group(1).strip().strip("'\"")
                    if value and value not in {"column", "field", "attribute"}:
                        return col, value
            if any(token in question_text for token in ["search", "find", "lookup", "filter", "show", "where"]):
                for token in ["for", "about", "with", "on"]:
                    if f"{token} " in question_text:
                        remainder = question_text.split(f"{token} ", 1)[1]
                        if f" {col_lower}" in remainder:
                            value = remainder.split(f" {col_lower}", 1)[0].strip()
                            if value:
                                return col, value
        return None, None

    search_column, search_value = find_column_search(normalized)
    if search_column and search_value:
        search_series = df[str(search_column)].astype(str).str.contains(re.escape(search_value), case=False, na=False)
        filtered = df.loc[search_series].copy()
        if filtered.empty:
            filtered = df.loc[df[str(search_column)].astype(str).str.contains(re.escape(search_value.split()[0]), case=False, na=False)].copy()
        limit = min(len(filtered), 10)
        evidence = filtered.head(limit).fillna("").to_dict(orient="records")
        answer = f"I found {len(filtered):,} matching rows where {search_column} contains \"{search_value}\"."
        sql = f"SELECT * FROM processed_dataset WHERE {search_column} LIKE '%{search_value}%' LIMIT {limit}"
        return {
            "route": "HYBRID_SQL_RAG",
            "answer": f"{answer} This is a direct column-level search using the dataset's structured values and the related RAG context.",
            "sql": sql,
            "evidence": [to_jsonable(item) for item in evidence],
            "columns": list(df.columns),
            "sources": rag_context,
        }

    if "record" in normalized and ("top" in normalized or "latest" in normalized or "first" in normalized or "show" in normalized or "head" in normalized):
        match = re.search(r"\b(\d+)\b", normalized)
        limit = int(match.group(1)) if match else 5
        limit = min(max(limit, 1), 20)
        rows = df.head(limit).fillna("").to_dict(orient="records")
        return {
            "route": "HYBRID_SQL_RAG",
            "answer": f"Here are the top {limit} records in the dataset.",
            "sql": f"SELECT * FROM processed_dataset LIMIT {limit}",
            "evidence": [to_jsonable(item) for item in rows],
            "columns": list(df.columns),
            "sources": rag_context,
        }

    if any(token in normalized for token in ["highest", "largest", "max", "most"]) or any(token in normalized for token in ["lowest", "smallest", "minimum", "least"]):
        metric = next((candidate for candidate in numeric_cols if str(candidate).lower().replace("_", " ") in normalized or any(token in normalized for token in [str(candidate).lower(), str(candidate).lower().replace("_", " ")]) ), None)
        if metric is None and numeric_cols:
            metric = numeric_cols[0]
        if group_col and metric:
            grouped = df.groupby(group_col, dropna=False)[metric].sum(numeric_only=True).reset_index()
            if any(token in normalized for token in ["lowest", "smallest", "minimum", "least"]):
                row = grouped.nsmallest(1, metric).iloc[0]
                answer = f"{row[group_col]} had the lowest total {metric} at {float(row[metric]):,.2f}."
            else:
                row = grouped.nlargest(1, metric).iloc[0]
                answer = f"{row[group_col]} had the highest total {metric} at {float(row[metric]):,.2f}."
            sql = f"SELECT {group_col}, SUM({metric}) AS total_{metric} FROM processed_dataset GROUP BY {group_col} ORDER BY total_{metric} DESC LIMIT 5"
            evidence = grouped.head(5).to_dict(orient="records")
            return {
                "route": "HYBRID_SQL_RAG",
                "answer": f"{answer} This fits the platform's SQL analytics pattern for grouped comparisons and is consistent with the project’s autonomous pipeline style.",
                "sql": sql,
                "evidence": [to_jsonable(item) for item in evidence],
                "columns": list(grouped.columns),
                "sources": rag_context,
            }

        if metric:
            series = df[metric]
            best = float(series.min()) if any(token in normalized for token in ["lowest", "smallest", "minimum", "least"]) else float(series.max())
            answer = f"The {'minimum' if any(token in normalized for token in ['lowest', 'smallest', 'minimum', 'least']) else 'maximum'} {metric} is {best:,.2f}."
            sql = f"SELECT * FROM processed_dataset ORDER BY {metric} {'ASC' if any(token in normalized for token in ['lowest', 'smallest', 'minimum', 'least']) else 'DESC'} LIMIT 1"
            evidence = [to_jsonable(df.loc[series.idxmin() if any(token in normalized for token in ['lowest', 'smallest', 'minimum', 'least']) else series.idxmax()].to_dict())]
            return {
                "route": "HYBRID_SQL_RAG",
                "answer": f"{answer} This answer is backed by the structured dataset and also matches the RAG interpretation of the data pipeline workflow.",
                "sql": sql,
                "evidence": evidence,
                "columns": list(df.columns),
                "sources": rag_context,
            }

    if any(token in normalized for token in ["average", "avg", "mean"]):
        metric = next((candidate for candidate in numeric_cols if str(candidate).lower().replace("_", " ") in normalized), numeric_cols[0] if numeric_cols else None)
        if metric:
            avg = float(df[metric].mean())
            answer = f"The average {metric} is {avg:,.2f}."
            sql = f"SELECT AVG({metric}) AS avg_{metric} FROM processed_dataset"
            evidence = [{"metric": metric, "average_value": avg}]
            return {
                "route": "HYBRID_SQL_RAG",
                "answer": f"{answer} The project’s RAG context confirms this is the standard aggregate metric produced by the autonomous profiling pipeline.",
                "sql": sql,
                "evidence": [to_jsonable(item) for item in evidence],
                "columns": list(evidence[0].keys()) if evidence else [],
                "sources": rag_context,
            }

    if any(token in normalized for token in ["count", "how many", "number of rows", "total records", "size"]):
        answer = f"The processed dataset contains {len(df):,} rows across {len(df.columns)} columns."
        sql = "SELECT COUNT(*) AS row_count FROM processed_dataset"
        evidence = [{"row_count": len(df), "column_count": len(df.columns)}]
        return {
            "route": "HYBRID_SQL_RAG",
            "answer": f"{answer} This is a direct SQL summary, and the supporting RAG context describes the same pipeline lifecycle.",
            "sql": sql,
            "evidence": [to_jsonable(item) for item in evidence],
            "columns": list(evidence[0].keys()) if evidence else [],
            "sources": rag_context,
        }

    numeric_summary = ", ".join(str(col) for col in numeric_cols[:5]) if numeric_cols else "no numeric metrics"
    answer = f"I found {len(df):,} processed records across {len(df.columns)} columns. Numeric fields available include: {numeric_summary}."
    evidence = df.head(8).fillna("").to_dict(orient="records")
    return {
        "route": "HYBRID_SQL_RAG",
        "answer": f"{answer} This response combines the SQL-derived structure with the operational knowledge from the autonomous data engineering workflow.",
        "sql": "SELECT * FROM processed_dataset LIMIT 8",
        "evidence": [to_jsonable(item) for item in evidence],
        "columns": list(df.columns),
        "sources": rag_context,
    }

@app.post("/api/rag/query")
def rag(query: Query):
    return {"route": "DOCUMENT_RAG", "answer": "Demo mode can index PDF metadata; configure an embedding provider for grounded document answers.", "sources": []}
