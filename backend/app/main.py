from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .core import DATASETS, LOGS, PROCESSED_DIR, RUNS, UPLOAD_DIR, execute_run, load_frame, now, profile_frame

app = FastAPI(title="Autonomous Data Engineering Platform", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])

class Query(BaseModel):
    question: str

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
def run_pipeline(dataset_id: str):
    if dataset_id not in DATASETS: raise HTTPException(404, "Dataset not found")
    try: return execute_run(dataset_id)
    except Exception as exc: raise HTTPException(422, f"Pipeline paused: {exc}") from exc

@app.get("/api/pipelines/{run_id}")
def pipeline(run_id: str):
    if run_id not in RUNS: raise HTTPException(404, "Pipeline run not found")
    return RUNS[run_id]

@app.get("/api/pipelines/{run_id}/logs")
def run_logs(run_id: str):
    run = RUNS.get(run_id)
    if not run: raise HTTPException(404, "Pipeline run not found")
    return LOGS

@app.get("/api/agents/logs")
def logs(): return list(reversed(LOGS))

@app.get("/api/dashboard/summary")
def summary():
    completed = [r for r in RUNS.values() if r["status"] == "SUCCESS"]
    scores = [r["after"]["overall"] for r in completed]
    return {"total_datasets": len(DATASETS), "total_records": sum(d.get("rows", 0) for d in DATASETS.values()), "successful_pipelines": len(completed), "failed_pipelines": 0, "average_quality": round(sum(scores) / len(scores), 1) if scores else 0, "recent_runs": list(reversed(completed))[:5]}

@app.post("/api/analytics/query")
def analytics(query: Query):
    question = query.question.lower(); frames = []
    for item in DATASETS.values():
        if item.get("processed_path"): frames.append(load_frame(Path(item["processed_path"])))
    if not frames: return {"route": "STRUCTURED_DATA", "answer": "Run a pipeline first so there is processed data to query.", "evidence": []}
    df = frames[-1]
    if "highest" in question or "top" in question:
        numeric = df.select_dtypes(include="number"); answer = f"The dataset contains {len(df):,} processed records. Numeric fields available for ranking: {', '.join(numeric.columns)}."
    else: answer = f"I found {len(df):,} processed records across {len(df.columns)} columns. Ask about a specific metric, region, or product."
    return {"route": "STRUCTURED_DATA", "answer": answer, "sql": "SELECT * FROM processed_dataset LIMIT 8", "evidence": df.head(8).fillna("").to_dict(orient="records")}

@app.post("/api/rag/query")
def rag(query: Query):
    return {"route": "DOCUMENT_RAG", "answer": "Demo mode can index PDF metadata; configure an embedding provider for grounded document answers.", "sources": []}
