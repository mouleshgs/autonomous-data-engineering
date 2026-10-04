from pathlib import Path

import pandas as pd

from app.agents.data_cleaning import DataCleaningAgent
from app.agents.data_profiler import DataProfilerAgent
from app.agents.pipeline_planner import PipelinePlannerAgent
from app.agents.source_analyzer import SourceAnalyzerAgent
from app.agents.transformation import TransformationAgent
from app.agents.validation import ValidationAgent
from app.agents.storage import StorageAgent
from app.agents.model_benchmark import ModelBenchmarkAgent
from app.schemas.pipeline_state import PipelineState
from app.main import app
from fastapi.testclient import TestClient
import app.core as core


def test_agent_pipeline_components_are_present():
    state = PipelineState(
        dataset_id="demo-123",
        dataset_name="demo.csv",
        source_type="CSV",
        source_path="/tmp/demo.csv",
    )

    source = SourceAnalyzerAgent()
    source.run(state, dataframe=pd.DataFrame({"region": [" india ", "US", "US"], "value": [10, None, 10]}))

    profiler = DataProfilerAgent()
    profiler.run(state)

    planner = PipelinePlannerAgent()
    planner.run(state)

    cleaning = DataCleaningAgent()
    cleaning.run(state)

    transformed = TransformationAgent()
    transformed.run(state)

    validation = ValidationAgent()
    validation.run(state)

    storage = StorageAgent()
    storage.run(state)

    benchmark = ModelBenchmarkAgent()
    state.model_type = "logistic_regression"
    state.target_column = "value"
    benchmark.run(state)

    assert len(state.plan) >= 3
    assert state.status == "SUCCESS"
    assert state.processed_path is not None
    assert Path(state.processed_path).exists()
    assert any(stage.name == "Validation" for stage in state.stages)
    assert any(stage.name == "Storage" for stage in state.stages)
    assert any(stage.name == "Model Benchmark" for stage in state.stages)


def test_analytics_query_detects_summary_questions_for_processed_data(monkeypatch, tmp_path):
    csv_path = tmp_path / "processed.csv"
    pd.DataFrame(
        {
            "region": ["us", "us", "eu"],
            "revenue": [120, 80, 200],
        }
    ).to_csv(csv_path, index=False)

    monkeypatch.setitem(
        core.DATASETS,
        "dataset-analytics",
        {
            "id": "dataset-analytics",
            "name": "sales.csv",
            "processed_path": str(csv_path),
        },
    )

    response = TestClient(app).post(
        "/api/analytics/query",
        json={"question": "Which region had the highest total revenue?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "HYBRID_SQL_RAG"
    assert "us" in body["answer"].lower() or "eu" in body["answer"].lower()
    assert body["sql"]
    assert len(body["evidence"]) >= 1


def test_model_benchmark_auto_selects_target_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "RUNS", {})
    monkeypatch.setattr(core, "LOGS", [])
    pd.DataFrame(
        {
            "status": ["approved", "rejected", "approved"],
            "amount": [120, 50, 150],
            "region": ["us", "eu", "us"],
        }
    ).to_csv(tmp_path / "auto.csv", index=False)

    run = core.run_langgraph_pipeline(
        "auto-benchmark",
        {"name": "auto.csv", "source_type": "CSV", "path": str(tmp_path / "auto.csv")},
        model_type="logistic_regression",
        target_column=None,
    )

    assert run["model_benchmark"] is not None
    assert run["model_benchmark"]["model_type"] == "logistic_regression"
    assert any(stage["name"] == "Model Benchmark" for stage in run["stages"])


def test_analytics_top_records_question_returns_rows_in_table_form(monkeypatch, tmp_path):
    csv_path = tmp_path / "top_records.csv"
    pd.DataFrame(
        {
            "region": ["us", "eu", "us", "apac"],
            "revenue": [120, 200, 80, 150],
        }
    ).to_csv(csv_path, index=False)

    monkeypatch.setitem(
        core.DATASETS,
        "dataset-top-records",
        {
            "id": "dataset-top-records",
            "name": "top_records.csv",
            "processed_path": str(csv_path),
        },
    )

    response = TestClient(app).post(
        "/api/analytics/query",
        json={"question": "Show me the top 5 records in the dataset."},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["route"] == "HYBRID_SQL_RAG"
    assert len(body["evidence"]) >= 1
    assert body["sql"]
    assert "region" in str(body["evidence"][0]).lower() or "revenue" in str(body["evidence"][0]).lower()


def test_analytics_search_query_filters_by_column_value(monkeypatch, tmp_path):
    csv_path = tmp_path / "company_lookup.csv"
    pd.DataFrame(
        {
            "company": ["Apple", "Acer", "HP", "Dell"],
            "revenue": [120, 80, 200, 90],
        }
    ).to_csv(csv_path, index=False)

    monkeypatch.setitem(
        core.DATASETS,
        "dataset-company-search",
        {
            "id": "dataset-company-search",
            "name": "company_lookup.csv",
            "processed_path": str(csv_path),
        },
    )

    response = TestClient(app).post(
        "/api/analytics/query",
        json={"question": "search for hp in company column"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "HYBRID_SQL_RAG"
    assert len(body["evidence"]) >= 1
    assert any("hp" in str(row).lower() for row in [str(body["evidence"][0]), str(body["evidence"][0]).lower()])
    assert "like" in body["sql"].lower() or "where" in body["sql"].lower()
