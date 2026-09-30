from pathlib import Path

import pandas as pd
import app.graph.agent_graph as agent_graph
import app.core as core
from app.tools import data_tools
from app.core import plan_for, profile_frame, quality
from app.tools.data_tools import correct_negative_values, fill_missing_values, normalize_strings, remove_duplicates, standardize_dates
from fastapi.testclient import TestClient
from app.main import app

def test_profile_and_adaptive_plan():
    frame = pd.DataFrame({"price": [10, None, 10], "region": [" india ", "US", "US"]})
    profile = profile_frame(frame)
    assert profile["missing_values"] == 1
    assert profile["duplicates"] == 0
    assert {step["operation"] for step in plan_for(profile)} >= {"fill_missing", "normalize_strings", "validate"}

def test_quality_improves_after_cleaning():
    before = quality(pd.DataFrame({"value": [1, None, 1]}))
    after = quality(pd.DataFrame({"value": [1, 2]}))
    assert after["overall"] > before["overall"]


def test_planner_flags_negative_values_stored_as_text():
    frame = pd.DataFrame({"adjustment": ["$-4.00", "$8.00"]})
    profile = profile_frame(frame)
    operations = {step["operation"] for step in plan_for(profile)}

    assert "coerce_numeric_columns" in operations
    assert "correct_negative_values" in operations


def test_empty_frame_profile_is_json_safe_and_upload_is_rejected():
    profile = profile_frame(pd.DataFrame(columns=["name"]))
    assert profile["column_stats"][0]["null_pct"] == 0.0

    response = TestClient(app).post(
        "/api/datasets/upload",
        files={"file": ("empty.csv", b"name\n", "text/csv")},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Dataset contains no data rows"


def test_sample_sales_pipeline_normalizes_data():
    sample_path = Path(__file__).resolve().parents[2] / "data" / "sample_sales.csv"
    frame = pd.read_csv(sample_path)
    plan = plan_for(profile_frame(frame))

    assert "standardize_dates" in {step["operation"] for step in plan}

    cleaned = remove_duplicates(frame)
    cleaned = fill_missing_values(cleaned)
    cleaned = normalize_strings(cleaned)
    cleaned = standardize_dates(cleaned)

    assert len(frame) == 6
    assert len(cleaned) == 5
    assert cleaned["price"].isna().sum() == 0
    assert cleaned.loc[cleaned["order_id"] == 1002, "price"].item() == 13.75
    assert cleaned["region"].tolist().count("US") == 2
    assert cleaned["order_date"].tolist() == [
        "2025-01-05",
        "2025-01-06",
        "2025-01-07",
        "2025-01-08",
        "2025-01-09",
    ]


def test_messy_sales_graph_corrects_negative_quantity(tmp_path, monkeypatch):
    sample_path = Path(__file__).resolve().parents[2] / "data" / "sample_sales_all_tools.csv"
    monkeypatch.setattr(agent_graph, "PROCESSED_DIR", tmp_path)
    dataset = {
        "name": "sample_sales_messy.csv",
        "source_type": "CSV",
        "path": str(sample_path.resolve()),
    }

    run = agent_graph.run_langgraph_pipeline("negative-quantity-regression", dataset)
    processed = pd.read_csv(dataset["processed_path"])
    operations = {step["operation"] for step in run["plan"]}

    assert run["status"] == "SUCCESS"
    assert {
        "remove_empty_rows",
        "remove_empty_columns",
        "remove_duplicates",
        "coerce_numeric_columns",
        "normalize_boolean_columns",
        "fill_missing",
        "correct_negative_values",
        "normalize_strings",
        "standardize_dates",
        "standardize_column_names",
        "validate",
        "store",
    }.issubset(operations)
    assert processed["quantity"].ge(0).all()
    assert processed.loc[processed["order_id"] == "D-2004", "quantity"].item() == 0
    assert len(processed) == 9
    assert "unused_notes" not in processed.columns
    assert processed.isna().sum().sum() == 0
    assert not processed.duplicated().any()
    assert processed["review_score"].dtype.kind in "fi"
    assert set(processed["approved"]) == {"Yes", "No"}
    assert all(name == name.lower() for name in processed.columns)
    assert run["after"]["validity"] == 100.0
    validation_log = next(log for log in run["logs"] if log["agent"] == "Validation Agent")
    assert "'issues': []" in validation_log["output"]


def test_ollama_plan_cannot_omit_required_negative_correction(monkeypatch):
    class FakeResponse:
        content = '[{"operation":"validate","reason":"Check the result"}]'

    class FakeChatOllama:
        def __init__(self, **_kwargs):
            pass

        def invoke(self, _messages):
            return FakeResponse()

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setattr(data_tools, "ChatOllama", FakeChatOllama)
    profile = {
        "duplicates": 0,
        "missing_values": 0,
        "column_stats": [{"name": "quantity", "dtype": "int64", "min": -2, "max": 8}],
    }

    plan = data_tools.plan_with_llm(profile)

    assert "correct_negative_values" in {step["operation"] for step in plan}
    assert "validate" in {step["operation"] for step in plan}


def test_execute_run_persists_graph_decisions(monkeypatch):
    dataset_id = "logging-regression"
    decision = {
        "agent": "Planner Agent",
        "action": "Generated adaptive plan",
        "reason": "Found missing values",
        "tool": "plan_with_llm",
        "status": "SUCCESS",
        "output": "fill_missing",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    monkeypatch.setattr(core, "LOGS", [])
    monkeypatch.setattr(core, "RUNS", {})
    monkeypatch.setitem(core.DATASETS, dataset_id, {"name": "sample.csv"})
    monkeypatch.setattr(
        core,
        "run_langgraph_pipeline",
        lambda _dataset_id, _dataset: {"logs": [decision], "status": "SUCCESS"},
    )

    run = core.execute_run(dataset_id)

    assert run["logs"][0]["agent"] == "Planner Agent"
    assert core.LOGS[0]["run_id"] == run["id"]
    assert core.LOGS[0]["dataset_id"] == dataset_id
