import pandas as pd
from fastapi.testclient import TestClient

from app import core
from app.main import app
from app.tools.rag_engine import HybridRAGEngine, build_knowledge_base, retrieve_relevant_sources


def test_rag_knowledge_base_and_retrieval(tmp_path):
    df = pd.DataFrame({
        "product": ["Laptop", "Mouse", "Keyboard", "Monitor"],
        "price": [1200.0, 25.0, 75.0, 300.0],
        "category": ["Computers", "Accessories", "Accessories", "Displays"],
    })
    meta = {"name": "tech_store.csv", "id": "tech-1"}
    run_meta = {
        "model_type": "logistic_regression",
        "before": {"overall": 70},
        "after": {"overall": 95},
        "model_benchmark": {
            "model_name": "Logistic Regression",
            "target_column": "category",
            "baseline_score": 72.0,
            "model_aware_score": 85.5,
            "lift": 13.5,
        },
        "plan": [
            {"operation": "clip_outliers", "reason": "Bound leverage points"},
            {"operation": "standard_scaler", "reason": "Zero-mean unit-variance for gradient descent"},
        ],
    }

    kb = build_knowledge_base(df, meta, run_meta=run_meta)
    assert len(kb) >= 4

    # Test retrieval for schema
    sources = retrieve_relevant_sources("what are the price and products?", kb, top_k=2)
    assert len(sources) == 2
    assert any("price" in s["title"].lower() or "price" in s["snippet"].lower() for s in sources)

    # Test retrieval for policy / optimization
    sources_policy = retrieve_relevant_sources("why outlier clipping and standard scaling?", kb, top_k=2)
    assert any("policy" in s["title"].lower() or "tma" in s["title"].lower() or "lineage" in s["title"].lower() for s in sources_policy)


def test_rag_analytics_custom_questions(monkeypatch, tmp_path):
    csv_path = tmp_path / "sales_rag.csv"
    pd.DataFrame({
        "region": ["North", "South", "East", "West"],
        "sales": [5000, 8000, 3000, 6500],
        "rating": [4.5, 4.8, 3.9, 4.2],
    }).to_csv(csv_path, index=False)

    monkeypatch.setitem(
        core.DATASETS,
        "dataset-rag-custom",
        {
            "id": "dataset-rag-custom",
            "name": "sales_rag.csv",
            "processed_path": str(csv_path),
        },
    )

    client = TestClient(app)

    # 1. Ask about columns / schema
    r_schema = client.post("/api/analytics/query", json={"question": "What are the columns in this dataset?"})
    assert r_schema.status_code == 200
    b_schema = r_schema.json()
    assert b_schema["route"] == "HYBRID_SQL_RAG"
    assert "region" in str(b_schema["answer"]).lower() or "sales" in str(b_schema["answer"]).lower()
    assert len(b_schema["sources"]) >= 1

    # 2. Ask about highest sales
    r_high = client.post("/api/analytics/query", json={"question": "Which region had the highest total sales?"})
    assert r_high.status_code == 200
    b_high = r_high.json()
    assert "south" in b_high["answer"].lower()
    assert len(b_high["evidence"]) >= 1

    # 3. Ask about average rating
    r_avg = client.post("/api/analytics/query", json={"question": "Average rating"})
    assert r_avg.status_code == 200
    b_avg = r_avg.json()
    assert "4." in b_avg["answer"] or "rating" in b_avg["answer"].lower()
