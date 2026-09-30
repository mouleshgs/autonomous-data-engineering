from pathlib import Path

import pandas as pd

from app.agents.data_cleaning import DataCleaningAgent
from app.agents.data_profiler import DataProfilerAgent
from app.agents.pipeline_planner import PipelinePlannerAgent
from app.agents.source_analyzer import SourceAnalyzerAgent
from app.agents.transformation import TransformationAgent
from app.schemas.pipeline_state import PipelineState


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

    assert len(state.plan) >= 3
    assert state.status == "SUCCESS"
    assert state.processed_path is not None
    assert Path(state.processed_path).exists()
