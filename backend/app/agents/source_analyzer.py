from __future__ import annotations

from pathlib import Path

from app.agents.base import BaseAgent
from app.schemas.pipeline_state import PipelineState
from app.tools.data_tools import load_frame


class SourceAnalyzerAgent(BaseAgent):
    name = "Source Analyzer"

    def run(self, state: PipelineState, dataframe=None, file_path: str | None = None) -> PipelineState:
        source_path = file_path or state.source_path
        if not source_path:
            raise ValueError("Source path is required for analysis")

        if dataframe is None:
            dataframe = load_frame(source_path)

        state.dataframe = dataframe
        state.rows = len(dataframe)
        state.columns = len(dataframe.columns)
        state.source_type = state.source_type or Path(source_path).suffix.upper().replace(".", "")

        self.log(
            state,
            "Inspected uploaded source",
            f"Detected {state.source_type} with {state.rows} rows",
            "load_frame",
            output=str(source_path),
        )
        self.add_stage(state, self.name, f"Read {state.source_type} source")
        return state
