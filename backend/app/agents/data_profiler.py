from __future__ import annotations

from app.agents.base import BaseAgent
from app.schemas.pipeline_state import PipelineState
from app.tools.data_tools import load_frame, profile_frame, quality


class DataProfilerAgent(BaseAgent):
    name = "Profiler Agent"

    def run(self, state: PipelineState) -> PipelineState:
        dataframe = state.dataframe
        if dataframe is None and state.source_path:
            dataframe = load_frame(state.source_path)
        if dataframe is None:
            raise ValueError("No dataframe available for profiling")

        profile = profile_frame(dataframe)
        state.dataframe = dataframe
        state.profile = profile
        state.before_quality = quality(dataframe)

        self.log(
            state,
            "Profiled dataset",
            f"Found {profile['missing_values']} missing values and {profile['duplicates']} duplicates",
            "profile_frame",
            output=str(state.before_quality),
        )
        self.add_stage(state, self.name, "Generated column-level profile")
        return state
