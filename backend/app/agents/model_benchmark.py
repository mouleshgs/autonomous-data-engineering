from __future__ import annotations

from app.agents.base import BaseAgent
from app.schemas.pipeline_state import PipelineState
from app.tools.data_tools import evaluate_downstream_model


class ModelBenchmarkAgent(BaseAgent):
    name = "Model Benchmark Agent"

    def run(self, state: PipelineState) -> PipelineState:
        if state.dataframe is None:
            raise ValueError("No dataframe available for benchmarking")

        if state.model_type == "generic" or not state.target_column:
            self.add_stage(state, self.name, "Skipped benchmark because no target model was selected")
            return state

        benchmark = evaluate_downstream_model(
            state.dataframe,
            target_col=state.target_column,
            model_type=state.model_type,
        )
        state.model_benchmark = benchmark

        self.log(
            state,
            "Evaluated downstream model",
            f"Trained {benchmark['model_name']} on target '{benchmark['target_column']}'",
            "evaluate_downstream_model",
            output=f"Baseline: {benchmark['baseline_score']}% | Model-Aware: {benchmark['model_aware_score']}% | Research Lift: +{benchmark['lift']}%",
        )
        self.add_stage(
            state,
            "Model Benchmark",
            f"{benchmark['model_name']} ({benchmark['metric_name']}): {benchmark['model_aware_score']}% (+{benchmark['lift']}% lift)",
        )
        return state
