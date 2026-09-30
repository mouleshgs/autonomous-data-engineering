from __future__ import annotations

from app.agents.base import BaseAgent
from app.schemas.pipeline_state import PipelineState
from app.tools.data_tools import fill_missing_values, normalize_strings, remove_duplicates, validate_dataframe


class DataCleaningAgent(BaseAgent):
    name = "Cleaning Agent"

    def run(self, state: PipelineState) -> PipelineState:
        frame = state.dataframe.copy()
        applied = []
        for step in state.plan:
            operation = step["operation"]
            if operation == "remove_duplicates":
                frame = remove_duplicates(frame)
                applied.append("remove_duplicates")
                self.log(state, operation, step["reason"], "drop_duplicates", output=str(len(frame)))
            elif operation == "fill_missing":
                frame = fill_missing_values(frame)
                applied.append("fill_missing")
                self.log(state, operation, step["reason"], "fill_missing_values", output=str(frame.isna().sum().sum()))
            elif operation == "normalize_strings":
                frame = normalize_strings(frame)
                applied.append("normalize_strings")
                self.log(state, operation, step["reason"], "normalize_strings", output=str(frame.shape))
            elif operation == "validate":
                validation = validate_dataframe(frame)
                self.log(state, operation, step["reason"], "quality", output=str(validation["quality"]))
                state.after_quality = validation["quality"]
                applied.append("validate")

        state.dataframe = frame
        self.add_stage(state, "Cleaning", "; ".join(applied) if applied else "No clean-up steps needed")
        return state
