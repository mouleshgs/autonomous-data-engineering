from __future__ import annotations

from app.agents.base import BaseAgent
from app.schemas.pipeline_state import PipelineState
from app.tools.data_tools import validate_dataframe


class ValidationAgent(BaseAgent):
    name = "Validation Agent"

    def run(self, state: PipelineState) -> PipelineState:
        if state.dataframe is None:
            raise ValueError("No dataframe available for validation")

        validation = validate_dataframe(state.dataframe)
        state.after_quality = validation["quality"]
        state.status = "SUCCESS"

        self.log(
            state,
            "Validated pipeline output",
            "Checked schema, nulls, duplicates, and negative values",
            "validate_dataframe",
            status="SUCCESS" if validation["validated"] else "WARNING",
            output=str(validation),
        )
        self.add_stage(
            state,
            "Validation",
            "Quality checks passed" if validation["validated"] else f"Issues remain: {', '.join(validation['issues'])}",
            status="SUCCESS",
        )
        return state
