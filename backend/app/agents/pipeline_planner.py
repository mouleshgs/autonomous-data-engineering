from __future__ import annotations

from app.agents.base import BaseAgent
from app.schemas.pipeline_state import PipelineState
from app.tools.data_tools import build_plan, plan_with_llm, profile_frame


class PipelinePlannerAgent(BaseAgent):
    name = "Planner Agent"

    def run(self, state: PipelineState) -> PipelineState:
        if not state.profile:
            state.profile = profile_frame(state.dataframe)
        plan = plan_with_llm(state.profile)
        state.plan = plan

        self.log(
            state,
            "Generated adaptive plan",
            f"Selected {len(plan)} safe operations from observed issues",
            "plan_with_llm",
            output=str(plan),
        )
        self.add_stage(state, self.name, f"Selected {len(plan)} safe operations")
        return state
