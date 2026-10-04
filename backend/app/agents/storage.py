from __future__ import annotations

from pathlib import Path

from app.agents.base import BaseAgent
from app.schemas.pipeline_state import PipelineState
from app.tools.data_tools import PROCESSED_DIR, persist_dataframe


class StorageAgent(BaseAgent):
    name = "Storage Agent"

    def run(self, state: PipelineState) -> PipelineState:
        if state.dataframe is None:
            raise ValueError("No dataframe available for storage")

        output_path = Path(PROCESSED_DIR) / f"{state.dataset_id}.csv"
        state.processed_path = persist_dataframe(state.dataframe, output_path)

        self.log(
            state,
            "Persisted processed dataset",
            f"Saved {len(state.dataframe)} rows to processed output",
            "persist_dataframe",
            output=state.processed_path,
        )
        self.add_stage(state, "Storage", f"Stored processed dataset: {state.processed_path}")
        return state
