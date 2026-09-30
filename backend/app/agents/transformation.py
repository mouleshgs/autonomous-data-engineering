from __future__ import annotations

from pathlib import Path

from app.agents.base import BaseAgent
from app.schemas.pipeline_state import PipelineState
from app.tools.data_tools import PROCESSED_DIR, persist_dataframe, quality


class TransformationAgent(BaseAgent):
    name = "Transformation Agent"

    def run(self, state: PipelineState) -> PipelineState:
        frame = state.dataframe.copy()
        state.after_quality = quality(frame)
        output_path = Path(PROCESSED_DIR) / f"{state.dataset_id}.csv"
        state.processed_path = persist_dataframe(frame, output_path)
        state.status = "SUCCESS"

        self.log(
            state,
            "Prepared final dataset",
            f"Saved {len(frame)} rows to processed output",
            "save_processed_csv",
            output=state.processed_path,
        )
        self.add_stage(state, "Ingestion", "Loaded source into internal dataframe")
        self.add_stage(state, "Validation", "Schema retained; deterministic quality checks completed")
        self.add_stage(state, "Storage", f"Processed dataset stored: {state.processed_path}")
        return state
