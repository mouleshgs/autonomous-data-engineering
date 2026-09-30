from __future__ import annotations

from uuid import uuid4

from app.schemas.pipeline_state import PipelineStage, PipelineState


class BaseAgent:
    name = "Agent"

    def __init__(self, name: str | None = None) -> None:
        if name:
            self.name = name

    def log(
        self,
        state: PipelineState,
        action: str,
        reason: str,
        tool: str,
        status: str = "SUCCESS",
        output: str = "",
    ) -> None:
        state.logs.append(
            {
                "id": str(uuid4()),
                "agent": self.name,
                "action": action,
                "reason": reason,
                "tool": tool,
                "status": status,
                "output": output,
                "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            }
        )

    def add_stage(self, state: PipelineState, name: str, detail: str, status: str = "SUCCESS") -> None:
        state.stages.append(PipelineStage(name=name, status=status, detail=detail))
