from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional
from .agent_base import AgentResult

class RuntimeStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_HUMAN = "waiting_human"
    MAX_ITERATIONS = "max_iterations"

@dataclass
class RuntimeState:
    status: RuntimeStatus = RuntimeStatus.RUNNING
    current_agent: Optional[str] = None
    iteration_count: int = 0
    iteration_count_round: int = 0
    max_iterations: int = 20
    context: Dict[str, Any] = field(default_factory=dict)
    private_contexts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    final_result: Any = None
    final_meta: Any = None
    error_message: str = ""

    def add_to_history(self, agent_name: str, result: AgentResult):
        self.history.append(
            {
                "agent": agent_name,
                "status": result.status.value,
                "message": result.message,
                "iteration": self.iteration_count,
                "data": result.data,
                "metadata": result.metadata,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "current_agent": self.current_agent,
            "iteration_count": self.iteration_count,
            "iteration_count_round": self.iteration_count_round,
            "max_iterations": self.max_iterations,
            "context": self.context,
            "private_contexts": self.private_contexts,
            "history": self.history,
            "final_result": self.final_result,
            "final_meta": self.final_meta,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeState":
        state = cls(max_iterations=data.get("max_iterations", 20))
        state.status = RuntimeStatus(data.get("status", "running"))
        state.current_agent = data.get("current_agent")
        state.iteration_count = data.get("iteration_count", 0)
        state.iteration_count_round = data.get("iteration_count_round", 0)
        state.context = data.get("context", {})
        state.private_contexts = data.get("private_contexts", {})
        state.history = data.get("history", [])
        state.final_result = data.get("final_result")
        state.final_meta = data.get("final_meta")
        state.error_message = data.get("error_message", "")
        return state
