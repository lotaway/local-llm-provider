from abc import ABC, abstractmethod
from schemas.evolution_trace import WorkflowTrace, AgentExecutionTrace

class TraceCollector(ABC):
    @abstractmethod
    def start_workflow(self, session_id: str, query: str) -> None:
        pass

    @abstractmethod
    def record_agent_execution(self, trace: AgentExecutionTrace) -> None:
        pass

    @abstractmethod
    def complete_workflow(self, final_answer: str) -> WorkflowTrace:
        pass
