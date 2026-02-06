from datetime import datetime
from typing import Dict, List, Optional, Any
from schemas.evolution_trace import WorkflowTrace, AgentExecutionTrace, TraceStatus
from services.interfaces.trace_collector import TraceCollector

class MoltTraceCollector(TraceCollector):
    def __init__(self, repository: Any):
        self._repository = repository
        self._current_workflow: Optional[Dict] = None

    def start_workflow(self, session_id: str, query: str) -> None:
        self._current_workflow = {
            "session_id": session_id,
            "initial_query": query,
            "steps": [],
            "start_time": datetime.now()
        }

    def record_agent_execution(self, trace: AgentExecutionTrace) -> None:
        if self._is_workflow_active():
            self._current_workflow["steps"].append(trace)

    def persist_workflow(self, final_answer: str) -> None:
        if not self._is_workflow_active():
            raise RuntimeError("WorkflowNotStarted")
        
        workflow = self._extract_workflow_trace(final_answer)
        self._repository.save_trace(workflow)
        self._current_workflow = None

    def _is_workflow_active(self) -> bool:
        return self._current_workflow is not None

    def _extract_workflow_trace(self, final_answer: str) -> WorkflowTrace:
        start_time = self._current_workflow["start_time"]
        duration = datetime.now() - start_time
        return WorkflowTrace(
            session_id=self._current_workflow["session_id"],
            initial_query=self._current_workflow["initial_query"],
            execution_steps=self._current_workflow["steps"],
            final_answer=final_answer,
            total_latency_ms=duration.total_seconds() * 1000
        )
