from typing import Any, Dict
from datetime import datetime
from pymongo.collection import Collection
from schemas.evolution_trace import WorkflowTrace, AgentExecutionTrace

class EvolutionRepository:
    def __init__(self, db: Any):
        self._traces: Collection = db["evolution_traces"]
        self._create_indexes()

    def _create_indexes(self) -> None:
        self._traces.create_index("session_id")
        self._traces.create_index("created_at")

    def save_trace(self, trace: WorkflowTrace) -> None:
        trace_data = self._serialize_trace(trace)
        self._traces.insert_one(trace_data)

    def _serialize_trace(self, trace: WorkflowTrace) -> Dict[str, Any]:
        return {
            "session_id": trace.session_id,
            "initial_query": trace.initial_query,
            "steps": [self._serialize_step(step) for step in trace.execution_steps],
            "final_answer": trace.final_answer,
            "total_latency_ms": trace.total_latency_ms,
            "created_at": trace.created_at
        }

    def _serialize_step(self, step: AgentExecutionTrace) -> Dict[str, Any]:
        return {
            "agent_name": step.agent_name,
            "input_payload": step.input_payload,
            "output_message": step.output_message,
            "status": step.status.value,
            "thought_process": step.thought_process,
            "tool_calls": [
                {
                    "tool": tc.tool_name,
                    "args": tc.arguments,
                    "observation": tc.raw_observation,
                    "latency": tc.latency_ms,
                    "time": tc.timestamp
                }
                for tc in step.tool_calls
            ],
            "timestamp": step.timestamp
        }
