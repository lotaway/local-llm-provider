from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

class TraceStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    PENDING = "pending"

@dataclass(frozen=True)
class ToolCallTrace:
    tool_name: str
    arguments: Dict[str, Any]
    raw_observation: Any
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass(frozen=True)
class AgentExecutionTrace:
    agent_name: str
    input_payload: Any
    output_message: str
    status: TraceStatus
    thought_process: Optional[str] = None
    tool_calls: List[ToolCallTrace] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass(frozen=True)
class WorkflowTrace:
    session_id: str
    initial_query: str
    execution_steps: List[AgentExecutionTrace] = field(default_factory=list)
    final_answer: Optional[str] = None
    total_latency_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
