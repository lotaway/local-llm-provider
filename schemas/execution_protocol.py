from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_HUMAN = "waiting_human"
    WAITING_CLIENT = "waiting_client"


@dataclass
class ExecutionRequest:
    session_id: str
    user_message: str
    context: dict | None = None
    stream: bool = False
    start_agent: str = "qa"


@dataclass
class ExecutionResponse:
    status: ExecutionStatus
    content: str
    tool_calls: list | None = None
    context: dict | None = None
    session_id: str = ""
    error_message: str = ""
