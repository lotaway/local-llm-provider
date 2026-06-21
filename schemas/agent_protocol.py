from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import json
import time
from fastapi.responses import JSONResponse


class AgentErrorCode(Enum):
    AGENT_NOT_INITIALIZED = "agent_not_initialized"
    SESSION_NOT_FOUND = "session_not_found"
    TOOL_NOT_FOUND = "tool_not_found"
    PERMISSION_DENIED = "permission_denied"
    EXECUTION_FAILED = "execution_failed"
    INVALID_REQUEST = "invalid_request"
    TIMEOUT = "timeout"


class StreamEventType(Enum):
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    STATUS = "status"
    ERROR = "error"


@dataclass
class StreamEvent:
    event_type: StreamEventType
    content: str
    agent_name: Optional[str] = None
    tool_name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": self.event_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        if self.agent_name:
            result["agent"] = self.agent_name
        if self.tool_name:
            result["tool"] = self.tool_name
        return result
    
    def to_sse_format(self) -> str:
        return f"data: {json.dumps(self.to_dict())}\n\n"


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, Any]
    
    def to_openai_format(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


class AgentProtocolException(Exception):
    def __init__(self, error_code: AgentErrorCode, message: str):
        self.error_code = error_code
        self.message = message
    
    def to_http_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self._get_http_status(),
            content={
                "error": {
                    "code": self.error_code.value,
                    "message": self.message,
                }
            }
        )
    
    def _get_http_status(self) -> int:
        mapping = {
            AgentErrorCode.AGENT_NOT_INITIALIZED: 400,
            AgentErrorCode.SESSION_NOT_FOUND: 404,
            AgentErrorCode.TOOL_NOT_FOUND: 404,
            AgentErrorCode.PERMISSION_DENIED: 403,
            AgentErrorCode.EXECUTION_FAILED: 500,
            AgentErrorCode.INVALID_REQUEST: 400,
            AgentErrorCode.TIMEOUT: 504,
        }
        return mapping.get(self.error_code, 500)