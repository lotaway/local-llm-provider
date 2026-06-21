from typing import List, Dict, Any
from interfaces.agent_protocol_types import (
    AgentMetadataProvider,
    AgentInfo,
    StreamEventConverter,
)
from schemas.agent_protocol import (
    StreamEvent,
    StreamEventType,
    ToolSchema,
    AgentErrorCode,
)
from schemas.capability import Capability
import json


class AgentMetadataService(AgentMetadataProvider):
    def __init__(self, runtime):
        self._runtime = runtime
    
    def get_available_agents(self) -> List[AgentInfo]:
        return [
            AgentInfo(
                name="qa",
                description="问答解析助手",
                supported_task_types=["fact_query", "operation", "analysis"],
                capabilities=["intent_parsing", "entity_extraction"],
            ),
            AgentInfo(
                name="planning",
                description="任务规划助手",
                supported_task_types=["planning", "decomposition"],
                capabilities=["task_decomposition", "dependency_analysis"],
            ),
            AgentInfo(
                name="router",
                description="路由分发助手",
                supported_task_types=["routing"],
                capabilities=["agent_routing"],
            ),
            AgentInfo(
                name="verification",
                description="结果验证助手",
                supported_task_types=["verification"],
                capabilities=["quality_check", "fact_verification"],
            ),
            AgentInfo(
                name="risk",
                description="风险评估助手",
                supported_task_types=["risk_assessment"],
                capabilities=["safety_evaluation", "permission_check"],
            ),
            AgentInfo(
                name="error_handler",
                description="异常处理助手",
                supported_task_types=["error_handling"],
                capabilities=["root_cause_analysis", "auto_recovery"],
            ),
        ]
    
    def get_available_tools(self) -> List[ToolSchema]:
        tools = self._runtime.state.context.get("available_mcp_tools", [])
        return [self._convert_tool_schema(t) for t in tools]
    
    def get_capabilities(self) -> List[Capability]:
        return self._runtime.state.context.get("capabilities", [])
    
    def _convert_tool_schema(self, tool_name: str) -> ToolSchema:
        return ToolSchema(
            name=tool_name,
            description=f"Execute {tool_name} tool",
            parameters={
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Task description"}
                },
                "required": ["description"],
            },
            returns={"type": "string"},
        )


class OpenAIStreamEventConverter(StreamEventConverter):
    def convert_chunk(self, chunk: str, agent_name: str) -> StreamEvent:
        return StreamEvent(
            event_type=StreamEventType.MESSAGE,
            content=chunk,
            agent_name=agent_name,
        )
    
    def convert_tool_call(self, tool_name: str, arguments: Dict) -> StreamEvent:
        return StreamEvent(
            event_type=StreamEventType.TOOL_CALL,
            content=json.dumps(arguments),
            tool_name=tool_name,
        )
    
    def convert_status(self, status: str) -> StreamEvent:
        return StreamEvent(
            event_type=StreamEventType.STATUS,
            content=status,
        )
    
    def convert_error(self, error_code: AgentErrorCode, message: str) -> StreamEvent:
        return StreamEvent(
            event_type=StreamEventType.ERROR,
            content=json.dumps({"code": error_code.value, "message": message}),
        )