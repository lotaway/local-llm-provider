import pytest
import json
from schemas.agent_protocol import (
    AgentErrorCode,
    AgentProtocolException,
    StreamEventType,
    StreamEvent,
    ToolSchema,
)


class TestAgentErrorCode:
    def test_error_code_value(self):
        assert AgentErrorCode.AGENT_NOT_INITIALIZED.value == "agent_not_initialized"
    
    def test_error_code_count(self):
        codes = list(AgentErrorCode)
        assert len(codes) == 7


class TestAgentProtocolException:
    def test_error_code_mapping(self):
        exception = AgentProtocolException(
            AgentErrorCode.AGENT_NOT_INITIALIZED,
            "Runtime not initialized"
        )
        assert exception._get_http_status() == 400
    
    def test_error_response_format(self):
        exception = AgentProtocolException(
            AgentErrorCode.TOOL_NOT_FOUND,
            "Tool xyz not found"
        )
        response = exception.to_http_response()
        assert response.status_code == 404
        body = json.loads(response.body)
        assert body["error"]["code"] == "tool_not_found"
        assert body["error"]["message"] == "Tool xyz not found"


class TestStreamEvent:
    def test_stream_event_to_dict(self):
        event = StreamEvent(
            event_type=StreamEventType.MESSAGE,
            content="Hello",
            agent_name="qa"
        )
        result = event.to_dict()
        assert result["type"] == "message"
        assert result["content"] == "Hello"
        assert result["agent"] == "qa"
        assert "timestamp" in result
    
    def test_stream_event_sse_format(self):
        event = StreamEvent(
            event_type=StreamEventType.MESSAGE,
            content="test",
            agent_name="qa"
        )
        sse = event.to_sse_format()
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        data = json.loads(sse[6:-2])
        assert data["type"] == "message"


class TestToolSchema:
    def test_tool_schema_openai_format(self):
        schema = ToolSchema(
            name="read_file",
            description="Read file content",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            },
            returns={"type": "string"}
        )
        result = schema.to_openai_format()
        assert result["type"] == "function"
        assert result["function"]["name"] == "read_file"
        assert result["function"]["parameters"]["type"] == "object"


from interfaces.agent_protocol_types import AgentInfo


class TestAgentInfo:
    def test_agent_info_to_dict(self):
        info = AgentInfo(
            name="qa",
            description="问答解析助手",
            supported_task_types=["fact_query", "operation"],
            capabilities=["intent_parsing"]
        )
        result = info.to_dict()
        assert result["name"] == "qa"
        assert result["description"] == "问答解析助手"
        assert result["supported_task_types"] == ["fact_query", "operation"]


from unittest.mock import Mock
from services.agent_protocol_service import AgentMetadataService, OpenAIStreamEventConverter


class TestAgentMetadataService:
    def test_get_available_agents(self):
        runtime = Mock()
        runtime.agents = {"qa": Mock(), "planning": Mock()}
        service = AgentMetadataService(runtime)
        agents = service.get_available_agents()
        assert len(agents) >= 2
        assert agents[0].name == "qa"
    
    def test_get_available_tools_empty(self):
        runtime = Mock()
        runtime.state = Mock()
        runtime.state.context = {"available_mcp_tools": []}
        service = AgentMetadataService(runtime)
        tools = service.get_available_tools()
        assert len(tools) == 0


class TestOpenAIStreamEventConverter:
    def test_convert_chunk(self):
        converter = OpenAIStreamEventConverter()
        event = converter.convert_chunk("Hello", "qa")
        assert event.event_type == StreamEventType.MESSAGE
        assert event.content == "Hello"
        assert event.agent_name == "qa"
    
    def test_convert_tool_call(self):
        converter = OpenAIStreamEventConverter()
        event = converter.convert_tool_call("read_file", {"path": "/tmp"})
        assert event.event_type == StreamEventType.TOOL_CALL
        assert event.tool_name == "read_file"
        data = json.loads(event.content)
        assert data["path"] == "/tmp"
    
    def test_convert_status(self):
        converter = OpenAIStreamEventConverter()
        event = converter.convert_status("completed")
        assert event.event_type == StreamEventType.STATUS
        assert event.content == "completed"
    
    def test_convert_error(self):
        converter = OpenAIStreamEventConverter()
        event = converter.convert_error(AgentErrorCode.TOOL_NOT_FOUND, "Tool missing")
        assert event.event_type == StreamEventType.ERROR
        data = json.loads(event.content)
        assert data["code"] == "tool_not_found"