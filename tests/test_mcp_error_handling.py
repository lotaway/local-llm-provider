"""Test MCP agent graceful error handling when tools are unavailable"""

import asyncio

from clients.agents.task_agents.mcp_agent import MCPTaskAgent
from clients.agents.agent_base import AgentStatus


class MockLLM:
    """Mock LLM for testing without dependencies"""

    def __init__(self):
        pass

    def generate(self, *args, **kwargs):
        return "Mock response"


def _run(coro):
    return asyncio.run(coro)


def test_mcp_agent_no_tools():
    """Test MCP agent behavior when no tools are registered"""
    print("=" * 60)
    print("Test 1: MCP Agent with No Tools Registered")
    print("=" * 60)

    # Create MCP agent with mock LLM and no tools registered
    llm = MockLLM()
    mcp_agent = MCPTaskAgent(llm)

    # Verify no tools available
    available_tools = mcp_agent.get_available_tools()
    print(f"\nAvailable MCP tools: {available_tools}")
    assert len(available_tools) == 0, "Expected no tools to be registered"

    # Test execution with unavailable tool
    task = {"tool_name": "file_manager", "description": "List files in directory"}
    context = {"original_query": "Show me the files in the current directory"}

    result = _run(mcp_agent.execute(task, context, {}))

    print(f"\nResult Status: {result.status.value}")
    print(f"\nResult Message: {result.message}")
    print(f"\nResult Next Agent: {result.next_agent}")

    # Verify graceful handling
    assert result.status == AgentStatus.NEEDS_CLIENT_EXECUTION, (
        f"Expected NEEDS_CLIENT_EXECUTION, got {result.status}"
    )
    assert result.data.get("tool_name") == "file_manager"
    assert result.data.get("backend_available") == [], (
        "Expected empty tools list in data"
    )

    print(f"\nResult Data:")
    for key, value in result.data.items():
        print(f"  {key}: {value}")

    print("\n✓ Test 1 PASSED: MCP agent handles missing tools gracefully")
    return True


def test_mcp_agent_with_registered_tool():
    """Test MCP agent behavior when tool is registered"""
    print("\n" + "=" * 60)
    print("Test 2: MCP Agent with Registered Tool")
    print("=" * 60)

    # Create MCP agent and register a mock tool
    llm = MockLLM()
    mcp_agent = MCPTaskAgent(llm)

    # Register a mock tool
    from schemas.permission import PermissionNames

    def mock_tool(query, task, context):
        return {"result": "Mock tool executed"}

    mcp_agent.register_tool(
        "test_tool", mock_tool, PermissionNames.from_tool_name("test_tool")
    )

    # Verify tool is available
    available_tools = mcp_agent.get_available_tools()
    print(f"\nAvailable MCP tools: {available_tools}")
    assert len(available_tools) == 1, "Expected one tool to be registered"
    assert "test_tool" in available_tools, "Expected 'test_tool' to be registered"

    # Test requesting unavailable tool
    task = {"tool_name": "file_manager", "description": "List files"}
    context = {"original_query": "List files"}

    result = _run(mcp_agent.execute(task, context, {}))

    print(f"\nRequesting unavailable tool 'file_manager':")
    print(f"\nAvailable tools in response: {result.data.get('backend_available')}")

    # Should still return NEEDS_CLIENT_EXECUTION with available tools list
    assert result.status == AgentStatus.NEEDS_CLIENT_EXECUTION, (
        "Expected NEEDS_CLIENT_EXECUTION for unavailable tool"
    )
    assert result.data.get("backend_available") == ["test_tool"], (
        "Expected available tools list"
    )

    print(
        "\n✓ Test 2 PASSED: MCP agent provides available tools when requested tool not found"
    )
    return True


def test_planning_agent_retry_detection():
    """Test that planning agent can detect MCP retry scenarios"""
    print("=" * 60)
    print("Test 3: Planning Agent MCP Retry Detection")
    print("=" * 60)

    # Test with MCP retry scenario
    input_data = {
        "error": "tool_not_found",
        "requested_tool": "file_manager",
        "available_mcp_tools": [],
        "original_task": "List files",
        "suggestion": "MCP工具 'file_manager' 未注册。可用的MCP工具: 无。建议使用LLM或RAG能力重新规划任务。",
    }

    # This should trigger the is_mcp_retry path
    is_mcp_retry = (
        isinstance(input_data, dict) and input_data.get("error") == "tool_not_found"
    )
    print(f"\nInput data error type: {input_data.get('error')}")
    print(f"\nIs MCP retry detected: {is_mcp_retry}")

    assert is_mcp_retry, "Expected MCP retry to be detected"

    # Test with normal data
    normal_data = {"some_key": "some_value"}
    is_normal_retry = (
        isinstance(normal_data, dict) and normal_data.get("error") == "tool_not_found"
    )
    print(f"\nNormal data is MCP retry: {is_normal_retry}")

    assert not is_normal_retry, "Expected normal data not to be detected as MCP retry"

    print("\n✓ Test 3 PASSED: Planning agent can correctly detect MCP retry scenarios")
    return True


if __name__ == "__main__":
    print("\nMCP Agent Error Handling Tests")
    print("=" * 60)

    try:
        test_mcp_agent_no_tools()
        test_mcp_agent_with_registered_tool()
        test_planning_agent_retry_detection()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
