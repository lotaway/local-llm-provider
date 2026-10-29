import json
import subprocess
import sys
import textwrap

import pytest

from utils.mcp_loader import MCPConnection, MCPToolSpec, _select_tool_specs
from utils.mcp_protocol import (
    JsonRpcTransport,
    MCPRemoteError,
    MCPToolExecutionError,
    MCPValidationError,
)

SERVER_SCRIPT = textwrap.dedent(
    """
    import json
    import sys

    for line in sys.stdin:
        request = json.loads(line)
        if "id" not in request:
            continue
        method = request.get("method")
        if method == "initialize":
            result = {"protocolVersion": "2024-11-05"}
        elif method == "tools/list":
            result = {"tools": [{"name": f"tool-{index}"} for index in range(2000)]}
        elif method == "tools/call":
            name = request["params"]["name"]
            if name == "remote-error":
                response = {"jsonrpc": "2.0", "id": request["id"], "error": {"code": -32000, "message": "boom"}}
                sys.stdout.write(json.dumps(response))
                sys.stdout.flush()
                continue
            result = {"content": [{"type": "text", "text": str(request["params"]["arguments"])}], "isError": name == "fail"}
        else:
            result = {}
        response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
        sys.stdout.write(json.dumps(response))
        sys.stdout.flush()
    """
)


@pytest.fixture
def transport():
    process = subprocess.Popen(
        [sys.executable, "-c", SERVER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    client = JsonRpcTransport(process, timeout=5, max_retries=0)
    yield client
    client.close()
    process.terminate()
    process.wait(timeout=5)


class TestJsonRpcTransport:
    def test_reads_large_json_without_newline_delimiters(self, transport):
        result = transport.request("tools/list", {})

        assert len(result["tools"]) == 2000
        assert result["tools"][-1]["name"] == "tool-1999"

    def test_calls_tool_with_validated_arguments(self, transport):
        spec = {
            "name": "echo",
            "inputSchema": {
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            },
        }

        result = transport.call_tool(spec, {"query": "hello"})

        assert result["isError"] is False

    def test_rejects_invalid_arguments_before_sending(self, transport):
        spec = {
            "name": "echo",
            "inputSchema": {
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            },
        }

        with pytest.raises(MCPValidationError):
            transport.call_tool(spec, {"query": 1})

    def test_classifies_remote_tool_errors(self, transport):
        with pytest.raises(MCPRemoteError) as error:
            transport.request("tools/call", {"name": "remote-error", "arguments": {}})

        assert error.value.kind.value == "TOOL_EXECUTION"

    def test_classifies_result_level_tool_failure(self, transport):
        with pytest.raises(MCPToolExecutionError):
            transport.call_tool({"name": "fail", "inputSchema": {}}, {})


class TestToolSpecPriority:
    def test_discovered_tools_override_config_conflicts(self):
        connection = MCPConnection(
            name="server",
            command="python",
            tool_specs={"shared": MCPToolSpec(name="shared", source="config"), "config-only": MCPToolSpec(name="config-only")},
        )
        discovered = {"shared": MCPToolSpec(name="shared", source="discovered")}

        selected = _select_tool_specs(connection, discovered)

        assert selected["shared"].source == "discovered"
        assert "config-only" in selected

    def test_config_tools_are_used_when_discovery_is_empty(self):
        connection = MCPConnection(
            name="server",
            command="python",
            tool_specs={"configured": MCPToolSpec(name="configured")},
        )

        selected = _select_tool_specs(connection, {})

        assert list(selected) == ["configured"]
