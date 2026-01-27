import os
import sys
import tempfile
import yaml

from agents.task_agents.mcp_agent import MCPTaskAgent
from utils.mcp_loader import load_from_env, ConnectionStatus


class MockLLM:
    def generate(self, *args, **kwargs):
        return "ok"


def test_mcp_dead_process_not_registered():
    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    cfg = {
        "connections": [
            {
                "name": "dead_conn",
                "command": sys.executable,
                "args": ["-c", "import sys; sys.exit(0)"],
                "tools": ["dead_tool"],
                "safety": "HIGH",
            }
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    os.environ["LLP_MCP_CONNECTIONS_FILE"] = path
    llm = MockLLM()
    agent = MCPTaskAgent(llm)
    conns = load_from_env(agent)
    tools = agent.get_available_tools()
    assert "dead_tool" not in tools
    os.remove(path)
    return True
