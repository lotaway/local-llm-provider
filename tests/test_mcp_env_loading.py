import os
from utils import mcp_loader


class DummyAgent:
    def __init__(self):
        self.tools = {}

    def register_tool(self, tool_name, tool_callable, permission_name=None, safety_level=None):
        self.tools[tool_name] = {
            "callable": tool_callable,
            "permission": permission_name,
            "safety_level": safety_level,
        }


def test_mcp_env_loading(monkeypatch, tmp_path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text('{"connections": [{"name": "test_mcp", "command": "python", "args": ["-c", "import time; time.sleep(30)"], "safety": "HIGH", "tools": ["dummy_tool"]}]}', encoding="utf-8")
    monkeypatch.setenv("LLP_MCP_CONNECTIONS_FILE", str(cfg))
    monkeypatch.setenv("LLP_MCP_DEFAULT_SAFETY", "MEDIUM")

    agent = DummyAgent()
    try:
        conns = mcp_loader.load_from_env(agent)
    finally:
        mcp_loader.stop_all()

    assert conns
    conn = conns[0]
    assert conn.safety == "HIGH"
    assert "dummy_tool" in agent.tools
    # safety passed through
    assert agent.tools["dummy_tool"]["safety_level"] == "HIGH"
