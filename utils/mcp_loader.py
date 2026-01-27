import os
import json
import yaml
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ConnectionStatus(Enum):
    INIT = "INIT"
    READY = "READY"
    DEGRADED = "DEGRADED"
    DEAD = "DEAD"


@dataclass
class MCPConnection:
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    workdir: Optional[str] = None
    safety: str = "MEDIUM"
    status: ConnectionStatus = ConnectionStatus.INIT
    process: Optional[subprocess.Popen] = None
    last_heartbeat: float = 0.0
    tools: List[str] = field(default_factory=list)


_connections: Dict[str, MCPConnection] = {}


def _parse_config() -> List[Dict[str, Any]]:
    file_path = os.getenv("LLP_MCP_CONNECTIONS_FILE")
    dir_path = os.getenv("LLP_MCP_CONNECTIONS_DIR")
    configs: List[Dict[str, Any]] = []
    if file_path and os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            data = yaml.safe_load(text) if file_path.endswith((".yml", ".yaml")) else json.loads(text)
            items = data.get("connections") if isinstance(data, dict) else data
            if isinstance(items, list):
                configs.extend(items)
    if dir_path and os.path.isdir(dir_path):
        for fn in os.listdir(dir_path):
            p = os.path.join(dir_path, fn)
            if not os.path.isfile(p):
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    text = f.read()
                data = yaml.safe_load(text) if p.endswith((".yml", ".yaml")) else json.loads(text)
                if isinstance(data, dict) and "connections" in data and isinstance(data["connections"], list):
                    configs.extend(data["connections"])
                elif isinstance(data, dict):
                    configs.append(data)
                elif isinstance(data, list):
                    configs.extend(data)
            except Exception:
                continue
    return configs


def _start_connection(cfg: Dict[str, Any]) -> Optional[MCPConnection]:
    name = cfg.get("name")
    command = cfg.get("command")
    args = cfg.get("args") or []
    env = cfg.get("env") or {}
    workdir = cfg.get("workdir")
    safety = str(cfg.get("safety", "MEDIUM")).upper()
    tools = cfg.get("tools") or []
    if not name or not command:
        return None
    try:
        proc_env = os.environ.copy()
        proc_env.update({str(k): str(v) for k, v in env.items()})
        proc = subprocess.Popen(
            [command] + list(args),
            cwd=workdir or None,
            env=proc_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        conn = MCPConnection(
            name=name,
            command=command,
            args=list(args),
            env={str(k): str(v) for k, v in env.items()},
            workdir=workdir or None,
            safety=safety,
            status=ConnectionStatus.READY,
            process=proc,
            last_heartbeat=time.time(),
            tools=[str(t) for t in tools],
        )
        _connections[name] = conn
        return conn
    except Exception:
        return None


def _heartbeat(conn: MCPConnection) -> None:
    if conn.process is None:
        conn.status = ConnectionStatus.DEAD
        return
    code = conn.process.poll()
    if code is not None:
        conn.status = ConnectionStatus.DEAD
        return
    conn.last_heartbeat = time.time()


def load_from_env(mcp_agent) -> List[MCPConnection]:
    configs = _parse_config()
    conns: List[MCPConnection] = []
    for cfg in configs:
        conn = _start_connection(cfg)
        if not conn:
            continue
        _heartbeat(conn)
        if conn.status == ConnectionStatus.READY:
            for t in conn.tools:
                perm = f"mcp.{t}"
                mcp_agent.register_tool(t, lambda query, task, context: {"ok": True}, perm)
            conns.append(conn)
    return conns


def stop_all():
    for conn in list(_connections.values()):
        if conn.process:
            try:
                conn.process.terminate()
            except Exception:
                pass
            try:
                conn.process.kill()
            except Exception:
                pass
        conn.status = ConnectionStatus.DEAD
