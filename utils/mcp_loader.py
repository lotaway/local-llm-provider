import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import yaml

from utils.mcp_protocol import JsonRpcTransport, MCPError

logger = logging.getLogger(__name__)

INITIALIZE_TIMEOUT_SECONDS = 3.0


class ConnectionStatus(Enum):
    INIT = "INIT"
    READY = "READY"
    DEGRADED = "DEGRADED"
    DEAD = "DEAD"


@dataclass
class MCPToolSpec:
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    source: str = "config"

    def as_payload(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "source": self.source,
        }


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
    transport: Optional[JsonRpcTransport] = field(default=None, repr=False)
    last_heartbeat: float = 0.0
    tools: List[str] = field(default_factory=list)
    tool_specs: Dict[str, MCPToolSpec] = field(default_factory=dict)


_connections: Dict[str, MCPConnection] = {}


def _get_default_safety() -> str:
    val = os.getenv("LLP_MCP_DEFAULT_SAFETY", "MEDIUM")
    val = str(val).strip().upper()
    if val in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        return val
    return "MEDIUM"


def _normalize_safety(val: Any) -> str:
    if val is None:
        return _get_default_safety()
    safety = str(val).strip().upper()
    if safety in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        return safety
    if safety == "SAFE":
        return "LOW"
    return _get_default_safety()


def _parse_config() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    file_path = os.getenv("LLP_MCP_CONNECTIONS_FILE")
    dir_path = os.getenv("LLP_MCP_CONNECTIONS_DIR")
    if file_path:
        configs.extend(_load_config_file(file_path))
    if dir_path and os.path.isdir(dir_path):
        for filename in sorted(os.listdir(dir_path)):
            configs.extend(_load_config_file(os.path.join(dir_path, filename)))
    return configs


def _load_config_file(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        data = yaml.safe_load(text) if path.endswith((".yml", ".yaml")) else json.loads(text)
    except (OSError, ValueError, yaml.YAMLError) as error:
        logger.warning("failed to read MCP config", extra={"path": path, "error": str(error)})
        return []
    return _normalize_configs(data)


def _normalize_configs(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    connections = data.get("connections")
    if isinstance(connections, list):
        return [item for item in connections if isinstance(item, dict)]
    return [data]


def _parse_tool_specs(tools: Any, source: str) -> Dict[str, MCPToolSpec]:
    specs: Dict[str, MCPToolSpec] = {}
    for item in tools or []:
        spec = _parse_tool_spec(item, source)
        if spec:
            specs[spec.name] = spec
    return specs


def _parse_tool_spec(item: Any, source: str) -> Optional[MCPToolSpec]:
    if isinstance(item, str):
        return MCPToolSpec(name=item, source=source)
    if not isinstance(item, dict):
        return None
    name = item.get("name") or item.get("tool") or item.get("id")
    if not name:
        return None
    schema = item.get("inputSchema") or item.get("input_schema") or {}
    return MCPToolSpec(
        name=str(name),
        description=str(item.get("description", "")),
        input_schema=schema if isinstance(schema, dict) else {},
        source=source,
    )


def _start_connection(cfg: Dict[str, Any]) -> Optional[MCPConnection]:
    name = cfg.get("name")
    command = cfg.get("command")
    if not name or not command:
        logger.warning("MCP connection config missing name or command", extra={"config": cfg})
        return None
    try:
        process = _spawn_process(cfg)
        connection = _build_connection(cfg, process)
        connection.transport = JsonRpcTransport(process)
        _connections[connection.name] = connection
        return connection
    except (OSError, MCPError) as error:
        logger.error("failed to start MCP connection", extra={"name": name, "error": str(error)})
        return None


def _spawn_process(cfg: Dict[str, Any]) -> subprocess.Popen:
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in (cfg.get("env") or {}).items()})
    return subprocess.Popen(
        [cfg["command"]] + list(cfg.get("args") or []),
        cwd=cfg.get("workdir") or None,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _build_connection(cfg: Dict[str, Any], process: subprocess.Popen) -> MCPConnection:
    specs = _parse_tool_specs(cfg.get("tools"), "config")
    return MCPConnection(
        name=str(cfg["name"]),
        command=str(cfg["command"]),
        args=list(cfg.get("args") or []),
        env={str(key): str(value) for key, value in (cfg.get("env") or {}).items()},
        workdir=cfg.get("workdir") or None,
        safety=_normalize_safety(cfg.get("safety")),
        process=process,
        last_heartbeat=time.time(),
        tools=list(specs),
        tool_specs=specs,
    )


def _discover_tools_via_mcp(conn: MCPConnection, timeout: float = INITIALIZE_TIMEOUT_SECONDS) -> Dict[str, MCPToolSpec]:
    if conn.transport is None:
        return {}
    try:
        conn.transport.request("initialize", _initialize_params(), timeout=timeout)
        conn.transport.notify("notifications/initialized")
        result = conn.transport.request("tools/list", {}, timeout=timeout)
    except MCPError as error:
        logger.warning(
            "MCP tool discovery degraded",
            extra={"name": conn.name, "kind": error.kind.value, "error": str(error)},
        )
        return {}
    tools = result.get("tools", []) if isinstance(result, dict) else []
    return _parse_tool_specs(tools, "discovered")


def _initialize_params() -> Dict[str, Any]:
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "local-llm-provider", "version": "1.0.0"},
    }


def _select_tool_specs(conn: MCPConnection, discovered: Dict[str, MCPToolSpec]) -> Dict[str, MCPToolSpec]:
    if not discovered:
        return dict(conn.tool_specs)
    conflicts = sorted(set(discovered) & set(conn.tool_specs))
    if conflicts:
        logger.warning(
            "MCP tool name conflict resolved in favor of discovery",
            extra={"name": conn.name, "tools": conflicts},
        )
    merged = dict(conn.tool_specs)
    merged.update(discovered)
    return merged


def _heartbeat(conn: MCPConnection) -> None:
    if conn.process is None or conn.process.poll() is not None:
        conn.status = ConnectionStatus.DEAD
        return
    conn.last_heartbeat = time.time()


def _register_tool(mcp_agent, conn: MCPConnection, spec: MCPToolSpec) -> None:
    from schemas.permission import PermissionNames

    permission = PermissionNames.from_tool_name(spec.name)
    callable_tool = _build_tool_callable(conn, spec)
    try:
        mcp_agent.register_tool(
            spec.name, callable_tool, permission, safety_level=conn.safety
        )
    except TypeError:
        mcp_agent.register_tool(spec.name, callable_tool, permission)


def _build_tool_callable(conn: MCPConnection, spec: MCPToolSpec):
    def invoke(query: str, task: str, context: Optional[Dict[str, Any]] = None):
        arguments = (context or {}).get("tool_arguments") or {"query": query, "task": task}
        return conn.transport.call_tool(spec.as_payload(), arguments)

    return invoke


def get_connections() -> Dict[str, MCPConnection]:
    return dict(_connections)


def load_from_env(mcp_agent) -> List[MCPConnection]:
    connections: List[MCPConnection] = []
    for cfg in _parse_config():
        conn = _start_connection(cfg)
        if conn is None:
            continue
        _heartbeat(conn)
        if conn.status == ConnectionStatus.DEAD:
            continue
        discovered = _discover_tools_via_mcp(conn)
        specs = _select_tool_specs(conn, discovered)
        if not specs:
            conn.status = ConnectionStatus.DEAD
            continue
        conn.status = ConnectionStatus.READY if discovered else ConnectionStatus.DEGRADED
        conn.tool_specs = specs
        conn.tools = list(specs)
        for spec in specs.values():
            _register_tool(mcp_agent, conn, spec)
        connections.append(conn)
    return connections


def stop_all() -> None:
    for conn in list(_connections.values()):
        if conn.transport:
            conn.transport.close()
        _stop_process(conn.process)
    _connections.clear()


def _stop_process(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        process.kill()
