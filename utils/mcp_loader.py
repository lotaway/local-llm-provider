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


def _get_default_safety() -> str:
    val = os.getenv("LLP_MCP_DEFAULT_SAFETY", "MEDIUM")
    val = str(val).strip().upper()
    if val in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        return val
    return "MEDIUM"


def _normalize_safety(val: Any) -> str:
    if val is None:
        return _get_default_safety()
    s = str(val).strip().upper()
    if s in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        return s
    # backward compat
    if s == "SAFE":
        return "LOW"
    return _get_default_safety()


def _parse_config() -> List[Dict[str, Any]]:
    file_path = os.getenv("LLP_MCP_CONNECTIONS_FILE")
    dir_path = os.getenv("LLP_MCP_CONNECTIONS_DIR")
    configs: List[Dict[str, Any]] = []
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            data = yaml.safe_load(text) if file_path.endswith((".yml", ".yaml")) else json.loads(text)
            items = data.get("connections") if isinstance(data, dict) else data
            if isinstance(items, list):
                configs.extend(items)
        except Exception:
            pass
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


def _read_jsonrpc_response(proc: subprocess.Popen, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    if not proc.stdout:
        return None
    import select

    start = time.time()
    buf = b""
    while time.time() - start < timeout:
        r, _, _ = select.select([proc.stdout], [], [], 0.2)
        if not r:
            continue
        try:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            # try to parse line-based JSON
            if b"\n" in buf:
                for line in buf.split(b"\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        return json.loads(line.decode("utf-8"))
                    except Exception:
                        continue
                buf = b""
        except Exception:
            break
    # try whole buffer
    try:
        if buf.strip():
            return json.loads(buf.strip().decode("utf-8"))
    except Exception:
        pass
    return None


def _discover_tools_via_mcp(conn: MCPConnection, timeout: float = 3.0) -> List[str]:
    proc = conn.process
    if not proc or proc.stdin is None or proc.stdout is None:
        return []
    try:
        # initialize
        req_init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        proc.stdin.write((json.dumps(req_init) + "\n").encode("utf-8"))
        proc.stdin.flush()
        _read_jsonrpc_response(proc, timeout=timeout)

        # tools/list
        req_list = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        proc.stdin.write((json.dumps(req_list) + "\n").encode("utf-8"))
        proc.stdin.flush()
        resp = _read_jsonrpc_response(proc, timeout=timeout)
        if not resp:
            return []
        result = resp.get("result") or {}
        tools = result.get("tools") or []
        out: List[str] = []
        for t in tools:
            if isinstance(t, dict):
                name = t.get("name") or t.get("tool") or t.get("id")
                if name:
                    out.append(str(name))
            elif isinstance(t, str):
                out.append(t)
        return out
    except Exception:
        return []


def _start_connection(cfg: Dict[str, Any]) -> Optional[MCPConnection]:
    name = cfg.get("name")
    command = cfg.get("command")
    args = cfg.get("args") or []
    env = cfg.get("env") or {}
    workdir = cfg.get("workdir")
    safety = _normalize_safety(cfg.get("safety"))
    tools_cfg = cfg.get("tools") or []
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
            status=ConnectionStatus.INIT,
            process=proc,
            last_heartbeat=time.time(),
            tools=[str(t) for t in tools_cfg],
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


def get_connections() -> Dict[str, MCPConnection]:
    return dict(_connections)


def load_from_env(mcp_agent) -> List[MCPConnection]:
    from schemas.permission import PermissionNames

    configs = _parse_config()
    conns: List[MCPConnection] = []
    for cfg in configs:
        conn = _start_connection(cfg)
        if not conn:
            continue
        _heartbeat(conn)
        if conn.status == ConnectionStatus.DEAD:
            continue

        discovered = _discover_tools_via_mcp(conn)
        tool_list = discovered if discovered else [t for t in conn.tools]
        if discovered:
            conn.tools = discovered
            conn.status = ConnectionStatus.READY
        else:
            if tool_list:
                conn.status = ConnectionStatus.DEGRADED
            else:
                conn.status = ConnectionStatus.DEAD
                continue

        for t in tool_list:
            perm = PermissionNames.from_tool_name(t)
            try:
                mcp_agent.register_tool(t, lambda query, task, context: {"ok": True}, perm, safety_level=conn.safety)
            except TypeError:
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
    _connections.clear()
