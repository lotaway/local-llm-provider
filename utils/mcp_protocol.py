import json
import logging
import threading
import time
from enum import Enum
from queue import Empty, Queue
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_NO_MESSAGE = object()
_RETRY_MESSAGE = object()


class MCPErrorKind(str, Enum):
    TIMEOUT = "TIMEOUT"
    TRANSPORT = "TRANSPORT"
    PROTOCOL = "PROTOCOL"
    VALIDATION = "VALIDATION"
    TOOL_EXECUTION = "TOOL_EXECUTION"


class MCPError(RuntimeError):
    kind = MCPErrorKind.PROTOCOL

    def __init__(self, message: str, **context: Any):
        super().__init__(message)
        self.context = context


class MCPTimeoutError(MCPError):
    kind = MCPErrorKind.TIMEOUT


class MCPTransportError(MCPError):
    kind = MCPErrorKind.TRANSPORT


class MCPProtocolError(MCPError):
    kind = MCPErrorKind.PROTOCOL


class MCPValidationError(MCPError):
    kind = MCPErrorKind.VALIDATION


class MCPRemoteError(MCPError):
    def __init__(self, message: str, code: Optional[int] = None, data: Any = None):
        kind = MCPErrorKind.TOOL_EXECUTION if code == -32000 else MCPErrorKind.PROTOCOL
        super().__init__(message, code=code, data=data)
        self.kind = kind


class MCPToolExecutionError(MCPError):
    kind = MCPErrorKind.TOOL_EXECUTION


def validate_arguments(schema: Dict[str, Any], arguments: Dict[str, Any]) -> None:
    if not isinstance(arguments, dict):
        raise MCPValidationError("tool arguments must be an object")
    missing = [name for name in schema.get("required", []) if name not in arguments]
    if missing:
        raise MCPValidationError("missing required tool arguments", missing=missing)
    properties = schema.get("properties", {})
    for name, value in arguments.items():
        expected = (properties.get(name) or {}).get("type")
        if expected and not _matches_type(value, expected):
            raise MCPValidationError(
                "tool argument type mismatch", argument=name, expected=expected
            )


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return True


class JsonRpcTransport:
    READ_CHUNK_SIZE = 65536
    RETRY_BACKOFF_SECONDS = 0.2

    def __init__(self, process, timeout: float = 30.0, max_retries: int = 1):
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise MCPTransportError("mcp process pipes unavailable")
        self._process = process
        self._timeout = timeout
        self._max_retries = max_retries
        self._decoder = json.JSONDecoder()
        self._buffer = b""
        self._buffer_lock = threading.Lock()
        self._responses: Dict[int, Queue] = {}
        self._responses_lock = threading.Lock()
        self._request_id = 0
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._stderr_reader = threading.Thread(target=self._drain_stderr, daemon=True)
        self._reader.start()
        self._stderr_reader.start()

    def request(self, method: str, params: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Any:
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._request_once(method, params, timeout)
            except (MCPTimeoutError, MCPTransportError) as error:
                last_error = error
                if attempt < self._max_retries:
                    time.sleep(self.RETRY_BACKOFF_SECONDS * (attempt + 1))
        raise last_error

    def call_tool(self, tool_spec: Dict[str, Any], arguments: Dict[str, Any]) -> Any:
        schema = tool_spec.get("inputSchema") or tool_spec.get("input_schema") or {}
        validate_arguments(schema, arguments)
        result = self.request(
            "tools/call", {"name": tool_spec["name"], "arguments": arguments}
        )
        if isinstance(result, dict) and result.get("isError"):
            raise MCPToolExecutionError(
                "mcp tool returned an error", tool=tool_spec["name"], result=result
            )
        return result

    def notify(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def close(self) -> None:
        self._running = False

    def _request_once(self, method: str, params: Optional[Dict[str, Any]], timeout: Optional[float]) -> Any:
        request_id = self._next_request_id()
        response_queue: Queue = Queue(maxsize=1)
        with self._responses_lock:
            self._responses[request_id] = response_queue
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        try:
            response = response_queue.get(timeout=timeout or self._timeout)
        except Empty as error:
            raise MCPTimeoutError("mcp request timed out", method=method) from error
        finally:
            with self._responses_lock:
                self._responses.pop(request_id, None)
        return self._unpack_response(response, method)

    def _write(self, payload: Dict[str, Any]) -> None:
        try:
            self._process.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
            self._process.stdin.flush()
        except (OSError, ValueError) as error:
            raise MCPTransportError("failed to write mcp request") from error

    def _unpack_response(self, response: Dict[str, Any], method: str) -> Any:
        error = response.get("error")
        if error:
            raise MCPRemoteError(
                error.get("message", "mcp remote error"),
                code=error.get("code"),
                data=error.get("data"),
            )
        if "result" not in response:
            raise MCPProtocolError("mcp response missing result", method=method)
        return response["result"]

    def _next_request_id(self) -> int:
        with self._responses_lock:
            self._request_id += 1
            return self._request_id

    def _read_loop(self) -> None:
        while self._running:
            chunk = self._process.stdout.read(self.READ_CHUNK_SIZE)
            if not chunk:
                break
            with self._buffer_lock:
                self._buffer += chunk
            self._drain_messages()

    def _drain_messages(self) -> None:
        while self._running:
            message = self._pop_message()
            if message is _NO_MESSAGE:
                return
            if message is _RETRY_MESSAGE:
                continue
            self._dispatch(message)

    def _pop_message(self) -> Any:
        with self._buffer_lock:
            data = self._buffer
        stripped = data.lstrip()
        if not stripped:
            return _NO_MESSAGE
        try:
            text = stripped.decode("utf-8")
            message, end = self._decoder.raw_decode(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._discard_or_wait(data, stripped)
        consumed = len(data) - len(stripped) + len(text[:end].encode("utf-8"))
        with self._buffer_lock:
            self._buffer = data[consumed:]
        return message

    def _discard_or_wait(self, data: bytes, stripped: bytes) -> Any:
        newline = data.find(b"\n")
        if newline != -1:
            logger.warning("discarding non-JSON MCP stdout line", extra={"line": data[:newline][:200]})
            with self._buffer_lock:
                self._buffer = data[newline + 1 :]
            return _RETRY_MESSAGE
        if stripped.startswith((b"{", b"[")):
            return _NO_MESSAGE
        logger.warning("discarding non-JSON MCP stdout prefix", extra={"prefix": stripped[:200]})
        with self._buffer_lock:
            self._buffer = b""
        return _RETRY_MESSAGE

    def _dispatch(self, message: Any) -> None:
        if not isinstance(message, dict):
            return
        if "id" not in message:
            logger.debug("mcp notification received", extra={"method": message.get("method")})
            return
        with self._responses_lock:
            queue = self._responses.get(message["id"])
        if queue is None:
            logger.warning("mcp response without pending request", extra={"id": message["id"]})
            return
        queue.put(message)

    def _drain_stderr(self) -> None:
        for line in iter(self._process.stderr.readline, b""):
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                logger.warning("mcp stderr output", extra={"text": text})
