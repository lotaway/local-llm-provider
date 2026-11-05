from enum import Enum
from typing import Any


class SafetyLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


SAFETY_LEVEL_NAMES = tuple(level.name for level in SafetyLevel)
UNKNOWN_SAFETY_LEVEL = "UNKNOWN"
LEGACY_SAFE_LEVEL = "SAFE"


def normalize_safety_level(value: Any) -> str:
    name = str(value or "").strip().upper()
    if name in SAFETY_LEVEL_NAMES:
        return name
    if name == LEGACY_SAFE_LEVEL:
        return SafetyLevel.LOW.name
    return UNKNOWN_SAFETY_LEVEL


class PermissionNames:
    LLM_QUERY = "llm.query"
    RAG_QUERY = "rag.query"
    MCP_WEB_SEARCH = "mcp.web_search"
    MCP_FILE_READ = "mcp.file_read"
    MCP_FILE_WRITE = "mcp.file_write"
    MCP_FILE_DELETE = "mcp.file_delete"
    MCP_SYSTEM_COMMAND = "mcp.system_command"
    MCP_OCR = "mcp.ocr"
    MCP_IMAGE_RECOGNITION = "mcp.image_recognition"
    MCP_AUDIO_RECOGNITION = "mcp.audio_recognition"

    @staticmethod
    def from_tool_name(tool_name: str) -> str:
        return f"mcp.{tool_name}"
