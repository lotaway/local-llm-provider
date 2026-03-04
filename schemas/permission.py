"""Permission names and safety level definitions."""

from enum import Enum

class SafetyLevel(Enum):
    """Tool safety levels"""
    SAFE = 0        # Read-only, no side effects
    LOW = 1         # Minimal risk (e.g., simple calculations)
    MEDIUM = 2      # Moderate risk (e.g., file read)
    HIGH = 3        # High risk (e.g., file write, network requests)
    CRITICAL = 4    # Critical risk (e.g., system commands, delete operations)


class PermissionNames:
    """Unique permission identifiers to avoid hardcoding strings project-wide."""
    
    # LLM operations
    LLM_QUERY = "llm.query"
    
    # RAG operations
    RAG_QUERY = "rag.query"
    
    # MCP tools
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
        """Standardize mapping from tool name to permission name."""
        return f"mcp.{tool_name}"
