"""
Interface: Chat DTOs.
Defines stable API contracts.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field


class MessageContent(BaseModel):
    """Message content."""
    type: Literal["text", "image_url"] = "text"
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None


class Message(BaseModel):
    """Message."""
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[MessageContent]]
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None


class GenerationConfigDTO(BaseModel):
    """Generation config DTO."""
    max_new_tokens: int = Field(default=4096, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    repetition_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    stop: Optional[List[str]] = None
    seed: Optional[int] = None


class ChatRequestDTO(BaseModel):
    """Chat request DTO."""
    model: str
    messages: List[Message]
    stream: bool = False
    config: Optional[GenerationConfigDTO] = None
    session_id: Optional[str] = None
    enable_rag: bool = False
    files: Optional[List[str]] = None


class ChatResponseDTO(BaseModel):
    """Chat response DTO (non-stream)."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class ChatStreamChunkDTO(BaseModel):
    """Stream chunk DTO."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


class ModelInfoDTO(BaseModel):
    """Model info DTO."""
    id: str
    name: str
    type: str
    path: str
    size_bytes: Optional[int] = None
    architecture: Optional[str] = None
    description: str = ""
    tags: List[str] = []
    loaded: bool = False


class LoadModelRequestDTO(BaseModel):
    """Load model request DTO."""
    model_id: str
    engine: Literal["auto", "vllm", "transformers", "llama_cpp"] = "auto"
    context_length: Optional[int] = None
    quantization: Optional[str] = None
    vllm_options: Optional[Dict[str, Any]] = None


class LoadModelResponseDTO(BaseModel):
    """Load model response DTO."""
    model_id: str
    engine_type: str
    context_length: int
    status: str = "loaded"


class HealthResponseDTO(BaseModel):
    """Health check response DTO."""
    status: str
    model_loaded: bool
    current_model: Optional[str] = None
    engine_type: Optional[str] = None
    gpu_memory_used: Optional[str] = None
    uptime_seconds: float