# Controllers package for organizing FastAPI routes
from .agent_controller import router as agent_router
from .llm_controller import router as llm_router
from .rag_controller import router as rag_router
from .file_controller import router as file_router
from .voice_controller import router as voice_router

__all__ = ["agent_router", "llm_router", "rag_router", "file_router", "voice_router"]
