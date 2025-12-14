import os
from model_providers import LocalLLModel, PoeModelProvider, ComfyUIProvider
from model_providers.multimodal_provider import MultimodalFactory, models as vlm_models
from rag import LocalRAG
from agents import AgentRuntime
from agents.agent_runtime import RuntimeStatus
from agents.context_storage import create_context_storage

# Global variables
comfyui_provider = ComfyUIProvider()
poe_model_provider = None
local_rag = None
agent_runtime = None
permission_manager = None
context_storage = None

# Multimodal configuration
MULTIMODAL_PROVIDER_URL = os.getenv("MULTIMODAL_PROVIDER_URL")
remote_multimodal_status = False
multimodal_model = None
default_vlm = os.getenv("DEFAULT_MULTIMODAL_MODEL", "deepseek-janus:7b")

# Preload multimodal model if configured
if os.getenv("PRELOAD_MULTIONDAL", "False").lower() == "true":
    multimodal_model = MultimodalFactory.get_model(default_vlm)
    multimodal_model.load_model()
