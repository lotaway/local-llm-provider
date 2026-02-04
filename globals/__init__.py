import os
from model_providers import ComfyUIProvider
from remote_providers import PoeModelProvider
from model_providers.multimodal_provider import MultimodalFactory
from rag import LocalRAG

comfyui_provider = ComfyUIProvider()
poe_model_provider: PoeModelProvider | None = None
local_rag: LocalRAG | None = None
agent_runtime = None
permission_manager = None
context_storage = None

MULTIMODAL_PROVIDER_URL = os.getenv("MULTIMODAL_PROVIDER_URL")
remote_multimodal_status = False
multimodal_model = None
default_vlm = os.getenv("PRELOAD_MULTIMODAL_MODEL")

if default_vlm is not None:
    multimodal_model = MultimodalFactory.get_model(default_vlm)
    multimodal_model.load_model()
