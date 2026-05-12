from fastapi import APIRouter
from typing import cast
from model_providers import LocalLLModel, local_model
from model_providers.provider_registry import (
    build_default_registry,
    ProviderRegistrySettings,
)
from constants import (
    CUSTOM_LLM_API_KEY,
    CUSTOM_LLM_BASE_URL,
    CUSTOM_LLM_MODEL,
    CUSTOM_LLM_PROTOCOL,
)

router = APIRouter(prefix="/api", tags=["api"])

DEFAULT_MODEL_INFO = {
    "name": "unknown",
    "version": "1.0.0",
    "object": "model",
    "owned_by": "lotaway",
    "api_version": "v1",
}


@router.post("/show")
async def api_show():
    return {"ok": True}


@router.get("/tags")
async def api_tags():
    settings = ProviderRegistrySettings(
        custom_llm_api_key=CUSTOM_LLM_API_KEY,
        custom_llm_base_url=CUSTOM_LLM_BASE_URL,
        custom_llm_model=CUSTOM_LLM_MODEL,
        custom_llm_protocol=CUSTOM_LLM_PROTOCOL,
    )
    registry = build_default_registry(settings)
    return [{**DEFAULT_MODEL_INFO, "name": m} for m in registry.list_models()]

@router.get("/version")
async def api_version():
    tags = await api_tags()
    if not local_model or not local_model.cur_model_name:
        return DEFAULT_MODEL_INFO
    name = cast(LocalLLModel, local_model).cur_model_name
    return next((t for t in tags if t["name"] == name), DEFAULT_MODEL_INFO)
