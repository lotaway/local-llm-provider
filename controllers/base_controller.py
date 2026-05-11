from fastapi import APIRouter
from typing import cast
from model_providers import LocalLLModel, local_model
from model_providers.provider_registry import (
    build_default_registry,
    ProviderRegistrySettings,
)
from constants import (
    POE_API_KEY,
    POE_DEFAULT_MODEL,
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
    li = []
    registry = build_default_registry(
        ProviderRegistrySettings(
            poe_api_key=POE_API_KEY,
            poe_default_model=POE_DEFAULT_MODEL,
            custom_llm_api_key=CUSTOM_LLM_API_KEY,
            custom_llm_base_url=CUSTOM_LLM_BASE_URL,
            custom_llm_model=CUSTOM_LLM_MODEL,
            custom_llm_protocol=CUSTOM_LLM_PROTOCOL,
        )
    )
    for model in registry.list_models():
        li.append(
            {
                **DEFAULT_MODEL_INFO,
                "name": model,
            }
        )
    return li


@router.get("/version")
async def api_version():
    li = await api_tags()
    if local_model is None or local_model.cur_model_name == "":
        return DEFAULT_MODEL_INFO
    _local_model = cast(LocalLLModel, local_model)
    cur = next(model for model in li if model["name"] == _local_model.cur_model_name)
    return cur
