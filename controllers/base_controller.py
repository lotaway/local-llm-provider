from fastapi import APIRouter
from typing import cast
from model_providers import LocalLLModel, local_model
from model_providers.provider_registry import (
    build_default_registry,
    ProviderRegistrySettings,
)
from constants import (
    POE_API_KEY,
    OPENAI_API_KEY,
    POE_DEFAULT_MODEL,
    OPENAI_BASE_URL,
    OPENAI_ORGANIZATION,
    OPENAI_PROJECT,
    OPENAI_PROXY_URL,
    OPENAI_TIMEOUT,
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
            openai_api_key=OPENAI_API_KEY,
            poe_default_model=POE_DEFAULT_MODEL,
            openai_base_url=OPENAI_BASE_URL,
            openai_organization=OPENAI_ORGANIZATION,
            openai_project=OPENAI_PROJECT,
            openai_proxy_url=OPENAI_PROXY_URL,
            openai_timeout=OPENAI_TIMEOUT,
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
