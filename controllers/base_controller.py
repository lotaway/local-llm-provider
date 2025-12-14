from fastapi import APIRouter
from typing import cast
from globals import local_model
from model_providers import LocalLLModel

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
    for model in LocalLLModel.get_models():
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
