import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
    FileResponse,
    PlainTextResponse,
)
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
import json
import httpx
import logging
from typing import cast
import asyncio
from contextlib import asynccontextmanager
from rag import LocalRAG

# import triton
# import triton.language as tl
from dotenv import load_dotenv

load_dotenv()

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

import globals as backend_globals
from model_providers import LocalLLModel, PoeModelProvider
from model_providers.multimodal_provider import MultimodalFactory
from routers import version_router
from controllers.base_controller import router as base_router
import auth
from auth import get_multimodal_headers


async def check_multimodal_health():
    if not backend_globals.MULTIMODAL_PROVIDER_URL:
        return

    async with httpx.AsyncClient(trust_env=False) as client:
        while True:
            try:
                resp = await client.post(
                    f"{backend_globals.MULTIMODAL_PROVIDER_URL}/api/show",
                    headers=get_multimodal_headers(),
                    timeout=5,
                )
                if resp.status_code == 200 and resp.json().get("ok"):
                    backend_globals.remote_multimodal_status = True
                else:
                    backend_globals.remote_multimodal_status = False
            except Exception:
                backend_globals.remote_multimodal_status = False

            await asyncio.sleep(10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if backend_globals.MULTIMODAL_PROVIDER_URL:
        asyncio.create_task(check_multimodal_health())
    PRELOAD_MODEL = os.getenv("PRELOAD_MODEL")
    if PRELOAD_MODEL:
        available_models = LocalLLModel.get_models()
        if PRELOAD_MODEL in available_models:
            logger.info(f"Preloading model: {PRELOAD_MODEL}")
            LocalLLModel.init_local_model(PRELOAD_MODEL).load_model()
        else:
            logger.error(
                f"PRELOAD_MODEL '{PRELOAD_MODEL}' not found in available models: {available_models}. Skipping preload."
            )
    yield
    # Shutdown
    import model_providers

    if getattr(model_providers, "local_model", None):
        print("Shutting down... unloading local model.")
        model_providers.local_model.unload_model()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include version router
app.include_router(version_router)
app.include_router(base_router)


DEFAULT_MODEL_INFO = {
    "name": "unknown",
    "version": "1.0.0",
    "object": "model",
    "owned_by": "lotaway",
    "api_version": "v1",
}
VERSION = DEFAULT_MODEL_INFO["api_version"]


@app.middleware("http")
async def verify_admin_token(request: Request, call_next):
    if request.url.path.startswith(f"/{VERSION}"):
        if request.method == "OPTIONS":
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return JSONResponse(
                status_code=401, content={"detail": "Missing Authorization header"}
            )

        token = auth_header
        if token.startswith("Bearer "):
            token = token[7:]

        if token != auth.ADMIN_TOKEN:
            return JSONResponse(
                status_code=401, content={"detail": "Invalid admin token"}
            )

    return await call_next(request)


@app.get("/")
async def index():
    return "Hello, Local LLM Provider!"


@app.get("/manifest.json")
async def manifest():
    return FileResponse("manifest.json")


@app.get("/mcp")
async def query_rag(request: Request):

    query = request.query_params.get("query")
    if query is None:
        raise HTTPException(
            status_code=400, detail="Either query or request parameter is required"
        )

    try:
        if backend_globals.local_rag is None:
            local_model = LocalLLModel.init_local_model()
            backend_globals.local_rag = LocalRAG(local_model)
        result = await backend_globals.local_rag.generate_answer(query)

        if isinstance(result, str):
            return PlainTextResponse(result)

        async def event_stream():
            yield f"data: [DONE]{result}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except json.JSONDecodeError as e:
        error_msg = (
            f"模型配置文件解析失败: {str(e)}\n\n"
            "可能的原因:\n"
            "1. Embedding 模型缓存文件损坏\n"
            "2. 模型下载不完整\n\n"
            "建议解决方案:\n"
            "1. 清理缓存: rm -rf ~/.cache/huggingface/hub/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct\n"
            "2. 重新启动服务以重新下载模型\n"
            "3. 检查网络连接"
        )
        print(f"错误: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    except ValueError as e:
        error_msg = f"配置错误: {str(e)}"
        print(f"错误: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)

    except RuntimeError as e:
        error_msg = f"运行时错误: {str(e)}"
        print(f"错误: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    except Exception as e:
        error_msg = f"未知错误: {type(e).__name__} - {str(e)}"
        print(f"错误: {error_msg}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

    return await backend_globals.comfyui_provider.proxy_request(request)


@app.post("/poe/v1/{path:path}")
async def poe(request: Request, path: str):
    if backend_globals.poe_model_provider is None:
        backend_globals.poe_model_provider = PoeModelProvider()
        res = backend_globals.poe_model_provider.ping()
        print(f"ping: {res}")
    url = str(request.url)
    print(f"url: {url}")
    try:
        resp = await backend_globals.poe_model_provider.handle_request(path, request)

        if resp is str:
            return StreamingResponse(content=resp, media_type="text/event-stream")

        def event_stream(resp: httpx.Response):
            for chunk in resp.iter_text():
                if chunk:
                    yield chunk
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            content=event_stream(cast(httpx.Response, resp)),
            media_type="text/event-stream",
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    default_port = 11435
    port = os.getenv("PORT", default_port)
    try:
        port = int(port)
    except ValueError:
        port = default_port
    uvicorn.run(app, host="0.0.0.0", port=port)
