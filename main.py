import os
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
    FileResponse,
    PlainTextResponse,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
import json
import httpx
import logging
import uuid
import secrets
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

from globals import (
    comfyui_provider,
    local_rag,
    MULTIMODAL_PROVIDER_URL,
    remote_multimodal_status,
)
from model_providers import LocalLLModel, PoeModelProvider
from model_providers.multimodal_provider import MultimodalFactory

# Import version router
from routers.version import router as version_router
from controllers.base_controller import router as base_router


def get_multimodal_headers():
    headers = {"Content-Type": "application/json"}
    MULTIMODAL_ADMIN_TOKEN = os.getenv("MULTIMODAL_ADMIN_TOKEN", ADMIN_TOKEN)
    if MULTIMODAL_ADMIN_TOKEN is None:
        raise ValueError("MULTIMODAL_ADMIN_TOKEN is not set")
    headers["Authorization"] = f"Bearer {MULTIMODAL_ADMIN_TOKEN}"
    return headers


async def check_multimodal_health():
    """Background task to check external multimodal provider health"""
    global remote_multimodal_status
    if not MULTIMODAL_PROVIDER_URL:
        return

    while True:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{MULTIMODAL_PROVIDER_URL}/api/show",
                    headers=get_multimodal_headers(),
                    timeout=5,
                )
                if resp.status_code == 200 and resp.json().get("ok"):
                    remote_multimodal_status = True
                else:
                    remote_multimodal_status = False
        except Exception:
            remote_multimodal_status = False

        await asyncio.sleep(10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if MULTIMODAL_PROVIDER_URL:
        asyncio.create_task(check_multimodal_health())
    yield
    # Shutdown (if needed in the future)


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

ADMIN_TOKEN = None


def ensure_admin_token():
    global ADMIN_TOKEN
    token_file = ".admin"
    if os.path.exists(token_file):
        with open(token_file, "r") as f:
            token = f.read().strip()
            # Check if token looks valid (starts with sk- and has content)
            if token and token.startswith("sk-") and len(token) >= 32:
                ADMIN_TOKEN = token
                print(f"Loaded admin token from {token_file}")
                return

    # Generate new token: sk- + 32 hex chars
    token = f"sk-{secrets.token_hex(16)}"
    with open(token_file, "w") as f:
        f.write(token)
    ADMIN_TOKEN = token
    print(f"Generated new admin token and saved to {token_file}: {token}")


# Initialize token on module load
ensure_admin_token()


@app.middleware("http")
async def verify_admin_token(request: Request, call_next):
    # Only check routes starting with /VERSION (e.g. /v1)
    if request.url.path.startswith(f"/{VERSION}"):
        # Skip OPTIONS requests (CORS preflight)
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

        if token != ADMIN_TOKEN:
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
    global local_rag

    query = request.query_params.get("query")
    if query is None:
        raise HTTPException(
            status_code=400, detail="Either query or request parameter is required"
        )

    try:
        if local_rag is None:
            local_model = LocalLLModel.init_local_model()
            local_rag = LocalRAG(local_model)
        result = await local_rag.generate_answer(query)

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

    return await comfyui_provider.proxy_request(request)


@app.post("/poe/v1/{path:path}")
async def poe(request: Request, path: str):
    global poe_model_provider
    if poe_model_provider is None:
        poe_model_provider = PoeModelProvider()
        res = poe_model_provider.ping()
        print(f"ping: {res}")
    url = str(request.url)
    print(f"url: {url}")
    try:
        resp = await poe_model_provider.handle_request(path, request)

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
    default_port = 11434
    port = os.getenv("PORT", default_port)
    try:
        port = int(port)
    except ValueError:
        port = default_port
    uvicorn.run(app, host="0.0.0.0", port=port)
