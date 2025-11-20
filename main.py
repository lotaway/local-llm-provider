import os
from fastapi import FastAPI, HTTPException, Request
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
from typing import cast

# import triton
# import triton.language as tl
from dotenv import load_dotenv

load_dotenv()

from poe_model_provider import PoeModelProvider
from model_provider import LocalLLModel
from comfyui_provider import ComfyUIProvider
from rag import LocalRAG

local_rag = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str


class PromptTokensDetail:
    cached_tokens: int
    audio_tokens: int


class CompletionTokensDetail:
    audio_tokens: int
    reasoning_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int


class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetail
    completion_tokens_details: CompletionTokensDetail
    input_tokens: int
    output_tokens: int
    input_tokens_details: dict


class ApiResponse:
    id: str
    object: str
    created: int
    model: str
    choices: list[dict]
    usage: Usage


class EmbeddingRequest(BaseModel):
    model: str
    input: list[str] | str


@app.get("/")
async def index():
    return "Hello, Local LLM Provider!"


@app.get("/manifest.json")
async def manifest():
    return FileResponse("manifest.json")


@app.get("/mcp")
async def query_rag(request: Request):
    global local_rag
    global local_model
    query = request.query_params.get("query")
    if query is None:
        raise HTTPException(
            status_code=400, detail="Either query or request parameter is required"
        )

    if local_rag is None:
        if local_model is None:
            local_model = LocalLLModel()
        data_path = os.getenv("DATA_PATH", "./docs")
        local_rag = LocalRAG(local_model, data_path=data_path)
    result = local_rag.rag_chain.invoke(query)

    if result is str:
        return PlainTextResponse(result)

    def event_stream():
        yield f"data: {result}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


comfyui_provider = ComfyUIProvider()


@app.api_route(
    "/comfyui/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
)
async def comfyui(request: Request, path: str):
    return await comfyui_provider.proxy_request(request)


poe_model_provider = None


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


local_model = None


@app.post("/api/show")
async def api_show():
    return {"ok": True}


@app.get("/api/tags")
async def api_tags():
    li = []
    for model in LocalLLModel.get_models():
        li.append(
            {
                "name": model,
                "version": "1.0.0",
                "object": "model",
                "owned_by": "lotaway",
                "api_version": "v1",
            }
        )
    return li


@app.get("/api/version")
async def api_version():
    li = await api_tags()
    if local_model is not None:
        return {
            "model_name": "unknown",
            "version": "1.0.0",
            "object": "model",
            "owned_by": "lotaway",
            "api_version": "v1",
        }
    _local_model = cast(LocalLLModel, local_model)
    cur = next(model for model in li if model["name"] == _local_model.cur_model_name)
    return cur


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest):
    if isinstance(req.input, str):
        texts = [req.input]
    else:
        texts = req.input
    global local_model
    if local_model is None:
        local_model = LocalLLModel(embedding_model_name=req.model)
    vectors = local_model.tokenizer.encode(texts).tolist()
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ],
        "model": req.model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """openai chat/edit/apply"""
    global local_model
    if local_model is None:
        local_model = LocalLLModel(req.model)
    if req.stream:
        streamer = local_model.chat([m.model_dump() for m in req.messages])

        def event_stream():
            for chunk in streamer:
                data = {
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [
                        {"delta": {"content": chunk}, "index": 0, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    output = local_model.chat_at_once([m.model_dump() for m in req.messages])
    response = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(output.split()),
            "total_tokens": len(output.split()),
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "input_tokens": 0,
            "output_tokens": 0,
            "input_tokens_details": None,
        },
    }
    return JSONResponse(content=response, headers={"Content-Type": "application/json"})


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    """openai autocompletions"""
    global local_model
    if local_model is None:
        local_model = LocalLLModel(req.model)
    output = local_model.complete_at_once(req.prompt)
    return {
        "id": "cmpl-1",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "text": output, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": len(req.prompt.split()),
            "completion_tokens": len(output.split()),
            "total_tokens": len(req.prompt.split()) + len(output.split()),
        },
    }


if __name__ == "__main__":
    default_port = 11434
    port = os.getenv("PORT", default_port)
    try:
        port = int(port)
    except ValueError:
        port = default_port
    uvicorn.run(app, host="0.0.0.0", port=port)
