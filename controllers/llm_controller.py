from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import logging
import uuid
import time
import asyncio
import httpx
import json
from typing import cast
import os

from globals import limiter
import globals as backend_globals
from auth import get_multimodal_headers
from model_providers import LocalLLModel, RemoteModelProvider
from remote_providers import OpenAIModelProvider, OpenAISettings
from constants import (
    CUSTOM_LLM_API_KEY,
    CUSTOM_LLM_BASE_URL,
    CUSTOM_LLM_MODEL,
    CUSTOM_LLM_PROTOCOL,
)
from schemas import Message, ChatRequest
from rag import LocalRAG
from utils import ContentType

router = APIRouter()
logger = logging.getLogger(__name__)
OTHER_VERSION = "v1"


def _has_value(value: str) -> bool:
    return value.strip() != ""


def _none_if_empty(value: str):
    return value.strip() or None


def _parse_timeout(value: str):
    if value.strip() == "":
        return None
    return float(value)


def _get_poe_models():
    # Poe is now just a specific case of custom OpenAI protocol
    if CUSTOM_LLM_PROTOCOL == "poe" and _has_value(CUSTOM_LLM_MODEL):
        return [CUSTOM_LLM_MODEL]
    return []


def _get_custom_models():
    if not _has_value(CUSTOM_LLM_MODEL):
        return []
    return [CUSTOM_LLM_MODEL]


def _is_remote_model(model: str) -> bool:
    return model in _get_poe_models() or model in _get_custom_models()


# Removed _build_poe_provider as Poe uses OpenAIModelProvider


def _build_custom_provider() -> OpenAIModelProvider:
    settings = OpenAISettings(
        api_key=CUSTOM_LLM_API_KEY,
        base_url=CUSTOM_LLM_BASE_URL,
        proxy_url=os.getenv("HTTP_PROXY"),
        timeout=60.0,
    )
    return OpenAIModelProvider(settings)


def _get_remote_provider(model: str):
    if model == CUSTOM_LLM_MODEL:
        return _build_custom_provider()
    raise HTTPException(status_code=400, detail="Unsupported remote model")


def _stream_response(resp: httpx.Response):
    def event_stream():
        for chunk in resp.iter_text():
            if chunk:
                yield chunk
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def _proxy_remote_chat(body_data: dict, request: Request, model: str):
    provider = _get_remote_provider(model)
    resp = await provider.handle_request("chat/completions", request, body_data)
    if isinstance(resp, str):
        return StreamingResponse(content=resp, media_type="text/event-stream")
    if body_data.get("stream", False):
        return _stream_response(cast(httpx.Response, resp))
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    enable_rag: bool = False


@router.post("/chat/completions", tags=["chat"])
@limiter.limit("20/minute")
async def chat_completions(req: ChatRequest, request: Request):
    _inject_files(req)
    
    if _is_remote_model(req.model) and not req.enable_rag:
        return await _proxy_remote_chat(req.model_dump(), request, req.model)

    if _has_images(req.messages):
        return await _handle_multimodal_flow(req)

    model = _get_llm_instance(req.model)
    _truncate_context(req, model)
    
    if req.enable_rag:
        return await _handle_rag_flow(req, model, request)
    return await _handle_standard_chat_flow(req, model, request)

def _inject_files(req: ChatRequest):
    if not req.files: return
    from utils import FileProcessor
    req.messages = FileProcessor().inject_file_context_to_messages(req.messages, req.files)

def _has_images(messages: list) -> bool:
    for m in messages:
        if not isinstance(m.content, list): continue
        for part in m.content:
            if isinstance(part, dict) and part.get("type") != ContentType.TEXT.value:
                return True
    return False

async def _handle_multimodal_flow(req: ChatRequest):
    prompt = "Analyze the provided image and return a JSON object describing visual content (objects, text, colors, spatial relations). Extract text if present."
    history = [{"role": "system", "content": prompt}] + [m.model_dump() for m in req.messages]
    
    if backend_globals.remote_multimodal_status and backend_globals.MULTIMODAL_PROVIDER_URL:
        try: return await _proxy_remote_multimodal(req, history)
        except Exception: logger.error("Remote multimodal failed, falling back")

    vlm = _get_vlm_instance(req.model)
    output = await asyncio.to_thread(lambda: vlm.chat(history))
    return JSONResponse(content=_format_vlm_response(output, req.model))

async def _proxy_remote_multimodal(req: ChatRequest, history: list):
    payload = req.model_dump()
    payload["messages"] = history
    if req.stream: return await _stream_remote_vlm(payload)
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{backend_globals.MULTIMODAL_PROVIDER_URL}/{OTHER_VERSION}/chat/completions", json=payload, headers=get_multimodal_headers(), timeout=60.0)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

async def _stream_remote_vlm(payload: dict):
    async def stream():
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", f"{backend_globals.MULTIMODAL_PROVIDER_URL}/{OTHER_VERSION}/chat/completions", json=payload, headers=get_multimodal_headers(), timeout=60.0) as resp:
                async for chunk in resp.aiter_text(): yield chunk
    return StreamingResponse(stream(), media_type="text/event-stream")

def _get_vlm_instance(model_name: str):
    is_vlm = model_name in backend_globals.vlm_models or any(x in model_name.lower() for x in ["janus", "llava", "vl"])
    target = model_name if is_vlm else backend_globals.default_vlm
    if backend_globals.multimodal_model is None or backend_globals.multimodal_model.model_name != target:
        backend_globals.multimodal_model = backend_globals.MultimodalFactory.get_model(target)
    return backend_globals.multimodal_model

def _format_vlm_response(output: str, model_name: str) -> dict:
    return {
        "id": f"chatcmpl-vlm-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()), "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": output}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": len(str(output).split()), "total_tokens": len(str(output).split())}
    }

def _get_llm_instance(model_name: str):
    if _is_remote_model(model_name): return RemoteModelProvider(model_name)
    return LocalLLModel.init_local_model(model_name)

def _truncate_context(req: ChatRequest, model):
    try:
        msgs = [m.model_dump() for m in req.messages]
        truncated = model.smart_truncate_messages(msgs)
        if len(truncated) < len(msgs) or (truncated and msgs and truncated[0]["content"] != msgs[0]["content"]):
            req.messages = [Message(**m) for m in truncated]
    except Exception as e:
        logger.warning(f"Truncation failed: {e}")

async def _handle_rag_flow(req: ChatRequest, model, request: Request):
    if backend_globals.local_rag is None or backend_globals.local_rag.llm != model:
        backend_globals.local_rag = LocalRAG(model)
    query = _extract_query(req.messages, model)
    if req.stream: return await _stream_rag_answer(query, req.model, request)
    result = await backend_globals.local_rag.generate_answer(query)
    return JSONResponse(content=_format_chat_response(result, req.model))

async def _handle_standard_chat_flow(req: ChatRequest, model, request: Request):
    if req.stream: return await _stream_chat_answer(model, req, request)
    result = await model.chat_at_once(req.messages, max_new_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p)
    return JSONResponse(content=_format_chat_response(result, req.model))

def _extract_query(messages: list, model) -> str:
    for m in reversed(messages):
        if m.role == "user": return model._extract_text_from_content(m.content)
    return model._extract_text_from_content(messages[-1].content) if messages else ""

async def _stream_rag_answer(query: str, model_name: str, request: Request):
    queue = asyncio.Queue()
    async def callback(chunk): await queue.put(chunk)
    async def run():
        try: await backend_globals.local_rag.generate_answer(query, stream_callback=callback)
        except asyncio.CancelledError: pass
        finally: await queue.put(None)
    task = asyncio.create_task(run())
    return StreamingResponse(_chat_stream_generator(queue, model_name, request, task), media_type="text/event-stream")

async def _stream_chat_answer(model, req: ChatRequest, request: Request):
    queue = asyncio.Queue()
    async def run():
        try:
            async for chunk in model.chat(req.messages, max_new_tokens=req.max_tokens, temperature=req.temperature, top_p=req.top_p):
                if not isinstance(chunk, int): await queue.put(chunk)
        except asyncio.CancelledError: pass
        finally: await queue.put(None)
    task = asyncio.create_task(run())
    return StreamingResponse(_chat_stream_generator(queue, req.model, request, task), media_type="text/event-stream")

async def _chat_stream_generator(queue: asyncio.Queue, model_name: str, request: Request, bg_task: asyncio.Task = None):
    last_reason = None
    while True:
        if await request.is_disconnected():
            if bg_task:
                bg_task.cancel()
            break
        try: chunk = await asyncio.wait_for(queue.get(), 0.1)
        except asyncio.TimeoutError: continue
        if chunk is None: break
        
        delta, reason = _parse_chat_chunk(chunk)
        if reason: last_reason = reason
        yield f"data: {json.dumps(_format_chat_chunk(delta, reason, model_name))}\n\n"
    
    yield f"data: {json.dumps(_format_chat_chunk({}, last_reason or 'stop', model_name))}\n\n"
    yield "data: [DONE]\n\n"

def _parse_chat_chunk(chunk):
    if isinstance(chunk, dict):
        return {k: v for k, v in chunk.items() if k != "finish_reason"}, chunk.get("finish_reason")
    return {"content": chunk}, None

def _format_chat_chunk(delta: dict, reason: str | None, model: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion.chunk", "created": int(time.time()),
        "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]
    }

def _format_chat_response(content: str, model: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": len(str(content).split()), "total_tokens": len(str(content).split())}
    }

@router.post("/completions", tags=["completions"])
async def completions(req: CompletionRequest):
    model = LocalLLModel.init_local_model(req.model)
    if req.enable_rag:
        if backend_globals.local_rag is None: backend_globals.local_rag = LocalRAG(model)
        result = await backend_globals.local_rag.generate_answer(req.prompt)
        return JSONResponse(content=_format_completion_response(result, req.model))
    result = await model.complete(req.prompt)
    return JSONResponse(content=_format_completion_response(result, req.model))

def _format_completion_response(text: str, model: str) -> dict:
    return {
        "id": f"cmpl-{uuid.uuid4().hex}", "object": "text_completion", "created": int(time.time()), "model": model,
        "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": len(str(text).split()), "total_tokens": len(str(text).split())}
    }
