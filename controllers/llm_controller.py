from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import logging
import uuid
import time
import asyncio
import httpx
import json
from typing import cast

# Import global variables and functions
from globals import local_rag, multimodal_model
from globals import MULTIMODAL_PROVIDER_URL, remote_multimodal_status, default_vlm
from globals import vlm_models, MultimodalFactory
from main import get_multimodal_headers
from model_providers import LocalLLModel
from schemas import Message, ChatRequest, CompletionRequest
from rag import LocalRAG
from utils import ContentType

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)
OTHER_VERSION = "v1"


@router.post("/completions")
async def chat_completions(req: ChatRequest, request: Request):
    global local_rag
    global multimodal_model

    if req.files:
        from utils import FileProcessor

        file_processor = FileProcessor()
        req.messages = file_processor.inject_file_context_to_messages(
            req.messages, req.files
        )

    has_images = False
    for m in req.messages:
        if isinstance(m.content, list):
            for part in m.content:
                if isinstance(part, dict) and (
                    part.get("type") != ContentType.TEXT.value
                ):
                    has_images = True
                    break
        if has_images:
            break

    if has_images:
        multimodal_prompt = "请分析所提供的图像，并返回一个包含视觉内容详细描述的 JSON 对象，内容涵盖物体、文字、颜色以及空间位置关系。如果图像中主体信息包含大量文字，请提取出文字内容。"

        system_msg = {"role": "system", "content": multimodal_prompt}
        msgs_dicts = [
            m.model_dump() if hasattr(m, "model_dump") else m for m in req.messages
        ]
        multimodal_messages = [system_msg] + msgs_dicts

        if remote_multimodal_status and MULTIMODAL_PROVIDER_URL:
            try:
                payload = req.model_dump()
                payload["messages"] = multimodal_messages

                if req.stream:

                    async def remote_stream():
                        async with httpx.AsyncClient() as client:
                            async with client.stream(
                                "POST",
                                f"{MULTIMODAL_PROVIDER_URL}/{OTHER_VERSION}/chat/completions",
                                json=payload,
                                headers=get_multimodal_headers(),
                                timeout=60.0,
                            ) as resp:
                                async for chunk in resp.aiter_text():
                                    yield chunk

                    return StreamingResponse(
                        remote_stream(), media_type="text/event-stream"
                    )
                else:
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(
                            f"{MULTIMODAL_PROVIDER_URL}/{OTHER_VERSION}/chat/completions",
                            json=payload,
                            headers=get_multimodal_headers(),
                            timeout=60.0,
                        )
                        return JSONResponse(
                            content=resp.json(), status_code=resp.status_code
                        )

            except Exception as e:
                logger.error(
                    f"Remote multimodal failed: {e}, falling back to local if available"
                )

        is_vlm_request = (
            req.model in vlm_models
            or "janus" in req.model.lower()
            or "llava" in req.model.lower()
            or "vl" in req.model.lower()
        )

        target_model_name = req.model if is_vlm_request else default_vlm

        if multimodal_model is None or multimodal_model.model_name != target_model_name:
            multimodal_model = MultimodalFactory.get_model(target_model_name)

        def run_multimodal():
            return multimodal_model.chat(multimodal_messages)

        output = await asyncio.to_thread(run_multimodal)

        response = {
            "id": f"chatcmpl-janus-{uuid.uuid4().hex}",
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
                "completion_tokens": len(str(output).split()),
                "total_tokens": len(str(output).split()),
            },
        }
        return JSONResponse(content=response)

    local_model = LocalLLModel.init_local_model(req.model)

    try:
        msgs_dicts = [
            m.model_dump() if hasattr(m, "model_dump") else m for m in req.messages
        ]
        truncated_dicts = local_model.smart_truncate_messages(msgs_dicts)
        if len(truncated_dicts) < len(msgs_dicts) or (
            len(truncated_dicts) > 0
            and len(msgs_dicts) > 0
            and truncated_dicts[0]["content"] != msgs_dicts[0]["content"]
        ):
            req.messages = [Message(**m) for m in truncated_dicts]

    except Exception as e:
        logger.warning(f"Context truncation process failed: {e}")

    if req.enable_rag:
        if local_rag is None:
            local_rag = LocalRAG(local_model)

        query = ""
        for m in reversed(req.messages):
            if m.role == "user":
                query = m.content
                break
        if not query and req.messages:
            query = req.messages[-1].content

        if req.stream:
            event_queue = asyncio.Queue()

            async def stream_callback(chunk):
                await event_queue.put(chunk)

            async def run_rag():
                try:
                    await cast(LocalRAG, local_rag).generate_answer(
                        query, stream_callback=stream_callback
                    )
                except Exception as e:
                    print(f"RAG Error: {e}")
                finally:
                    await event_queue.put(None)

            asyncio.create_task(run_rag())

            async def event_stream():
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        try:
                            chunk = await asyncio.wait_for(
                                event_queue.get(), timeout=0.1
                            )
                        except asyncio.TimeoutError:
                            continue

                        if chunk is None:
                            break

                        data = {
                            "id": f"chatcmpl-{uuid.uuid4().hex}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": req.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    except Exception as e:
                        logger.error(f"Stream error: {e}")
                        break

                data = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")
        else:
            try:
                result = await local_rag.generate_answer(query)
                response = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": result},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": len(str(result).split()),
                        "total_tokens": len(str(result).split()),
                    },
                }
                return JSONResponse(content=response)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    else:
        if req.stream:
            event_queue = asyncio.Queue()

            async def stream_callback(chunk):
                await event_queue.put(chunk)

            async def run_chat():
                try:
                    await local_model.chat(
                        req.messages, stream_callback=stream_callback
                    )
                except Exception as e:
                    print(f"Chat Error: {e}")
                finally:
                    await event_queue.put(None)

            asyncio.create_task(run_chat())

            async def event_stream():
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        try:
                            chunk = await asyncio.wait_for(
                                event_queue.get(), timeout=0.1
                            )
                        except asyncio.TimeoutError:
                            continue

                        if chunk is None:
                            break

                        data = {
                            "id": f"chatcmpl-{uuid.uuid4().hex}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": req.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    except Exception as e:
                        logger.error(f"Stream error: {e}")
                        break

                data = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")
        else:
            try:
                result = await local_model.chat(req.messages)
                response = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": result},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": len(str(result).split()),
                        "total_tokens": len(str(result).split()),
                    },
                }
                return JSONResponse(content=response)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))


@router.post("/completions")
async def completions(req: CompletionRequest):
    global local_rag

    local_model = LocalLLModel.init_local_model(req.model)

    if req.enable_rag:
        if local_rag is None:
            local_rag = LocalRAG(local_model)

        try:
            result = await local_rag.generate_answer(req.prompt)
            response = {
                "id": f"cmpl-{uuid.uuid4().hex}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [
                    {
                        "text": result,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(str(result).split()),
                    "total_tokens": len(str(result).split()),
                },
            }
            return JSONResponse(content=response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        try:
            result = await local_model.complete(req.prompt)
            response = {
                "id": f"cmpl-{uuid.uuid4().hex}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [
                    {
                        "text": result,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(str(result).split()),
                    "total_tokens": len(str(result).split()),
                },
            }
            return JSONResponse(content=response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
