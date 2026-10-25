"""
接口层：聊天路由
"""
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from interfaces.http.dto.chat_dto import (
    ChatRequestDTO, ChatResponseDTO, ChatStreamChunkDTO,
    LoadModelRequestDTO, LoadModelResponseDTO,
    ModelInfoDTO, HealthResponseDTO,
)
from application.inference.handlers import ChatUseCase, CompletionUseCase, TokenCountUseCase
from application.model.handlers import LoadModelUseCase, UnloadModelUseCase, ListModelsUseCase
from domain.model.entities import GenerationConfig, EngineType
from domain.inference.ports import InferenceChunk

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])

# 依赖注入：实际运行时通过容器注入
_chat_use_case: Optional[ChatUseCase] = None
_completion_use_case: Optional[CompletionUseCase] = None
_token_count_use_case: Optional[TokenCountUseCase] = None
_load_model_use_case: Optional[LoadModelUseCase] = None
_unload_model_use_case: Optional[UnloadModelUseCase] = None
_list_models_use_case: Optional[ListModelsUseCase] = None


def set_dependencies(
    chat_use_case: ChatUseCase,
    completion_use_case: CompletionUseCase,
    token_count_use_case: TokenCountUseCase,
    load_model_use_case: LoadModelUseCase,
    unload_model_use_case: UnloadModelUseCase,
    list_models_use_case: ListModelsUseCase,
):
    """设置依赖（由容器调用）"""
    global _chat_use_case, _completion_use_case, _token_count_use_case
    global _load_model_use_case, _unload_model_use_case, _list_models_use_case
    _chat_use_case = chat_use_case
    _completion_use_case = completion_use_case
    _token_count_use_case = token_count_use_case
    _load_model_use_case = load_model_use_case
    _unload_model_use_case = unload_model_use_case
    _list_models_use_case = list_models_use_case


def _to_domain_config(dto) -> GenerationConfig:
    """DTO 转领域配置"""
    if dto is None:
        return GenerationConfig()
    return GenerationConfig(
        max_new_tokens=dto.max_new_tokens,
        temperature=dto.temperature,
        top_p=dto.top_p,
        top_k=dto.top_k,
        repetition_penalty=dto.repetition_penalty,
        presence_penalty=dto.presence_penalty,
        frequency_penalty=dto.frequency_penalty,
        min_p=dto.min_p,
        stop_sequences=tuple(dto.stop) if dto.stop else (),
        seed=dto.seed,
    )


def _format_chunk(chunk: InferenceChunk, model: str, chunk_id: str) -> ChatStreamChunkDTO:
    """格式化流式响应片段"""
    delta = {}
    if chunk.content:
        delta["content"] = chunk.content
    if chunk.reasoning_content:
        delta["reasoning_content"] = chunk.reasoning_content
    
    return ChatStreamChunkDTO(
        id=chunk_id,
        created=int(time.time()),
        model=model,
        choices=[{
            "index": 0,
            "delta": delta,
            "finish_reason": chunk.finish_reason,
        }],
    )


@router.post("/chat/completions")
async def chat_completions(request: ChatRequestDTO, http_request: Request):
    """聊天完成接口（OpenAI 兼容）"""
    if _chat_use_case is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    config = _to_domain_config(request.config)
    session_id = request.session_id or str(uuid.uuid4())
    
    if request.stream:
        return StreamingResponse(
            _stream_chat(request.model, request.messages, config, session_id, http_request),
            media_type="text/event-stream",
        )
    else:
        result = await _chat_use_case.execute_once(ChatCommand(
            messages=[msg.model_dump() for msg in request.messages],
            config=config,
            stream=False,
            session_id=session_id,
        ))
        return ChatResponseDTO(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop",
            }],
            usage={"prompt_tokens": 0, "completion_tokens": len(result.split()), "total_tokens": len(result.split())},
        )


async def _stream_chat(
    model: str, 
    messages: list, 
    config: GenerationConfig, 
    session_id: str,
    http_request: Request
) -> AsyncGenerator[str, None]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    
    try:
        async for chunk in _chat_use_case.execute(ChatCommand(
            messages=messages,
            config=config,
            stream=True,
            session_id=session_id,
        )):
            if await http_request.is_disconnected():
                break
            
            dto = _format_chunk(chunk, model, chunk_id)
            yield f"data: {dto.model_dump_json()}\n\n"
            
            if chunk.is_final:
                break
    except Exception as e:
        logger.error(f"Stream chat error: {e}")
        error_chunk = ChatStreamChunkDTO(
            id=chunk_id,
            created=int(time.time()),
            model=model,
            choices=[{"index": 0, "delta": {}, "finish_reason": "error"}],
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
    
    yield "data: [DONE]\n\n"


@router.post("/completions")
async def completions(request: ChatRequestDTO, http_request: Request):
    """文本补全接口（OpenAI 兼容）"""
    if _completion_use_case is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # 兼容 completions 格式：prompt 字符串
    prompt = request.messages[-1].content if request.messages else ""
    if isinstance(prompt, list):
        prompt = " ".join(p.get("text", "") for p in prompt if isinstance(p, dict))
    
    config = _to_domain_config(request.config)
    session_id = request.session_id or str(uuid.uuid4())
    
    if request.stream:
        return StreamingResponse(
            _stream_completion(prompt, config, session_id, http_request),
            media_type="text/event-stream",
        )
    else:
        result = await _completion_use_case.execute_once(CompletionCommand(
            prompt=prompt,
            config=config,
            stream=False,
            session_id=session_id,
        ))
        return ChatResponseDTO(
            id=f"cmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model,
            choices=[{"text": result, "index": 0, "finish_reason": "stop"}],
            usage={"prompt_tokens": 0, "completion_tokens": len(result.split()), "total_tokens": len(result.split())},
        )


async def _stream_completion(
    prompt: str, 
    config: GenerationConfig, 
    session_id: str,
    http_request: Request
) -> AsyncGenerator[str, None]:
    chunk_id = f"cmpl-{uuid.uuid4().hex}"
    
    try:
        async for chunk in _completion_use_case.execute(CompletionCommand(
            prompt=prompt,
            config=config,
            stream=True,
            session_id=session_id,
        )):
            if await http_request.is_disconnected():
                break
            
            dto = ChatStreamChunkDTO(
                id=chunk_id,
                created=int(time.time()),
                model="",
                choices=[{
                    "index": 0,
                    "text": chunk.content,
                    "finish_reason": chunk.finish_reason,
                }],
            )
            yield f"data: {dto.model_dump_json()}\n\n"
            
            if chunk.is_final:
                break
    except Exception as e:
        logger.error(f"Stream completion error: {e}")
    
    yield "data: [DONE]\n\n"


@router.post("/models/load", response_model=LoadModelResponseDTO)
async def load_model(request: LoadModelRequestDTO):
    """加载模型"""
    if _load_model_use_case is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        engine_type = EngineType(request.engine)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid engine: {request.engine}")
    
    event = _load_model_use_case.execute(LoadModelCommand(
        model_id=request.model_id,
        engine_preference=engine_type,
        context_length=request.context_length,
        quantization=request.quantization,
        vllm_options=request.vllm_options,
    ))
    
    return LoadModelResponseDTO(
        model_id=event.model_id,
        engine_type=event.engine_type,
        context_length=event.context_length,
    )


@router.post("/models/unload")
async def unload_model():
    """卸载当前模型"""
    if _unload_model_use_case is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    success = _unload_model_use_case.execute(UnloadModelCommand())
    return {"success": success}


@router.get("/models", response_model=List[ModelInfoDTO])
async def list_models():
    """列出可用模型"""
    if _list_models_use_case is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    models = _list_models_use_case.execute(ListModelsQuery())
    return [ModelInfoDTO(
        id=m.id.value,
        name=m.name,
        type=m.model_type.value,
        path=str(m.path),
        size_bytes=m.metadata.get("size_bytes"),
        architecture=m.metadata.get("architecture"),
        description=m.metadata.get("description", ""),
        tags=m.metadata.get("tags", []),
        loaded=m.is_loaded,
    ) for m in models]


@router.get("/health", response_model=HealthResponseDTO)
async def health_check():
    """健康检查"""
    # TODO: 从推理服务获取状态
    return HealthResponseDTO(
        status="healthy",
        model_loaded=False,
    )