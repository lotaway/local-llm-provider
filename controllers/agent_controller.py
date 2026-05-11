from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import logging
import uuid
import asyncio
import time
from constants import DATA_PATH

from globals import (
    limiter,
    local_rag,
    agent_runtime,
    context_storage,
    permission_manager,
)
from model_providers import LocalLLModel
from schemas import ChatRequest
from agents.context_storage import create_context_storage, MemoryContextStorage
from agents.agent_runtime import AgentRuntime
from agents.runtime_factory import RuntimeFactory
from utils import ContentType
from rag import LocalRAG

router = APIRouter(prefix="/agents", tags=["agent"])
logger = logging.getLogger(__name__)


class AgentRequest(BaseModel):
    model: str
    messages: list[str]
    session_id: str = None
    stream: bool = False
    files: list[str] = []


class AgentDecisionRequest(BaseModel):
    approved: bool
    feedback: str = ""
    data: dict = None


class AgentExecutionResultRequest(BaseModel):
    success: bool
    data: Any = None
    error: str = ""


@router.post("/run")
@limiter.limit("10/minute")
async def query_agent(agentRequest: AgentRequest, request: Request):
    storage = _get_context_storage()
    session_id = agentRequest.session_id or str(uuid.uuid4())
    runtime = _get_agent_runtime(session_id, storage)
    initial_context = _load_initial_context(agentRequest.files)
    
    if agentRequest.stream:
        return await _stream_agent_execution(runtime, agentRequest.messages, initial_context, request)
    
    state = await runtime.execute(agentRequest.messages, start_agent="qa", initial_context=initial_context)
    return JSONResponse(content=state.to_dict())

def _get_context_storage():
    global context_storage
    if context_storage is None:
        try:
            context_storage = create_context_storage()
        except Exception:
            context_storage = MemoryContextStorage()
    return context_storage

def _get_agent_runtime(session_id: str, storage):
    global agent_runtime, local_rag
    if agent_runtime is None:
        model = LocalLLModel.init_local_model()
        local_rag = local_rag or LocalRAG(model)
        agent_runtime = RuntimeFactory.create_with_all_agents(
            model, rag_instance=local_rag, permission_manager=permission_manager,
            context_storage=storage, session_id=session_id
        )
    elif agent_runtime.session_id != session_id:
        agent_runtime.session_id = session_id
        agent_runtime._load_saved_state()
    return agent_runtime

def _load_initial_context(files: list[str]) -> dict:
    if not files: return {}
    file_map = {}
    from utils import FileProcessor
    processor = FileProcessor()
    for fid in files:
        try:
            file_map[fid] = processor.get_file_content(fid)
        except Exception:
            continue
    return {"files": file_map}

async def _stream_agent_execution(runtime, query, context, request):
    queue = asyncio.Queue()
    async def callback(chunk): await queue.put(chunk)
    
    async def run():
        try: await runtime.execute(query, stream_callback=callback, initial_context=context)
        finally: await queue.put(None)
    
    asyncio.create_task(run())
    return StreamingResponse(_event_generator(queue, request), media_type="text/event-stream")

async def _event_generator(queue, request):
    while True:
        if await request.is_disconnected(): break
        try:
            chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
            if chunk is None: break
            yield f"data: {chunk}\n\n"
        except asyncio.TimeoutError: continue


@router.post("/decision")
@limiter.limit("5/minute")
async def agent_decision(req: AgentDecisionRequest, request: Request):
    global agent_runtime

    if agent_runtime is None:
        raise HTTPException(status_code=400, detail="Agent runtime not initialized")

    try:
        result = await agent_runtime.handle_decision(
            req.approved, req.feedback, req.data
        )
        return {"success": True, "result": result.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute")
@limiter.limit("20/minute")
async def agent_execute_result(req: AgentExecutionResultRequest, request: Request):
    global agent_runtime

    if agent_runtime is None:
        raise HTTPException(status_code=400, detail="Agent runtime not initialized")

    try:
        result = await agent_runtime.handle_client_result(
            req.data, req.success, req.error
        )
        return {"success": True, "result": result.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
@limiter.limit("15/minute")
async def agent_chat(req: ChatRequest, request: Request):
    runtime = _ensure_chat_runtime()
    query = _extract_query_from_messages(req.messages)
    
    if _has_images(req.messages):
        query = await _process_multimodal_query(query, req.messages)
    
    state = await runtime.execute(query, start_agent="qa")
    return JSONResponse(content=_format_openai_response(state, req.model))

def _ensure_chat_runtime():
    global agent_runtime, local_rag
    if agent_runtime is None:
        model = LocalLLModel.init_local_model()
        local_rag = local_rag or LocalRAG(model, data_path=DATA_PATH)
        agent_runtime = RuntimeFactory.create_with_all_agents(model, rag_instance=local_rag)
    return agent_runtime

def _extract_query_from_messages(messages) -> str:
    model = LocalLLModel.init_local_model()
    for m in reversed(messages):
        if m.role == "user":
            return model._extract_text_from_content(m.content)
    return model._extract_text_from_content(messages[-1].content) if messages else ""

def _has_images(messages) -> bool:
    for m in messages:
        if isinstance(m.content, list):
            for part in m.content:
                if isinstance(part, dict) and part.get("type") != ContentType.TEXT.value:
                    return True
    return False

async def _process_multimodal_query(query, messages) -> str:
    from globals import default_vlm, MultimodalFactory, multimodal_model
    vlm = multimodal_model
    if vlm is None or vlm.model_name != default_vlm:
        vlm = MultimodalFactory.get_model(default_vlm)
    
    prompt = "Analyze the provided image and return a JSON object describing visual content (objects, text, colors, spatial relations). Extract text if present."
    history = [{"role": "system", "content": prompt}] + [m.model_dump() if hasattr(m, "model_dump") else m for m in messages]
    
    description = await asyncio.to_thread(lambda: vlm.chat(history))
    return f"{query}\n\n[Image Analysis]: {description}"

def _format_openai_response(state, model_name) -> dict:
    answer = state.final_result if state.status.value == "completed" else f"Workflow {state.status.value}: {state.error_message}"
    return {
        "id": f"agent-chat-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": len(str(answer).split()), "total_tokens": len(str(answer).split())},
        "agent_metadata": {"status": state.status.value, "iterations": state.iteration_count, "history_length": len(state.history), "core_meta": state.final_meta or {}}
    }


@router.get("/status")
async def agent_status():
    if agent_runtime is None:
        return {"status": "not_initialized"}
    state = agent_runtime.get_state()
    return {
        "status": state.status.value, "current_agent": state.current_agent,
        "iteration_count": state.iteration_count, "max_iterations": state.max_iterations,
        "history_length": len(state.history), "context_keys": list(state.context.keys())
    }
