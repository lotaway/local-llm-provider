from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import logging
import uuid
import asyncio
import time
from constants import DATA_PATH

from globals import (
    local_rag,
    agent_runtime,
    context_storage,
    permission_manager,
)
from model_providers import LocalLLModel
from schemas import ChatRequest
from agents.context_storage import create_context_storage, MemoryContextStorage
from agents.agent_runtime import AgentRuntime
from utils import ContentType
from rag import LocalRAG

router = APIRouter(prefix="/agents", tags=["agent"])
logger = logging.getLogger(__name__)


class AgentRequest(BaseModel):
    model: str
    messages: list[str]
    session_id: str = None
    files: list[str] = []


class AgentDecisionRequest(BaseModel):
    approved: bool
    feedback: str = ""
    data: dict = None


@router.post("/run")
async def query_agent(agentRequest: AgentRequest, request: Request):
    global local_rag
    global agent_runtime
    global context_storage

    query = agentRequest.messages
    if query is None:
        raise HTTPException(status_code=400, detail="Query parameter is required")

    if context_storage is None:
        try:
            context_storage = create_context_storage()
            logger.info(
                f"Initialized context storage: {type(context_storage).__name__}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize context storage: {e}")
            context_storage = MemoryContextStorage()

    session_id = agentRequest.session_id or str(uuid.uuid4())
    logger.info(f"Processing request for session: {session_id}")

    if agent_runtime is None:
        local_model = LocalLLModel.init_local_model()
        if local_rag is None:
            local_rag = LocalRAG(local_model)
        agent_runtime = AgentRuntime.create_with_all_agents(
            local_model,
            rag_instance=local_rag,
            permission_manager=permission_manager,
            context_storage=context_storage,
            session_id=session_id,
        )
    else:
        agent_runtime.session_id = session_id
        loaded_state = agent_runtime._load_state()
        if loaded_state:
            agent_runtime.state = loaded_state
            logger.info(f"Resumed existing session {session_id}")

    initial_context = {}
    if agentRequest.files:
        file_map = {}
        available_files = []
        from utils import FileProcessor

        file_processor = FileProcessor()
        for file_id in agentRequest.files:
            try:
                file_content = file_processor.get_file_content(file_id)
                file_map[file_id] = file_content
                available_files.append(file_id)
            except Exception as e:
                logger.warning(f"Failed to load file {file_id}: {e}")

        if available_files:
            initial_context["files"] = file_map
            logger.info(f"Included {len(available_files)} files in context")

    if agentRequest.stream:
        event_queue = asyncio.Queue()

        async def stream_callback(chunk):
            await event_queue.put(chunk)

        async def run_agent():
            try:
                await agent_runtime.execute_async(
                    query, initial_context, stream_callback
                )
            except Exception as e:
                logger.error(f"Agent execution failed: {e}")
            finally:
                await event_queue.put(None)

        asyncio.create_task(run_agent())

        async def event_stream():
            while True:
                if await request.is_disconnected():
                    break
                try:
                    chunk = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                    if chunk is None:
                        break
                    yield f"data: {chunk}\n\n"
                except asyncio.TimeoutError:
                    continue

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        try:
            state = agent_runtime.execute(
                query, start_agent="qa", initial_context=initial_context
            )
            return JSONResponse(content=state.to_dict())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/decision")
async def agent_decision(req: AgentDecisionRequest):
    global agent_runtime

    if agent_runtime is None:
        raise HTTPException(status_code=400, detail="Agent runtime not initialized")

    try:
        result = await agent_runtime.handle_decision(
            req.approved, req.feedback, req.data
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def agent_chat(req: ChatRequest):
    global agent_runtime
    global local_rag

    if agent_runtime is None:
        local_model = LocalLLModel.init_local_model()
        if local_rag is None:
            import os

            data_path = DATA_PATH
            local_rag = LocalRAG(local_model, data_path=data_path)
        from agents.agent_runtime import AgentRuntime

        agent_runtime = AgentRuntime.create_with_all_agents(
            local_model, rag_instance=local_rag
        )

    query = ""
    for m in reversed(req.messages):
        if m.role == "user":
            query = m.content
            break
    if not query and req.messages:
        query = req.messages[-1].content

    has_images = False
    multimodal_messages = []
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
        from globals import multimodal_model, default_vlm, MultimodalFactory

        multimodal_prompt = "请分析所提供的图像，并返回一个包含视觉内容详细描述的 JSON 对象，内容涵盖物体、文字、颜色以及空间位置关系。如果图像中主体信息包含大量文字，请提取出文字内容。"

        system_msg = {"role": "system", "content": multimodal_prompt}
        msgs_dicts = [
            m.model_dump() if hasattr(m, "model_dump") else m for m in req.messages
        ]
        multimodal_messages = [system_msg] + msgs_dicts

        # Determine which model to use for description
        target_vlm_name = default_vlm

        if multimodal_model is None or multimodal_model.model_name != target_vlm_name:
            multimodal_model = MultimodalFactory.get_model(target_vlm_name)

        def run_multimodal():
            return multimodal_model.chat(multimodal_messages)

        description = await asyncio.to_thread(run_multimodal)
        text_query = ""
        for part in query:
            if isinstance(part, dict) and part.get("type") == ContentType.TEXT.value:
                text_query += part.get(ContentType.TEXT.value, "") + "\n"

        query = f"{text_query}\n\n[Image Analysis]: {description}"

    try:
        state = agent_runtime.execute(query, start_agent="qa")

        if state.status.value == "completed":
            answer = state.final_result
            status = "success"
        else:
            answer = f"Workflow {state.status.value}: {state.error_message}"
            status = state.status.value

        response = {
            "id": f"agent-chat-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(str(answer).split()),
                "total_tokens": len(str(answer).split()),
            },
            "agent_metadata": {
                "status": status,
                "iterations": state.iteration_count,
                "history_length": len(state.history),
                "core_meta": state.final_meta or {},
            },
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def agent_status():
    """Get agent runtime status"""
    global agent_runtime

    if agent_runtime is None:
        return {"status": "not_initialized"}

    state = agent_runtime.get_state()

    return {
        "status": state.status.value,
        "current_agent": state.current_agent,
        "iteration_count": state.iteration_count,
        "iteration_count_round": state.iteration_count_round,
        "max_iterations": state.max_iterations,
        "history_length": len(state.history),
        "context_keys": list(state.context.keys()),
    }
