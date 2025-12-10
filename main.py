import os
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
import shutil
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

# import triton
# import triton.language as tl
from dotenv import load_dotenv

load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from model_providers import LocalLLModel, PoeModelProvider, ComfyUIProvider
from rag import LocalRAG
from permission_manager import PermissionManager, SafetyLevel

# Import agents
from agents import AgentRuntime
from agents.agent_runtime import RuntimeStatus
from agents.qa_agent import QAAgent
from agents.planning_agent import PlanningAgent
from agents.router_agent import RouterAgent
from agents.verification_agent import VerificationAgent
from agents.risk_agent import RiskAgent, RiskLevel
from agents.task_agents.llm_agent import LLMTaskAgent
from agents.task_agents.rag_agent import RAGTaskAgent
from agents.task_agents.mcp_agent import MCPTaskAgent
from agents.context_storage import create_context_storage

local_rag = None
agent_runtime = None
permission_manager = None
context_storage = None  # Global context storage instance

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
            return JSONResponse(status_code=401, content={"detail": "Missing Authorization header"})
        
        token = auth_header
        if token.startswith("Bearer "):
            token = token[7:]
            
        if token != ADMIN_TOKEN:
            return JSONResponse(status_code=401, content={"detail": "Invalid admin token"})
            
    return await call_next(request)


class Message(BaseModel):
    role: str
    content: str | list


import queue
import threading

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    use_single: bool = False
    enable_rag: bool = False
    files: list[str] = []  # List of file IDs from upload endpoint


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    enable_rag: bool = False
    
    
class AgentRequest(BaseModel):
    model: str
    messages: list[str]
    session_id: str = None  # Optional session ID for context persistence
    files: list[str] = []  # List of file paths to include in context


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


class AgentDecisionRequest(BaseModel):
    approved: bool
    feedback: str = ""
    data: dict = None


class ImportDocumentRequest(BaseModel):
    title: str
    source: str
    content: str
    contentType: str = 'md'
    bvid: str
    cid: int


class DocumentCheckRequest(BaseModel):
    bvid: str
    cid: int



@app.get("/")
async def index():
    return "Hello, Local LLM Provider!"


@app.get("/manifest.json")
async def manifest():
    return FileResponse("manifest.json")


@app.get("/mcp")
async def query_rag(request: Request):
    """Direct RAG query endpoint (original functionality)"""
    global local_rag
    global local_model
    query = request.query_params.get("query")
    if query is None:
        raise HTTPException(
            status_code=400, detail="Either query or request parameter is required"
        )

    try:
        if local_rag is None:
            if local_model is None:
                local_model = LocalLLModel()
            data_path = os.getenv("DATA_PATH", "./docs")
            print(f"初始化 RAG 系统，数据路径: {data_path}")
            local_rag = LocalRAG(local_model, data_path=data_path)
        result = local_rag.generate_answer(query)

        if isinstance(result, str):
            return PlainTextResponse(result)

        def event_stream():
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


@app.post(f"/{VERSION}/upload")
async def upload_file(file: UploadFile = File(...)):
    from utils import FileProcessor
    
    try:
        file_processor = FileProcessor()
        file_id, filename, _ = file_processor.save_uploaded_file(file.file, file.filename)
        
        return {
            "id": file_id,
            "filename": filename,
            "message": "File uploaded successfully"
        }
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.post(f"/{VERSION}/agents/run")
async def query_agent(agentRequest: AgentRequest, request: Request):
    """Agent-based query endpoint with streaming support"""
    global local_rag
    global local_model
    global agent_runtime
    global context_storage
    
    query = agentRequest.messages
    if query is None:
        raise HTTPException(
            status_code=400, detail="Query parameter is required"
        )
    
    # Initialize context storage if not already done
    if context_storage is None:
        try:
            context_storage = create_context_storage()
            logger.info(f"Initialized context storage: {type(context_storage).__name__}")
        except Exception as e:
            logger.error(f"Failed to initialize context storage: {e}")
            # Fall back to memory storage
            from agents.context_storage import MemoryContextStorage
            context_storage = MemoryContextStorage()
    
    # Generate session ID if not provided
    session_id = agentRequest.session_id or str(uuid.uuid4())
    logger.info(f"Processing request for session: {session_id}")

    # Initialize agent runtime if needed
    if agent_runtime is None:
        if local_model is None:
            local_model = LocalLLModel()
        if local_rag is None:
            data_path = os.getenv("DATA_PATH", "./docs")
            local_rag = LocalRAG(local_model, data_path=data_path)
        agent_runtime = AgentRuntime.create_with_all_agents(
            local_model, 
            rag_instance=local_rag,
            permission_manager=permission_manager,
            context_storage=context_storage,
            session_id=session_id
        )
    else:
        # Update session ID for existing runtime
        agent_runtime.session_id = session_id
        # Try to load existing state
        loaded_state = agent_runtime._load_state()
        if loaded_state:
            agent_runtime.state = loaded_state
            logger.info(f"Resumed existing session {session_id}")
    
    # Prepare initial context with files if provided
    initial_context = {}
    if agentRequest.files:
        file_map = {}
        available_files = []
        upload_dir = os.path.join(os.getcwd(), "uploads")
        
        for file_ref in agentRequest.files:
            # Check if it's a full path (legacy/testing) or an ID
            real_path = None
            file_name = None
            file_id = None
            
            if os.path.exists(file_ref):
                real_path = file_ref
                file_name = os.path.basename(file_ref)
                file_id = str(uuid.uuid4()) # Generate temp ID for direct paths
            else:
                # Assume it's an ID, look for file in uploads
                # Pattern: {ID}_{filename}
                if os.path.exists(upload_dir):
                    for f in os.listdir(upload_dir):
                        if f.startswith(file_ref + "_"):
                            real_path = os.path.join(upload_dir, f)
                            file_name = f[len(file_ref)+1:] # Remove ID_ prefix
                            file_id = file_ref
                            break
            
            if real_path and os.path.exists(real_path):
                file_map[file_id] = {
                    "path": real_path,
                    "name": file_name
                }
                available_files.append(f"File: {file_name} (ID: {file_id})")
                logger.info(f"Mapped file {file_id}: {file_name}")
            else:
                logger.warning(f"File reference not found: {file_ref}")
        
        if file_map:
            initial_context["file_map"] = file_map
            initial_context["available_files"] = available_files

    # Execute through agent system with streaming
    async def event_stream():
        """Generate Server-Sent Events for agent execution"""
        import queue
        import threading
        import asyncio
        
        # Create a queue for streaming events
        event_queue = queue.Queue()
        execution_complete = threading.Event()
        final_state = {"state": None, "error": None}
        
        def stream_callback(event_data):
            """Callback to receive streaming events from agents"""
            event_queue.put(event_data)
        
        def execute_agents():
            """Execute agents in a separate thread"""
            try:
                state = agent_runtime.execute(
                    query, 
                    start_agent="qa", 
                    stream_callback=stream_callback,
                    initial_context=initial_context
                )
                final_state["state"] = state
            except Exception as e:
                final_state["error"] = str(e)
            finally:
                execution_complete.set()
        
        # Start agent execution in background thread
        execution_thread = threading.Thread(target=execute_agents)
        execution_thread.start()
        
        # Stream events as they arrive
        while not execution_complete.is_set() or not event_queue.empty():
            if await request.is_disconnected():
                logger.info("Client disconnected, cancelling agent execution")
                # Attempt to stop the agent runtime
                if agent_runtime and agent_runtime.state:
                     if agent_runtime.state.status == RuntimeStatus.RUNNING:
                         agent_runtime.state.status = RuntimeStatus.FAILED
                         agent_runtime.state.error_message = "Cancelled by user disconnection"
                break

            try:
                # Use non-blocking get
                event_data = event_queue.get_nowait()
                
                event_type = event_data.get("event_type")
                
                if event_type == "agent_start":
                    # Agent start event
                    data = {
                        "id": f"agent-run-{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": agentRequest.model,
                        "choices": [{
                            "delta": {},
                            "index": 0,
                            "finish_reason": None
                        }],
                        "agent_metadata": {
                            "event_type": "agent_start",
                            "current_agent": event_data.get("agent_name"),
                            "iteration": event_data.get("iteration")
                        }
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                
                elif event_type == "llm_chunk":
                    # LLM chunk event
                    chunk = event_data.get("chunk", "")
                    if chunk:
                        data = {
                            "id": f"agent-run-{uuid.uuid4().hex}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": agentRequest.model,
                            "choices": [{
                                "delta": {"content": chunk},
                                "index": 0,
                                "finish_reason": None
                            }],
                            "agent_metadata": {
                                "event_type": "llm_chunk",
                                "current_agent": event_data.get("agent_name")
                            }
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                
                elif event_type == "agent_complete":
                    # Agent complete event
                    data = {
                        "id": f"agent-run-{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": agentRequest.model,
                        "choices": [{
                            "delta": {},
                            "index": 0,
                            "finish_reason": None
                        }],
                        "agent_metadata": {
                            "event_type": "agent_complete",
                            "current_agent": event_data.get("agent_name"),
                            "status": event_data.get("status"),
                            "next_agent": event_data.get("next_agent"),
                            "message": event_data.get("message")
                        }
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                
            except queue.Empty:
                # Wait a bit and continue to check disconnection
                await asyncio.sleep(0.1)
                continue
        
        # Wait for execution thread to complete (or timeout if we cancelled)
        # If cancelled, the thread should stop soon because we set status to FAILED
        execution_thread.join(timeout=2.0)
        
        # Send final completion event only if not disconnected
        if not await request.is_disconnected():
            if final_state["error"]:
                # Error occurred
                data = {
                    "id": f"agent-run-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": agentRequest.model,
                    "choices": [{
                        "delta": {"content": f"\n\nError: {final_state['error']}"},
                        "index": 0,
                        "finish_reason": "error"
                    }],
                    "agent_metadata": {
                        "event_type": "error",
                        "error": final_state["error"]
                    }
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            elif final_state["state"]:
                state = final_state["state"]
                # Send final result summary
                if state.status.value == "completed":
                    data = {
                        "id": f"agent-run-{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": agentRequest.model,
                        "choices": [{
                            "delta": {},
                            "index": 0,
                            "finish_reason": "stop"
                        }],
                        "agent_metadata": {
                            "event_type": "workflow_complete",
                            "status": state.status.value,
                            "iterations": state.iteration_count,
                            "iteration_count_round": state.iteration_count_round,
                            "history_length": len(state.history)
                        }
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                elif state.status.value == "waiting_human":
                    # Workflow paused for human intervention
                    data = {
                        "id": f"agent-run-{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": agentRequest.model,
                        "choices": [{
                            "delta": {
                                "content": f"\n\n⏸️ Workflow paused: Waiting for human decision.\nPlease call POST /v1/agents/decision to continue.",    
                                "resume_api": "/v1/agents/decision",
                                "resume_method": "POST",
                            },
                            "index": 0,
                            "finish_reason": "length"
                        }],
                        "agent_metadata": {
                            "event_type": "needs_decision",
                            "status": state.status.value,
                            "reason": "waiting_human",
                            "message": state.error_message or "Human intervention required",
                            "iterations": state.iteration_count,
                            "iteration_count_round": state.iteration_count_round,
                            "action_required": "POST /v1/agents/decision with approved/feedback/data"
                        }
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                elif state.status.value == "max_iterations":
                    # Workflow paused due to max iterations
                    data = {
                        "id": f"agent-run-{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": agentRequest.model,
                        "choices": [{
                            "delta": {
                                "content": f"\n\n⏸️ Workflow paused: Reached maximum iterations ({state.max_iterations}) in round {state.iteration_count_round}.\nPlease call POST /v1/agents/decision to continue or stop.",
                                "resume_api": "/v1/agents/decision",
                                "resume_method": "POST",
                            },
                            "index": 0,
                            "finish_reason": "length"
                        }],
                        "agent_metadata": {
                            "event_type": "needs_decision",
                            "status": state.status.value,
                            "reason": "max_iterations",
                            "message": state.error_message,
                            "iterations": state.iteration_count,
                            "iteration_count_round": state.iteration_count_round,
                            "max_iterations": state.max_iterations,
                            "action_required": "POST /v1/agents/decision with approved=true/false"
                        }
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                else:
                    # Workflow ended in other non-completed state (failed, etc.)
                    data = {
                        "id": f"agent-run-{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": agentRequest.model,
                        "choices": [{
                            "delta": {"content": f"\n\nWorkflow {state.status.value}: {state.error_message}"},
                            "index": 0,
                            "finish_reason": "stop"
                        }],
                        "agent_metadata": {
                            "event_type": "workflow_end",
                            "status": state.status.value,
                            "error_message": state.error_message,
                            "iterations": state.iteration_count,
                            "iteration_count_round": state.iteration_count_round
                        }
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        
        # Send [DONE] marker
        if not await request.is_disconnected():
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post(f"/{VERSION}/agents/decision")
async def agent_decision(req: AgentDecisionRequest):
    """Handle human decision for paused agent workflow or max iterations continuation"""
    global agent_runtime
    
    if agent_runtime is None:
        raise HTTPException(status_code=400, detail="Agent runtime not initialized")
    
    try:
        current_state = agent_runtime.get_state()
        
        # Check current runtime status and call appropriate resume method
        if current_state.status.value == "max_iterations":
            # Handle max_iterations: only continue if approved
            if not req.approved:
                current_state.status = agent_runtime.state.status.__class__.FAILED
                current_state.error_message = f"User declined to continue after max_iterations (round {current_state.iteration_count_round})"
                state = current_state
            else:
                # Resume execution after max_iterations
                state = agent_runtime.resume_after_max_iterations()
        elif current_state.status.value == "waiting_human":
            # Handle human intervention with decision data
            state = agent_runtime.resume(req.model_dump())
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot make decision: Runtime is in {current_state.status.value} state, expected 'waiting_human' or 'max_iterations'"
            )
        
        # Format response similar to chat completion
        if state.status.value == "completed":
            answer = state.final_result
            status = "success"
        elif state.status.value == "waiting_human":
            answer = "Waiting for further human input..."
            status = "waiting_human"
        elif state.status.value == "max_iterations":
            answer = f"Reached max_iterations again in round {state.iteration_count_round}. Continue?"
            status = "max_iterations"
        else:
            answer = f"Workflow {state.status.value}: {state.error_message}"
            status = state.status.value
        
        response = {
            "id": f"agent-decision-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "agent-decision",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": str(answer)},
                    "finish_reason": "stop" if status == "success" else "length",
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
                "iteration_count_round": state.iteration_count_round,
                "history_length": len(state.history)
            }
        }
        
        return JSONResponse(content=response)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"/{VERSION}/agents/chat")
async def agent_chat(req: ChatRequest):
    """Agent-based chat endpoint with full workflow"""
    global local_rag
    global local_model
    global agent_runtime
    
    if local_model is None:
        local_model = LocalLLModel(req.model)
    
    if agent_runtime is None:
        if local_rag is None:
            local_rag = LocalRAG(local_model)
        agent_runtime = AgentRuntime.create_with_all_agents(
            local_model,
            rag_instance=local_rag,
            permission_manager=permission_manager
        )
    
    # Extract user query from messages
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    query = user_messages[-1].content
    
    try:
        # Execute agent workflow
        state = agent_runtime.execute(query, start_agent="qa")
        
        # Format response
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
                "history_length": len(state.history)
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(f"/{VERSION}/agents/status")
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
        "context_keys": list(state.context.keys())
    }


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
                **DEFAULT_MODEL_INFO,
                "name": model,
            }
        )
    return li


@app.get("/api/version")
async def api_version():
    li = await api_tags()
    if local_model is None or local_model.cur_model_name == "":
        return DEFAULT_MODEL_INFO
    _local_model = cast(LocalLLModel, local_model)
    cur = next(model for model in li if model["name"] == _local_model.cur_model_name)
    return cur


@app.post(f"/{VERSION}/embeddings")
async def embeddings(req: EmbeddingRequest):
    if isinstance(req.input, str):
        texts = [req.input]
    else:
        texts = req.input
    global local_model
    if local_model is None:
        local_model = LocalLLModel(embedding_model_name=req.model)
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    
    vectors = cast(PreTrainedTokenizerBase, local_model.tokenizer).encode(texts)
    if not isinstance(vectors, list) and hasattr(vectors, "tolist"):
        vectors = vectors.tolist()
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ],
        "model": req.model,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@app.get(f"/{VERSION}/rag/document/check")
async def check_document(req: DocumentCheckRequest):
    global local_rag
    global local_model
    
    if local_model is None:
        local_model = LocalLLModel()
        
    if local_rag is None:
        local_rag = LocalRAG(local_model)
        
    exists = local_rag.check_document_exists(req.bvid, req.cid)
    return {"exists": exists}


@app.post(f"/{VERSION}/rag/document/import")
async def import_document(req: ImportDocumentRequest):
    global local_rag
    global local_model
    
    if local_model is None:
        local_model = LocalLLModel()
        
    if local_rag is None:
        local_rag = LocalRAG(local_model)
        
    try:
        if local_rag.check_document_exists(req.bvid, req.cid):
            return {"data": None, "message": "Document already exists", "exists": True}

        result = local_rag.add_document(
            title=req.title,
            content=req.content,
            source=req.source,
            content_type=req.contentType,
            bvid=req.bvid,
            cid=req.cid
        )
        return {"data": result, "exists": False}
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return {"error": str(e)}


@app.post(f"/{VERSION}/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    """openai chat/edit/apply with multimodal support"""
    global local_model
    global local_rag
    
    if local_model is None:
        local_model = LocalLLModel(req.model)
    
    # Process files if provided using FileProcessor
    if req.files:
        from utils import FileProcessor
        
        upload_dir = os.path.join(os.getcwd(), os.getenv("UPLOAD_DIR", "uploads"))
        file_processor = FileProcessor(upload_dir)
        
        # Inject file context to messages
        req.messages = file_processor.inject_file_context_to_messages(req.messages, req.files)
        
    if req.enable_rag:
        if local_rag is None:
            local_rag = LocalRAG(local_model)
            
        # Extract query from last user message
        query = ""
        for m in reversed(req.messages):
            if m.role == "user":
                query = m.content
                break
        if not query and req.messages:
            query = req.messages[-1].content
            
        if req.stream:
            event_queue = queue.Queue()
            
            def stream_callback(chunk):
                event_queue.put(chunk)
                
            def run_rag():
                try:
                    cast(LocalRAG, local_rag).generate_answer(query, stream_callback=stream_callback)
                except Exception as e:
                    print(f"RAG Error: {e}")
                finally:
                    event_queue.put(None)
                    
            threading.Thread(target=run_rag).start()
            
            async def event_stream():
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        chunk = event_queue.get_nowait()
                        if chunk is None:
                            break
                            
                        data = {
                            "id": f"chatcmpl-rag-{uuid.uuid4().hex}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": req.model,
                            "choices": [
                                {"delta": {"content": chunk}, "index": 0, "finish_reason": None}
                            ],
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    except queue.Empty:
                        await asyncio.sleep(0.01)
                yield "data: [DONE]\n\n"
                
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        else:
            output = local_rag.generate_answer(query)
            response = {
                "id": f"chatcmpl-rag-{uuid.uuid4().hex}",
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
                },
            }
            return JSONResponse(content=response, headers={"Content-Type": "application/json"})

    if req.stream:
        if req.use_single:
            streamer = local_model.chat([m.model_dump() for m in req.messages])

            async def event_stream():
                try:
                    for chunk in streamer:
                        if await request.is_disconnected():
                            print("Client disconnected, cancelling generation")
                            if hasattr(streamer, 'cancel'):
                                streamer.cancel()
                            break

                        data = {
                            "id": f"chatcmpl-{uuid.uuid4().hex}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": req.model,
                            "choices": [
                                {"delta": {"content": chunk}, "index": 0, "finish_reason": None}
                            ],
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except asyncio.CancelledError:
                    print("Stream cancelled, cancelling generation")
                    if hasattr(streamer, 'cancel'):
                        streamer.cancel()
                    raise

            return StreamingResponse(event_stream(), media_type="text/event-stream")
        
        generator = local_model.chat_in_scheduler([m.model_dump() for m in req.messages])
        rid = await generator.__anext__()
        async def event_stream():
            try:
                async for output_chunk in generator:
                    if await request.is_disconnected():
                        if isinstance(rid, int):
                            await cast(LocalLLModel, local_model).cancel_scheduler(rid, "client disconnected")
                        break
                    data = {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [
                            {"delta": {"content": output_chunk}, "index": 0, "finish_reason": None}
                        ],
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                print("Stream cancelled")
                raise
            
        return StreamingResponse(event_stream(), media_type="text/event-stream")
        
    output = local_model.chat_at_once([m.model_dump() for m in req.messages])
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
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


@app.post(f"/{VERSION}/completions")
async def completions(req: CompletionRequest):
    """openai autocompletions"""
    global local_model
    global local_rag

    if local_model is None:
        local_model = LocalLLModel(req.model)
    
    if req.enable_rag:
        if local_rag is None:
            local_rag = LocalRAG(local_model)
            
        # Use RAG for completion (treating prompt as query)
        output = local_rag.generate_answer(req.prompt)
    else:
        output = local_model.complete_at_once(req.prompt)

    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
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
