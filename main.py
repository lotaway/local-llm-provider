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

from model_providers import LocalLLModel, PoeModelProvider, ComfyUIProvider
from rag import LocalRAG
from permission_manager import PermissionManager, SafetyLevel

# Import agents
from agents import AgentRuntime
from agents.qa_agent import QAAgent
from agents.planning_agent import PlanningAgent
from agents.router_agent import RouterAgent
from agents.verification_agent import VerificationAgent
from agents.risk_agent import RiskAgent, RiskLevel
from agents.task_agents.llm_agent import LLMTaskAgent
from agents.task_agents.rag_agent import RAGTaskAgent
from agents.task_agents.mcp_agent import MCPTaskAgent

local_rag = None
agent_runtime = None
permission_manager = None

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


class AgentDecisionRequest(BaseModel):
    approved: bool
    feedback: str = ""
    data: dict = None


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


@app.get("/agents/run")
async def query_agent(request: Request):
    """Agent-based query endpoint with full workflow"""
    global local_rag
    global local_model
    global agent_runtime
    
    query = request.query_params.get("query")
    if query is None:
        raise HTTPException(
            status_code=400, detail="Query parameter is required"
        )

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
            permission_manager=permission_manager
        )
    
    # Execute through agent system
    try:
        state = agent_runtime.execute(query, start_agent="qa")
        
        if state.status.value == "completed":
            result = state.final_result
        else:
            result = f"Error: {state.error_message}"
        
        if isinstance(result, str):
            return PlainTextResponse(result)

        def event_stream():
            yield f"data: [DONE]{result}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        return PlainTextResponse(f"Error: {str(e)}")


@app.post("/agent/decision")
async def agent_decision(req: AgentDecisionRequest):
    """Handle human decision for paused agent workflow"""
    global agent_runtime
    
    if agent_runtime is None:
        raise HTTPException(status_code=400, detail="Agent runtime not initialized")
    
    try:
        # Resume execution with human decision
        state = agent_runtime.resume(req.model_dump())
        
        # Format response similar to chat completion
        if state.status.value == "completed":
            answer = state.final_result
            status = "success"
        elif state.status.value == "waiting_human":
            answer = "Waiting for further human input..."
            status = "waiting_human"
        else:
            answer = f"Workflow {state.status.value}: {state.error_message}"
            status = state.status.value
        
        response = {
            "id": "agent-decision-1",
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
                "history_length": len(state.history)
            }
        }
        
        return JSONResponse(content=response)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/chat")
async def agent_chat(req: ChatRequest):
    """Agent-based chat endpoint with full workflow"""
    global local_rag
    global local_model
    global agent_runtime
    
    if local_model is None:
        local_model = LocalLLModel(req.model)
    
    if agent_runtime is None:
        if local_rag is None:
            data_path = os.getenv("DATA_PATH", "./docs")
            local_rag = LocalRAG(local_model, data_path=data_path)
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
            "id": "agent-chat-1",
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


@app.get("/agent/status")
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
