from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
from model_provider import LocalModel
import time
import json

app = FastAPI()

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

class Usege:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetail
    completion_tokens_details: CompletionTokensDetail
    input_tokens: int
    output_tokens: int
    input_tokens_details: dict

class ApiRespnose:
    id: str
    object: str
    created: int
    model: str
    choices: list[dict]
    usage: Usege

class EmbeddingRequest(BaseModel):
    model: str
    input: list[str] | str

@app.post("/api/show")
async def api_show():
    return {"ok": True}

@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest):
    if isinstance(req.input, str):
        texts = [req.input]
    else:
        texts = req.input
    model = LocalModel(req.model)
    vectors = model.tokenizer.encode(texts).tolist()
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
    model = LocalModel(req.model)
    if req.stream:
        streamer = model.chat([m.model_dump() for m in req.messages])
        def event_stream():
            for chunk in streamer:
                data = {
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [
                        {"delta": {"content": chunk}, "index": 0, "finish_reason": None}
                    ]
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    output = model.chat_at_once([m.model_dump() for m in req.messages])
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
    model = LocalModel(req.model)
    output = model.complete_at_once(req.prompt)
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
    uvicorn.run(app, host="0.0.0.0", port=11434)
