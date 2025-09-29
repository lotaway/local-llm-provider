from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from model_provider import LocalModel

app = FastAPI()
model = LocalModel()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]

class CompletionRequest(BaseModel):
    model: str
    prompt: str

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    output = model.chat([m.model_dump() for m in req.messages])
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

@app.post("/v1/completions")
def completions(req: CompletionRequest):
    output = model.complete(req.prompt)
    return {
        "id": "cmpl-1",
        "object": "text_completion",
        "choices": [
            {
                "index": 0,
                "text": output,
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11434)
