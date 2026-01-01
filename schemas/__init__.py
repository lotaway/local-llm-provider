from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str | list


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    exclusive: bool = False
    enable_rag: bool = False
    files: list[str] = []
    max_tokens: int | None = None
    temperature: float = 0.7
    top_p: float = 0.9
