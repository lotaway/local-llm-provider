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


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    enable_rag: bool = False


class AgentRequest(BaseModel):
    model: str
    messages: list[str]
    session_id: str = None
    files: list[str] = []


class AgentDecisionRequest(BaseModel):
    approved: bool
    feedback: str = ""
    data: dict = None


class ImportDocumentRequest(BaseModel):
    title: str
    source: str
    content: str
    contentType: str = "md"
    bvid: str
    cid: int


class DocumentCheckRequest(BaseModel):
    bvid: str
    cid: int
