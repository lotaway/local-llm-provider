import uuid
from typing import Any
from acp.core import Agent, Client
from acp.schema import (
    InitializeRequest,
    InitializeResponse,
    Implementation,
    NewSessionRequest,
    NewSessionResponse,
    LoadSessionRequest,
    LoadSessionResponse,
    ListSessionsRequest,
    ListSessionsResponse,
    SessionInfo,
    PromptRequest,
    PromptResponse,
    CloseSessionRequest,
    CloseSessionResponse,
    ForkSessionRequest,
    ForkSessionResponse,
    ResumeSessionRequest,
    ResumeSessionResponse,
    AuthenticateRequest,
    AuthenticateResponse,
    SetSessionModeRequest,
    SetSessionModeResponse,
    SetSessionModelRequest,
    SetSessionModelResponse,
    TextContentBlock,
)
from acp import update_agent_message, text_block
from model_providers import LocalLLModel


class SessionRecord:
    def __init__(self, cwd: str):
        self.cwd = cwd


class SessionStore:
    def __init__(self):
        self._records: dict[str, SessionRecord] = {}

    def create(self, cwd: str, session_id: str | None = None) -> str:
        sid = session_id or str(uuid.uuid4())
        self._records[sid] = SessionRecord(cwd)
        return sid

    def has(self, session_id: str) -> bool:
        return session_id in self._records

    def remove(self, session_id: str) -> bool:
        return self._records.pop(session_id, None) is not None

    def list_sessions(self, cwd: str | None) -> list[SessionInfo]:
        result = []
        for sid, record in self._records.items():
            if cwd and record.cwd != cwd:
                continue
            result.append(SessionInfo(cwd=record.cwd, session_id=sid))
        return result


class ContentHandler:
    @staticmethod
    def extract_text(prompt: list) -> str:
        texts = []
        for block in prompt:
            if isinstance(block, TextContentBlock):
                texts.append(block.text)
        return "\n".join(texts)


class PromptResponder:
    def __init__(self, llm: LocalLLModel):
        self._llm = llm

    async def chat(self, text: str) -> str:
        messages = self._llm.format_messages([{"role": "user", "content": text}])
        return await self._llm.chat_at_once(messages)


class ProtocolBridge(Agent):
    def __init__(self, llm: LocalLLModel):
        self._llm = llm
        self._store = SessionStore()
        self._responder = PromptResponder(llm)
        self._connection: Client | None = None

    def on_connect(self, connection: Client):
        self._connection = connection

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities=None,
        client_info=None,
        **kwargs,
    ) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=protocol_version,
            capabilities=None,
            server_info=Implementation(
                name="local-llm-provider-backend", version="0.1.0"
            ),
        )

    async def new_session(
        self, cwd: str, additional_directories=None, mcp_servers=None, **kwargs
    ) -> NewSessionResponse:
        session_id = self._store.create(cwd)
        return NewSessionResponse(session_id=session_id)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories=None,
        mcp_servers=None,
        **kwargs,
    ) -> LoadSessionResponse | None:
        if not self._store.has(session_id):
            return None
        return LoadSessionResponse(session_id=session_id)

    async def list_sessions(
        self, additional_directories=None, cursor=None, cwd=None, **kwargs
    ) -> ListSessionsResponse:
        sessions = self._store.list_sessions(cwd)
        return ListSessionsResponse(sessions=sessions)

    async def prompt(
        self, prompt: list, session_id: str, message_id: str | None = None, **kwargs
    ) -> PromptResponse:
        text = ContentHandler.extract_text(prompt)
        if not text:
            return PromptResponse(stop_reason="end_turn")

        response = await self._responder.chat(text)

        if self._connection:
            await self._connection.session_update(
                session_id=session_id,
                update=update_agent_message(text_block(response)),
            )

        return PromptResponse(stop_reason="end_turn", user_message_id=message_id)

    async def close_session(
        self, session_id: str, **kwargs
    ) -> CloseSessionResponse | None:
        self._store.remove(session_id)
        return None

    async def cancel(self, session_id: str, **kwargs):
        pass

    async def fork_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories=None,
        mcp_servers=None,
        **kwargs,
    ) -> ForkSessionResponse:
        new_id = self._store.create(cwd)
        return ForkSessionResponse(session_id=new_id)

    async def resume_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories=None,
        mcp_servers=None,
        **kwargs,
    ) -> ResumeSessionResponse:
        if not self._store.has(session_id):
            self._store.create(cwd, session_id)
        return ResumeSessionResponse(session_id=session_id)

    async def authenticate(self, method_id: str, **kwargs) -> AuthenticateResponse:
        return AuthenticateResponse()

    async def set_session_mode(
        self, mode_id: str, session_id: str, **kwargs
    ) -> SetSessionModeResponse | None:
        return None

    async def set_session_model(
        self, model_id: str, session_id: str, **kwargs
    ) -> SetSessionModelResponse | None:
        return None

    async def set_config_option(
        self, config_id: str, session_id: str, value: str | bool, **kwargs
    ):
        return None

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def ext_notification(self, method: str, params: dict[str, Any]):
        pass
