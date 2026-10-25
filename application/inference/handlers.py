"""
Application: Inference use cases.
Orchestrates chat, completion, streaming generation.
"""
from dataclasses import dataclass
from typing import AsyncGenerator, Optional, List, Dict, Any

from domain.model.entities import GenerationConfig
from domain.inference.services import InferenceDomainService, InferenceNotReadyError
from domain.inference.ports import InferenceChunk


@dataclass(frozen=True)
class ChatCommand:
    """Chat command."""
    messages: List[Dict[str, Any]]
    config: GenerationConfig
    stream: bool = False
    session_id: Optional[str] = None


@dataclass(frozen=True)
class CompletionCommand:
    """Completion command."""
    prompt: str
    config: GenerationConfig
    stream: bool = False
    session_id: Optional[str] = None


class ChatUseCase:
    """Chat use case."""

    def __init__(self, inference_service: InferenceDomainService):
        self._inference = inference_service

    async def execute(self, command: ChatCommand) -> AsyncGenerator[InferenceChunk, None]:
        if not self._inference.is_ready:
            raise InferenceNotReadyError("No model loaded")

        prompt = self._inference.apply_chat_template(
            command.messages,
            add_generation_prompt=True
        )

        async for chunk in self._inference.generate_stream(
            prompt=prompt,
            config=command.config,
            session_id=command.session_id,
        ):
            yield chunk

    async def execute_once(self, command: ChatCommand) -> str:
        """Non-stream chat, return complete response."""
        chunks = []
        async for chunk in self.execute(command):
            if chunk.content:
                chunks.append(chunk.content)
            if chunk.is_final:
                break
        return "".join(chunks)


class CompletionUseCase:
    """Completion use case."""

    def __init__(self, inference_service: InferenceDomainService):
        self._inference = inference_service

    async def execute(self, command: CompletionCommand) -> AsyncGenerator[InferenceChunk, None]:
        if not self._inference.is_ready:
            raise InferenceNotReadyError("No model loaded")

        async for chunk in self._inference.generate_stream(
            prompt=command.prompt,
            config=command.config,
            session_id=command.session_id,
        ):
            yield chunk

    async def execute_once(self, command: CompletionCommand) -> str:
        chunks = []
        async for chunk in self.execute(command):
            if chunk.content:
                chunks.append(chunk.content)
            if chunk.is_final:
                break
        return "".join(chunks)


class TokenCountUseCase:
    """Token count use case."""

    def __init__(self, inference_service: InferenceDomainService):
        self._inference = inference_service

    def execute(self, text: str) -> int:
        if not self._inference.is_ready:
            raise InferenceNotReadyError("No model loaded")
        return self._inference.count_tokens(text)


class TruncateMessagesUseCase:
    """Truncate messages use case."""

    def __init__(self, inference_service: InferenceDomainService):
        self._inference = inference_service

    def execute(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        reserve_tokens: int = 1024
    ) -> List[Dict[str, Any]]:
        if not self._inference.is_ready:
            raise InferenceNotReadyError("No model loaded")

        def count_msg(msg):
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            return self._inference.count_tokens(f"{msg.get('role', '')}: {content}")

        from domain.model.services import TokenBudgetService
        budget_service = TokenBudgetService(
            default_context_length=self._inference.current_model.context_length if self._inference.current_model else 8192,
            reserve_tokens=reserve_tokens,
        )

        budget = max_tokens - reserve_tokens
        return budget_service.truncate_messages(messages, count_msg, budget)