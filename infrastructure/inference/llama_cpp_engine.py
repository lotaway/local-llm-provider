"""
Infrastructure: llama.cpp engine implementation.
Adapts LlamaCppEngine to InferenceEngine interface.
"""
import asyncio
import logging
from typing import AsyncGenerator, Any, Dict, Optional, List
import httpx

from domain.inference.ports import InferenceEngine, InferenceRequest, InferenceChunk
from domain.model.entities import GenerationConfig

logger = logging.getLogger(__name__)


class LlamaCppEngineAdapter(InferenceEngine):
    """llama.cpp engine adapter."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        max_context_length: int = 4096,
    ):
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._max_context_length = max_context_length
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def engine_type(self) -> str:
        return "llama_cpp"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def max_context_length(self) -> int:
        return self._max_context_length

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=None, trust_env=False)
        return self._client

    async def generate_stream(
        self, 
        request: InferenceRequest
    ) -> AsyncGenerator[InferenceChunk, None]:
        client = await self._get_client()
        config = request.config

        messages = request.metadata.get("messages", [{"role": "user", "content": request.prompt}])

        payload = {
            "messages": messages,
            "stream": True,
            "model": self._model_id,
            "max_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k if config.top_k > 0 else -1,
            "repeat_penalty": config.repetition_penalty,
            "presence_penalty": config.presence_penalty,
            "frequency_penalty": config.frequency_penalty,
            "min_p": config.min_p,
            "stop": list(config.stop_sequences) if config.stop_sequences else None,
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        buffer = []
        chunk_count = 0

        async with client.stream(
            "POST", f"{self._base_url}/v1/chat/completions", json=payload
        ) as response:
            if response.status_code != 200:
                error_data = await response.aread()
                raise RuntimeError(f"llama.cpp server error: {error_data.decode()}")

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    line_content = line[6:].strip()
                    if line_content == "[DONE]":
                        break

                    try:
                        import json
                        data = json.loads(line_content)
                        choices = data.get("choices", [])
                        if not choices:
                            continue

                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        reasoning = delta.get("reasoning_content")
                        finish_reason = choices[0].get("finish_reason")

                        if reasoning:
                            if buffer:
                                yield InferenceChunk(
                                    content="".join(buffer),
                                    token_count=chunk_count,
                                )
                                buffer = []
                                chunk_count = 0
                            yield InferenceChunk(
                                content="",
                                reasoning_content=reasoning,
                            )
                        elif content:
                            buffer.append(content)
                            chunk_count += len(content)

                        if finish_reason and finish_reason != "null":
                            if buffer:
                                yield InferenceChunk(
                                    content="".join(buffer),
                                    finish_reason=finish_reason,
                                    token_count=chunk_count,
                                )
                            else:
                                yield InferenceChunk(
                                    content="",
                                    finish_reason=finish_reason,
                                )
                            break
                    except json.JSONDecodeError:
                        continue

        if buffer:
            yield InferenceChunk(
                content="".join(buffer),
                token_count=chunk_count,
            )

    async def generate(self, request: InferenceRequest) -> InferenceChunk:
        client = await self._get_client()
        config = request.config

        messages = request.metadata.get("messages", [{"role": "user", "content": request.prompt}])

        payload = {
            "messages": messages,
            "stream": False,
            "model": self._model_id,
            "max_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        response = await client.post(f"{self._base_url}/v1/chat/completions", json=payload)

        if response.status_code != 200:
            raise RuntimeError(f"llama.cpp server error: {response.text}")

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return InferenceChunk(content="", finish_reason="stop")

        message = choices[0].get("message", {})
        content = message.get("content", "")
        finish_reason = choices[0].get("finish_reason", "stop")
        usage = data.get("usage", {})

        return InferenceChunk(
            content=content,
            finish_reason=finish_reason,
            token_count=usage.get("completion_tokens", 0),
        )

    def count_tokens(self, text: str) -> int:
        # Simple estimation: ~4 chars/token for English, ~1.5 for Chinese
        # Production should call llama.cpp tokenize endpoint
        return len(text) // 3

    def apply_chat_template(self, messages: List[dict], add_generation_prompt: bool = True) -> str:
        # llama.cpp server handles template, return raw prompt for reference
        prompt = ""
        for msg in messages:
            prompt += f"{msg['role'].capitalize()}: {msg.get('content', '')}\n"
        if add_generation_prompt:
            prompt += "Assistant:"
        return prompt

    def unload(self) -> None:
        if self._client:
            import asyncio
            asyncio.create_task(self._client.aclose())
            self._client = None

    def health_check(self) -> bool:
        return True

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "engine_type": self.engine_type,
            "model_id": self._model_id,
            "base_url": self._base_url,
            "max_context_length": self._max_context_length,
        }


class LlamaCppModelLoader:
    """llama.cpp model loader."""

    def __init__(self, settings):
        self._settings = settings

    def can_load(self, model_path: str) -> bool:
        import os
        return model_path.endswith(".gguf")

    def supported_types(self) -> list[str]:
        return ["gguf"]

    def load(self, model_path: str, config: Dict[str, Any]) -> LlamaCppEngineAdapter:
        import os
        import subprocess
        import socket
        import atexit

        llama_config = config.get("llama_cpp", {})

        # Reuse existing LlamaCppEngine startup logic
        from model_providers.llama_cpp_engine import LlamaCppEngine
        from constants import PROJECT_ROOT

        n_ctx = llama_config.get("n_ctx", 4096)
        n_gpu_layers = llama_config.get("n_gpu_layers", -1)
        batch_size = llama_config.get("batch_size", 2048)

        engine = LlamaCppEngine(
            model_path=model_path,
            project_root=PROJECT_ROOT,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            batch_size=batch_size,
        )

        port = engine.port
        base_url = f"http://localhost:{port}"

        return LlamaCppEngineAdapter(
            base_url=base_url,
            model_id=os.path.basename(model_path),
            max_context_length=n_ctx,
        )

    def unload(self, engine: LlamaCppEngineAdapter) -> None:
        engine.unload()