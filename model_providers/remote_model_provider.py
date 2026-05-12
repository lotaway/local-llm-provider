import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from dataclasses import dataclass
from providers.provider_base import ModelProvider
from remote_providers import OpenAIModelProvider, OpenAISettings
from constants import (
    CUSTOM_LLM_API_KEY,
    CUSTOM_LLM_BASE_URL,
    CUSTOM_LLM_MODEL,
    CUSTOM_LLM_PROTOCOL,
)

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RemoteProviderConfig:
    provider_id: str
    api_key: str
    model_names: List[str]
    base_url: Optional[str] = None
    protocol: str = "openai"

class RemoteModelProvider(ModelProvider):
    def __init__(self, config: Union[RemoteProviderConfig, str] = None):
        import os
        if isinstance(config, str):
            self.cur_model_name = config
            self._config = None
            self.provider = self._build_custom_provider()
        elif isinstance(config, RemoteProviderConfig):
            self._config = config
            self.cur_model_name = config.model_names[0] if config.model_names else ""
            self.provider = self._build_provider_from_config(config)
        else:
            self.cur_model_name = CUSTOM_LLM_MODEL
            self._config = None
            self.provider = self._build_custom_provider()

    def _build_custom_provider(self):
        import os
        return OpenAIModelProvider(OpenAISettings(
            api_key=CUSTOM_LLM_API_KEY, base_url=CUSTOM_LLM_BASE_URL,
            proxy_url=os.getenv("HTTP_PROXY"), timeout=60.0
        ))

    def _build_provider_from_config(self, config: RemoteProviderConfig):
        import os
        if config.protocol in ["openai", "poe"] or config.provider_id in ["custom", "poe"]:
            return OpenAIModelProvider(OpenAISettings(
                api_key=config.api_key, base_url=config.base_url or "https://api.openai.com/v1",
                proxy_url=os.getenv("HTTP_PROXY"), timeout=60.0
            ))
        return None

    def is_available(self) -> bool:
        if self._config: return self._config.api_key.strip() != ""
        return self.provider is not None and self.provider.is_available()

    def list_models(self) -> List[str]:
        if self._config: return list(self._config.model_names)
        return [self.cur_model_name] if self.cur_model_name else []

    def format_messages(self, prompt_content) -> list[dict]:
        if hasattr(prompt_content, "to_messages"): return self.format_messages(prompt_content.to_messages())
        if hasattr(prompt_content, "messages"): return self.format_messages(prompt_content.messages)
        msgs = prompt_content if isinstance(prompt_content, list) else [prompt_content]
        return [{"role": m.get("role", "user") if isinstance(m, dict) else "user", 
                 "content": m.get("content", str(m)) if isinstance(m, dict) else str(m)} for m in msgs]

    def extract_after_think(self, text: str) -> str:
        pos = text.find("</think>")
        return text[pos + 8:].strip() if pos != -1 else text.strip()

    def extract_thought(self, text: str) -> str:
        import re
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_text_from_content(self, content: Any) -> str:
        if isinstance(content, str): return content
        if isinstance(content, list):
            return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        return str(content)

    def smart_truncate_messages(self, messages: list[dict], max_tokens: int = None) -> list[dict]:
        return messages

    async def chat_at_once(self, messages: list[dict], **kwargs) -> str:
        if not self.provider: raise RuntimeError("Remote provider not initialized")
        payload = {"model": self.cur_model_name, "messages": messages, "stream": False, **kwargs}
        resp = await self.provider.handle_request("chat/completions", None, payload)
        return resp.json()["choices"][0]["message"]["content"] if hasattr(resp, "json") else str(resp)

    async def chat(self, messages: list[dict], **kwargs) -> AsyncGenerator[str, None]:
        if not self.provider: raise RuntimeError("Remote provider not initialized")
        payload = {"model": self.cur_model_name, "messages": messages, "stream": True, **kwargs}
        resp = await self.provider.handle_request("chat/completions", None, payload)
        if isinstance(resp, str): yield resp; return
        async for line in resp.aiter_lines():
            if line.startswith("data: ") and line[6:] != "[DONE]":
                try: yield json.loads(line[6:])["choices"][0].get("delta", {}).get("content", "")
                except: continue
