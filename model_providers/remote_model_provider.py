import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from dataclasses import dataclass
from providers.provider_base import ModelProvider
from remote_providers import OpenAIModelProvider, OpenAISettings, PoeModelProvider
from constants import (
    CUSTOM_LLM_API_KEY,
    CUSTOM_LLM_BASE_URL,
    CUSTOM_LLM_MODEL,
    CUSTOM_LLM_PROTOCOL,
    POE_API_KEY,
    POE_DEFAULT_MODEL,
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
    """
    Enhanced RemoteModelProvider that supports both legacy Poe/OpenAI configs
    and the new CUSTOM_LLM configuration.
    """
    
    def __init__(self, config: Union[RemoteProviderConfig, str] = None):
        if isinstance(config, str):
            # If passed a model name string, treat as CUSTOM_LLM or lookup
            self.cur_model_name = config
            self._config = None
            self.provider = self._build_custom_provider(config)
        elif isinstance(config, RemoteProviderConfig):
            self._config = config
            self.cur_model_name = config.model_names[0] if config.model_names else ""
            self.provider = self._build_provider_from_config(config)
        else:
            # Default to CUSTOM_LLM
            self.cur_model_name = CUSTOM_LLM_MODEL
            self._config = None
            self.provider = self._build_custom_provider(self.cur_model_name)

    def _build_custom_provider(self, model_name: str):
        if CUSTOM_LLM_PROTOCOL == "openai":
            settings = OpenAISettings(
                api_key=CUSTOM_LLM_API_KEY,
                base_url=CUSTOM_LLM_BASE_URL,
                timeout=60.0
            )
            return OpenAIModelProvider(settings)
        return None

    def _build_provider_from_config(self, config: RemoteProviderConfig):
        if config.provider_id == "poe":
            return PoeModelProvider()
        if config.protocol == "openai":
            settings = OpenAISettings(
                api_key=config.api_key,
                base_url=config.base_url or "https://api.openai.com/v1",
                timeout=60.0
            )
            return OpenAIModelProvider(settings)
        return None

    def is_available(self) -> bool:
        if self._config:
            return self._config.api_key.strip() != ""
        return self.provider is not None and self.provider.is_available()

    def list_models(self) -> List[str]:
        if self._config:
            return list(self._config.model_names)
        if self.cur_model_name:
            return [self.cur_model_name]
        return []

    def format_messages(self, prompt_content) -> list[dict]:
        if hasattr(prompt_content, "to_messages"):
            return self.format_messages(prompt_content.to_messages())
        if hasattr(prompt_content, "messages"):
            return self.format_messages(prompt_content.messages)
        if isinstance(prompt_content, dict):
            prompt_content = [prompt_content]
        elif not isinstance(prompt_content, list):
            text = str(prompt_content)
            prompt_content = [{"role": "user", "content": text}]
        
        formatted_messages: list[dict] = []
        for msg in prompt_content:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = "user"
                content = str(msg)
            formatted_messages.append({"role": role, "content": content})
        return formatted_messages

    def extract_after_think(self, text: str) -> str:
        think_pos = text.find("</think>")
        if think_pos != -1:
            return text[think_pos + len("</think>") :].strip()
        return text.strip()

    def extract_thought(self, text: str) -> str:
        import re
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_text_from_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            return "".join(text_parts)
        return str(content)

    def smart_truncate_messages(
        self, messages: list[dict], max_tokens: int = None
    ) -> list[dict]:
        return messages

    async def chat_at_once(self, messages: list[dict], **kwargs) -> str:
        if not self.provider:
            raise RuntimeError("Remote provider not initialized")
            
        payload = {
            "model": self.cur_model_name,
            "messages": messages,
            "stream": False,
        }
        for k, v in kwargs.items():
            if k not in payload and v is not None:
                payload[k] = v
        
        try:
            # Note: handle_request expects (path, request, body_data)
            resp = await self.provider.handle_request("chat/completions", None, payload)
            if hasattr(resp, "json"):
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            return str(resp)
        except Exception as e:
            logger.error(f"Remote chat_at_once failed: {e}")
            raise

    async def chat(self, messages: list[dict], **kwargs) -> AsyncGenerator[str, None]:
        if not self.provider:
            raise RuntimeError("Remote provider not initialized")

        payload = {
            "model": self.cur_model_name,
            "messages": messages,
            "stream": True,
        }
        for k, v in kwargs.items():
            if k not in payload and v is not None:
                payload[k] = v

        try:
            resp = await self.provider.handle_request("chat/completions", None, payload)
            if isinstance(resp, str):
                yield resp
                return

            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except:
                        continue
        except Exception as e:
            logger.error(f"Remote chat stream failed: {e}")
            raise
