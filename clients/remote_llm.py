import json
import re
import httpx
from typing import Any, AsyncGenerator


BACKEND_URL_DEFAULT = "http://localhost:8434"


class RemoteLLM:
    def __init__(self, backend_url: str = BACKEND_URL_DEFAULT):
        self.backend_url = backend_url.rstrip("/")
        self.cur_model_name = "remote"
        self._client = httpx.AsyncClient(timeout=120.0)

    async def chat(self, messages: list[dict], **kwargs) -> AsyncGenerator[str, None]:
        payload = {
            "model": self.cur_model_name,
            "messages": messages,
            "stream": True,
            **kwargs,
        }
        url = f"{self.backend_url}/v1/chat/completions"

        async with self._client.stream("POST", url, json=payload) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def chat_at_once(self, messages: list[dict], **kwargs) -> str:
        payload = {
            "model": self.cur_model_name,
            "messages": messages,
            "stream": False,
            **kwargs,
        }
        url = f"{self.backend_url}/v1/chat/completions"

        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content.strip()

    def format_messages(self, prompt_content) -> list[dict]:
        if hasattr(prompt_content, "to_messages"):
            return self.format_messages(prompt_content.to_messages())
        if hasattr(prompt_content, "messages"):
            return self.format_messages(prompt_content.messages)
        if isinstance(prompt_content, dict):
            prompt_content = [prompt_content]
        elif not isinstance(prompt_content, list):
            prompt_content = [{"role": "user", "content": str(prompt_content)}]
        result = []
        for msg in prompt_content:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                result.append({"role": msg.role, "content": msg.content})
            elif hasattr(msg, "type") and hasattr(msg, "content"):
                role = (
                    "user"
                    if msg.type == "human"
                    else ("assistant" if msg.type == "ai" else "system")
                )
                result.append({"role": role, "content": msg.content})
            elif isinstance(msg, dict):
                result.append(
                    {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                )
            else:
                result.append({"role": "user", "content": str(msg)})
        return result

    def extract_after_think(self, text: str) -> str:
        pos = text.find("</think>")
        return text[pos + 8 :].strip() if pos != -1 else text.strip()

    def extract_thought(self, text: str) -> str:
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def smart_truncate_messages(
        self, messages: list[dict], max_tokens: int | None = None
    ) -> list[dict]:
        return messages
