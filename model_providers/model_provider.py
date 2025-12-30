from sentence_transformers import SentenceTransformer
import torch
import os
import gc
from typing import cast
from utils import (
    DeviceUtils,
    Scheduler,
    ContentType,
    discover_models,
)
from .inference_engine import InferenceEngine
from .unified_model_loader import UnifiedModelLoader
import asyncio

models = discover_models()


class GenerateHelper:
    def __init__(self):
        self.token_cache = []

    def save(self, inputs):
        self.token_cache.append(inputs)

    def clear(self):
        self.token_cache = []


local_model = None


class LocalLLModel:

    cur_model_name: str = ""
    embedding_model_name: str = ""
    embedding_model: SentenceTransformer | None = None
    engine: InferenceEngine | None = None

    @staticmethod
    def get_models():
        return list(models.keys())

    @staticmethod
    def init_local_model(model_name: str | None = None):
        global local_model
        if local_model is None:
            local_model = LocalLLModel()
        elif model_name is not None and local_model.cur_model_name != model_name:
            local_model.unload_model()
        return cast(LocalLLModel, local_model)

    def __init__(
        self,
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        embedding_model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    ):
        self.embedding_model_name = embedding_model_name
        self.cur_model_name = model_name
        self.engine = None
        self.scheduler = Scheduler(handler=self.generate_next_token)
        self._state = {}
        self.task = asyncio.create_task(self.scheduler.loop())

    def get_token_usage(self, messages: list[dict]) -> tuple[int, int]:
        self.load_model()
        tokenizer = None
        if hasattr(self.engine, "tokenizer"):
            tokenizer = self.engine.tokenizer

        if tokenizer is None:
            return 0, 32768

        prompt = self.format_prompt(messages)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = inputs.input_ids.shape[1]
        max_len = getattr(tokenizer, "model_max_length", 32768)
        if max_len > 1_000_000:
            max_len = 32768

        return input_len, max_len

    def smart_truncate_messages(
        self, messages: list[dict], max_tokens: int = None
    ) -> list[dict]:
        current_tokens, limit = self.get_token_usage(messages)
        if max_tokens is None:
            max_tokens = limit - 1024
        if current_tokens <= max_tokens:
            return messages
        if not messages:
            return []

        system_msg = None
        start_idx = 0
        if messages[0].get("role") == "system":
            system_msg = messages[0]
            start_idx = 1
        history_messages = messages[start_idx:]

        while len(history_messages) > 1:
            temp_msgs = []
            if system_msg:
                temp_msgs.append(system_msg)
            temp_msgs.extend(history_messages)
            curr, _ = self.get_token_usage(temp_msgs)
            if curr <= max_tokens:
                break
            history_messages.pop(0)

        final_msgs = []
        if system_msg:
            final_msgs.append(system_msg)
        final_msgs.extend(history_messages)
        return final_msgs

    def extract_after_think(self, text: str) -> str:
        think_pos = text.find("</think>")
        if think_pos != -1:
            return text[think_pos + len("</think>") :].strip()
        return text.strip()

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
            if hasattr(msg, "type") and hasattr(msg, "content"):
                role = (
                    "user"
                    if msg.type == "human"
                    else "assistant" if msg.type == "ai" else "system"
                )
                content = msg.content
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = "user"
                content = str(msg)
            formatted_messages.append({"role": role, "content": content})
        return formatted_messages

    def load_model(self):
        if self.engine is not None:
            return

        model_path = models.get(self.cur_model_name)
        if model_path is None:
            model_path = self.cur_model_name

        if not os.path.exists(model_path):
            raise ValueError(f"Model path '{model_path}' does not exist")

        options = {
            "use_gpu": not DeviceUtils.platform_is_mac(),
            "context_length": 4096,
        }

        loader = UnifiedModelLoader(model_path, options)
        self.engine = loader.get_engine()

    def unload_model(self):
        if self.engine is not None:
            self.engine.unload()
            del self.engine
            self.engine = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("Successfully unloaded model and cleared resources.")

    def _extract_text_from_content(self, content: str | list) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == ContentType.TEXT.value:
                        text_parts.append(part.get(ContentType.TEXT.value, ""))
            return "".join(text_parts)
        return str(content)

    def group_turns(self, messages: list[dict]) -> list[str]:
        turns: list[str] = []
        buf: list[str] = []
        for m in messages:
            content_text = self._extract_text_from_content(m["content"])
            buf.append(f"{m['role']}: {content_text}")
            if m["role"] == "assistant":
                turns.append("\n".join(buf))
                buf = []
        if buf:
            turns.append("\n".join(buf))
        return turns

    def embed(self, text: str) -> torch.Tensor:
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                cache_folder=os.getenv("CACHE_PATH", "./cache"),
            )
        return self.embedding_model.encode(text)

    def format_prompt(self, messages: list[dict]):
        tokenizer = getattr(self.engine, "tokenizer", None)
        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        prompt = ""
        for msg in messages:
            content_text = self._extract_text_from_content(msg["content"])
            prompt += f"{msg['role'].capitalize()}: {content_text}\n"
        prompt += "Assistant:"
        return prompt

    async def _make_generator(self, request_id, payload):
        prompt = payload["prompt"]
        kwargs = payload.get("kwargs", {})

        async for token in self.engine.generate_stream(prompt, **kwargs):
            yield token

    async def generate_next_token(
        self, batch: list[tuple[int, dict, asyncio.Queue[dict]]]
    ):
        out = {}
        for rid, payload, _ in batch:
            if rid not in self._state:
                self._state[rid] = self._make_generator(rid, payload)
            gen = self._state[rid]
            try:
                token = await gen.__anext__()

                if token is not None:
                    out[rid] = token
                else:
                    out[rid] = None
                    self._state.pop(rid, None)
            except StopAsyncIteration:
                out[rid] = None
                self._state.pop(rid, None)
            except Exception as e:
                import logging

                logging.getLogger(__name__).error(
                    f"Error in generate_next_token for {rid}: {e}"
                )
                out[rid] = None
                self._state.pop(rid, None)
        return out

    async def _generate_stream(self, prompt: str, **kwargs):
        self.load_model()
        rid, q = await self.scheduler.register(
            {
                "prompt": prompt,
                "kwargs": kwargs,
            }
        )
        yield rid
        while True:
            t = await q.get()
            if t is None:
                break
            yield cast(str, t)

    async def chat(self, messages: list[dict], **kwargs):
        prompt = self.format_prompt(messages)
        async for chunk in self._generate_stream(prompt, **kwargs):
            yield chunk

    async def chat_at_once(self, messages: list[dict], **kwargs) -> str:
        response = []
        async for chunk in self.chat(messages, **kwargs):
            if isinstance(chunk, int):
                continue
            response.append(chunk)
        return "".join(response).strip()

    async def complete(self, prompt: str, **kwargs):
        async for chunk in self._generate_stream(prompt, **kwargs):
            yield chunk

    async def complete_at_once(self, prompt: str) -> str:
        response = []
        async for chunk in self.complete(prompt):
            if isinstance(chunk, int):
                continue
            response.append(chunk)
        return "".join(response).strip()
