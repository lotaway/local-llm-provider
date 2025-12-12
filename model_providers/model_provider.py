from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    StoppingCriteriaList,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from sentence_transformers import SentenceTransformer
import torch
from threading import Thread
import os
import gc
from typing import cast
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import (
    DeviceUtils,
    Scheduler,
    CancellableStreamer,
    ContentType,
)
import asyncio

project_root = os.path.abspath(os.path.dirname(__file__))
models = {
    "gpt-oss:20b": os.path.join(project_root, "..", "models", "openai", "gpt-oss-20b"),
    "deepseek-r1:16b": os.path.join(
        project_root, "..", "models", "deepseek-ai", "DeepSeek-R1-Distill-Qwen-16B"
    ),
    "deepseek-r1:32b": os.path.join(
        project_root, "..", "models", "deepseek-ai", "DeepSeek-R1-Distill-Qwen-32B"
    ),
}


class GenerateHelper:
    def __init__(self):
        self.token_cache = []

    def save(self, inputs):
        self.token_cache.append(inputs)

    def clear(self):
        self.token_cache = []


class LocalLLModel:

    cur_model_name: str = ""
    embedding_model_name: str = ""
    embedding_model: SentenceTransformer | None = None
    tokenizer: PreTrainedTokenizerBase | None
    # mode: _BaseModelWithGenerate | None

    @staticmethod
    def get_models():
        return list(models.keys())

    def __init__(
        self,
        model_name="deepseek-r1:16b",
        embedding_model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    ):
        self.embedding_model_name = embedding_model_name
        self.cur_model_name = model_name
        self.model = None
        self.tokenizer = None
        self.scheduler = Scheduler(handler=self.generate_next_token)
        self._state = {}
        self.task = asyncio.create_task(self.scheduler.loop())

    def get_token_usage(self, messages: list[dict]) -> tuple[int, int]:
        self.load_model()
        if self.tokenizer is None:
            return 0, 4096

        prompt = self.format_prompt(messages)
        inputs = cast(PreTrainedTokenizerBase, self.tokenizer)(
            prompt, return_tensors="pt"
        )
        input_len = inputs.input_ids.shape[1]
        if hasattr(self.model, "config") and hasattr(
            self.model.config, "max_position_embeddings"
        ):
            max_len = self.model.config.max_position_embeddings
        else:
            max_len = getattr(self.tokenizer, "model_max_length", 4096)
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
        print(f"Truncating context: {current_tokens} > {max_tokens}")
        system_msg = None
        start_idx = 0
        if messages[0].get("role") == "system":
            system_msg = messages[0]
            start_idx = 1
        history_messages = messages[start_idx:]

        # Phase 1: Drop history messages
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

        current_tokens, _ = self.get_token_usage(final_msgs)
        if current_tokens > max_tokens and final_msgs:
            target_msg = final_msgs[-1]
            content = self._extract_text_from_content(target_msg["content"])
            rest_tokens = 0
            if len(final_msgs) > 1:
                rest_msgs = final_msgs[:-1]
                rest_tokens, _ = self.get_token_usage(rest_msgs)

            available_tokens = max_tokens - rest_tokens - 100
            if available_tokens < 100:
                available_tokens = 100
            tokenizer = cast(PreTrainedTokenizerBase, self.tokenizer)
            tokenized_content = tokenizer.encode(content, add_special_tokens=False)
            if len(tokenized_content) > available_tokens:
                print(
                    f"Content truncation: {len(tokenized_content)} -> {available_tokens}"
                )
                truncated_ids = tokenized_content[:available_tokens]
                new_content = tokenizer.decode(truncated_ids, skip_special_tokens=False)
                new_content += "\n...[Context Truncated due to length limit]..."
                target_msg["content"] = new_content

        return final_msgs

    def load_model(self):
        if self.model is not None and self.tokenizer is not None:
            return

        model_path = models[self.cur_model_name]
        tokenizer_model_name = models["deepseek-r1:16b"]

        if model_path is None:
            raise ValueError("Model name not found")
        if not os.path.exists(model_path) or not os.listdir(model_path):
            raise ValueError(f"Model path '{model_path}' does not exist or is empty")

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_name, local_files_only=True
        )
        if hasattr(self.tokenizer, "pad_token_id"):
            cast(PreTrainedTokenizerBase, self.tokenizer).pad_token_id = cast(
                PreTrainedTokenizerBase, self.tokenizer
            ).eos_token_id
        is_mac = DeviceUtils.platform_is_mac()
        if is_mac:
            device_map = "auto"
        else:
            device_map = {
                "transformer.word_embeddings": 0,
                "transformer.final_layernorm": 0,
                # "transformer.h": "cpu",
                "model.embed_tokens": 0,
                "model.layers": 0,
                "model.norm": 0,
                "lm_head": 0,
            }
        max_memory = DeviceUtils.get_available_memory()
        if self.cur_model_name.startswith("gpt"):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=torch.bfloat16,
                device_map="auto",
                max_memory=max_memory,
                low_cpu_mem_usage=True,
            )
        else:
            quantization_config = (
                BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                if is_mac is False
                else None
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=torch.float16 if is_mac else None,
                device_map=device_map,
                max_memory=max_memory,
                quantization_config=quantization_config,
                # low_cpu_mem_usage=True,  # 启用低内存模式
                # offload_folder="./offload",  # 将部分权重卸载到磁盘
                # offload_state_dict=True,  # 卸载状态字典
            )

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    def _extract_text_from_content(self, content: str | list) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == ContentType.TEXT.value:
                        text_parts.append(part.get(ContentType.TEXT.value, ""))
                    # Optional: Add placeholder for images if needed for text models
                    elif part.get("type") == ContentType.IMAGE_URL.value:
                        text_parts.append(
                            f"[{part.get(ContentType.IMAGE_URL.value, '')}]"
                        )
                    # Optional: Add placeholder for images if needed for text models
                    elif part.get("type") == ContentType.INPUT_IMAGE.value:
                        text_parts.append(
                            f"[{part.get(ContentType.INPUT_IMAGE.value, '')}]"
                        )
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

    def chunk_turns(self, turns: list[str]) -> list[str]:
        all_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        for t in turns:
            parts = text_splitter.split_text(t)
            all_chunks.extend(parts)
        return all_chunks

    def embed(self, text: str) -> torch.Tensor:
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                # device="cpu",
                cache_folder=os.getenv("CACHE_PATH", "./cache"),
            )
        return self.embedding_model.encode(text)

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
                content = msg.content if msg.content is not None else ""
            else:
                role = "user"
                content = str(msg)
            formatted_messages.append({"role": role, "content": content})
        return formatted_messages

    def format_prompt(self, messages: list[dict]):
        if hasattr(self.tokenizer, "apply_chat_template"):
            # TODO: Handle potential errors if tokenizer doesn't support list content
            try:
                # Try handling list content directly (for VLM support)
                prompt = cast(
                    PreTrainedTokenizerBase, self.tokenizer
                ).apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback: flatten content to text for text-only tokenizers
                flattened_messages = []
                for m in messages:
                    flattened_messages.append(
                        {
                            "role": m["role"],
                            "content": self._extract_text_from_content(m["content"]),
                        }
                    )
                prompt = cast(
                    PreTrainedTokenizerBase, self.tokenizer
                ).apply_chat_template(
                    flattened_messages, tokenize=False, add_generation_prompt=True
                )
        else:
            prompt = ""
            for msg in messages:
                content_text = self._extract_text_from_content(msg["content"])
                if msg["role"] == "system":
                    prompt += f"System: {content_text}\n"
                elif msg["role"] == "user":
                    prompt += f"User: {content_text}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {content_text}\n"
            prompt += "Assistant:"
        return prompt

    async def _make_generator(self, request_id, payload):
        prompt = payload["prompt"]
        kwargs = payload.get("kwargs", {})
        inputs = cast(PreTrainedTokenizerBase, self.tokenizer)(
            prompt, return_tensors="pt"
        ).to(self.model.device)

        # Prepare initial inputs
        curr_input_ids = inputs.input_ids
        curr_attention_mask = inputs.get("attention_mask")
        past_key_values = None

        generate_helper = GenerateHelper()

        while True:
            model_inputs = {"input_ids": curr_input_ids}
            if past_key_values is not None:
                model_inputs["past_key_values"] = past_key_values

            if curr_attention_mask is not None:
                model_inputs["attention_mask"] = curr_attention_mask

            with torch.no_grad():
                out = self.model(**model_inputs, use_cache=True, **kwargs)

            past_key_values = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1], dim=-1).unsqueeze(0)

            token_id = next_token[0].item()
            generate_helper.save(token_id)

            text = cast(PreTrainedTokenizerBase, self.tokenizer).decode(
                generate_helper.token_cache, skip_special_tokens=False
            )

            is_eos = (
                token_id == cast(PreTrainedTokenizerBase, self.tokenizer).eos_token_id
            )

            if is_eos:
                yield text
                break

            if text.endswith("\ufffd") and len(generate_helper.token_cache) < 10:
                yield ""
            else:
                yield text
                generate_helper.clear()

            # Update inputs for next iteration: only pass the new token
            curr_input_ids = next_token

            # Update attention mask if it exists
            if curr_attention_mask is not None:
                curr_attention_mask = torch.cat(
                    [
                        curr_attention_mask,
                        torch.ones(
                            (curr_attention_mask.shape[0], 1),
                            device=curr_attention_mask.device,
                            dtype=curr_attention_mask.dtype,
                        ),
                    ],
                    dim=1,
                )

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
                out[rid] = token
            except StopAsyncIteration:
                out[rid] = None
                self._state.pop(rid, None)
        return out

    async def cancel_scheduler(self, rid: int, reason: str = "unkown reason"):
        print(f"{reason}, cancelling generation, scheduler id: {rid}")
        self._state.pop(rid, None)
        return self.scheduler.quit(rid)

    async def chat(self, messages: list[dict], **kwargs):
        self.load_model()
        prompt = self.format_prompt(messages)
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

    def chat_in_exclusive(self, messages: list[dict], **kwargs):
        self.load_model()
        prompt = self.format_prompt(messages)
        inputs = cast(PreTrainedTokenizerBase, self.tokenizer)(
            prompt, return_tensors="pt"
        ).to(self.model.device)
        streamer = CancellableStreamer(
            cast(PreTrainedTokenizerBase, self.tokenizer),
            skip_prompt=True,
            skip_special_tokens=False,
        )
        stopping_criteria = StoppingCriteriaList([streamer.stopping_criteria])
        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=2000,
            stopping_criteria=stopping_criteria,
        )
        generation_kwargs.update(kwargs)

        def safe_generate():
            try:
                self.model.generate(**generation_kwargs)
            except Exception as e:
                print(f"Generation error: {e}")
            finally:
                # streamer.on_finalized_text("<|END|>")
                streamer.end()

        thread = Thread(target=safe_generate)
        thread.start()
        return streamer

    def chat_at_once(self, messages: list[dict], **kwargs) -> str:
        self.load_model()
        prompt = self.format_prompt(messages)
        inputs = cast(PreTrainedTokenizerBase, self.tokenizer)(
            prompt, return_tensors="pt"
        ).to(self.model.device)

        result_queue = asyncio.Queue()

        def generate_and_put():
            if "max_new_tokens" not in kwargs:
                kwargs["max_new_tokens"] = 3000
            outputs = self.model.generate(**inputs, **kwargs)
            # transform input_ids to list to get length
            input_len = inputs.input_ids.shape[1]
            # slice only generated tokens
            generated_tokens = outputs[0][input_len:]
            response = cast(PreTrainedTokenizerBase, self.tokenizer).decode(
                generated_tokens, skip_special_tokens=False
            )
            result_queue.put(response)

        thread = Thread(target=generate_and_put)
        thread.start()
        thread.join()

        return result_queue.get().strip()

    def complete(self, prompt: str):
        self.load_model()
        inputs = cast(PreTrainedTokenizerBase, self.tokenizer)(
            prompt, return_tensors="pt"
        ).to(self.model.device)
        streamer = TextIteratorStreamer(
            cast(PreTrainedTokenizerBase, self.tokenizer),
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=200)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def complete_at_once(self, prompt: str) -> str:
        self.load_model()
        inputs = cast(PreTrainedTokenizerBase, self.tokenizer)(
            prompt, return_tensors="pt"
        ).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=3000)
        return cast(PreTrainedTokenizerBase, self.tokenizer).decode(
            outputs[0], skip_special_tokens=True
        )

    def extract_after_think(self, text: str) -> str:
        think_pos = text.find("</think>")
        if think_pos != -1:
            return text[think_pos + len("</think>") :].strip()
        return text.strip()
