from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from sentence_transformers import SentenceTransformer
import torch
from threading import Thread
import os
import psutil
import gc
from typing import cast
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import platform_is_mac, Scheduler
from enum import Enum
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


class ContentType(Enum):
    TEXT = "text"
    IMAGE_URL = "image_url"
    INPUT_IMAGE = "input_url"
    INPUT_AUDIO = "input_audio"
    OUTPUT_AUDIO = "output_audio"


class CancellationStoppingCriteria(StoppingCriteria):
    def __init__(self):
        self.cancelled = False

    def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
        batch_size = input_ids.shape[0]
        t = torch.tensor(
            [self.cancelled] * batch_size, dtype=torch.bool, device=input_ids.device
        )
        return cast(torch.BoolTensor, t)

    def cancel(self):
        self.cancelled = True


class CancellableStreamer(TextIteratorStreamer):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.stopping_criteria = CancellationStoppingCriteria()

    def cancel(self):
        self.stopping_criteria.cancel()


class LocalLLModel:

    cur_model_name: str = ""
    embedding_model_name: str = ""
    embedding_model: SentenceTransformer | None = None
    tokenizer: PreTrainedTokenizerBase | None
    # mode: _BaseModelWithGenerate | None

    @staticmethod
    def get_available_memory():
        """
        自动检测系统可用内存配置
        返回格式: {0: "20GiB", "cpu": "60GiB"}
        """
        max_memory = {}
        is_mac = platform_is_mac()

        if is_mac:
            total_memory = psutil.virtual_memory().total
            reserved_gb = 4
            available_memory_gb = max(1, (total_memory / (1024**3)) - reserved_gb)
            max_memory["cpu"] = f"{int(available_memory_gb)}GiB"
            if torch.backends.mps.is_available():
                print(f"MPS 加速可用")
        else:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                available_gpu_gb = max(1, (gpu_memory / (1024**3)) - 4)
                max_memory[0] = f"{int(available_gpu_gb)}GiB"
            total_memory = psutil.virtual_memory().total
            available_cpu_gb = max(1, (total_memory / (1024**3)) - 8)
            max_memory["cpu"] = f"{int(available_cpu_gb)}GiB"
        if max_memory.get("cpu") == "0GiB" and max_memory.get(0) == "0GiB":
            print("内存检测异常，使用默认配置")
            max_memory[0] = f"{24 - 4}GiB"
            max_memory["cpu"] = f"{60}GiB"

        return max_memory

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
        is_mac = platform_is_mac()
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
        max_memory = self.get_available_memory()
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
                    if msg.get("type") == "human"
                    else "assistant" if msg.get("type") == "ai" else "system"
                )
                content = msg.get("content")
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
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

            text = cast(PreTrainedTokenizerBase, self.tokenizer).decode(
                next_token[0], skip_special_tokens=False
            )
            yield text

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

            if (
                next_token[0].item()
                == cast(PreTrainedTokenizerBase, self.tokenizer).eos_token_id
            ):
                break

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

    async def chat_in_scheduler(self, messages: list[dict], **kwargs):
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

    def chat(self, messages: list[dict], **kwargs):
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
