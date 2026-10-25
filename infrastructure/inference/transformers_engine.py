"""
Infrastructure: Transformers engine implementation.
Adapts TransformersEngine to InferenceEngine interface.
"""
import asyncio
import logging
from typing import AsyncGenerator, Any, Dict, Optional, List
import torch

from domain.inference.ports import InferenceEngine, InferenceRequest, InferenceChunk
from domain.model.entities import GenerationConfig

logger = logging.getLogger(__name__)


class TransformersEngineAdapter(InferenceEngine):
    """Transformers engine adapter."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_id: str,
        max_context_length: int = 8192,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._model_id = model_id
        self._max_context_length = max_context_length
        self._device = model.device
        if hasattr(self._model, "config"):
            self._model.config.use_cache = True

    @property
    def engine_type(self) -> str:
        return "transformers"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def max_context_length(self) -> int:
        return self._max_context_length

    async def generate_stream(
        self, 
        request: InferenceRequest
    ) -> AsyncGenerator[InferenceChunk, None]:
        from transformers import TextIteratorStreamer
        from threading import Thread

        config = request.config

        inputs = self._tokenizer(request.prompt, return_tensors="pt").to(self._device)
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.temperature > 0,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else None,
            repetition_penalty=config.repetition_penalty,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            use_cache=True,
        )

        if config.stop_sequences:
            generation_kwargs["stopping_criteria"] = self._create_stopping_criteria(config.stop_sequences)

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        streamer_iter = iter(streamer)
        chunk_count = 0
        prev_text = ""

        while True:
            new_text = await asyncio.to_thread(next, streamer_iter, None)
            if new_text is None:
                finish_reason = "stop"
                if chunk_count >= config.max_new_tokens - 5:
                    finish_reason = "length"
                yield InferenceChunk(
                    content="",
                    finish_reason=finish_reason,
                    token_count=chunk_count,
                )
                break

            if new_text:
                chunk_count += len(new_text)
                delta = new_text[len(prev_text):]
                prev_text = new_text
                if delta:
                    yield InferenceChunk(
                        content=delta,
                        token_count=len(delta),
                    )

            await asyncio.sleep(0)

    async def generate(self, request: InferenceRequest) -> InferenceChunk:
        config = request.config

        inputs = self._tokenizer(request.prompt, return_tensors="pt").to(self._device)

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.temperature > 0,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else None,
            repetition_penalty=config.repetition_penalty,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            use_cache=True,
        )

        if config.stop_sequences:
            generation_kwargs["stopping_criteria"] = self._create_stopping_criteria(config.stop_sequences)

        with torch.no_grad():
            outputs = self._model.generate(**generation_kwargs)

        generated = outputs[0][inputs.input_ids.shape[1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)

        return InferenceChunk(
            content=text,
            finish_reason="stop",
            token_count=len(generated),
        )

    def _create_stopping_criteria(self, stop_sequences):
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopSequenceCriteria(StoppingCriteria):
            def __init__(self, tokenizer, stop_sequences, prompt_length):
                self.tokenizer = tokenizer
                self.stop_sequences = stop_sequences
                self.prompt_length = prompt_length

            def __call__(self, input_ids, scores, **kwargs):
                decoded = self.tokenizer.decode(
                    input_ids[0][self.prompt_length:], 
                    skip_special_tokens=True
                )
                return any(stop in decoded for stop in self.stop_sequences)

        return StoppingCriteriaList([
            StopSequenceCriteria(self._tokenizer, stop_sequences, inputs.input_ids.shape[1])
        ])

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def apply_chat_template(self, messages: List[dict], add_generation_prompt: bool = True) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        # Fallback simple template
        prompt = ""
        for msg in messages:
            prompt += f"{msg['role'].capitalize()}: {msg.get('content', '')}\n"
        if add_generation_prompt:
            prompt += "Assistant:"
        return prompt

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def health_check(self) -> bool:
        return self._model is not None

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "engine_type": self.engine_type,
            "model_id": self._model_id,
            "device": str(self._device),
            "max_context_length": self._max_context_length,
        }


class TransformersModelLoader:
    """Transformers model loader implementation."""

    def __init__(self, settings):
        self._settings = settings

    def can_load(self, model_path: str) -> bool:
        """Check if path is HF format model directory."""
        import os
        return os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json"))

    def supported_types(self) -> list[str]:
        return ["hf", "awq", "gptq", "fp8"]

    def load(self, model_path: str, config: Dict[str, Any]) -> TransformersEngineAdapter:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import os

        model_loader_config = config.get("model_loader", {})
        vllm_config = config.get("vllm", {})

        quantization = model_loader_config.get("quantization")
        torch_dtype = model_loader_config.get("torch_dtype")
        use_gpu = model_loader_config.get("use_gpu", True)

        transformers_kwargs = model_loader_config.get("transformers_kwargs", {}).copy()
        tokenizer_path = model_loader_config.get("tokenizer_path", model_path)

        if quantization == "4bit" and use_gpu:
            compute_dtype = torch.float16
            if torch_dtype == "bfloat16":
                compute_dtype = torch.bfloat16
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            transformers_kwargs["quantization_config"] = quantization_config
        elif quantization == "8bit":
            transformers_kwargs["load_in_8bit"] = True

        if use_gpu:
            transformers_kwargs.setdefault("device_map", "auto")
            transformers_kwargs.setdefault("low_cpu_mem_usage", True)
            if torch_dtype and torch_dtype != "auto":
                dtype_map = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                }
                transformers_kwargs["torch_dtype"] = dtype_map.get(torch_dtype, torch.float16)
            else:
                transformers_kwargs.setdefault("torch_dtype", torch.float16)
        else:
            transformers_kwargs.setdefault("device_map", "cpu")
            transformers_kwargs.setdefault("torch_dtype", torch.float32)

        transformers_kwargs.setdefault("trust_remote_code", True)

        model = AutoModelForCausalLM.from_pretrained(model_path, **transformers_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        context_length = model_loader_config.get("context_length", 8192)

        return TransformersEngineAdapter(
            model=model,
            tokenizer=tokenizer,
            model_id=os.path.basename(model_path),
            max_context_length=context_length,
        )

    def unload(self, engine: TransformersEngineAdapter) -> None:
        engine.unload()