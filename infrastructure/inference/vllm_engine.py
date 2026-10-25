"""
Infrastructure: vLLM engine implementation.
Adapts VllmEngine to InferenceEngine interface.
"""
import asyncio
import logging
from typing import AsyncGenerator, Any, Dict, Optional, List

from domain.inference.ports import InferenceEngine, InferenceRequest, InferenceChunk
from domain.model.entities import GenerationConfig

logger = logging.getLogger(__name__)


class VllmEngineAdapter(InferenceEngine):
    """vLLM engine adapter."""

    def __init__(
        self,
        model_path: str,
        model_id: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        quantization: Optional[str] = None,
        trust_remote_code: bool = True,
        enforce_eager: bool = False,
        **kwargs,
    ):
        self._model_path = model_path
        self._model_id = model_id
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._quantization = quantization
        self._trust_remote_code = trust_remote_code
        self._enforce_eager = enforce_eager
        self._extra_kwargs = kwargs

        self._engine = None
        self._tokenizer = None
        self._initialized = False

    def _init_engine(self):
        if self._initialized:
            return

        try:
            from vllm import AsyncLLMEngine
            from vllm.engine.arg_utils import AsyncEngineArgs
            from transformers import AutoTokenizer
        except ImportError as e:
            raise RuntimeError(f"vLLM not installed: {e}")

        engine_args = AsyncEngineArgs(
            model=self._model_path,
            tensor_parallel_size=self._tensor_parallel_size,
            gpu_memory_utilization=self._gpu_memory_utilization,
            max_model_len=self._max_model_len,
            quantization=self._quantization,
            trust_remote_code=self._trust_remote_code,
            enforce_eager=self._enforce_eager,
            **self._extra_kwargs,
        )

        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=self._trust_remote_code
        )
        self._initialized = True
        logger.info(f"vLLM engine loaded: {self._model_path}")

    @property
    def engine_type(self) -> str:
        return "vllm"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def max_context_length(self) -> int:
        return self._max_model_len

    async def generate_stream(
        self, 
        request: InferenceRequest
    ) -> AsyncGenerator[InferenceChunk, None]:
        if not self._initialized:
            self._init_engine()

        from vllm import SamplingParams
        from vllm.utils import random_uuid

        config = request.config

        sampling_params = SamplingParams(
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else -1,
            repetition_penalty=config.repetition_penalty,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            min_p=config.min_p,
            stop=list(config.stop_sequences) if config.stop_sequences else None,
            skip_special_tokens=True,
            spaces_between_special_tokens=False,
            seed=config.seed,
        )

        request_id = random_uuid()
        results_generator = self._engine.generate(
            request.prompt, sampling_params, request_id
        )

        prev_text = ""
        chunk_count = 0

        async for request_output in results_generator:
            if not request_output.outputs:
                continue

            text = request_output.outputs[0].text
            delta = text[len(prev_text):]
            prev_text = text

            if delta:
                chunk_count += len(delta)
                yield InferenceChunk(
                    content=delta,
                    token_count=len(delta),
                )

            if request_output.finished:
                finish_reason = request_output.outputs[0].finish_reason
                yield InferenceChunk(
                    content="",
                    finish_reason=finish_reason,
                    token_count=chunk_count,
                )
                break

            await asyncio.sleep(0)

    async def generate(self, request: InferenceRequest) -> InferenceChunk:
        if not self._initialized:
            self._init_engine()

        from vllm import SamplingParams
        from vllm.utils import random_uuid

        config = request.config

        sampling_params = SamplingParams(
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else -1,
            repetition_penalty=config.repetition_penalty,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            min_p=config.min_p,
            stop=list(config.stop_sequences) if config.stop_sequences else None,
            seed=config.seed,
        )

        request_id = random_uuid()
        results_generator = self._engine.generate(
            request.prompt, sampling_params, request_id
        )

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        if final_output and final_output.outputs:
            text = final_output.outputs[0].text
            finish_reason = final_output.outputs[0].finish_reason
            return InferenceChunk(
                content=text,
                finish_reason=finish_reason,
                token_count=len(text),
            )

        return InferenceChunk(content="", finish_reason="stop")

    def count_tokens(self, text: str) -> int:
        if not self._initialized:
            self._init_engine()
        return len(self._tokenizer.encode(text))

    def apply_chat_template(self, messages: List[dict], add_generation_prompt: bool = True) -> str:
        if not self._initialized:
            self._init_engine()
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def unload(self) -> None:
        if self._engine:
            self._engine = None
        if self._tokenizer:
            self._tokenizer = None
        self._initialized = False
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"vLLM engine unloaded: {self._model_path}")

    def health_check(self) -> bool:
        return self._initialized and self._engine is not None

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "engine_type": self.engine_type,
            "model_id": self._model_id,
            "model_path": self._model_path,
            "tensor_parallel_size": self._tensor_parallel_size,
            "gpu_memory_utilization": self._gpu_memory_utilization,
            "max_model_len": self._max_model_len,
            "quantization": self._quantization,
            "max_context_length": self._max_model_len,
        }


class VllmModelLoader:
    """vLLM model loader."""

    def __init__(self, settings):
        self._settings = settings

    def can_load(self, model_path: str) -> bool:
        import os
        return os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "config.json"))

    def supported_types(self) -> list[str]:
        return ["hf", "awq", "gptq", "fp8"]

    def load(self, model_path: str, config: Dict[str, Any]) -> VllmEngineAdapter:
        import os

        vllm_config = config.get("vllm", {})
        model_loader_config = config.get("model_loader", {})

        return VllmEngineAdapter(
            model_path=model_path,
            model_id=os.path.basename(model_path),
            tensor_parallel_size=vllm_config.get("tensor_parallel_size", 1),
            gpu_memory_utilization=vllm_config.get("gpu_memory_utilization", 0.9),
            max_model_len=vllm_config.get("max_model_len", model_loader_config.get("context_length", 8192)),
            quantization=vllm_config.get("quantization"),
            trust_remote_code=vllm_config.get("trust_remote_code", True),
            enforce_eager=vllm_config.get("enforce_eager", False),
            dtype=vllm_config.get("dtype", "auto"),
            kv_cache_dtype=vllm_config.get("kv_cache_dtype", "auto"),
            max_num_batched_tokens=vllm_config.get("max_num_batched_tokens"),
            max_num_seqs=vllm_config.get("max_num_seqs", 256),
            disable_log_stats=vllm_config.get("disable_log_stats", True),
        )

    def unload(self, engine: VllmEngineAdapter) -> None:
        engine.unload()