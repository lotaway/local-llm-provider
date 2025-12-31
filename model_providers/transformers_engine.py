import torch
import gc
import asyncio
from typing import AsyncGenerator, Any
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from .inference_engine import InferenceEngine


class TransformersEngine(InferenceEngine):
    def __init__(self, model: Any, tokenizer: PreTrainedTokenizerBase):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True

    @property
    def model_type(self) -> str:
        return "transformers"

        return [], None, None, None

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        from transformers import TextIteratorStreamer
        from threading import Thread

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=kwargs.get("max_new_tokens", 512),
            do_sample=kwargs.get("do_sample", True),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            if new_text:
                yield new_text
            await asyncio.sleep(0)  # Yield control to async loop

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
