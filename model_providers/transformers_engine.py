import torch
import gc
import asyncio
from typing import AsyncGenerator, Any
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from model_providers import InferenceEngine


class TransformersEngine(InferenceEngine):
    def __init__(self, model: Any, tokenizer: PreTrainedTokenizerBase):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    @property
    def model_type(self) -> str:
        return "transformers"

    def _generate_next_chunk_sync(
        self, curr_input_ids, past_key_values, curr_attention_mask, chunk_size, **kwargs
    ):
        generated_tokens = []

        for _ in range(chunk_size):
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
            generated_tokens.append(token_id)

            curr_input_ids = next_token
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

            if token_id == self.tokenizer.eos_token_id:
                break

        return generated_tokens, curr_input_ids, past_key_values, curr_attention_mask

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        from constants import CHUNK_SIZE

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        curr_input_ids = inputs.input_ids
        curr_attention_mask = inputs.get("attention_mask")
        past_key_values = None

        while True:
            res = await asyncio.to_thread(
                self._generate_next_chunk_sync,
                curr_input_ids,
                past_key_values,
                curr_attention_mask,
                CHUNK_SIZE,
                **kwargs,
            )

            new_tokens, curr_input_ids, past_key_values, curr_attention_mask = res

            if not new_tokens:
                break

            text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
            yield text

            if new_tokens[-1] == self.tokenizer.eos_token_id:
                break

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
