from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
    TextIteratorStreamer,
)
from accelerate import init_empty_weights, infer_auto_device_map
import torch
from threading import Thread

models = {
    "gpt-oss:20b": "./models/gpt-oss-20b",
    "deepseek-r1:16b": "./models/deepseek-ai/DeepSeek-R1-Distill-Qwen-16B",
    "deepseek-r1:32b": "./models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
}
model_cache = {}

class LocalModel:
    def __init__(self, simple_model_name="deepseek-r1:16b"):
        if simple_model_name in model_cache:
            self.model, self.tokenizer = model_cache[simple_model_name]
        else:
            model_name = models[simple_model_name]
            tokenizer_model_name = models["deepseek-r1:16b"]
            if model_name is None:
                raise ValueError("Model name not found")

            # config = AutoConfig.from_pretrained(model_name)
            # with init_empty_weights():
            #     model = AutoModelForCausalLM.from_config(config)

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_model_name, local_files_only=True
            )
            quantization_config = BitsAndBytesConfig(
                # load_in_8bit=True,
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            # device_map = infer_auto_device_map(
            #     self.model,
            #     max_memory={0: "20GiB", "cpu": "60GiB"},
            #     no_split_module_classes=["DeepseekLayer"],  # follow real model
            # )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=True,
                dtype=torch.float16,
                device_map={
                    "transformer.word_embeddings": 0,
                    "transformer.final_layernorm": 0,
                    "transformer.h": "cpu",
                    "model.embed_tokens": 0,  # 嵌入层
                    "model.layers": 0,
                    # "model.layers": "cpu",  # 中间层
                    "model.norm": 0,  # LayerNorm
                    "lm_head": 0,  # 输出层
                },
                # device_map=device_map,
                quantization_config=quantization_config,
                max_memory={0: "20GiB", "cpu": "60GiB"},
            )
            model_cache[simple_model_name] = (self.model, self.tokenizer)

    def chat(self, messages: list[dict]):
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
            prompt += "Assistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=2000)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def chat_at_once(self, messages: list[dict]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
            prompt += "Assistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Assistant:")[-1].strip()
    
    def complete(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=200)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def complete_at_once(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
