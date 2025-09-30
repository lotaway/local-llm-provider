from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
)
from accelerate import init_empty_weights, infer_auto_device_map
import torch


class LocalModel:
    def __init__(self, model_name="./models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"):
        tokenizer_model_name = "./models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

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
                "model.embed_tokens": 0,  # 明确分配嵌入层到GPU0
                "model.layers": "cpu",  # 中间层分配到CPU
                "model.norm": 0,  # 最后的LayerNorm留在GPU
                "lm_head": 0,  # 输出层留在GPU
            },
            # device_map=device_map,
            quantization_config=quantization_config,
            max_memory={0: "20GiB", "cpu": "60GiB"},
        )

    def chat(self, messages: list[dict]) -> str:
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

    def complete(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
