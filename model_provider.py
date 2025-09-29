from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from accelerate import init_empty_weights
import torch

class LocalModel:
    def __init__(self, model_name="facebook/galactica-30b"):
        tokenizer_model_name = "Xenova/claude-tokenizer"

        # config = AutoConfig.from_pretrained(model_name)
        # with init_empty_weights():
        #     model = AutoModelForCausalLM.from_config(config)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
        quantization_config = BitsAndBytesConfig(
            # load_in_8bit=True,
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config,
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
