from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
    TextIteratorStreamer,
)
from sentence_transformers import SentenceTransformer
from accelerate import init_empty_weights, infer_auto_device_map
import torch
from threading import Thread
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

project_root = os.path.abspath(os.path.dirname(__file__))
models = {
    "gpt-oss:20b": os.path.join(project_root, "models", "openai", "gpt-oss-20b"),
    "deepseek-r1:16b": os.path.join(
        project_root, "models", "deepseek-ai", "DeepSeek-R1-Distill-Qwen-16B"
    ),
    "deepseek-r1:32b": os.path.join(
        project_root, "models", "deepseek-ai", "DeepSeek-R1-Distill-Qwen-32B"
    ),
}


class LocalLLModel:

    cur_model_name = "deepseek-r1:16b"
    embedding_model_name: str = ""

    @staticmethod
    def get_models():
        return list(models.keys())

    def __init__(
        self,
        model_name="deepseek-r1:16b",
        embedding_model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    ):
        if self.embedding_model_name is not embedding_model_name:
            self.embedding_model = SentenceTransformer(
                embedding_model_name,
                # device="cpu",
            )
            self.embedding_model_name = embedding_model_name
        if model_name != self.cur_model_name:
            self.cur_model_name = model_name
            model_path = models[model_name]
            tokenizer_model_name = models["deepseek-r1:16b"]
            if model_path is None:
                raise ValueError("Model name not found")
            if not os.path.exists(model_path) or not os.listdir(model_path):
                raise ValueError(
                    f"Model path '{model_path}' does not exist or is empty"
                )

            # config = AutoConfig.from_pretrained(model_path)
            # with init_empty_weights():
            #     model = AutoModelForCausalLM.from_config(config)

            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_model_name, local_files_only=True
            )
            device_map = {
                "transformer.word_embeddings": 0,
                "transformer.final_layernorm": 0,
                # "transformer.h": "cpu",
                "model.embed_tokens": 0,
                "model.layers": 0,
                "model.norm": 0,
                "lm_head": 0,
            }
            max_memory = {0: "20GiB", "cpu": "60GiB"}
            if model_name.startswith("gpt"):
                # quantization_config = Mxfp4Config(
                #     device_map="auto"
                # )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    dtype=torch.bfloat16,
                    device_map="auto",
                    max_memory=max_memory,
                    # quantization_config=quantization_config,
                    low_cpu_mem_usage=True,
                )
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    # dtype=torch.float16,
                    device_map=device_map,
                    max_memory=max_memory,
                    quantization_config=quantization_config,
                )

    def group_turns(self, messages: list[dict]) -> list[str]:
        turns: list[str] = []
        buf: list[str] = []
        for m in messages:
            buf.append(f"{m['role']}: {m['content']}")
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
        return self.embedding_model.encode(text)

    def tokenize(self, messages: list[dict]):
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
            
    def generate(self, prompt_content):
        if hasattr(prompt_content, 'to_messages'):
            messages = prompt_content.to_messages()
        else:
            text = str(prompt_content)
            messages = [{"role": "user", "content": text}]
        
        # 转换为模型需要的 messages 格式
        formatted_messages: list[dict] = []
        for msg in messages:
            if hasattr(msg, 'type'):
                role = "user" if msg.type == "human" else "assistant" if msg.type == "ai" else "system"
                content = msg.content
            else:
                role = "user"
                content = str(msg)
            formatted_messages.append({"role": role, "content": content})
        return self.chat(formatted_messages)

    def chat(self, messages: list[dict]):
        prompt = self.tokenize(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=2000)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def chat_at_once(self, messages: list[dict]) -> str:
        prompt = self.tokenize(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=3000)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Assistant:")[-1].strip()

    def complete(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=200)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def complete_at_once(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=3000)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
