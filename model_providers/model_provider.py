from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from sentence_transformers import SentenceTransformer
from accelerate import init_empty_weights, infer_auto_device_map
import torch
from threading import Thread
import os
import psutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import platform_is_mac

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

class CancellationStoppingCriteria(StoppingCriteria):
    def __init__(self):
        self.cancelled = False

    def __call__(self, input_ids, scores, **kwargs):
        return self.cancelled
    
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
    # tokenizer: AutoTokenizer

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
        self.model = None
        self.tokenizer = None

    def load_model(self):
        if self.model is not None and self.tokenizer is not None:
            return

        if self.cur_model_name == "":
             self.cur_model_name = "deepseek-r1:16b"

        model_path = models[self.cur_model_name]
        tokenizer_model_name = models["deepseek-r1:16b"]
        
        if model_path is None:
            raise ValueError("Model name not found")
        if not os.path.exists(model_path) or not os.listdir(model_path):
            raise ValueError(
                f"Model path '{model_path}' does not exist or is empty"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_name, local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
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
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True,
            ) if is_mac is False else None
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
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                # device="cpu",
                cache_folder=os.getenv("CACHE_PATH", "./cache")
            )
        return self.embedding_model.encode(text)
            
    def format_messages(self, prompt_content):
        if hasattr(prompt_content, 'to_messages'):
            return self.format_messages(prompt_content.to_messages())
        if  hasattr(prompt_content, 'messages'):
            return self.format_messages(prompt_content.messages)
        if isinstance(prompt_content, dict):
            return prompt_content
        text = str(prompt_content)
        prompt_content = [{"role": "user", "content": text}]
        
        # 转换为模型需要的 messages 格式
        formatted_messages: list[dict] = []
        for msg in prompt_content:
            if hasattr(msg, 'type'):
                role = "user" if msg.type == "human" else "assistant" if msg.type == "ai" else "system"
                content = msg.content
            else:
                role = "user"
                content = str(msg)
            formatted_messages.append({"role": role, "content": content})
        return formatted_messages

    def format_prompt(self, messages: list[dict]):
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
        return prompt

    def chat(self, messages: list[dict], **kwargs):
        self.load_model()
        prompt = self.format_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = CancellableStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        stopping_criteria = StoppingCriteriaList([streamer.stopping_criteria])
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=2000, stopping_criteria=stopping_criteria)
        generation_kwargs.update(kwargs)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def chat_at_once(self, messages: list[dict], **kwargs) -> str:
        self.load_model()
        prompt = self.format_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        from queue import Queue
        result_queue = Queue()
        
        def generate_and_put():
            if "max_new_tokens" not in kwargs:
                kwargs["max_new_tokens"] = 3000
            outputs = self.model.generate(**inputs, **kwargs)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            result_queue.put(response)
            
        thread = Thread(target=generate_and_put)
        thread.start()
        thread.join()
        
        return result_queue.get().split("Assistant:")[-1].strip()

    def complete(self, prompt: str):
        self.load_model()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=200)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def complete_at_once(self, prompt: str) -> str:
        self.load_model()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=3000)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def extract_after_think(self, text: str) -> str:
        think_pos = text.find("</think>")
        if think_pos != -1:
            return text[think_pos + len("</think>"):].strip()
        return text.strip()
