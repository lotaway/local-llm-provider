import torch
import logging
from typing import Any
from model_providers.transformers_engine import TransformersEngine
from model_providers.llama_cpp_engine import LlamaCppEngine
from constants import PROJECT_ROOT, OFFLOAD_DIR
from utils import DeviceUtils
import os


class UnifiedModelLoader:
    def __init__(self, model_path: str, options: dict = None):
        """
        Unified Model Loader
        :param model_path: Path to the model file or directory
        :param options: Unified configuration dictionary.
        """
        self.options = options or {}
        self.model_path = model_path
        self.project_root = PROJECT_ROOT

        # Unified parameters
        self.use_gpu = self.options.get("use_gpu", True)
        self.context_length = self.options.get("context_length", 4096)

        self.engine = None
        if model_path.endswith(".gguf"):
            self._load_gguf()
        elif model_path.endswith(".onnx"):
            self._load_onnx()
        else:
            self._load_transformers()

    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "half": torch.float16,
        }
        return mapping.get(dtype_str.lower(), torch.float16)

    def _load_gguf(self):
        n_gpu_layers = self.options.get("n_gpu_layers", -1 if self.use_gpu else 0)
        self.engine = LlamaCppEngine(
            model_path=self.model_path,
            project_root=self.project_root,
            n_ctx=self.context_length,
            n_gpu_layers=n_gpu_layers,
        )

    def _load_onnx(self):
        raise NotImplementedError(
            "ONNX support is not yet updated to the new engine interface"
        )

    def _load_transformers(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        transformers_kwargs = self.options.get("transformers_kwargs", {}).copy()
        tokenizer_model_name = self.options.get("tokenizer_path", self.model_path)

        quantization = self.options.get("quantization")
        if quantization == "4bit":
            compute_dtype = torch.float16
            if self.options.get("torch_dtype") == torch.bfloat16:
                compute_dtype = torch.bfloat16

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            transformers_kwargs["quantization_config"] = quantization_config
        elif quantization == "8bit":
            transformers_kwargs["load_in_8bit"] = True

        if self.use_gpu:
            transformers_kwargs.setdefault("device_map", "auto")
            transformers_kwargs.setdefault("low_cpu_mem_usage", False)

            if "max_memory" not in self.options:
                transformers_kwargs["max_memory"] = DeviceUtils.get_available_memory()
            else:
                transformers_kwargs["max_memory"] = self.options["max_memory"]

            offload_folder = os.path.join(PROJECT_ROOT, OFFLOAD_DIR)
            os.makedirs(offload_folder, exist_ok=True)
            transformers_kwargs.setdefault("offload_folder", offload_folder)

            if "torch_dtype" in self.options:
                val = self.options["torch_dtype"]
                if val != "auto":
                    transformers_kwargs["torch_dtype"] = self._get_torch_dtype(val)
            else:
                transformers_kwargs.setdefault("torch_dtype", torch.float16)
        else:
            transformers_kwargs.setdefault("device_map", "cpu")
            transformers_kwargs.setdefault("low_cpu_mem_usage", False)
            if "torch_dtype" in self.options:
                val = self.options["torch_dtype"]
                if val != "auto":
                    transformers_kwargs["torch_dtype"] = self._get_torch_dtype(val)
            else:
                transformers_kwargs.setdefault("torch_dtype", torch.float32)

        transformers_kwargs.setdefault("trust_remote_code", True)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **transformers_kwargs
        )

        logger = logging.getLogger(__name__)
        logger.info(f"Loaded HF model: {self.model_path}")
        logger.info(f"Model device: {model.device}")

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_name, local_files_only=True
        )
        if hasattr(tokenizer, "pad_token_id"):
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self.engine = TransformersEngine(model, tokenizer)

    def get_engine(self) -> Any:
        return self.engine
