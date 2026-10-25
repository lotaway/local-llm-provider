import torch
import logging
import sys
import os
import json
from typing import Any, Optional

from model_providers.transformers_engine import TransformersEngine
from model_providers.llama_cpp_engine import LlamaCppEngine
from constants import PROJECT_ROOT, OFFLOAD_DIR
from utils import DeviceUtils


logger = logging.getLogger(__name__)


# vLLM 支持的架构列表
# 来源: vllm/model_executor/models/__init__.py 的 _MODEL_ARCHS
# 需随 vLLM 版本更新维护
VLLM_SUPPORTED_ARCHS = {
    # 文本生成模型
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen2MoeForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "Phi3ForCausalLM",
    "PhiForCausalLM",
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "YiForCausalLM",
    "InternLM2ForCausalLM",
    "InternLM3ForCausalLM",
    "BaichuanForCausalLM",
    "ChatGLMForCausalLM",
    "GLMForCausalLM",
    "MptForCausalLM",
    "FalconForCausalLM",
    "BloomForCausalLM",
    "GPTBigCodeForCausalLM",
    "Starcoder2ForCausalLM",
    "CohereForCausalLM",
    "DbrxForCausalLM",
    "JambaForCausalLM",
    "MambaForCausalLM",
    "Nemotron3ForCausalLM",
    "Qwen2MoeForCausalLM",
    "OlmoForCausalLM",
    "Olmo2ForCausalLM",
    "MinicpmForCausalLM",
    "Minicpm3ForCausalLM",
    # 视觉语言模型
    "LlavaForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Phi3VForCausalLM",
    "InternVLForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "LlavaNextVideoForConditionalGeneration",
    "LlavaOnevisionForConditionalGeneration",
    "MllamaForConditionalGeneration",
    "PixtralForConditionalGeneration",
    # 音频模型
    "UltravoxForConditionalGeneration",
    "Qwen2AudioForConditionalGeneration",
}


class UnifiedModelLoader:
    def __init__(self, model_path: str, options: dict = None):
        """
        Unified Model Loader
        :param model_path: Path to the model file or directory
        :param options: Unified configuration dictionary.
            - engine: "auto" | "vllm" | "transformers" | "llama_cpp"
            - vllm_options: vLLM 特定配置字典
            - transformers_kwargs: Transformers 额外参数
        """
        self.options = options or {}
        self.model_path = model_path
        self.project_root = PROJECT_ROOT

        # Unified parameters
        self.use_gpu = self.options.get("use_gpu", True)
        self.context_length = self.options.get("context_length", 4096)

        # Engine selection preference
        self.engine_preference = self.options.get("engine", "auto")

        self.engine = None
        if model_path.endswith(".gguf"):
            self._load_gguf()
        elif model_path.endswith(".onnx"):
            self._load_onnx()
        else:
            # HF 格式目录：智能选择引擎
            self._load_hf_model_smart()

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

    def _load_hf_model_smart(self):
        """智能选择 HF 模型的推理引擎"""
        preference = self.engine_preference

        # 1. 显式指定 vLLM
        if preference == "vllm":
            logger.info("Engine preference: vLLM (explicit)")
            self._load_vllm()
            return

        # 2. 显式指定 Transformers
        if preference == "transformers":
            logger.info("Engine preference: Transformers (explicit)")
            self._load_transformers()
            return

        # 3. 显式指定 llama.cpp (仅适用于 GGUF，HF 目录忽略)
        if preference == "llama_cpp":
            logger.warning("llama_cpp engine only supports .gguf files, falling back to Transformers")
            self._load_transformers()
            return

        # 4. Auto 模式：尝试 vLLM，失败回退 Transformers
        if preference == "auto":
            logger.info("Engine preference: auto (try vLLM -> fallback Transformers)")
            if self._try_load_vllm():
                return
            logger.info("vLLM unavailable/unsupported, falling back to Transformers")
            self._load_transformers()
            return

        # 默认回退
        logger.info(f"Unknown engine preference '{preference}', defaulting to Transformers")
        self._load_transformers()

    def _try_load_vllm(self) -> bool:
        """尝试加载 vLLM，返回是否成功"""
        try:
            # 预检查：架构是否受支持
            if not self._is_vllm_compatible():
                logger.debug("Model architecture not natively supported by vLLM")
                return False

            # 检查 vLLM 是否安装
            try:
                import vllm
                logger.debug(f"vLLM version: {vllm.__version__}")
            except ImportError:
                logger.debug("vLLM not installed")
                return False

            # 检查 GPU 可用性 (vLLM 强依赖 CUDA)
            import torch
            if not torch.cuda.is_available():
                logger.debug("CUDA not available, vLLM requires GPU")
                return False

            # 实际加载
            self._load_vllm()
            return True

        except Exception as e:
            logger.debug(f"vLLM load failed: {e}")
            return False

    def _is_vllm_compatible(self) -> bool:
        """检查模型架构是否在 vLLM 支持列表中"""
        try:
            config_path = os.path.join(self.model_path, "config.json")
            if not os.path.exists(config_path):
                logger.debug(f"No config.json found at {config_path}")
                return False

            with open(config_path, "r") as f:
                config = json.load(f)

            architectures = config.get("architectures", [])
            if not architectures:
                logger.debug("No architectures field in config.json")
                return False

            # 检查是否有任一架构受支持
            for arch in architectures:
                if arch in VLLM_SUPPORTED_ARCHS:
                    logger.debug(f"Architecture '{arch}' is supported by vLLM")
                    return True

            logger.debug(f"Architectures {architectures} not in vLLM supported list")
            return False

        except Exception as e:
            logger.debug(f"Architecture check failed: {e}")
            return False

    def _load_vllm(self):
        """加载 vLLM 引擎"""
        from .vllm_engine import VllmEngine

        # 从 options 读取 vLLM 特定参数
        vllm_options = self.options.get("vllm_options", {})

        # 合并环境变量配置（环境变量优先级低于显式传参）
        vllm_options = self._merge_env_config(vllm_options)

        self.engine = VllmEngine(
            model_path=self.model_path,
            tensor_parallel_size=vllm_options.get("tensor_parallel_size", 1),
            gpu_memory_utilization=vllm_options.get("gpu_memory_utilization", 0.9),
            max_model_len=vllm_options.get("max_model_len", self.context_length),
            quantization=vllm_options.get("quantization"),
            trust_remote_code=vllm_options.get("trust_remote_code", True),
            enforce_eager=vllm_options.get("enforce_eager", False),
            disable_custom_all_reduce=vllm_options.get("disable_custom_all_reduce", False),
            dtype=vllm_options.get("dtype", "auto"),
            kv_cache_dtype=vllm_options.get("kv_cache_dtype", "auto"),
            max_num_batched_tokens=vllm_options.get("max_num_batched_tokens"),
            max_num_seqs=vllm_options.get("max_num_seqs", 256),
            disable_log_stats=vllm_options.get("disable_log_stats", True),
        )
        logger.info(f"Loaded model with vLLM engine: {self.model_path}")

    def _merge_env_config(self, vllm_options: dict) -> dict:
        """合并环境变量配置"""
        env_mapping = {
            "tensor_parallel_size": ("VLLM_TENSOR_PARALLEL", int),
            "gpu_memory_utilization": ("VLLM_GPU_MEMORY_UTIL", float),
            "max_model_len": ("VLLM_MAX_MODEL_LEN", int),
            "quantization": ("VLLM_QUANTIZATION", str),
            "enforce_eager": ("VLLM_ENFORCE_EAGER", lambda x: x.lower() == "true"),
            "dtype": ("VLLM_DTYPE", str),
        }

        for key, (env_var, converter) in env_mapping.items():
            if key not in vllm_options:
                env_val = os.getenv(env_var)
                if env_val:
                    try:
                        vllm_options[key] = converter(env_val)
                        logger.debug(f"vLLM config from env: {key}={vllm_options[key]}")
                    except Exception as e:
                        logger.warning(f"Failed to parse env {env_var}={env_val}: {e}")

        return vllm_options

    def _load_transformers(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        transformers_kwargs = self.options.get("transformers_kwargs", {}).copy()
        tokenizer_model_name = self.options.get("tokenizer_path", self.model_path)

        quantization = self.options.get("quantization")
        if quantization == "4bit" and not (
            sys.platform == "win32"
            and (
                torch.cuda.is_available()
                and "amd" in torch.cuda.get_device_name(0).lower()
            )
        ):
            from transformers import BitsAndBytesConfig

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