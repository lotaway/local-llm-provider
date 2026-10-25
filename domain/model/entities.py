"""
领域层：模型实体与值对象
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any
import uuid


class ModelType(Enum):
    """模型类型枚举"""
    HF = "hf"           # Hugging Face 格式目录
    GGUF = "gguf"       # GGUF 单文件
    ONNX = "onnx"       # ONNX 格式
    AWQ = "awq"         # AWQ 量化
    GPTQ = "gptq"       # GPTQ 量化
    FP8 = "fp8"         # FP8 量化
    REMOTE = "remote"   # 远程 API


class EngineType(Enum):
    """推理引擎类型"""
    TRANSFORMERS = "transformers"
    LLAMA_CPP = "llama_cpp"
    VLLM = "vllm"
    ONNX_RUNTIME = "onnx_runtime"
    AUTO = "auto"


@dataclass(frozen=True)
class ModelId:
    """模型标识值对象"""
    value: str
    
    def __post_init__(self):
        if not self.value or not self.value.strip():
            raise ValueError("ModelId cannot be empty")
    
    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ModelPath:
    """模型路径值对象"""
    value: Path
    
    def __post_init__(self):
        if not self.value:
            raise ValueError("ModelPath cannot be empty")
    
    def __str__(self) -> str:
        return str(self.value)
    
    @property
    def is_file(self) -> bool:
        return self.value.is_file()
    
    @property
    def is_dir(self) -> bool:
        return self.value.is_dir()
    
    @property
    def suffix(self) -> str:
        return self.value.suffix.lower()


@dataclass(frozen=True)
class TokenCount:
    """Token 计数值对象"""
    value: int
    
    def __post_init__(self):
        if self.value < 0:
            raise ValueError("TokenCount cannot be negative")
    
    def __add__(self, other: "TokenCount") -> "TokenCount":
        return TokenCount(self.value + other.value)
    
    def __sub__(self, other: "TokenCount") -> "TokenCount":
        return TokenCount(max(0, self.value - other.value))
    
    def __int__(self) -> int:
        return self.value


@dataclass(frozen=True)
class GenerationConfig:
    """生成配置值对象"""
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    min_p: float = 0.0
    stop_sequences: tuple = ()
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be in [0, 1]")


@dataclass
class Model:
    """模型聚合根"""
    id: ModelId
    name: str
    path: ModelPath
    model_type: ModelType
    engine_type: EngineType = EngineType.AUTO
    context_length: int = 8192
    config: Dict[str, Any] = field(default_factory=dict)  # config.json 内容
    metadata: Dict[str, Any] = field(default_factory=dict)
    _loaded: bool = field(default=False, init=False)
    _engine_info: Dict[str, Any] = field(default_factory=dict, init=False)
    
    def mark_loaded(self, engine_info: Dict[str, Any]) -> None:
        """标记为已加载（领域事件触发点）"""
        self._loaded = True
        self._engine_info = engine_info
        # TODO: 发布 ModelLoaded 领域事件
    
    def mark_unloaded(self) -> None:
        """标记为已卸载"""
        self._loaded = False
        self._engine_info = {}
        # TODO: 发布 ModelUnloaded 领域事件
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    @property
    def engine_info(self) -> Dict[str, Any]:
        return self._engine_info.copy()
    
    def supports_engine(self, engine: EngineType) -> bool:
        """检查模型是否支持指定引擎"""
        if engine == EngineType.AUTO:
            return True
        if self.model_type == ModelType.GGUF:
            return engine in (EngineType.LLAMA_CPP, EngineType.VLLM)
        if self.model_type in (ModelType.HF, ModelType.AWQ, ModelType.GPTQ, ModelType.FP8):
            return engine in (EngineType.TRANSFORMERS, EngineType.VLLM)
        return False
    
    def preferred_engine(self) -> EngineType:
        """获取推荐引擎"""
        if self.model_type == ModelType.GGUF:
            return EngineType.LLAMA_CPP
        return EngineType.VLLM if self._is_vllm_compatible() else EngineType.TRANSFORMERS
    
    def _is_vllm_compatible(self) -> bool:
        """检查是否兼容 vLLM（基于架构）"""
        architectures = self.config.get("architectures", [])
        if not architectures:
            return False
        
        # vLLM 支持的架构列表
        VLLM_SUPPORTED = {
            "LlamaForCausalLM", "MistralForCausalLM", "MixtralForCausalLM",
            "Qwen2ForCausalLM", "Qwen3ForCausalLM", "Qwen2MoeForCausalLM",
            "GemmaForCausalLM", "Gemma2ForCausalLM", "Gemma3ForCausalLM",
            "Phi3ForCausalLM", "PhiForCausalLM",
            "DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM",
            "YiForCausalLM", "InternLM2ForCausalLM", "InternLM3ForCausalLM",
            "BaichuanForCausalLM", "ChatGLMForCausalLM", "GLMForCausalLM",
            "MptForCausalLM", "FalconForCausalLM", "BloomForCausalLM",
            "GPTBigCodeForCausalLM", "Starcoder2ForCausalLM",
            "CohereForCausalLM", "DbrxForCausalLM", "JambaForCausalLM",
            "MambaForCausalLM", "Nemotron3ForCausalLM",
            "LlavaForConditionalGeneration", "Qwen2VLForConditionalGeneration",
            "Phi3VForCausalLM", "InternVLForConditionalGeneration",
        }
        
        return any(arch in VLLM_SUPPORTED for arch in architectures)
    
    @classmethod
    def create(cls, model_id: str, path: Path, model_type: ModelType, **kwargs) -> "Model":
        """工厂方法创建模型"""
        return cls(
            id=ModelId(model_id),
            name=kwargs.get("name", model_id),
            path=ModelPath(path),
            model_type=model_type,
            engine_type=kwargs.get("engine_type", EngineType.AUTO),
            context_length=kwargs.get("context_length", 8192),
            config=kwargs.get("config", {}),
            metadata=kwargs.get("metadata", {}),
        )


@dataclass
class InferenceSession:
    """推理会话聚合根"""
    id: str
    model_id: ModelId
    config: GenerationConfig
    created_at: float
    message_count: int = 0
    total_tokens: int = 0
    _active: bool = field(default=True, init=False)
    
    @classmethod
    def create(cls, model_id: ModelId, config: GenerationConfig) -> "InferenceSession":
        import time
        return cls(
            id=str(uuid.uuid4()),
            model_id=model_id,
            config=config,
            created_at=time.time(),
        )
    
    def record_generation(self, token_count: int) -> None:
        self.message_count += 1
        self.total_tokens += token_count
    
    def close(self) -> None:
        self._active = False
    
    @property
    def is_active(self) -> bool:
        return self._active