"""
统一配置管理
替代 constants.py + 分散的 os.getenv() 调用
"""
from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass(frozen=True)
class ModelLoaderConfig:
    """模型加载器配置"""
    engine_preference: str = "auto"  # auto | vllm | transformers | llama_cpp
    use_gpu: bool = True
    context_length: int = 8192
    quantization: Optional[str] = "4bit"
    torch_dtype: Optional[str] = None
    tokenizer_path: Optional[str] = None
    transformers_kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VllmConfig:
    """vLLM 引擎配置"""
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    quantization: Optional[str] = None  # awq, gptq, fp8
    trust_remote_code: bool = True
    enforce_eager: bool = False
    disable_custom_all_reduce: bool = False
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    disable_log_stats: bool = True


@dataclass(frozen=True)
class LlamaCppConfig:
    """llama.cpp 引擎配置"""
    n_ctx: int = 4096
    batch_size: int = 2048
    n_gpu_layers: int = -1
    run_mode: str = "github"  # github | docker | compose
    image: str = "rocm/llama.cpp:llama.cpp-b6356_rocm6.4.3_ubuntu24.04_full"
    gfx_version: str = "11.0.0"


@dataclass(frozen=True)
class EmbeddingConfig:
    """嵌入模型配置"""
    model_name: str = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    device: str = "cpu"
    cache_path: str = "./cache"


@dataclass(frozen=True)
class RagConfig:
    """RAG 配置"""
    db_host: str = "localhost"
    db_port: int = 19530
    db_collection: str = "rag_docs"
    es_host: str = "localhost"
    es_port: int = 9200
    es_index_name: str = "rag_docs"
    neo4j_bolt_url: str = "bolt://localhost:7687"
    neo4j_auth: str = "neo4j/123123123"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db_name: str = "rag_system"
    docs_path: str = "./docs"


@dataclass(frozen=True)
class ServerConfig:
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8434
    workers: int = 1
    log_level: str = "info"


@dataclass(frozen=True)
class Settings:
    """应用总配置 - 单一配置入口"""
    model_loader: ModelLoaderConfig = field(default_factory=ModelLoaderConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)
    llama_cpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    
    # 项目路径
    project_root: str = field(default_factory=lambda: os.path.abspath(os.path.dirname(__file__)))
    model_dir: str = "models"
    offload_dir: str = "offload"
    upload_dir: str = "uploads"
    
    # 功能开关
    learning_enabled: bool = False
    scheduler_enabled: bool = True
    root_task: str = ""
    root_task_interval: int = 3600
    
    # 远程模型
    custom_llm_api_key: str = ""
    custom_llm_base_url: str = ""
    custom_llm_model: str = ""
    custom_llm_protocol: str = "openai"
    
    # 多模态
    multimodal_provider_url: str = ""
    remote_multimodal_status: bool = False
    default_vlm: str = ""
    
    # TTS
    redis_host: str = "localhost"
    redis_port: int = 6379
    vibevoice_dir: str = ""
    vibevoice_model: str = "microsoft/VibeVoice-ASR"
    
    # 扩展
    chunk_size: int = 16
    
    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量构建配置（工厂方法）"""
        return cls(
            model_loader=ModelLoaderConfig(
                engine_preference=os.getenv("ENGINE_PREFERENCE", "auto"),
                use_gpu=os.getenv("USE_GPU", "true").lower() == "true",
                context_length=int(os.getenv("CONTEXT_LENGTH", "8192")),
                quantization=os.getenv("QUANTIZATION", "4bit"),
                torch_dtype=os.getenv("TORCH_DTYPE") or None,
            ),
            vllm=VllmConfig(
                tensor_parallel_size=int(os.getenv("VLLM_TENSOR_PARALLEL", "1")),
                gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTIL", "0.9")),
                max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")),
                quantization=os.getenv("VLLM_QUANTIZATION") or None,
                enforce_eager=os.getenv("VLLM_ENFORCE_EAGER", "false").lower() == "true",
                dtype=os.getenv("VLLM_DTYPE", "auto"),
            ),
            llama_cpp=LlamaCppConfig(
                n_ctx=int(os.getenv("LLAMA_N_CTX", "4096")),
                batch_size=int(os.getenv("LLAMA_BATCH_SIZE", "2048")),
                n_gpu_layers=int(os.getenv("LLAMA_N_GPU_LAYERS", "-1")),
                run_mode=os.getenv("LLAMA_CPP_RUN_MODE", "github"),
                image=os.getenv("LLAMA_CPP_IMAGE", "rocm/llama.cpp:llama.cpp-b6356_rocm6.4.3_ubuntu24.04_full"),
                gfx_version=os.getenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0"),
            ),
            embedding=EmbeddingConfig(
                model_name=os.getenv("EMBEDDING_MODEL", "Alibaba-NLP/gte-Qwen2-1.5B-instruct"),
                device=os.getenv("EMBEDDING_DEVICE", "cpu"),
                cache_path=os.getenv("CACHE_PATH", "./cache"),
            ),
            rag=RagConfig(
                db_host=os.getenv("DB_HOST", "localhost"),
                db_port=int(os.getenv("DB_PORT", "19530")),
                db_collection=os.getenv("DB_COLLECTION", "rag_docs"),
                es_host=os.getenv("ES_HOST", "localhost"),
                es_port=int(os.getenv("ES_PORT1", "9200")),
                es_index_name=os.getenv("ES_INDEX_NAME", "rag_docs"),
                neo4j_bolt_url=os.getenv("NEO4J_BOLT_URL", "bolt://localhost:7687"),
                neo4j_auth=os.getenv("NEO4J_AUTH", "neo4j/123123123"),
                mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
                mongo_db_name=os.getenv("MONGO_DB_NAME", "rag_system"),
                docs_path=os.getenv("DOCS_PATH", "./docs"),
            ),
            server=ServerConfig(
                host=os.getenv("HOST", "0.0.0.0"),
                port=int(os.getenv("PORT", "8434")),
                workers=int(os.getenv("WORKERS", "1")),
                log_level=os.getenv("LOG_LEVEL", "info"),
            ),
            project_root=os.getenv("PROJECT_ROOT", os.path.abspath(os.path.dirname(__file__))),
            model_dir=os.getenv("MODEL_DIR", "models"),
            offload_dir=os.getenv("OFFLOAD_DIR", "offload"),
            upload_dir=os.getenv("UPLOAD_DIR", "uploads"),
            learning_enabled=os.getenv("LEARNING", "false").lower() == "true",
            scheduler_enabled=os.getenv("ENABLE_SCHEDULER", "true").lower() == "true",
            root_task=os.getenv("ROOT_TASK", ""),
            root_task_interval=int(os.getenv("ROOT_TASK_INTERVAL", "3600")),
            custom_llm_api_key=os.getenv("CUSTOM_LLM_API_KEY", ""),
            custom_llm_base_url=os.getenv("CUSTOM_LLM_BASE_URL", ""),
            custom_llm_model=os.getenv("CUSTOM_LLM_MODEL", ""),
            custom_llm_protocol=os.getenv("CUSTOM_LLM_PROTOCOL", "openai"),
            multimodal_provider_url=os.getenv("MULTIMODAL_PROVIDER_URL", ""),
            remote_multimodal_status=os.getenv("REMOTE_MULTIMODAL_STATUS", "false").lower() == "true",
            default_vlm=os.getenv("DEFAULT_VLM", ""),
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            vibevoice_dir=os.getenv("VIBEVOICE_DIR", ""),
            vibevoice_model=os.getenv("VIBEVOICE_MODEL", "microsoft/VibeVoice-ASR"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "16")),
        )
    
    def model_dir_path(self) -> str:
        return os.path.join(self.project_root, self.model_dir)
    
    def offload_dir_path(self) -> str:
        return os.path.join(self.project_root, self.offload_dir)


# 全局配置实例（仅用于向后兼容，新代码应使用依赖注入）
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """获取全局配置实例（单例模式，仅用于渐进式迁移）"""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def set_settings(settings: Settings) -> None:
    """设置全局配置（仅用于测试）"""
    global _settings
    _settings = settings