"""
基础设施层：模型加载器工厂
根据模型类型选择合适的加载器
"""
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from domain.inference.ports import ModelLoader, InferenceEngine
from domain.model.entities import ModelType, EngineType
from domain.model.repositories import ModelRepository, ModelInfo
from infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


class ModelLoaderFactory:
    """模型加载器工厂"""
    
    def __init__(self, settings: Settings):
        self._settings = settings
        self._loaders: List[ModelLoader] = []
        self._init_loaders()
    
    def _init_loaders(self):
        """初始化加载器（按优先级顺序）"""
        # vLLM 优先（高性能）
        try:
            from infrastructure.inference.vllm_engine import VllmModelLoader
            self._loaders.append(VllmModelLoader(self._settings))
            logger.info("VllmModelLoader registered")
        except ImportError:
            logger.debug("vLLM not available, skipping")
        
        # Transformers 通用回退
        try:
            from infrastructure.inference.transformers_engine import TransformersModelLoader
            self._loaders.append(TransformersModelLoader(self._settings))
            logger.info("TransformersModelLoader registered")
        except ImportError:
            logger.warning("Transformers not available")
        
        # llama.cpp 用于 GGUF
        try:
            from infrastructure.inference.llama_cpp_engine import LlamaCppModelLoader
            self._loaders.append(LlamaCppModelLoader(self._settings))
            logger.info("LlamaCppModelLoader registered")
        except ImportError:
            logger.debug("llama.cpp not available")
    
    def get_loader_for_model(self, model_path: str, engine_preference: EngineType = EngineType.AUTO) -> Optional[ModelLoader]:
        """根据模型路径和引擎偏好选择加载器"""
        path = Path(model_path)
        
        # GGUF 文件强制使用 llama.cpp
        if path.suffix.lower() == ".gguf":
            for loader in self._loaders:
                if loader.can_load(model_path) and "gguf" in loader.supported_types():
                    return loader
            return None
        
        # 目录模型：根据偏好选择
        if path.is_dir():
            if engine_preference == EngineType.VLLM:
                for loader in self._loaders:
                    if hasattr(loader, '__class__') and 'Vllm' in loader.__class__.__name__:
                        if loader.can_load(model_path):
                            return loader
            elif engine_preference == EngineType.TRANSFORMERS:
                for loader in self._loaders:
                    if hasattr(loader, '__class__') and 'Transformers' in loader.__class__.__name__:
                        if loader.can_load(model_path):
                            return loader
            elif engine_preference == EngineType.LLAMA_CPP:
                for loader in self._loaders:
                    if hasattr(loader, '__class__') and 'LlamaCpp' in loader.__class__.__name__:
                        if loader.can_load(model_path):
                            return loader
            
            # Auto 模式：按注册顺序尝试（vLLM -> Transformers -> llama.cpp）
            for loader in self._loaders:
                if loader.can_load(model_path):
                    return loader
        
        return None
    
    def load(self, model_path: str, config: Dict[str, Any], engine_preference: EngineType = EngineType.AUTO) -> InferenceEngine:
        """加载模型"""
        loader = self.get_loader_for_model(model_path, engine_preference)
        if loader is None:
            raise ModelLoadError(f"No suitable loader for model: {model_path}")
        
        logger.info(f"Loading model with {loader.__class__.__name__}: {model_path}")
        return loader.load(model_path, config)
    
    def unload(self, engine: InferenceEngine) -> None:
        """卸载模型（需要找到对应的加载器）"""
        for loader in self._loaders:
            try:
                loader.unload(engine)
                return
            except Exception:
                continue
        # 兜底：直接调用 engine.unload()
        engine.unload()


class FilesystemModelRepository(ModelRepository):
    """基于文件系统的模型仓储实现"""
    
    def __init__(self, settings: Settings):
        self._settings = settings
        self._cache: Dict[str, ModelInfo] = {}
        self._scan_done = False
    
    def _scan_models(self) -> None:
        """扫描模型目录"""
        if self._scan_done:
            return
        
        model_dir = Path(self._settings.model_dir_path())
        if not model_dir.exists():
            logger.warning(f"Model directory not found: {model_dir}")
            self._scan_done = True
            return
        
        self._cache.clear()
        
        # 扫描 GGUF 文件
        for gguf_file in model_dir.rglob("*.gguf"):
            rel_path = gguf_file.relative_to(model_dir)
            model_id = str(rel_path).replace("/", "-").replace(".gguf", "")
            
            stat = gguf_file.stat()
            self._cache[model_id] = ModelInfo(
                model_id=model_id,
                display_name=gguf_file.stem,
                model_path=gguf_file,
                model_type="gguf",
                size_bytes=stat.st_size,
                description=f"GGUF model: {gguf_file.name}",
                tags=["gguf", "quantized"],
            )
        
        # 扫描 HF 模型目录（包含 config.json）
        for config_file in model_dir.rglob("config.json"):
            model_dir_path = config_file.parent
            if model_dir_path == model_dir:
                continue  # 跳过根目录
            
            rel_path = model_dir_path.relative_to(model_dir)
            model_id = str(rel_path).replace("/", "-")
            
            # 读取 config.json 获取架构信息
            import json
            try:
                with open(config_file) as f:
                    config = json.load(f)
                architectures = config.get("architectures", [])
                arch = architectures[0] if architectures else None
            except Exception:
                arch = None
            
            # 计算目录大小
            total_size = sum(f.stat().st_size for f in model_dir_path.rglob("*") if f.is_file())
            
            self._cache[model_id] = ModelInfo(
                model_id=model_id,
                display_name=model_dir_path.name,
                model_path=model_dir_path,
                model_type="hf",
                size_bytes=total_size,
                architecture=arch,
                description=f"Hugging Face model: {model_dir_path.name}",
                tags=["hf", "transformers"] + (architectures if architectures else []),
            )
        
        self._scan_done = True
        logger.info(f"Scanned {len(self._cache)} models from {model_dir}")
    
    def list_models(self) -> List[ModelInfo]:
        self._scan_models()
        return list(self._cache.values())
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        self._scan_models()
        return self._cache.get(model_id)
    
    def find_by_path(self, path: Path) -> Optional[ModelInfo]:
        self._scan_models()
        for model_info in self._cache.values():
            if model_info.model_path == path or model_info.model_path.resolve() == path.resolve():
                return model_info
        return None
    
    def exists(self, model_id: str) -> bool:
        self._scan_models()
        return model_id in self._cache
    
    def refresh(self) -> None:
        self._scan_done = False
        self._scan_models()


class ModelLoadError(Exception):
    """模型加载失败异常"""
    pass