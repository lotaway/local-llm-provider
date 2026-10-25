"""
应用层：模型管理用例
编排模型加载、卸载、列表查询
"""
from dataclasses import dataclass
from typing import List, Optional

from domain.model.entities import Model, ModelId, EngineType
from domain.model.repositories import ModelRepository
from domain.model.services import ModelSelectionService, ModelNotFoundError
from domain.inference.ports import InferenceEngine
from infrastructure.model.loader.factory import ModelLoaderFactory, FilesystemModelRepository, ModelLoadError


@dataclass(frozen=True)
class LoadModelCommand:
    """加载模型命令"""
    model_id: str
    engine_preference: EngineType = EngineType.AUTO
    context_length: Optional[int] = None
    quantization: Optional[str] = None
    vllm_options: Optional[dict] = None


@dataclass(frozen=True)
class UnloadModelCommand:
    """卸载模型命令"""
    model_id: Optional[str] = None  # None 表示卸载当前模型


@dataclass(frozen=True)
class ListModelsQuery:
    """列出模型查询"""
    model_type: Optional[str] = None
    tags: Optional[List[str]] = None


@dataclass(frozen=True)
class ModelLoadedEvent:
    """模型加载完成事件"""
    model_id: str
    engine_type: str
    context_length: int


class LoadModelUseCase:
    """加载模型用例"""
    
    def __init__(
        self,
        model_repo: ModelRepository,
        loader_factory: ModelLoaderFactory,
        selection_service: ModelSelectionService,
    ):
        self._model_repo = model_repo
        self._loader_factory = loader_factory
        self._selection_service = selection_service
        self._current_engine: Optional[InferenceEngine] = None
        self._current_model: Optional[Model] = None
    
    def execute(self, command: LoadModelCommand) -> ModelLoadedEvent:
        # 1. 解析模型（领域逻辑：架构兼容性检查）
        model = self._selection_service.resolve_model(
            command.model_id, 
            command.engine_preference
        )
        
        # 2. 如果有当前模型，先卸载
        if self._current_engine:
            self._loader_factory.unload(self._current_engine)
            self._current_engine = None
            if self._current_model:
                self._current_model.mark_unloaded()
        
        # 3. 构建加载配置
        config = self._build_load_config(model, command)
        
        # 4. 加载模型（基础设施逻辑）
        engine = self._loader_factory.load(model.path.value, config, command.engine_preference)
        
        # 5. 更新状态
        self._current_engine = engine
        self._current_model = model
        model.mark_loaded(engine.get_model_info())
        
        return ModelLoadedEvent(
            model_id=model.id.value,
            engine_type=engine.engine_type,
            context_length=model.context_length,
        )
    
    def _build_load_config(self, model: Model, command: LoadModelCommand) -> dict:
        config = {
            "model_loader": {
                "engine_preference": command.engine_preference.value,
                "context_length": command.context_length or model.context_length,
                "quantization": command.quantization,
            }
        }
        
        if command.vllm_options:
            config["vllm"] = command.vllm_options
        
        return config


class UnloadModelUseCase:
    """卸载模型用例"""
    
    def __init__(self, loader_factory: ModelLoaderFactory):
        self._loader_factory = loader_factory
        self._current_engine: Optional[InferenceEngine] = None
        self._current_model: Optional[Model] = None
    
    def set_current(self, engine: InferenceEngine, model: Model):
        self._current_engine = engine
        self._current_model = model
    
    def execute(self, command: UnloadModelCommand) -> bool:
        if self._current_engine is None:
            return False
        
        self._loader_factory.unload(self._current_engine)
        if self._current_model:
            self._current_model.mark_unloaded()
        self._current_engine = None
        self._current_model = None
        return True


class ListModelsUseCase:
    """列出模型用例"""
    
    def __init__(self, model_repo: ModelRepository):
        self._model_repo = model_repo
    
    def execute(self, query: ListModelsQuery) -> List[Model]:
        models = self._model_repo.list_models()
        
        if query.model_type:
            models = [m for m in models if m.model_type == query.model_type]
        
        if query.tags:
            models = [m for m in models if any(tag in m.tags for tag in query.tags)]
        
        return models


class GetModelInfoUseCase:
    """获取模型信息用例"""
    
    def __init__(self, model_repo: ModelRepository):
        self._model_repo = model_repo
    
    def execute(self, model_id: str) -> Optional[Model]:
        return self._model_repo.get_model(model_id)