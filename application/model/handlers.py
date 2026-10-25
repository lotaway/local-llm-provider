"""
Application: Model management use cases.
Orchestrates model loading, unloading, listing.
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
    """Load model command."""
    model_id: str
    engine_preference: EngineType = EngineType.AUTO
    context_length: Optional[int] = None
    quantization: Optional[str] = None
    vllm_options: Optional[dict] = None


@dataclass(frozen=True)
class UnloadModelCommand:
    """Unload model command."""
    model_id: Optional[str] = None


@dataclass(frozen=True)
class ListModelsQuery:
    """List models query."""
    model_type: Optional[str] = None
    tags: Optional[List[str]] = None


@dataclass(frozen=True)
class ModelLoadedEvent:
    """Model loaded event."""
    model_id: str
    engine_type: str
    context_length: int


class LoadModelUseCase:
    """Load model use case."""

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
        model = self._selection_service.resolve_model(
            command.model_id,
            command.engine_preference
        )

        if self._current_engine:
            self._loader_factory.unload(self._current_engine)
            self._current_engine = None
            if self._current_model:
                self._current_model.mark_unloaded()

        config = self._build_load_config(model, command)

        engine = self._loader_factory.load(model.path.value, config, command.engine_preference)

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
    """Unload model use case."""

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
    """List models use case."""

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
    """Get model info use case."""

    def __init__(self, model_repo: ModelRepository):
        self._model_repo = model_repo

    def execute(self, model_id: str) -> Optional[Model]:
        return self._model_repo.get_model(model_id)