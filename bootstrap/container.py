"""
Dependency injection container (lazy loading + optional dependency graceful degradation).
Supports single-file testing: no forced heavy dependency loading.
"""
import logging
from collections.abc import Callable
from typing import Any

from infrastructure.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class Container:
    """Dependency injection container - lazy loading mode."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._overrides: dict[type, Any] = {}
        self._singletons: dict[type, Any] = {}
        self._initialized = False

    def register_override(self, interface: type, instance: Any) -> None:
        """Register test double (Mock/Stub)."""
        self._overrides[interface] = instance
        logger.debug(f"Override registered: {interface.__name__} -> {type(instance).__name__}")

    def clear_overrides(self) -> None:
        """Clear all overrides."""
        self._overrides.clear()
        self._singletons.clear()

    def _get_or_create(self, key: type, factory: Callable[["Container"], Any]) -> Any:
        """Get singleton, create on first call."""
        if key in self._overrides:
            return self._overrides[key]

        if key not in self._singletons:
            self._singletons[key] = factory(self)
        return self._singletons[key]

    @property
    def settings(self) -> Settings:
        return self._settings

    def model_repo(self):
        from domain.model.repositories import ModelRepository
        from infrastructure.model.loader.factory import FilesystemModelRepository
        return self._get_or_create(ModelRepository, lambda c: FilesystemModelRepository(c.settings))

    def loader_factory(self):
        from infrastructure.model.loader.factory import ModelLoaderFactory
        return self._get_or_create(ModelLoaderFactory, lambda c: ModelLoaderFactory(c.settings))

    def task_scheduler(self):
        from domain.scheduling.ports import TaskScheduler
        from infrastructure.scheduling.async_scheduler import AsyncTaskScheduler
        return self._get_or_create(TaskScheduler, lambda c: AsyncTaskScheduler(max_concurrent=10))

    def batch_scheduler(self):
        from domain.scheduling.ports import BatchScheduler
        from infrastructure.scheduling.async_scheduler import AsyncBatchScheduler
        return self._get_or_create(BatchScheduler, lambda c: AsyncBatchScheduler(max_batch_size=32))

    def embedding_provider(self):
        """Embedding provider - optional dependency, graceful degradation."""
        from domain.embedding.ports import EmbeddingProvider

        def factory(c):
            try:
                from infrastructure.embedding.sentence_transformer_provider import (
                    SentenceTransformerEmbeddingProvider,
                )
                return SentenceTransformerEmbeddingProvider(c.settings)
            except ImportError as e:
                logger.warning(f"sentence-transformers not available: {e}, using NullEmbeddingProvider")
                from infrastructure.embedding.null_provider import NullEmbeddingProvider
                return NullEmbeddingProvider()

        return self._get_or_create(EmbeddingProvider, factory)

    def vector_store(self):
        """Vector store - optional dependency, graceful degradation."""
        from domain.rag.ports import VectorStore

        def factory(c):
            try:
                from infrastructure.rag.milvus_store import MilvusVectorStore
                return MilvusVectorStore(c.settings)
            except ImportError as e:
                logger.warning(f"pymilvus not available: {e}, using NullVectorStore")
                from infrastructure.rag.null_store import NullVectorStore
                return NullVectorStore()

        return self._get_or_create(VectorStore, factory)

    def selection_service(self):
        from domain.model.services import ModelSelectionService
        return self._get_or_create(ModelSelectionService, lambda c: ModelSelectionService(c.model_repo()))

    def inference_service(self):
        from domain.inference.services import InferenceDomainService
        return self._get_or_create(InferenceDomainService,
            lambda c: InferenceDomainService(
                model_repo=c.model_repo(),
                batch_scheduler=c.batch_scheduler(),
            ))

    def load_model_use_case(self):
        from application.model.handlers import LoadModelUseCase
        return self._get_or_create(LoadModelUseCase,
            lambda c: LoadModelUseCase(
                model_repo=c.model_repo(),
                loader_factory=c.loader_factory(),
                selection_service=c.selection_service(),
            ))

    def unload_model_use_case(self):
        from application.model.handlers import UnloadModelUseCase
        return self._get_or_create(UnloadModelUseCase,
            lambda c: UnloadModelUseCase(c.loader_factory()))

    def list_models_use_case(self):
        from application.model.handlers import ListModelsUseCase
        return self._get_or_create(ListModelsUseCase,
            lambda c: ListModelsUseCase(c.model_repo()))

    def get_model_info_use_case(self):
        from application.model.handlers import GetModelInfoUseCase
        return self._get_or_create(GetModelInfoUseCase,
            lambda c: GetModelInfoUseCase(c.model_repo()))

    def chat_use_case(self):
        from application.inference.handlers import ChatUseCase
        return self._get_or_create(ChatUseCase,
            lambda c: ChatUseCase(c.inference_service()))

    def completion_use_case(self):
        from application.inference.handlers import CompletionUseCase
        return self._get_or_create(CompletionUseCase,
            lambda c: CompletionUseCase(c.inference_service()))

    def token_count_use_case(self):
        from application.inference.handlers import TokenCountUseCase
        return self._get_or_create(TokenCountUseCase,
            lambda c: TokenCountUseCase(c.inference_service()))

    def truncate_messages_use_case(self):
        from application.inference.handlers import TruncateMessagesUseCase
        return self._get_or_create(TruncateMessagesUseCase,
            lambda c: TruncateMessagesUseCase(c.inference_service()))

    def rag_service(self):
        """RAG service - depends on inference service, created on demand."""
        from domain.rag.ports import RAGService

        def factory(c):
            try:
                from infrastructure.rag.local_rag_service import LocalRAGService
                return LocalRAGService(
                    settings=c.settings,
                    vector_store=c.vector_store(),
                    inference_service=c.inference_service(),
                    embedding_provider=c.embedding_provider(),
                )
            except ImportError as e:
                logger.warning(f"RAG dependencies not available: {e}, using NullRAGService")
                from infrastructure.rag.null_service import NullRAGService
                return NullRAGService()

        return self._get_or_create(RAGService, factory)

    async def start_schedulers(self):
        """Start schedulers (explicit call, not automatic)."""
        scheduler = self.task_scheduler()
        batch_scheduler = self.batch_scheduler()
        await scheduler.start()
        await batch_scheduler.start()
        self._initialized = True
        logger.info("Schedulers started")

    async def shutdown(self):
        """Shutdown all resources."""
        logger.info("Shutting down container...")

        from domain.inference.services import InferenceDomainService
        from infrastructure.scheduling.async_scheduler import (
            AsyncBatchScheduler,
            AsyncTaskScheduler,
        )

        if InferenceDomainService in self._singletons:
            inference = self._singletons[InferenceDomainService]
            if hasattr(inference, 'clear_engine'):
                inference.clear_engine()

        if AsyncTaskScheduler in self._singletons:
            await self._singletons[AsyncTaskScheduler].shutdown(graceful=True)

        if AsyncBatchScheduler in self._singletons:
            await self._singletons[AsyncBatchScheduler].shutdown()

        self._singletons.clear()
        self._initialized = False
        logger.info("Container shutdown complete")

    def wire_routes(self):
        """Wire routes (production entry point)."""
        from interfaces.http.routes.chat_routes import set_dependencies

        set_dependencies(
            chat_use_case=self.chat_use_case(),
            completion_use_case=self.completion_use_case(),
            token_count_use_case=self.token_count_use_case(),
            load_model_use_case=self.load_model_use_case(),
            unload_model_use_case=self.unload_model_use_case(),
            list_models_use_case=self.list_models_use_case(),
        )

        unload_uc = self.unload_model_use_case()
        inference_svc = self.inference_service()
        unload_uc.set_current(
            inference_svc._engine if inference_svc else None,
            inference_svc._current_model if inference_svc else None,
        )

        logger.info("Routes wired successfully")


# Global container instance (singleton)
_container: Container | None = None


def get_container() -> Container:
    """Get global container instance (lazy creation)."""
    global _container
    if _container is None:
        _container = Container()
    return _container


def set_container(container: Container) -> None:
    """Set global container (test only)."""
    global _container
    _container = container


async def initialize_container(settings: Settings | None = None) -> Container:
    """Initialize global container (production entry point)."""
    global _container
    _container = Container(settings)
    await _container.start_schedulers()
    _container.wire_routes()
    return _container


async def shutdown_container():
    """Shutdown global container."""
    global _container
    if _container:
        await _container.shutdown()
        _container = None