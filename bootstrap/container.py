"""
启动层：依赖注入容器（懒加载 + 可选依赖优雅降级）
支持单文件测试：不强制加载重型依赖
"""
import logging
import asyncio
from typing import Optional, Callable, Any, Dict, Type

from infrastructure.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class Container:
    """
    依赖注入容器 - 懒加载模式
    
    核心原则：
    1. __init__ 只存配置，不创建任何实例
    2. get_xxx() 首次调用时才创建（懒加载）
    3. 重型/可选依赖：捕获 ImportError，返回空实现或抛出清晰错误
    4. 测试时可通过 register_override 替换任意组件
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._overrides: Dict[Type, Any] = {}  # 测试用覆盖
        self._singletons: Dict[Type, Any] = {}  # 单例缓存
        self._initialized = False
    
    # ===== 注册覆盖（测试专用）=====
    def register_override(self, interface: Type, instance: Any) -> None:
        """注册测试替身（Mock/Stub）"""
        self._overrides[interface] = instance
        logger.debug(f"Override registered: {interface.__name__} -> {type(instance).__name__}")
    
    def clear_overrides(self) -> None:
        """清除所有覆盖"""
        self._overrides.clear()
        self._singletons.clear()
    
    # ===== 核心获取方法（懒加载）=====
    def _get_or_create(self, key: Type, factory: Callable[["Container"], Any]) -> Any:
        """获取单例，首次调用时创建"""
        if key in self._overrides:
            return self._overrides[key]
        
        if key not in self._singletons:
            self._singletons[key] = factory(self)
        return self._singletons[key]
    
    # ===== 基础设施层 =====
    @property
    def settings(self) -> Settings:
        return self._settings
    
    def model_repo(self):
        from domain.model.repositories import ModelRepository
        from infrastructure.model.loader.factory import FilesystemModelRepository
        return self._get_or_create(ModelRepository, lambda c: FilesystemModelRepository(c.settings))
    
    def loader_factory(self):
        from domain.inference.ports import ModelLoader
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
        """嵌入提供者 - 可选依赖，优雅降级"""
        from domain.embedding.ports import EmbeddingProvider
        
        def factory(c):
            try:
                from infrastructure.embedding.sentence_transformer_provider import SentenceTransformerEmbeddingProvider
                return SentenceTransformerEmbeddingProvider(c.settings)
            except ImportError as e:
                logger.warning(f"sentence-transformers not available: {e}, using NullEmbeddingProvider")
                from infrastructure.embedding.null_provider import NullEmbeddingProvider
                return NullEmbeddingProvider()
        
        return self._get_or_create(EmbeddingProvider, factory)
    
    def vector_store(self):
        """向量存储 - 可选依赖，优雅降级"""
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
    
    # ===== 领域服务层 =====
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
    
    # ===== 应用层用例 =====
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
    
    # ===== RAG 服务（按需创建）=====
    def rag_service(self):
        """RAG 服务 - 依赖推理服务，按需创建"""
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
    
    # ===== 生命周期管理 =====
    async def start_schedulers(self):
        """启动调度器（显式调用，非自动）"""
        scheduler = self.task_scheduler()
        batch_scheduler = self.batch_scheduler()
        await scheduler.start()
        await batch_scheduler.start()
        self._initialized = True
        logger.info("Schedulers started")
    
    async def shutdown(self):
        """关闭所有资源"""
        logger.info("Shutting down container...")
        
        # 关闭推理引擎
        if InferenceDomainService in self._singletons:
            inference = self._singletons[InferenceDomainService]
            if hasattr(inference, 'clear_engine'):
                inference.clear_engine()
        
        # 关闭调度器
        if AsyncTaskScheduler in self._singletons:
            await self._singletons[AsyncTaskScheduler].shutdown(graceful=True)
        
        if AsyncBatchScheduler in self._singletons:
            await self._singletons[AsyncBatchScheduler].shutdown()
        
        self._singletons.clear()
        self._initialized = False
        logger.info("Container shutdown complete")
    
    # ===== 生产环境一键注入 =====
    def wire_routes(self):
        """一键注入到路由模块（生产环境调用）"""
        from interfaces.http.routes.chat_routes import set_dependencies
        
        set_dependencies(
            chat_use_case=self.chat_use_case(),
            completion_use_case=self.completion_use_case(),
            token_count_use_case=self.token_count_use_case(),
            load_model_use_case=self.load_model_use_case(),
            unload_model_use_case=self.unload_model_use_case(),
            list_models_use_case=self.list_models_use_case(),
        )
        
        # 同步 unload use case 状态
        unload_uc = self.unload_model_use_case()
        inference_svc = self.inference_service()
        unload_uc.set_current(
            inference_svc._engine if inference_svc else None,
            inference_svc._current_model if inference_svc else None,
        )
        
        logger.info("Routes wired successfully")


# 全局容器实例（单例）
_container: Optional[Container] = None


def get_container() -> Container:
    """获取全局容器实例（懒创建）"""
    global _container
    if _container is None:
        _container = Container()
    return _container


def set_container(container: Container) -> None:
    """设置全局容器（测试专用）"""
    global _container
    _container = container


async def initialize_container(settings: Optional[Settings] = None) -> Container:
    """初始化全局容器（生产入口调用）"""
    global _container
    _container = Container(settings)
    await _container.start_schedulers()
    _container.wire_routes()
    return _container


async def shutdown_container():
    """关闭全局容器"""
    global _container
    if _container:
        await _container.shutdown()
        _container = None