"""
领域层：推理领域服务
编排推理流程、会话管理
"""
import asyncio
from typing import Optional, AsyncGenerator
from domain.inference.ports import InferenceEngine, InferenceRequest, InferenceChunk
from domain.model.entities import Model, GenerationConfig, InferenceSession
from domain.model.repositories import ModelRepository
from domain.scheduling.ports import BatchScheduler


class InferenceDomainService:
    """推理领域服务"""
    
    def __init__(
        self,
        model_repo: ModelRepository,
        batch_scheduler: BatchScheduler,
    ):
        self._model_repo = model_repo
        self._batch_scheduler = batch_scheduler
        self._engine: Optional[InferenceEngine] = None
        self._current_model: Optional[Model] = None
        self._session: Optional[InferenceSession] = None
    
    def set_engine(self, engine: InferenceEngine, model: Model) -> None:
        """设置当前引擎（由应用层调用）"""
        self._engine = engine
        self._current_model = model
        model.mark_loaded(engine.get_model_info())
    
    def clear_engine(self) -> None:
        """清除当前引擎"""
        if self._current_model:
            self._current_model.mark_unloaded()
        if self._engine:
            self._engine.unload()
        self._engine = None
        self._current_model = None
        self._session = None
    
    @property
    def current_model(self) -> Optional[Model]:
        return self._current_model
    
    @property
    def is_ready(self) -> bool:
        return self._engine is not None and self._engine.health_check()
    
    async def generate_stream(
        self, 
        prompt: str, 
        config: GenerationConfig,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[InferenceChunk, None]:
        """流式生成"""
        if not self.is_ready:
            raise InferenceNotReadyError("No model loaded or engine unhealthy")
        
        request = InferenceRequest(
            prompt=prompt,
            config=config,
            session_id=session_id,
        )
        
        # 使用批处理调度器
        if session_id:
            queue = await self._batch_scheduler.register(session_id, {
                "prompt": prompt,
                "config": config,
            })
            try:
                async for chunk in self._process_queue(queue):
                    if self._session:
                        self._session.record_generation(chunk.token_count)
                    yield chunk
            finally:
                await self._batch_scheduler.unregister(session_id)
        else:
            # 直接生成
            async for chunk in self._engine.generate_stream(request):
                if self._session:
                    self._session.record_generation(chunk.token_count)
                yield chunk
    
    async def generate(
        self, 
        prompt: str, 
        config: GenerationConfig
    ) -> InferenceChunk:
        """非流式生成"""
        if not self.is_ready:
            raise InferenceNotReadyError("No model loaded or engine unhealthy")
        
        request = InferenceRequest(prompt=prompt, config=config)
        return await self._engine.generate(request)
    
    async def _process_queue(self, queue: asyncio.Queue) -> AsyncGenerator[InferenceChunk, None]:
        """处理调度器队列"""
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
    
    def count_tokens(self, text: str) -> int:
        """计算 token"""
        if not self.is_ready:
            raise InferenceNotReadyError("No model loaded")
        return self._engine.count_tokens(text)
    
    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        """应用聊天模板"""
        if not self.is_ready:
            raise InferenceNotReadyError("No model loaded")
        return self._engine.apply_chat_template(messages, add_generation_prompt)
    
    def create_session(self, config: GenerationConfig) -> InferenceSession:
        """创建推理会话"""
        if not self._current_model:
            raise InferenceNotReadyError("No model loaded")
        session = InferenceSession.create(self._current_model.id, config)
        self._session = session
        return session
    
    def get_session(self) -> Optional[InferenceSession]:
        return self._session


class InferenceNotReadyError(Exception):
    """推理引擎未就绪异常"""
    pass