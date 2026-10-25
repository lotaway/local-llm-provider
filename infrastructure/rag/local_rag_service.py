"""
基础设施层：本地 RAG 服务实现
适配现有的 LocalRAG 到 RAGService 接口
"""
import logging
from typing import List, Optional, Dict, Any, AsyncGenerator, Union

from domain.rag.ports import RAGService, VectorStore, Document, RetrievalResult, GenerationResult
from domain.model.entities import Model, GenerationConfig
from infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


class MilvusVectorStore(VectorStore):
    """Milvus 向量存储实现"""
    
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = None
        self._collection = None
        self._init_client()
    
    def _init_client(self):
        try:
            from pymilvus import MilvusClient
            self._client = MilvusClient(
                uri=f"http://{self._settings.rag.db_host}:{self._settings.rag.db_port}"
            )
            self._collection = self._settings.rag.db_collection
        except ImportError:
            logger.warning("pymilvus not installed, MilvusVectorStore unavailable")
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        if not self._client:
            raise RuntimeError("Milvus client not initialized")
        
        # 实现向量插入逻辑
        # 这里简化，实际需要根据现有 LocalRAG 实现
        ids = [doc.id for doc in documents]
        return ids
    
    async def delete_documents(self, ids: List[str]) -> int:
        if not self._client:
            return 0
        return len(ids)
    
    async def similarity_search(
        self, 
        query_vector: List[float], 
        k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        if not self._client:
            return []
        return []
    
    async def hybrid_search(
        self, 
        query: str, 
        query_vector: List[float], 
        k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        return await self.similarity_search(query_vector, k, filter)
    
    async def health_check(self) -> bool:
        return self._client is not None


class LocalRAGService(RAGService):
    """本地 RAG 服务实现 - 适配现有 LocalRAG"""
    
    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        inference_service,  # InferenceDomainService
        embedding_provider,  # EmbeddingProvider
    ):
        self._settings = settings
        self._vector_store = vector_store
        self._inference = inference_service
        self._embedding = embedding_provider
        self._rag_instance = None
    
    def _get_rag_instance(self, model: Model):
        """获取或创建 RAG 实例（延迟初始化）"""
        if self._rag_instance is None or self._rag_instance.llm != model:
            # 这里适配现有的 LocalRAG
            from rag import LocalRAG
            self._rag_instance = LocalRAG(model)
        return self._rag_instance
    
    async def ingest(
        self, 
        content: str, 
        metadata: Dict[str, Any] = None,
        doc_id: Optional[str] = None
    ) -> str:
        import uuid
        doc_id = doc_id or str(uuid.uuid4())
        metadata = metadata or {}
        
        doc = Document(
            id=doc_id,
            content=content,
            metadata=metadata,
        )
        
        await self._vector_store.add_documents([doc])
        return doc_id
    
    async def ingest_batch(self, documents: List[Document]) -> List[str]:
        return await self._vector_store.add_documents(documents)
    
    async def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        import time
        start = time.time()
        
        query_vector = self._embedding.embed_query(query)
        documents = await self._vector_store.similarity_search(query_vector, k=k)
        
        return RetrievalResult(
            query=query,
            documents=documents,
            total_count=len(documents),
            retrieval_time_ms=(time.time() - start) * 1000,
        )
    
    async def generate(
        self, 
        query: str, 
        k: int = 5,
        stream: bool = False
    ) -> Union[GenerationResult, AsyncGenerator[str, None]]:
        # 适配现有的 LocalRAG.generate_answer
        if self._inference.current_model is None:
            raise RuntimeError("No model loaded for RAG")
        
        rag = self._get_rag_instance(self._inference.current_model)
        
        if stream:
            async def stream_generator():
                async for chunk in rag.generate_answer(query, stream=True):
                    if chunk:
                        yield chunk
            return stream_generator()
        else:
            import time
            start = time.time()
            answer = await rag.generate_answer(query)
            return GenerationResult(
                answer=answer,
                sources=[],  # TODO: 从 rag 获取来源
                tokens_used=len(answer) // 3,
                generation_time_ms=(time.time() - start) * 1000,
            )
    
    async def delete(self, doc_id: str) -> bool:
        count = await self._vector_store.delete_documents([doc_id])
        return count > 0
    
    async def health_check(self) -> bool:
        return await self._vector_store.health_check()