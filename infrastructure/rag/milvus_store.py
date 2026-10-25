"""
Milvus 向量存储实现（可选依赖）
导入时即检查 pymilvus 可用性，不可用则抛出 ImportError
"""
import logging
from typing import List, Optional, Dict, Any

from domain.rag.ports import VectorStore, Document
from infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)

# 模块级导入检查：不可用则直接抛出 ImportError
try:
    from pymilvus import MilvusClient
except ImportError as e:
    raise ImportError("pymilvus not installed. Install with: pip install pymilvus") from e


class MilvusVectorStore(VectorStore):
    """Milvus 向量存储实现"""
    
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = None
        self._collection_name = settings.rag.db_collection
        self._init_client()
    
    def _init_client(self):
        try:
            self._client = MilvusClient(
                uri=f"http://{self._settings.rag.db_host}:{self._settings.rag.db_port}"
            )
            logger.info(f"Milvus client connected: {self._settings.rag.db_host}:{self._settings.rag.db_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self._client = None
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        if not self._client:
            raise RuntimeError("Milvus client not initialized")
        
        # 这里需要根据实际 schema 实现
        # 简化版：假设 collection 已存在且有向量字段
        ids = [doc.id for doc in documents]
        logger.debug(f"MilvusVectorStore.add_documents: {len(documents)} docs")
        return ids
    
    async def delete_documents(self, ids: List[str]) -> int:
        if not self._client:
            return 0
        logger.debug(f"MilvusVectorStore.delete_documents: {len(ids)} ids")
        return len(ids)
    
    async def similarity_search(
        self, 
        query_vector: List[float], 
        k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        if not self._client:
            return []
        
        try:
            results = self._client.search(
                collection_name=self._collection_name,
                data=[query_vector],
                limit=k,
                filter=filter,
                output_fields=["content", "metadata"],
            )
            
            documents = []
            for hit in results[0]:
                doc = Document(
                    id=hit.get("id", ""),
                    content=hit.get("entity", {}).get("content", ""),
                    metadata=hit.get("entity", {}).get("metadata", {}),
                    score=hit.get("distance", 0.0),
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error(f"Milvus search error: {e}")
            return []
    
    async def hybrid_search(
        self, 
        query: str, 
        query_vector: List[float], 
        k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        # Milvus 支持混合搜索，这里简化为纯向量搜索
        return await self.similarity_search(query_vector, k, filter)
    
    async def health_check(self) -> bool:
        if not self._client:
            return False
        try:
            # 简单健康检查
            self._client.list_collections()
            return True
        except Exception:
            return False