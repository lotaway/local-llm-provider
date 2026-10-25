"""
空实现：向量存储（无重型依赖时的降级）
"""
import logging
from typing import List, Optional, Dict, Any

from domain.rag.ports import VectorStore, Document

logger = logging.getLogger(__name__)


class NullVectorStore(VectorStore):
    """空向量存储 - 无操作实现"""
    
    def __init__(self):
        logger.info("NullVectorStore initialized")
    
    async def add_documents(self, documents: List[Document]) -> List[str]:
        logger.debug(f"NullVectorStore.add_documents called with {len(documents)} docs")
        return [doc.id for doc in documents]
    
    async def delete_documents(self, ids: List[str]) -> int:
        logger.debug(f"NullVectorStore.delete_documents called with {len(ids)} ids")
        return len(ids)
    
    async def similarity_search(
        self, 
        query_vector: List[float], 
        k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        logger.debug(f"NullVectorStore.similarity_search called (k={k})")
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
        return True