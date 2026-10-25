"""
空实现：RAG 服务（无重型依赖时的降级）
"""
import logging
from typing import List, Optional, Dict, Any, AsyncGenerator, Union

from domain.rag.ports import RAGService, Document, RetrievalResult, GenerationResult

logger = logging.getLogger(__name__)


class NullRAGService(RAGService):
    """空 RAG 服务 - 返回空结果"""
    
    def __init__(self):
        logger.info("NullRAGService initialized")
    
    async def ingest(
        self, 
        content: str, 
        metadata: Dict[str, Any] = None,
        doc_id: Optional[str] = None
    ) -> str:
        logger.debug(f"NullRAGService.ingest called")
        return doc_id or "null-doc-id"
    
    async def ingest_batch(self, documents: List[Document]) -> List[str]:
        return [doc.id for doc in documents]
    
    async def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        logger.debug(f"NullRAGService.retrieve called: {query[:50]}...")
        return RetrievalResult(
            query=query,
            documents=[],
            total_count=0,
            retrieval_time_ms=0.0,
        )
    
    async def generate(
        self, 
        query: str, 
        k: int = 5,
        stream: bool = False
    ) -> Union[GenerationResult, AsyncGenerator[str, None]]:
        logger.debug(f"NullRAGService.generate called: {query[:50]}...")
        
        if stream:
            async def empty_generator():
                yield ""
                return
            return empty_generator()
        
        return GenerationResult(
            answer="[RAG service not available - install pymilvus and sentence-transformers]",
            sources=[],
            tokens_used=0,
            generation_time_ms=0.0,
        )
    
    async def delete(self, doc_id: str) -> bool:
        return True
    
    async def health_check(self) -> bool:
        return True