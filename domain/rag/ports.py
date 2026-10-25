"""
领域层：RAG 服务端口
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncGenerator, Union
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """文档值对象"""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


@dataclass(frozen=True)
class RetrievalResult:
    """检索结果"""
    query: str
    documents: List[Document]
    total_count: int
    retrieval_time_ms: float


@dataclass(frozen=True)
class GenerationResult:
    """生成结果"""
    answer: str
    sources: List[Document]
    tokens_used: int
    generation_time_ms: float


class VectorStore(ABC):
    """向量存储接口"""
    
    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档，返回 ID 列表"""
        ...
    
    @abstractmethod
    async def delete_documents(self, ids: List[str]) -> int:
        """删除文档，返回删除数量"""
        ...
    
    @abstractmethod
    async def similarity_search(
        self, 
        query_vector: List[float], 
        k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """向量相似度搜索"""
        ...
    
    @abstractmethod
    async def hybrid_search(
        self, 
        query: str, 
        query_vector: List[float], 
        k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """混合搜索（向量 + 关键词）"""
        ...
    
    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        ...


class RAGService(ABC):
    """RAG 服务接口"""
    
    @abstractmethod
    async def ingest(
        self, 
        content: str, 
        metadata: Dict[str, Any] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """摄入文档"""
        ...
    
    @abstractmethod
    async def ingest_batch(
        self, 
        documents: List[Document]
    ) -> List[str]:
        """批量摄入"""
        ...
    
    @abstractmethod
    async def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        """检索相关文档"""
        ...
    
    @abstractmethod
    async def generate(
        self, 
        query: str, 
        k: int = 5,
        stream: bool = False
    ) -> Union[GenerationResult, AsyncGenerator[str, None]]:
        """基于检索生成回答"""
        ...
    
    @abstractmethod
    async def delete(self, doc_id: str) -> bool:
        """删除文档"""
        ...
    
    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        ...