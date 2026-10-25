"""
领域层：嵌入向量提供者端口（零第三方依赖）
使用 typing.Sequence[float] 替代 np.ndarray
"""
from abc import ABC, abstractmethod
from typing import List, Sequence


# 向量类型别名：领域层不依赖 numpy
EmbeddingVector = Sequence[float]


class EmbeddingProvider(ABC):
    """嵌入向量提供者接口"""
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称"""
        ...
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度"""
        ...
    
    @property
    @abstractmethod
    def max_sequence_length(self) -> int:
        """最大序列长度"""
        ...
    
    @abstractmethod
    def embed(self, text: str) -> EmbeddingVector:
        """单文本嵌入"""
        ...
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[EmbeddingVector]:
        """批量文本嵌入"""
        ...
    
    @abstractmethod
    def similarity(self, vec1: EmbeddingVector, vec2: EmbeddingVector) -> float:
        """计算两个向量的相似度"""
        ...
    
    def embed_query(self, query: str) -> EmbeddingVector:
        """嵌入查询文本（可重写以添加查询前缀）"""
        return self.embed(query)
    
    def embed_documents(self, documents: List[str]) -> List[EmbeddingVector]:
        """嵌入文档列表（可重写以添加文档前缀）"""
        return self.embed_batch(documents)