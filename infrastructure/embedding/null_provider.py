"""
空实现：嵌入向量提供者（无重型依赖时的降级）
零第三方依赖，返回标准 Python list[float]
"""
import logging
from typing import List

from domain.embedding.ports import EmbeddingProvider, EmbeddingVector

logger = logging.getLogger(__name__)


class NullEmbeddingProvider(EmbeddingProvider):
    """空嵌入提供者 - 返回零向量，用于无重型依赖环境"""
    
    def __init__(self, dimension: int = 768):
        self._dimension = dimension
        self._zero_vector = [0.0] * dimension
        logger.info(f"NullEmbeddingProvider initialized (dim={dimension})")
    
    @property
    def model_name(self) -> str:
        return "null-embedding"
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def max_sequence_length(self) -> int:
        return 512
    
    def embed(self, text: str) -> EmbeddingVector:
        return self._zero_vector.copy()
    
    def embed_batch(self, texts: List[str]) -> List[EmbeddingVector]:
        return [self._zero_vector.copy() for _ in texts]
    
    def similarity(self, vec1: EmbeddingVector, vec2: EmbeddingVector) -> float:
        return 0.0
    
    def embed_query(self, query: str) -> EmbeddingVector:
        return self._zero_vector.copy()
    
    def embed_documents(self, documents: List[str]) -> List[EmbeddingVector]:
        return [self._zero_vector.copy() for _ in documents]