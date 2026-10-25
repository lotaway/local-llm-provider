"""
基础设施层：SentenceTransformer 嵌入向量提供者实现
内部使用 numpy，对外返回 Sequence[float]（领域层类型）
"""
import logging
from typing import List, Sequence
import numpy as np

from domain.embedding.ports import EmbeddingProvider, EmbeddingVector
from infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """SentenceTransformer 实现的嵌入向量提供者"""
    
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None
        self._dimension = None
        self._max_seq_length = None
        self._init_model()
    
    def _init_model(self):
        from sentence_transformers import SentenceTransformer
        
        self._model = SentenceTransformer(
            self._settings.embedding.model_name,
            cache_folder=self._settings.embedding.cache_path,
            device=self._settings.embedding.device,
        )
        self._dimension = self._model.get_sentence_embedding_dimension()
        self._max_seq_length = self._model.max_seq_length
        logger.info(f"Embedding model loaded: {self._settings.embedding.model_name}, dim={self._dimension}")
    
    @property
    def model_name(self) -> str:
        return self._settings.embedding.model_name
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def max_sequence_length(self) -> int:
        return self._max_seq_length
    
    def _to_sequence(self, arr: np.ndarray) -> EmbeddingVector:
        """numpy 数组转领域层序列类型"""
        return arr.astype(np.float32).tolist()
    
    def embed(self, text: str) -> EmbeddingVector:
        if self._model is None:
            self._init_model()
        vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return self._to_sequence(vec)
    
    def embed_batch(self, texts: List[str]) -> List[EmbeddingVector]:
        if self._model is None:
            self._init_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [self._to_sequence(emb) for emb in embeddings]
    
    def similarity(self, vec1: EmbeddingVector, vec2: EmbeddingVector) -> float:
        # 转 numpy 计算点积
        return float(np.dot(np.asarray(vec1, dtype=np.float32), np.asarray(vec2, dtype=np.float32)))
    
    def embed_query(self, query: str) -> EmbeddingVector:
        return self.embed(query)
    
    def embed_documents(self, documents: List[str]) -> List[EmbeddingVector]:
        return self.embed_batch(documents)