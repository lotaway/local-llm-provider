"""Reranker - Multi-stage reranking pipeline for better relevance"""

from typing import List, Tuple
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", use_cross_encoder: bool = True, cross_encoder_threshold: int = 100):
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder_threshold = cross_encoder_threshold
        self.cross_encoder = None
        if use_cross_encoder:
            try: self.cross_encoder = CrossEncoder(cross_encoder_model)
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}")
                self.use_cross_encoder = False
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5, stage: str = "full") -> List[Document]:
        if len(documents) <= top_k: return documents
        if stage in ["bm25", "full"]: documents = self._bm25_rerank(query, documents, min(len(documents), top_k * 3))
        if stage == "full" and self.use_cross_encoder and len(documents) <= self.cross_encoder_threshold:
            return self._cross_encoder_rerank(query, documents, top_k)
        return documents[:top_k]
    
    def _bm25_rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        tokenized_docs = [doc.page_content.split() for doc in documents]
        scores = BM25Okapi(tokenized_docs).get_scores(query.split())
        return [documents[i] for i in np.argsort(scores)[::-1][:top_k]]
    
    def _cross_encoder_rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        if not self.cross_encoder: return documents[:top_k]
        scores = self.cross_encoder.predict([[query, doc.page_content] for doc in documents])
        return [documents[i] for i in np.argsort(scores)[::-1][:top_k]]
    
    def adaptive_rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        doc_count = len(documents)
        if doc_count <= 50: return self.rerank(query, documents, top_k, stage="full")
        if doc_count <= 100:
            bm25_docs = self._bm25_rerank(query, documents, min(doc_count, top_k * 2))
            return self._cross_encoder_rerank(query, bm25_docs, top_k) if self.use_cross_encoder else bm25_docs[:top_k]
        return self._bm25_rerank(query, documents, top_k)
