"""Reranker - Multi-stage reranking pipeline for better relevance"""

from typing import List, Tuple
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np


class Reranker:
    """
    Multi-stage reranking pipeline: BM25 → Embedding → Cross-Encoder
    Configurable for different document scales
    """
    
    def __init__(
        self,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_cross_encoder: bool = True,
        cross_encoder_threshold: int = 100
    ):
        """
        Initialize reranker
        
        Args:
            cross_encoder_model: Cross-encoder model name
            use_cross_encoder: Whether to use cross-encoder (expensive for large doc sets)
            cross_encoder_threshold: Only use cross-encoder if candidates < this number
        """
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder_threshold = cross_encoder_threshold
        self.cross_encoder = None
        
        if use_cross_encoder:
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_model)
            except Exception as e:
                print(f"Warning: Failed to load cross-encoder: {e}")
                self.use_cross_encoder = False
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
        stage: str = "full"
    ) -> List[Document]:
        """
        Rerank documents using multi-stage pipeline
        
        Args:
            query: Search query
            documents: Candidate documents
            top_k: Number of documents to return
            stage: Reranking stage ("bm25", "embedding", "full")
            
        Returns:
            Reranked documents
        """
        if not documents:
            return []
        
        if len(documents) <= top_k:
            return documents
        
        # Stage 1: BM25 reranking
        if stage in ["bm25", "full"]:
            documents = self._bm25_rerank(query, documents, min(len(documents), top_k * 3))
        
        # Stage 2: Embedding-based reranking (already done by vector search)
        # Skip this stage as it's redundant with initial retrieval
        
        # Stage 3: Cross-encoder reranking (most expensive, most accurate)
        if stage == "full" and self.use_cross_encoder and len(documents) <= self.cross_encoder_threshold:
            documents = self._cross_encoder_rerank(query, documents, top_k)
        else:
            documents = documents[:top_k]
        
        return documents
    
    def _bm25_rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        """Rerank using BM25"""
        # Tokenize documents
        tokenized_docs = [doc.page_content.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        
        # Get BM25 scores
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        
        # Sort by score
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [documents[i] for i in ranked_indices]
    
    def _cross_encoder_rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        """Rerank using cross-encoder"""
        if not self.cross_encoder:
            return documents[:top_k]
        
        # Create query-document pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Sort by score
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [documents[i] for i in ranked_indices]
    
    def adaptive_rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> List[Document]:
        """
        Adaptive reranking based on document count
        
        - Few docs (< 50): Use full pipeline with cross-encoder
        - Medium docs (50-100): Use BM25 + cross-encoder
        - Many docs (> 100): Use BM25 only
        
        Args:
            query: Search query
            documents: Candidate documents
            top_k: Number of documents to return
            
        Returns:
            Reranked documents
        """
        doc_count = len(documents)
        
        if doc_count <= 50:
            # Use full pipeline
            return self.rerank(query, documents, top_k, stage="full")
        elif doc_count <= 100:
            # Use BM25 + cross-encoder
            bm25_docs = self._bm25_rerank(query, documents, min(doc_count, top_k * 2))
            if self.use_cross_encoder:
                return self._cross_encoder_rerank(query, bm25_docs, top_k)
            return bm25_docs[:top_k]
        else:
            # Use BM25 only for large doc sets
            return self._bm25_rerank(query, documents, top_k)
