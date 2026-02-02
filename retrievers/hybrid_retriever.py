"""Hybrid Retriever - Combines vector and keyword search"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining vector similarity and BM25 keyword search.
    Uses weighted linear fusion to combine results.
    Refactored to support external BM25 retrievers (e.g. Elasticsearch).
    """

    # Declare fields as class attributes for Pydantic
    vectorstore: Any
    bm25_retriever: Any
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    k: int = 5
    feedback_service: Optional[Any] = None

    # Allow arbitrary types (for vectorstore and bm25_retriever)
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        vectorstore,
        bm25_retriever,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        k: int = 5,
        feedback_service: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize hybrid retriever

        Args:
            vectorstore: Vector store for semantic search
            bm25_retriever: Retriever specifically for BM25 (must support get_documents_with_scores)
            vector_weight: Weight for vector search results (0-1)
            bm25_weight: Weight for BM25 results (0-1)
            k: Number of documents to retrieve
            feedback_service: Optional feedback service for tracking recalls
        """
        super().__init__(
            vectorstore=vectorstore,
            bm25_retriever=bm25_retriever,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            k=k,
            feedback_service=feedback_service,
            **kwargs,
        )
        self.feedback_service = feedback_service

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Retrieve documents using hybrid search

        Args:
            query: Search query
            run_manager: Callback manager

        Returns:
            List of retrieved documents
        """
        # Vector search
        try:
            # Fetch more candidates for reranking/fusion
            vector_docs = self.vectorstore.similarity_search_with_score(
                query, k=self.k * 2
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            vector_docs = []

        # BM25 search via external retriever
        try:
            # We assume the bm25_retriever has a method to get docs with scores
            # If standard retriever, we might not get scores, so we'd need to assume a default or adapt.
            # But our ESBM25Retriever has get_documents_with_scores.
            if hasattr(self.bm25_retriever, "get_documents_with_scores"):
                bm25_docs_with_scores = self.bm25_retriever.get_documents_with_scores(
                    query
                )
            else:
                # Fallback if standard retriever (no scores avail, assume 1.0 or rank-based)
                # This is a bit weak for fusion but a fallback.
                docs = self.bm25_retriever.get_relevant_documents(query)
                bm25_docs_with_scores = [(d, 1.0 / (i + 1)) for i, d in enumerate(docs)]

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            bm25_docs_with_scores = []

        # Normalize scores
        vector_scores_norm = self._normalize_scores([score for _, score in vector_docs])
        bm25_scores_norm = self._normalize_scores(
            [score for _, score in bm25_docs_with_scores]
        )

        # Combine results with weighted fusion
        doc_scores = {}

        # Add vector search results
        for i, (doc, score) in enumerate(vector_docs):
            doc_id = self._get_doc_id(doc)
            doc_scores[doc_id] = {
                "doc": doc,
                "score": vector_scores_norm[i] * self.vector_weight,
            }

        # Add BM25 results
        for i, (doc, score) in enumerate(bm25_docs_with_scores):
            doc_id = self._get_doc_id(doc)
            if doc_id in doc_scores:
                doc_scores[doc_id]["score"] += bm25_scores_norm[i] * self.bm25_weight
            else:
                doc_scores[doc_id] = {
                    "doc": doc,
                    "score": bm25_scores_norm[i] * self.bm25_weight,
                }

        # Sort by combined score and return top k
        sorted_docs = sorted(
            doc_scores.values(), key=lambda x: x["score"], reverse=True
        )
        results = [item["doc"] for item in sorted_docs[: self.k]]

        # Trigger feedback callback if feedback_service is available
        if self.feedback_service and results:
            self._trigger_recall_feedback(results)

        return results

    def _trigger_recall_feedback(self, docs: List[Document]):
        """Trigger feedback for recalled documents"""
        try:
            chunk_ids = []
            for doc in docs:
                chunk_id = doc.metadata.get("chunk_id")
                if chunk_id:
                    chunk_ids.append(chunk_id)

            if chunk_ids:
                self.feedback_service.on_recall_batch(chunk_ids)
                logger.debug(f"Triggered recall feedback for {len(chunk_ids)} chunks")
        except Exception as e:
            logger.error(f"Failed to trigger recall feedback: {e}")

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def _get_doc_id(self, doc: Document) -> str:
        """Get unique identifier for document"""
        # Use content hash as ID.
        # Ideally this should match between Vector Store and ES.
        # If we use the same content splitting, it should be fine.
        return str(hash(doc.page_content))
