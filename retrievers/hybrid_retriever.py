"""Hybrid Retriever - Combines vector and keyword search"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    vectorstore: Any
    bm25_retriever: Any
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    k: int = 5
    feedback_service: Optional[Any] = None
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, vectorstore, bm25_retriever, vector_weight: float = 0.7, bm25_weight: float = 0.3, k: int = 5, feedback_service: Optional[Any] = None, **kwargs):
        super().__init__(vectorstore=vectorstore, bm25_retriever=bm25_retriever, vector_weight=vector_weight, bm25_weight=bm25_weight, k=k, feedback_service=feedback_service, **kwargs)
        self.feedback_service = feedback_service

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None) -> List[Document]:
        try: vector_docs = self.vectorstore.similarity_search_with_score(query, k=self.k * 2)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            vector_docs = []

        try:
            if hasattr(self.bm25_retriever, "get_documents_with_scores"): bm25_docs = self.bm25_retriever.get_documents_with_scores(query)
            else: bm25_docs = [(d, 1.0 / (i + 1)) for i, d in enumerate(self.bm25_retriever.get_relevant_documents(query))]
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            bm25_docs = []

        v_norm = self._normalize_scores([s for _, s in vector_docs])
        b_norm = self._normalize_scores([s for _, s in bm25_docs])
        doc_scores = {}

        for i, (doc, score) in enumerate(vector_docs):
            doc_scores[self._get_doc_id(doc)] = {"doc": doc, "score": v_norm[i] * self.vector_weight}

        for i, (doc, score) in enumerate(bm25_docs):
            did = self._get_doc_id(doc)
            if did in doc_scores: doc_scores[did]["score"] += b_norm[i] * self.bm25_weight
            else: doc_scores[did] = {"doc": doc, "score": b_norm[i] * self.bm25_weight}

        results = [item["doc"] for item in sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)[: self.k]]
        if self.feedback_service and results: self._trigger_recall_feedback(results)
        return results

    def _trigger_recall_feedback(self, docs: List[Document]):
        try:
            chunk_ids = [d.metadata.get("chunk_id") for d in docs if d.metadata.get("chunk_id")]
            if chunk_ids: self.feedback_service.on_recall_batch(chunk_ids)
        except Exception as e: logger.error(f"Failed to trigger recall feedback: {e}")

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        if not scores: return []
        min_s, max_s = min(scores), max(scores)
        return [1.0] * len(scores) if max_s == min_s else [(s - min_s) / (max_s - min_s) for s in scores]

    def _get_doc_id(self, doc: Document) -> str:
        return str(hash(doc.page_content))
