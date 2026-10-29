import json
import logging
from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)

MAX_FILTER_CHUNK_IDS = 200


def build_chunk_expr(chunk_ids: List[str]) -> str:
    unique_ids = list(dict.fromkeys(str(item) for item in chunk_ids if item))
    limited = unique_ids[:MAX_FILTER_CHUNK_IDS]
    return f"chunk_id in [{', '.join(json.dumps(item) for item in limited)}]"


def similarity_search_with_chunk_filter(
    vectorstore: Any, query: str, k: int, chunk_ids: List[str]
) -> List[Any]:
    if not chunk_ids:
        return vectorstore.similarity_search_with_score(query, k=k)
    expr = build_chunk_expr(chunk_ids)
    try:
        return vectorstore.similarity_search_with_score(query, k=k, expr=expr)
    except (TypeError, ValueError) as error:
        logger.warning(
            "vector store rejected chunk filter, falling back to unfiltered search",
            extra={"error": str(error), "chunk_count": len(chunk_ids)},
        )
        return vectorstore.similarity_search_with_score(query, k=k)


class GraphFilteredVectorRetriever(BaseRetriever):
    vectorstore: Any
    k: int = 5
    chunk_id_provider: Any = None
    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        chunk_ids = self._chunk_ids(query)
        if not chunk_ids:
            return self.vectorstore.similarity_search(query, k=self.k)
        try:
            documents = self.vectorstore.similarity_search(
                query, k=self.k, expr=build_chunk_expr(chunk_ids)
            )
        except (TypeError, ValueError) as error:
            logger.warning(
                "vector store rejected chunk filter, using unfiltered retrieval",
                extra={"error": str(error), "chunk_count": len(chunk_ids)},
            )
            return self.vectorstore.similarity_search(query, k=self.k)
        return documents or self.vectorstore.similarity_search(query, k=self.k)

    def _chunk_ids(self, query: str) -> List[str]:
        if self.chunk_id_provider is None:
            return []
        try:
            return self.chunk_id_provider(query) or []
        except Exception as error:
            logger.warning(
                "graph chunk lookup failed", extra={"error": str(error)}, exc_info=True
            )
            return []
