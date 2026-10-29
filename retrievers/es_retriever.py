import logging
import os
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch, helpers
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from constants import ES_HOST, ES_INDEX_NAME, ES_PORT1

logger = logging.getLogger(__name__)

INDEX_MAPPINGS = {
    "properties": {
        "chunk_id": {"type": "keyword"},
        "doc_id": {"type": "keyword"},
        "filename": {"type": "keyword"},
        "text": {"type": "text"},
        "content": {"type": "text"},
        "metadata": {"type": "object", "enabled": True},
    }
}
HIGHLIGHT_FIELDS = {
    "text": {"fragment_size": 160, "number_of_fragments": 3},
    "content": {"fragment_size": 160, "number_of_fragments": 3},
}


class ESBM25Retriever(BaseRetriever):
    index_name: str = ""
    k: int = 5
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        index_name: str = ES_INDEX_NAME,
        host: str = ES_HOST,
        port: int = ES_PORT1,
        api_key: Optional[str] = os.getenv("ES_API_KEY"),
        k: int = 5,
    ):
        object.__setattr__(self, "index_name", index_name)
        object.__setattr__(self, "k", k)
        try:
            url = f"http://{host}:{port}"
            es = Elasticsearch(url, api_key=api_key) if api_key else Elasticsearch(url)
            object.__setattr__(self, "es_client", es)
            logger.info(f"Connected to Elasticsearch at {host}:{port}")
        except Exception as e:
            logger.error(f"ES connection failed: {e}")
            raise
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        try:
            if not self.es_client.indices.exists(index=self.index_name):
                self.es_client.indices.create(index=self.index_name, mappings=INDEX_MAPPINGS)
                return
            self.es_client.indices.put_mapping(index=self.index_name, properties=INDEX_MAPPINGS["properties"])
        except Exception as e:
            logger.warning(f"ES mapping update failed: {e}")

    def index_documents(self, documents: List[Document], batch_size: int = 500):
        actions = []
        for doc in documents:
            actions.append(self._index_action(doc))
            if len(actions) >= batch_size:
                self._bulk_index(actions)
                actions = []
        if actions:
            self._bulk_index(actions)

    def _index_action(self, doc: Document) -> Dict[str, Any]:
        metadata = doc.metadata or {}
        chunk_id = str(metadata.get("chunk_id", ""))
        source = {
            "chunk_id": chunk_id,
            "doc_id": str(metadata.get("doc_id", "")),
            "filename": str(metadata.get("filename") or metadata.get("source", "")),
            "text": doc.page_content,
            "content": doc.page_content,
            "metadata": metadata,
        }
        action = {"_index": self.index_name, "_source": source}
        if chunk_id:
            action["_id"] = chunk_id
        return action

    def _bulk_index(self, actions: List[Dict]):
        try:
            success, failed = helpers.bulk(self.es_client, actions, stats_only=True)
            if failed:
                raise RuntimeError(f"Elasticsearch bulk indexing failed for {failed} documents")
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}", exc_info=True)
            raise

    def _get_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List[Document]:
        return [document for document, _ in self._search(query)]

    def get_documents_with_scores(self, query: str) -> List[tuple[Document, float]]:
        return self._search(query)

    def delete_by_chunk_ids(self, chunk_ids: List[str]) -> int:
        if not chunk_ids:
            return 0
        try:
            result = self.es_client.delete_by_query(
                index=self.index_name,
                query={"terms": {"chunk_id": list(chunk_ids)}},
                refresh=True,
            )
            return int(result.get("deleted", 0))
        except Exception as e:
            logger.error(f"ES delete failed: {e}", exc_info=True)
            return 0

    def _search(self, query: str) -> List[tuple[Document, float]]:
        search_query = {
            "multi_match": {
                "query": query,
                "fields": ["text^2", "content"],
                "type": "best_fields",
            }
        }
        try:
            response = self.es_client.search(
                index=self.index_name,
                query=search_query,
                size=self.k,
                highlight={"fields": HIGHLIGHT_FIELDS},
            )
            return [self._parse_hit(hit) for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"ES search error: {e}", exc_info=True)
            return []

    def _parse_hit(self, hit: Dict[str, Any]) -> tuple[Document, float]:
        source = hit.get("_source", {})
        metadata = dict(source.get("metadata") or {})
        metadata.update(
            {
                "chunk_id": source.get("chunk_id", ""),
                "doc_id": source.get("doc_id", ""),
                "filename": source.get("filename", ""),
                "_es_score": hit.get("_score", 0.0),
                "_highlight": hit.get("highlight", {}),
            }
        )
        text = source.get("text") or source.get("content", "")
        return Document(page_content=text, metadata=metadata), float(hit.get("_score") or 0.0)
