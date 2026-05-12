import os
import logging
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch, helpers
from constants import ES_HOST, ES_PORT1, ES_INDEX_NAME
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ESBM25Retriever(BaseRetriever):
    index_name: str = ""
    k: int = 5
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, index_name: str = ES_INDEX_NAME, host: str = ES_HOST, port: int = ES_PORT1, api_key: Optional[str] = os.getenv("ES_API_KEY"), k: int = 5):
        object.__setattr__(self, "index_name", index_name)
        object.__setattr__(self, "k", k)
        try:
            es = Elasticsearch(f"http://{host}:{port}", api_key=api_key) if api_key else Elasticsearch(f"http://{host}:{port}")
            object.__setattr__(self, "es_client", es)
            logger.info(f"Connected to Elasticsearch at {host}:{port}")
        except Exception as e:
            logger.error(f"ES connection failed: {e}")
            raise
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        if not self.es_client.indices.exists(index=self.index_name):
            try:
                self.es_client.indices.create(
                    index=self.index_name,
                    settings={"analysis": {"analyzer": {"default": {"type": "standard"}}}, "similarity": {"default": {"type": "BM25"}}},
                    mappings={"properties": {"content": {"type": "text", "similarity": "default"}, "metadata": {"type": "object", "enabled": True}}}
                )
            except Exception as e:
                logger.error(f"Index creation failed: {e}")

    def index_documents(self, documents: List[Document], batch_size: int = 500):
        actions = []
        for doc in documents:
            actions.append({"_index": self.index_name, "_source": {"content": doc.page_content, "metadata": doc.metadata}})
            if len(actions) >= batch_size:
                self._bulk_index(actions)
                actions = []
        if actions: self._bulk_index(actions)

    def _bulk_index(self, actions: List[Dict]):
        try:
            success, failed = helpers.bulk(self.es_client, actions, stats_only=True)
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")

    def _get_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List[Document]:
        try:
            res = self.es_client.search(index=self.index_name, body={"query": {"match": {"content": query}}, "size": self.k})
            return [Document(page_content=h["_source"].get("content", ""), metadata={**h["_source"].get("metadata", {}), "_es_score": h["_score"]}) for h in res["hits"]["hits"]]
        except Exception as e:
            logger.error(f"ES search error: {e}")
            return []

    def get_documents_with_scores(self, query: str) -> List[tuple[Document, float]]:
        try:
            res = self.es_client.search(index=self.index_name, query={"match": {"content": query}}, size=self.k)
            return [(Document(page_content=h["_source"].get("content", ""), metadata=h["_source"].get("metadata", {})), h["_score"]) for h in res["hits"]["hits"]]
        except Exception as e:
            logger.error(f"ES search with scores error: {e}")
            return []
