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
    """
    Elasticsearch-based BM25 Retriever.
    Indexes documents into Elasticsearch and retrieves them using BM25 scoring.
    """

    index_name: str = ""
    k: int = 5

    # Allow arbitrary types for es_client
    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        index_name: str = ES_INDEX_NAME,
        host: str = ES_HOST,
        port: int = ES_PORT1,
        api_key: Optional[str] = os.getenv("ES_API_KEY"),
        k: int = 5,
    ):
        """
        Initialize the Elasticsearch client.

        Args:
            index_name: Name of the Elasticsearch index.
            host: Elasticsearch host.
            port: Elasticsearch port.
            api_key: Elasticsearch service account token.
            k: Number of documents to retrieve.
        """
        object.__setattr__(self, "index_name", index_name)
        object.__setattr__(self, "k", k)

        try:
            es_client = (
                Elasticsearch(f"http://{host}:{port}", api_key=api_key)
                if api_key
                else Elasticsearch(f"http://{host}:{port}")
            )
            object.__setattr__(self, "es_client", es_client)
            info = es_client.info()
            logger.info(
                f"Connected to Elasticsearch {info['version']['number']} at {host}:{port}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch at {host}:{port}: {e}")
            raise

        self._ensure_index_exists()

    def _ensure_index_exists(self):
        """Create the index with BM25 settings if it doesn't exist."""
        if not self.es_client.indices.exists(index=self.index_name):
            settings = {
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard"  # Simple standard analyzer for now
                        }
                    }
                },
                "similarity": {"default": {"type": "BM25"}},
            }
            mappings = {
                "properties": {
                    "content": {"type": "text", "similarity": "default"},
                    "metadata": {
                        "type": "object",
                        "enabled": True,
                    },  # Store metadata as object
                }
            }
            try:
                self.es_client.indices.create(
                    index=self.index_name, settings=settings, mappings=mappings
                )
                logger.info(f"Created Elasticsearch index '{self.index_name}'")
            except Exception as e:
                logger.error(f"Failed to create index '{self.index_name}': {e}")

    def index_documents(self, documents: List[Document], batch_size: int = 500):
        """
        Index a list of documents into Elasticsearch.

        Args:
            documents: List of LangChain Documents to index.
            batch_size: Number of documents to index in a single batch.
        """
        actions = []
        for doc in documents:
            # Use a hash of content as ID to prevent duplicates if needed,
            # or just simple metadata-based ID if available.
            # Here we let ES generate ID or use source/index combination if we want strict dedup.
            # For simplicity, we just insert.

            action = {
                "_index": self.index_name,
                "_source": {"content": doc.page_content, "metadata": doc.metadata},
            }
            actions.append(action)

            if len(actions) >= batch_size:
                self._bulk_index(actions)
                actions = []

        if actions:
            self._bulk_index(actions)

    def _bulk_index(self, actions: List[Dict]):
        try:
            success, failed = helpers.bulk(self.es_client, actions, stats_only=True)
            logger.info(f"Indexed {success} documents to ES. Failed: {failed}")
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Retrieve documents using BM25 search from Elasticsearch.

        Args:
            query: The search query.
            run_manager: Run manager for callbacks.

        Returns:
            List of documents matching the query.
        """
        search_query = {"query": {"match": {"content": query}}, "size": self.k}

        try:
            response = self.es_client.search(index=self.index_name, body=search_query)
            hits = response["hits"]["hits"]

            documents = []
            for hit in hits:
                source = hit["_source"]
                content = source.get("content", "")
                metadata = source.get("metadata", {})
                score = hit["_score"]

                # We can store the score in metadata if needed for later fusion
                metadata["_es_score"] = score

                documents.append(Document(page_content=content, metadata=metadata))

            return documents

        except Exception as e:
            logger.error(f"Error searching ES: {e}")
            return []

    def get_documents_with_scores(self, query: str) -> List[tuple[Document, float]]:
        """
        Get documents with their raw BM25 scores.
        Useful for hybrid search fusion.
        """
        search_query = {"query": {"match": {"content": query}}, "size": self.k}

        try:
            # elasticsearch-py 8.x/9.x: use direct parameters instead of body
            response = self.es_client.search(
                index=self.index_name,
                query=search_query["query"],
                size=search_query["size"],
            )
            hits = response["hits"]["hits"]

            results = []
            for hit in hits:
                source = hit["_source"]
                content = source.get("content", "")
                metadata = source.get("metadata", {})
                score = hit["_score"]

                results.append(
                    (Document(page_content=content, metadata=metadata), score)
                )

            return results
        except Exception as e:
            logger.error(f"Error searching ES with scores: {e}")
            return []
