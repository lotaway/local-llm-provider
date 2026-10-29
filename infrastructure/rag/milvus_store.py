import json
import logging
from typing import Any, Dict, List, Optional

from domain.rag.ports import Document, VectorStore
from infrastructure.config.settings import Settings

logger = logging.getLogger(__name__)

try:
    from pymilvus import MilvusClient
except ImportError as error:
    raise ImportError("pymilvus not installed. Install with: pip install pymilvus") from error

OUTPUT_FIELDS = ["content", "metadata", "chunk_id", "doc_id", "filename"]
ID_MAX_LENGTH = 512
INSERT_BATCH_SIZE = 128


class MilvusVectorStore(VectorStore):
    def __init__(self, settings: Settings, embedding_provider=None):
        self._settings = settings
        self._embedding = embedding_provider
        self._client: Optional[MilvusClient] = None
        self._collection_name = settings.rag.db_collection
        self._collection_ready = False
        self._init_client()

    def _init_client(self) -> None:
        uri = f"http://{self._settings.rag.db_host}:{self._settings.rag.db_port}"
        try:
            self._client = MilvusClient(uri=uri)
            logger.info(f"Milvus client connected: {uri}")
        except Exception as error:
            logger.error(f"Failed to connect to Milvus: {error}", exc_info=True)
            self._client = None

    def _require_client(self) -> MilvusClient:
        if self._client is None:
            raise RuntimeError("Milvus client not initialized")
        return self._client

    def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        client = self._require_client()
        if client.has_collection(self._collection_name):
            self._collection_ready = True
            return
        if self._embedding is None:
            raise RuntimeError("embedding provider is required to create Milvus collection")
        client.create_collection(
            collection_name=self._collection_name,
            dimension=int(self._embedding.dimension),
            primary_field_name="id",
            id_type="string",
            max_length=ID_MAX_LENGTH,
            vector_field_name="vector",
            metric_type="COSINE",
            auto_id=False,
            enable_dynamic_field=True,
        )
        self._collection_ready = True
        logger.info(f"Milvus collection created: {self._collection_name}")

    def _embed_documents(self, documents: List[Document]) -> List[List[float]]:
        if self._embedding is None:
            raise RuntimeError("embedding provider is required to add documents")
        texts = [doc.content for doc in documents]
        return [list(vector) for vector in self._embedding.embed_documents(texts)]

    async def add_documents(self, documents: List[Document]) -> List[str]:
        if not documents:
            return []
        self._ensure_collection()
        client = self._require_client()
        vectors = self._embed_documents(documents)
        rows = [self._row(document, vector) for document, vector in zip(documents, vectors)]
        for start in range(0, len(rows), INSERT_BATCH_SIZE):
            client.insert(self._collection_name, rows[start : start + INSERT_BATCH_SIZE])
        return [document.id for document in documents]

    def _row(self, document: Document, vector: List[float]) -> Dict[str, Any]:
        metadata = document.metadata or {}
        return {
            "id": str(document.id),
            "vector": vector,
            "content": document.content,
            "chunk_id": str(metadata.get("chunk_id", "")),
            "doc_id": str(metadata.get("doc_id", "")),
            "filename": str(metadata.get("filename") or metadata.get("source", "")),
            "metadata": json.dumps(metadata, ensure_ascii=False, default=str),
        }

    async def delete_documents(self, ids: List[str]) -> int:
        if not ids or self._client is None:
            return 0
        if not self._client.has_collection(self._collection_name):
            return 0
        result = self._client.delete(self._collection_name, ids=list(ids))
        return int(result.get("delete_count", len(ids)))

    async def similarity_search(
        self, query_vector: List[float], k: int = 10, filter: Optional[Dict] = None
    ) -> List[Document]:
        if self._client is None or not self._client.has_collection(self._collection_name):
            return []
        try:
            results = self._client.search(
                collection_name=self._collection_name,
                data=[list(query_vector)],
                limit=k,
                filter=self._build_filter(filter),
                output_fields=OUTPUT_FIELDS,
                search_params={"metric_type": "COSINE"},
            )
            return [self._to_document(hit) for hit in results[0]]
        except Exception as error:
            logger.error(f"Milvus search failed: {error}", exc_info=True)
            raise

    async def hybrid_search(
        self, query: str, query_vector: List[float], k: int = 10, filter: Optional[Dict] = None
    ) -> List[Document]:
        return await self.similarity_search(query_vector, k, filter)

    async def health_check(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.list_collections()
            return True
        except Exception as error:
            logger.warning(f"Milvus health check failed: {error}")
            return False

    def _to_document(self, hit: Dict[str, Any]) -> Document:
        entity = hit.get("entity") or {}
        return Document(
            id=str(hit.get("id", entity.get("id", ""))),
            content=entity.get("content", ""),
            metadata=self._decode_metadata(entity.get("metadata")),
            score=float(hit.get("distance", 0.0)),
        )

    def _decode_metadata(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value:
            try:
                decoded = json.loads(value)
                return decoded if isinstance(decoded, dict) else {}
            except json.JSONDecodeError:
                logger.warning("Milvus metadata is not valid JSON", extra={"metadata": value})
        return {}

    def _build_filter(self, filter: Optional[Dict]) -> str:
        if not filter:
            return ""
        if isinstance(filter, str):
            return filter
        clauses = [self._filter_clause(key, value) for key, value in filter.items()]
        return " and ".join(clause for clause in clauses if clause)

    def _filter_clause(self, key: str, value: Any) -> str:
        if isinstance(value, dict):
            return self._operator_clause(key, value)
        if isinstance(value, (list, tuple, set)):
            return f"{key} in {self._literal_list(value)}"
        return f"{key} == {self._literal(value)}"

    def _operator_clause(self, key: str, value: Dict[str, Any]) -> str:
        if "$in" in value:
            return f"{key} in {self._literal_list(value['$in'])}"
        if "$ne" in value:
            return f"{key} != {self._literal(value['$ne'])}"
        if "$eq" in value:
            return f"{key} == {self._literal(value['$eq'])}"
        raise ValueError(f"unsupported Milvus filter operator: {sorted(value)}")

    def _literal_list(self, values: Any) -> str:
        items = values if isinstance(values, (list, tuple, set)) else [values]
        return "[" + ", ".join(self._literal(item) for item in items) + "]"

    def _literal(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
