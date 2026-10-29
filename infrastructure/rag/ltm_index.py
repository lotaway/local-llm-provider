import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from interfaces.ltm_index_types import ILtmIndexRepository

KEYWORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}")
MAX_KEYWORDS = 20


class MilvusEsLtmIndex(ILtmIndexRepository):
    def __init__(
        self,
        embedding_provider,
        milvus_client,
        es_client,
        collection_name: str,
        index_name: str,
        ttl_days: int,
    ):
        self._embedding = embedding_provider
        self._milvus = milvus_client
        self._es = es_client
        self._collection_name = collection_name
        self._index_name = index_name
        self._ttl_days = ttl_days

    def save_ltm_vector(self, ltm: Dict[str, Any]) -> str:
        document = self._document(ltm)
        vector = list(self._embedding.embed(self._vector_text(ltm)))
        self._ensure_collection()
        self._milvus.upsert(
            collection_name=self._collection_name,
            data=[{"id": document["id"], "vector": vector, "metadata": json.dumps(document)}],
        )
        return document["id"]

    def index_ltm_keywords(self, ltm: Dict[str, Any]) -> str:
        document = self._document(ltm)
        self._ensure_index()
        self._es.index(index=self._index_name, id=document["id"], document=document)
        return document["id"]

    def _document(self, ltm: Dict[str, Any]) -> Dict[str, Any]:
        now = self._now()
        return {
            "id": self._document_id(ltm),
            "topic": ltm.get("topic", ""),
            "version": int(ltm.get("version", 1)),
            "conclusion": ltm.get("conclusion", ""),
            "conditions": ltm.get("conditions", []),
            "confidence": float(ltm.get("confidence", 0.0)),
            "sources": ltm.get("sources", []),
            "source_chunk_ids": ltm.get("source_chunk_ids", []),
            "keywords": self._keywords(ltm),
            "created_at": ltm.get("created_at") or now,
            "updated_at": ltm.get("updated_at") or now,
            "last_reinforced_at": ltm.get("last_reinforced_at") or now,
            "expires_at": ltm.get("expires_at") or self._expiry(now),
            "version_relations": ltm.get("version_relations", {}),
        }

    def _document_id(self, ltm: Dict[str, Any]) -> str:
        return f"ltm::{ltm.get('topic', '')}::v{int(ltm.get('version', 1))}"

    def _vector_text(self, ltm: Dict[str, Any]) -> str:
        return f"{ltm.get('topic', '')} {ltm.get('conclusion', '')}".strip()

    def _keywords(self, ltm: Dict[str, Any]) -> List[str]:
        provided = ltm.get("keywords")
        if provided:
            return list(dict.fromkeys(provided))[:MAX_KEYWORDS]
        corpus = self._vector_text(ltm)
        tokens = (token.lower() for token in KEYWORD_PATTERN.findall(corpus))
        return list(dict.fromkeys(tokens))[:MAX_KEYWORDS]

    def _expiry(self, now: str) -> str:
        return (datetime.fromisoformat(now) + timedelta(days=self._ttl_days)).isoformat()

    def _ensure_collection(self) -> None:
        if self._milvus.has_collection(self._collection_name):
            return
        self._milvus.create_collection(
            collection_name=self._collection_name,
            dimension=self._embedding.dimension,
            primary_field_name="id",
            vector_field_name="vector",
            id_type="string",
            max_length=512,
            auto_id=False,
            enable_dynamic_field=True,
        )

    def _ensure_index(self) -> None:
        if self._es.indices.exists(index=self._index_name):
            return
        self._es.indices.create(index=self._index_name, mappings=self._mappings())

    def _mappings(self) -> Dict[str, Any]:
        return {
            "properties": {
                "topic": {"type": "keyword"},
                "version": {"type": "integer"},
                "conclusion": {"type": "text"},
                "conditions": {"type": "keyword"},
                "confidence": {"type": "double"},
                "sources": {"type": "keyword"},
                "source_chunk_ids": {"type": "keyword"},
                "keywords": {"type": "keyword"},
                "created_at": {"type": "date"},
                "updated_at": {"type": "date"},
                "last_reinforced_at": {"type": "date"},
                "expires_at": {"type": "date"},
            }
        }

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
