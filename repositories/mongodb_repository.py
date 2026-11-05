"""
MongoDB Repository for RAG System
Handles documents and chunks as the source of truth
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from constants import MONGO_URI, MONGO_DB_NAME, MONGO_USER, MONGO_PASSWORD
from repositories.document_record import build_document_record

logger = logging.getLogger(__name__)


class MongoDBRepository:
    def __init__(self):
        mongo_uri = self._qualified_uri()
        self.client = MongoClient(mongo_uri)
        self.db: Database = self.client[MONGO_DB_NAME]
        self.documents: Collection = self.db["documents"]
        self.chunks: Collection = self.db["chunks"]
        self._create_indexes()
        logger.info(f"MongoDB connected to {mongo_uri}/{MONGO_DB_NAME}")

    def _qualified_uri(self) -> str:
        if MONGO_USER and MONGO_PASSWORD and "@" not in MONGO_URI:
            return self._authenticated_uri()
        return MONGO_URI

    def _authenticated_uri(self) -> str:
        parts = MONGO_URI.replace("mongodb://", "").split("/")
        host_port = parts[0]
        rest = "/" + "/".join(parts[1:]) if len(parts) > 1 else ""
        return f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{host_port}{rest}"

    def _create_indexes(self):
        self.documents.create_index("path", unique=True)
        self.documents.create_index("checksum")
        self.documents.create_index("created_at")
        self.chunks.create_index("chunk_id", unique=True)
        self.chunks.create_index("doc_id")
        self.chunks.create_index([("doc_id", 1), ("index", 1)])
        self.chunks.create_index("importance_score")
        self.chunks.create_index("last_reviewed")
        self.chunks.create_index("memory_type")
        self.chunks.create_index([("memory_type", 1), ("importance_score", -1)])
        logger.info("MongoDB indexes created")

    def save_document(
        self, doc_id: str, content: str, title: str, source: str, **fields: Any
    ) -> Dict[str, Any]:
        record = dict(fields, title=title, content=content, source=source)
        doc = self._build_document_record(doc_id, record)
        self.documents.update_one({"_id": doc_id}, {"$set": doc}, upsert=True)
        logger.info("document saved", extra={"doc_id": doc_id, "title": title})
        return doc

    def _build_document_record(self, doc_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        existing = self.documents.find_one({"_id": doc_id}, {"status": 1, "created_at": 1, "last_error": 1})
        return build_document_record(doc_id, fields, existing)

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.documents.find_one({"_id": doc_id})

    def get_document_by_path(self, path: str) -> Optional[Dict[str, Any]]:
        return self.documents.find_one({"path": path})

    def delete_document(self, doc_id: str) -> int:
        deleted_chunks = self.chunks.delete_many({"doc_id": doc_id}).deleted_count
        self.documents.delete_one({"_id": doc_id})
        logger.info(
            f"Document {doc_id} removed with {deleted_chunks} chunks",
            extra={"doc_id": doc_id, "chunk_count": deleted_chunks},
        )
        return int(deleted_chunks)

    def mark_document_error(self, doc_id: str, step: str, message: str) -> None:
        self.documents.update_one(
            {"_id": doc_id},
            {
                "$set": {
                    f"status.{step}": False,
                    "last_error": {
                        "step": step,
                        "message": str(message)[:1000],
                        "at": datetime.utcnow().isoformat() + "Z",
                    },
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                }
            },
        )

    def clear_document_error(self, doc_id: str) -> None:
        self.documents.update_one(
            {"_id": doc_id}, {"$unset": {"last_error": ""}}
        )

    def has_incomplete_documents(self) -> bool:
        return (
            self.documents.count_documents(
                {
                    "$or": [
                        {"status.chunked": False},
                        {"status.embedded": False},
                        {"status.graphed": False},
                    ]
                }
            )
            > 0
        )

    def document_exists(self, path: str) -> bool:
        return self.documents.find_one({"path": path}) is not None

    def update_document_status(self, doc_id: str, status_key: str, value: bool):
        self.documents.update_one(
            {"_id": doc_id},
            {
                "$set": {
                    f"status.{status_key}": value,
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                }
            },
        )

    def save_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        index: int,
        text: str,
        offset_start: int = 0,
        offset_end: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = datetime.utcnow()
        spec = {
            "chunk_id": chunk_id, "doc_id": doc_id, "index": index, "text": text,
            "offset_start": offset_start, "offset_end": offset_end, "metadata": metadata,
        }
        chunk = self._build_chunk_record(spec, now)
        self.chunks.update_one({"chunk_id": chunk_id}, {"$set": chunk}, upsert=True)
        return chunk

    def _build_chunk_record(self, spec: Dict[str, Any], now: datetime) -> Dict[str, Any]:
        return {
            "chunk_id": spec["chunk_id"],
            "doc_id": spec["doc_id"],
            "index": spec["index"],
            "offset_start": spec.get("offset_start", 0),
            "offset_end": spec.get("offset_end", 0),
            "text": spec["text"],
            "embedding_status": "pending",
            "metadata": spec.get("metadata") or {},
            "importance_score": 0.5,
            "last_reviewed": None,
            "decay_rate": 0.01,
            "review_count": 0,
            "memory_type": "episodic",
            "created_at": now,
            "updated_at": now,
        }

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        return self.chunks.find_one({"chunk_id": chunk_id})

    def get_chunks_by_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        return list(self.chunks.find({"doc_id": doc_id}).sort("index", 1))

    def update_chunk_embedding_status(self, chunk_id: str, status: str):
        self.chunks.update_one(
            {"chunk_id": chunk_id}, {"$set": {"embedding_status": status}}
        )

    def get_pending_chunks(self, limit: int = 100) -> List[Dict[str, Any]]:
        return list(self.chunks.find({"embedding_status": "pending"}).limit(limit))

    def update_importance_score(self, chunk_id: str, score: float):
        self.chunks.update_one(
            {"chunk_id": chunk_id},
            {"$set": {"importance_score": score, "updated_at": datetime.utcnow()}},
        )

    def increment_importance(self, chunk_id: str, delta: float = 0.1):
        self.chunks.update_one(
            {"chunk_id": chunk_id},
            {
                "$inc": {"importance_score": delta, "review_count": 1},
                "$set": {
                    "last_reviewed": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                },
            },
        )

    def update_last_reviewed(self, chunk_id: str):
        self.chunks.update_one(
            {"chunk_id": chunk_id},
            {
                "$set": {
                    "last_reviewed": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                }
            },
        )

    def get_chunks_by_importance(
        self, memory_type: str = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        query = {}
        if memory_type:
            query["memory_type"] = memory_type
        return list(self.chunks.find(query).sort("importance_score", -1).limit(limit))

    def get_chunks_by_memory_type(self, memory_type: str) -> List[Dict[str, Any]]:
        return list(
            self.chunks.find({"memory_type": memory_type}).sort("created_at", -1)
        )

    def get_low_importance_chunks(
        self, threshold: float = 0.1, memory_type: str = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        query = {"importance_score": {"$lt": threshold}}
        if memory_type:
            query["memory_type"] = memory_type
        return list(self.chunks.find(query).limit(limit))

    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        return self.get_chunk(chunk_id)

    def get_by_type(self, memory_type: str) -> List[Dict[str, Any]]:
        return self.get_chunks_by_memory_type(memory_type)

    def get_by_importance(
        self, memory_type: str = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        return self.get_chunks_by_importance(memory_type=memory_type, limit=limit)

    def get_low_importance(
        self, threshold: float = 0.1, limit: int = 100
    ) -> List[Dict[str, Any]]:
        return self.get_low_importance_chunks(threshold=threshold, limit=limit)

    def update_importance(self, chunk_id: str, score: float) -> None:
        self.update_importance_score(chunk_id, score)

    def mark_reviewed(self, chunk_id: str) -> None:
        self.update_last_reviewed(chunk_id)

    def mark_discarded(self, chunk_id: str) -> None:
        self.chunks.update_one(
            {"chunk_id": chunk_id},
            {
                "$set": {
                    "discarded": True,
                    "discarded_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                }
            },
        )

    def close(self):
        self.client.close()
        logger.info("MongoDB connection closed")
