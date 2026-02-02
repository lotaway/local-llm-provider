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

logger = logging.getLogger(__name__)


class MongoDBRepository:
    def __init__(self):
        mongo_uri = MONGO_URI
        db_name = MONGO_DB_NAME
        mongo_user = MONGO_USER
        mongo_password = MONGO_PASSWORD

        if mongo_user and mongo_password:
            if "@" not in mongo_uri:
                uri_parts = mongo_uri.replace("mongodb://", "").split("/")
                host_port = uri_parts[0]
                rest = "/" + "/".join(uri_parts[1:]) if len(uri_parts) > 1 else ""
                mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@{host_port}{rest}"

        self.client = MongoClient(mongo_uri)
        self.db: Database = self.client[db_name]

        self.documents: Collection = self.db["documents"]
        self.chunks: Collection = self.db["chunks"]

        self._create_indexes()

        logger.info(f"MongoDB connected to {mongo_uri}/{db_name}")

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
        self,
        doc_id: str,
        content: str,
        title: str,
        source: str,
        author: Optional[str] = None,
        summary: Optional[str] = None,
        path: Optional[str] = None,
        filename: Optional[str] = None,
        format: str = "markdown",
        checksum: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = datetime.utcnow().isoformat() + "Z"

        doc = {
            "_id": doc_id,
            "title": title,
            "source": source or "",
            "author": author or "",
            "summary": summary or "",
            "content": content,
            "created_at": now,
            "updated_at": now,
            "path": path or "",
            "filename": filename or "",
            "format": format,
            "checksum": checksum or "",
            "status": {"chunked": False, "embedded": False, "graphed": False},
            "metadata": metadata or {},
        }

        self.documents.update_one({"_id": doc_id}, {"$set": doc}, upsert=True)
        logger.info(f"Document {doc_id} saved to MongoDB with title: {title}")
        return doc

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self.documents.find_one({"_id": doc_id})

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
        chunk = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "index": index,
            "offset_start": offset_start,
            "offset_end": offset_end,
            "text": text,
            "embedding_status": "pending",
            "metadata": metadata or {},
            "importance_score": 0.5,
            "last_reviewed": None,
            "decay_rate": 0.01,
            "review_count": 0,
            "memory_type": "episodic",
            "created_at": now,
            "updated_at": now,
        }
        self.chunks.update_one({"chunk_id": chunk_id}, {"$set": chunk}, upsert=True)
        return chunk

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

    def close(self):
        self.client.close()
        logger.info("MongoDB connection closed")
