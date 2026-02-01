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
    """Repository for managing documents and chunks in MongoDB"""

    def __init__(self):
        mongo_uri = MONGO_URI
        db_name = MONGO_DB_NAME
        mongo_user = MONGO_USER
        mongo_password = MONGO_PASSWORD

        # Build connection URI with authentication if credentials provided
        if mongo_user and mongo_password:
            # Parse the URI and add credentials if not already present
            if "@" not in mongo_uri:
                # Insert user:password@ after mongodb://
                uri_parts = mongo_uri.replace("mongodb://", "").split("/")
                host_port = uri_parts[0]
                rest = "/" + "/".join(uri_parts[1:]) if len(uri_parts) > 1 else ""
                mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@{host_port}{rest}"

        self.client = MongoClient(mongo_uri)
        self.db: Database = self.client[db_name]

        # Collections
        self.documents: Collection = self.db["documents"]
        self.chunks: Collection = self.db["chunks"]

        # Create indexes for better performance
        self._create_indexes()

        logger.info(f"MongoDB connected to {mongo_uri}/{db_name}")

    def _create_indexes(self):
        """Create necessary indexes"""
        # Document indexes
        self.documents.create_index("path", unique=True)
        self.documents.create_index("checksum")
        self.documents.create_index("created_at")

        # Chunk indexes
        self.chunks.create_index("chunk_id", unique=True)
        self.chunks.create_index("doc_id")
        self.chunks.create_index([("doc_id", 1), ("index", 1)])

        logger.info("MongoDB indexes created")

    def save_document(
        self,
        doc_id: str,
        content: str,
        source: str,
        path: str,
        filename: str,
        format: str = "markdown",
        checksum: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Save or update a document in MongoDB"""
        now = datetime.utcnow().isoformat() + "Z"

        doc = {
            "_id": doc_id,
            "source": source,
            "path": path,
            "filename": filename,
            "format": format,
            "checksum": checksum or "",
            "created_at": now,
            "updated_at": now,
            "content": content,
            "status": {"chunked": False, "embedded": False, "graphed": False},
            "metadata": metadata or {},
        }

        result = self.documents.update_one({"_id": doc_id}, {"$set": doc}, upsert=True)

        logger.info(f"Document {doc_id} saved to MongoDB")
        return doc

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID"""
        return self.documents.find_one({"_id": doc_id})

    def document_exists(self, path: str) -> bool:
        """Check if a document exists by path"""
        return self.documents.find_one({"path": path}) is not None

    def update_document_status(self, doc_id: str, status_key: str, value: bool):
        """Update document processing status"""
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
        """Save a chunk to MongoDB"""
        chunk = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "index": index,
            "offset_start": offset_start,
            "offset_end": offset_end,
            "text": text,
            "embedding_status": "pending",
            "metadata": metadata or {},
        }

        result = self.chunks.update_one(
            {"chunk_id": chunk_id}, {"$set": chunk}, upsert=True
        )

        return chunk

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a chunk by ID"""
        return self.chunks.find_one({"chunk_id": chunk_id})

    def get_chunks_by_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document, sorted by index"""
        return list(self.chunks.find({"doc_id": doc_id}).sort("index", 1))

    def update_chunk_embedding_status(self, chunk_id: str, status: str):
        """Update chunk embedding status"""
        self.chunks.update_one(
            {"chunk_id": chunk_id}, {"$set": {"embedding_status": status}}
        )

    def get_pending_chunks(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get chunks that need embedding"""
        return list(self.chunks.find({"embedding_status": "pending"}).limit(limit))

    def close(self):
        """Close MongoDB connection"""
        self.client.close()
        logger.info("MongoDB connection closed")
