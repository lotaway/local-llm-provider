from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from interfaces.memory_types import ISensoryCache


class MongoSensoryCache(ISensoryCache):
    def __init__(self, database, collection_name: str, default_ttl_seconds: int):
        if default_ttl_seconds <= 0:
            raise ValueError("sensory cache ttl must be positive")
        self._collection = database[collection_name]
        self._default_ttl_seconds = default_ttl_seconds
        self._misses = 0
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._collection.create_index("expires_at", expireAfterSeconds=0)
        self._started = True

    def stop(self) -> None:
        self._started = False

    def remember(self, entry_id: str, payload: Dict[str, Any], ttl_seconds: Optional[int] = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl_seconds
        if ttl <= 0:
            raise ValueError("sensory cache ttl must be positive")
        now = self._now()
        self._collection.update_one(
            {"_id": entry_id},
            {
                "$set": {
                    "payload": payload,
                    "created_at": now,
                    "expires_at": now + timedelta(seconds=ttl),
                },
                "$setOnInsert": {"hits": 0},
            },
            upsert=True,
        )

    def recall(self, entry_id: str) -> Optional[Dict[str, Any]]:
        document = self._collection.find_one({"_id": entry_id})
        if document is None or self._is_expired(document):
            self._misses += 1
            self._collection.delete_one({"_id": entry_id})
            return None
        self._collection.update_one(
            {"_id": entry_id},
            {"$inc": {"hits": 1}, "$set": {"last_accessed_at": self._now()}},
        )
        return document.get("payload")

    def forget(self, entry_id: str) -> None:
        self._collection.delete_one({"_id": entry_id})

    def stats(self) -> Dict[str, Any]:
        return {
            "entries": self._collection.count_documents({}),
            "hits": self._total_hits(),
            "misses": self._misses,
            "ttl_seconds": self._default_ttl_seconds,
        }

    def _is_expired(self, document: Dict[str, Any]) -> bool:
        expires_at = document.get("expires_at")
        return expires_at is not None and expires_at <= self._now()

    def _total_hits(self) -> int:
        pipeline = [{"$group": {"_id": None, "total": {"$sum": "$hits"}}}]
        result = list(self._collection.aggregate(pipeline))
        return int(result[0]["total"]) if result else 0

    def _now(self) -> datetime:
        return datetime.now(timezone.utc).replace(tzinfo=None)
