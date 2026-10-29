import logging
from datetime import datetime
from typing import Any, Dict, Optional

from constants import MONGO_URI, TTS_MONGO_COLLECTION_PROFILES

logger = logging.getLogger(__name__)


class TTSPreferenceStore:
    def __init__(self, collection: Any = None):
        self._collection = collection or _profiles_collection()

    def save(self, user_id: str, preferences: Dict[str, Any]) -> None:
        self._collection.update_one(
            {"_id": user_id},
            {
                "$set": {
                    "user_id": user_id,
                    "preferences": preferences,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
            upsert=True,
        )

    def get(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self._collection.find_one({"_id": user_id})

    def delete(self, user_id: str) -> bool:
        result = self._collection.delete_one({"_id": user_id})
        return result.deleted_count > 0


def _profiles_collection() -> Any:
    from pymongo import MongoClient

    client = MongoClient(MONGO_URI)
    return client["rag_system"][TTS_MONGO_COLLECTION_PROFILES]
