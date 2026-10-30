import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from constants import MONGO_DB_NAME, MONGO_URI, TTS_MONGO_COLLECTION_PROFILES

logger = logging.getLogger(__name__)


class TTSPreferenceStore:
    def __init__(self, collection: Any = None, database: str = MONGO_DB_NAME):
        self._collection = collection
        self._database = database

    def _get_collection(self):
        if self._collection is None:
            from pymongo import MongoClient

            self._collection = MongoClient(MONGO_URI)[self._database][
                TTS_MONGO_COLLECTION_PROFILES
            ]
        return self._collection

    def save(self, user_id: str, preferences: Dict[str, Any]) -> None:
        self._get_collection().update_one(
            {"_id": user_id},
            {
                "$set": {
                    "user_id": user_id,
                    "preferences": preferences,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            },
            upsert=True,
        )

    def get(self, user_id: str) -> Optional[Dict[str, Any]]:
        return self._get_collection().find_one({"_id": user_id})

    def delete(self, user_id: str) -> bool:
        result = self._get_collection().delete_one({"_id": user_id})
        return bool(result.deleted_count)
