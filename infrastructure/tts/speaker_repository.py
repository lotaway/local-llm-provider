import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from constants import MONGO_DB_NAME, MONGO_URI, TTS_MONGO_COLLECTION_SPEAKERS
from interfaces.tts_types import ISpeakerRepository

logger = logging.getLogger(__name__)


class MongoSpeakerRepository(ISpeakerRepository):
    def __init__(self, collection: Any = None):
        self._collection = collection

    def _get_collection(self):
        if self._collection is None:
            from pymongo import MongoClient

            client = MongoClient(MONGO_URI)
            self._collection = client[MONGO_DB_NAME][TTS_MONGO_COLLECTION_SPEAKERS]
        return self._collection

    def save(self, speaker_id: str, record: Dict[str, Any]) -> None:
        payload = dict(record)
        payload["_id"] = speaker_id
        payload["speaker_id"] = speaker_id
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._get_collection().update_one({"_id": speaker_id}, {"$set": payload}, upsert=True)

    def get(self, speaker_id: str) -> Optional[Dict[str, Any]]:
        return self._get_collection().find_one({"_id": speaker_id})

    def list(self) -> List[Dict[str, Any]]:
        return list(self._get_collection().find({}, {"embedding": 0}))

    def delete(self, speaker_id: str) -> bool:
        result = self._get_collection().delete_one({"_id": speaker_id})
        return bool(result.deleted_count)
