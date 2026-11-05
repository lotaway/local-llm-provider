import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from interfaces.tts_types import ITTSCollection

logger = logging.getLogger(__name__)


class TTSPreferenceStoreUnconfigured(RuntimeError):
    pass


class TTSPreferenceStore:
    def __init__(self, collection: Optional[ITTSCollection] = None):
        self._collection = collection

    def _get_collection(self) -> ITTSCollection:
        if self._collection is None:
            raise TTSPreferenceStoreUnconfigured("a preference collection must be injected")
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
