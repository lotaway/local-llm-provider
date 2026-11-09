import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from constants import (
    MONGO_DB_NAME,
    MONGO_URI,
    TTS_MONGO_COLLECTION_QUALITY,
    TTS_QUALITY_RETENTION_DAYS,
)

logger = logging.getLogger(__name__)

ALLOWED_FIELDS = (
    "status",
    "duration_seconds",
    "duration_anomaly",
    "duplicate",
    "evaluated",
    "wer",
    "dnsmos",
    "speaker_similarity",
    "first_chunk_ms",
    "reason",
)


class MongoTTSQualityRecordRepository:
    def __init__(
        self,
        collection: Optional[Any] = None,
        retention_days: int = TTS_QUALITY_RETENTION_DAYS,
        clock: Optional[Callable[[], datetime]] = None,
        client_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        if retention_days < 1:
            raise ValueError(f"retention_days must be at least 1, got {retention_days}")
        self._collection = collection
        self._retention_days = retention_days
        self._clock = clock or self._utc_now
        self._client_factory = client_factory

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def save(self, record: Dict[str, Any]) -> None:
        payload = {
            key: record[key]
            for key in ALLOWED_FIELDS
            if key in record and record[key] is not None
        }
        payload["recorded_at"] = self._clock()
        self._get_collection().insert_one(payload)

    def _get_collection(self) -> Any:
        if self._collection is None:
            if self._client_factory is None:
                from pymongo import MongoClient

                self._client_factory = MongoClient
            database = self._client_factory(MONGO_URI)[MONGO_DB_NAME]
            self._ensure_timeseries(database)
            self._collection = database[TTS_MONGO_COLLECTION_QUALITY]
        return self._collection

    def _ensure_timeseries(self, database: Any) -> None:
        options = {
            "timeseries": {"timeField": "recorded_at", "granularity": "hours"},
            "expireAfterSeconds": self._retention_days * 86400,
        }
        try:
            database.create_collection(TTS_MONGO_COLLECTION_QUALITY, **options)
        except Exception as error:
            logger.debug(
                "tts_quality_collection_create_skipped",
                extra={"error_type": type(error).__name__},
            )
