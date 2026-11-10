import logging
import threading
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

TIME_FIELD = "recorded_at"
RETENTION_COMMAND = "collMod"
DAY_SECONDS = 86400


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
        self._lock = threading.Lock()

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @property
    def _expire_after_seconds(self) -> int:
        return self._retention_days * DAY_SECONDS

    def save(self, record: Dict[str, Any]) -> None:
        payload = {
            key: record[key]
            for key in ALLOWED_FIELDS
            if key in record and record[key] is not None
        }
        payload[TIME_FIELD] = self._clock()
        self._get_collection().insert_one(payload)

    def _get_collection(self) -> Any:
        with self._lock:
            if self._collection is None:
                self._collection = self._open_collection()
            return self._collection

    def _open_collection(self) -> Any:
        database = self._client(MONGO_URI)[MONGO_DB_NAME]
        self._apply_retention(database)
        return database[TTS_MONGO_COLLECTION_QUALITY]

    def _client(self, uri: str) -> Any:
        factory = self._client_factory or _default_client
        return factory(uri)

    def _apply_retention(self, database: Any) -> None:
        options = {
            "timeseries": {"timeField": TIME_FIELD, "granularity": "hours"},
            "expireAfterSeconds": self._expire_after_seconds,
        }
        try:
            database.create_collection(TTS_MONGO_COLLECTION_QUALITY, **options)
        except Exception as error:
            self._update_retention(database, error)

    def _update_retention(self, database: Any, create_error: Exception) -> None:
        try:
            database.command(
                RETENTION_COMMAND,
                TTS_MONGO_COLLECTION_QUALITY,
                expireAfterSeconds=self._expire_after_seconds,
            )
        except Exception as error:
            logger.warning(
                "tts_quality_retention_not_applied",
                extra={
                    "create_error": type(create_error).__name__,
                    "update_error": type(error).__name__,
                },
            )


def _default_client(uri: str) -> Any:
    from pymongo import MongoClient

    return MongoClient(uri)
