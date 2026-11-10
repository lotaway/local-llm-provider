import logging
import threading
from collections import deque
from typing import Deque

from constants import (
    TTS_AUDIO_HISTORY_LIMIT,
    TTS_AUDIO_MAX_DURATION_SECONDS,
    TTS_AUDIO_SAMPLE_RATE,
)
from model_providers.tts.audio_quality import (
    digest,
    duration_seconds,
    is_duration_anomaly,
    is_duplicate,
)

logger = logging.getLogger(__name__)


class TTSAudioQualityService:
    def __init__(
        self,
        sample_rate: int = TTS_AUDIO_SAMPLE_RATE,
        max_duration_seconds: float = TTS_AUDIO_MAX_DURATION_SECONDS,
        history_limit: int = TTS_AUDIO_HISTORY_LIMIT,
    ):
        if history_limit < 1:
            raise ValueError(f"history_limit must be at least 1, got {history_limit}")
        self._sample_rate = sample_rate
        self._max_duration_seconds = max_duration_seconds
        self._history: Deque[str] = deque(maxlen=history_limit)
        self._lock = threading.Lock()

    def duration_seconds(self, audio: bytes) -> float:
        return duration_seconds(audio, self._sample_rate)

    def is_duration_anomaly(self, audio: bytes) -> bool:
        return is_duration_anomaly(audio, self._sample_rate, self._max_duration_seconds)

    def has_duplicate(self, audio: bytes) -> bool:
        fingerprint = digest(audio)
        with self._lock:
            duplicate = is_duplicate(fingerprint, self._history)
            if not duplicate:
                self._history.append(fingerprint)
        if duplicate:
            logger.warning("duplicate tts audio detected", extra={"digest": fingerprint[:16]})
        return duplicate
