import logging
from typing import Dict, List

from constants import TTS_AUDIO_MAX_DURATION_SECONDS, TTS_AUDIO_SAMPLE_RATE
from model_providers.tts.audio_quality import digest, duration_seconds, is_duration_anomaly, is_duplicate

logger = logging.getLogger(__name__)


class TTSAudioQualityService:
    def __init__(self, sample_rate: int = TTS_AUDIO_SAMPLE_RATE, max_duration_seconds: float = TTS_AUDIO_MAX_DURATION_SECONDS):
        self._sample_rate = sample_rate
        self._max_duration_seconds = max_duration_seconds
        self._history: List[bytes] = []

    def duration_seconds(self, audio: bytes) -> float:
        return duration_seconds(audio, self._sample_rate)

    def is_duration_anomaly(self, audio: bytes) -> bool:
        return is_duration_anomaly(audio, self._sample_rate, self._max_duration_seconds)

    def has_duplicate(self, audio: bytes) -> bool:
        duplicate = is_duplicate(audio, self._history)
        if not duplicate:
            self._history.append(bytes(audio))
        else:
            logger.warning("duplicate tts audio detected", extra={"digest": digest(audio)[:16]})
        return duplicate
