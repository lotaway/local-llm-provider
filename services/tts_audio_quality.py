import hashlib
import logging
import time
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TTSAudioQualityService:
    def __init__(self, sample_rate: int = 24000, max_duration_seconds: float = 30.0):
        self._sample_rate = sample_rate
        self._max_duration_seconds = max_duration_seconds
        self._history: List[bytes] = []

    def duration_seconds(self, audio: bytes) -> float:
        samples = np.frombuffer(audio, dtype=np.float32)
        return float(len(samples) / self._sample_rate)

    def is_duration_anomaly(self, audio: bytes) -> bool:
        duration = self.duration_seconds(audio)
        if duration <= 0.0 or duration > self._max_duration_seconds:
            return True
        return False

    def has_duplicate(self, audio: bytes) -> bool:
        digest = hashlib.sha256(audio).hexdigest()
        if digest in {hashlib.sha256(item).hexdigest() for item in self._history}:
            return True
        self._history.append(bytes(audio))
        return False
