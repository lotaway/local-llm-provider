import logging
from typing import Optional

import numpy as np

from model_providers.tts.audio_processing import (
    breath_envelope,
    process_audio_bytes,
    process_samples,
)

logger = logging.getLogger(__name__)


class TTSPostProcessor:
    def __init__(self, seed: Optional[int] = None):
        self._seed = seed

    def apply(self, audio: bytes, emotion_params: Optional[dict] = None) -> bytes:
        return process_audio_bytes(audio, emotion_params, self._seed)

    def process(self, samples: np.ndarray, emotion_params: Optional[dict] = None) -> np.ndarray:
        return process_samples(samples, emotion_params, self._seed)

    def _breath_envelope(self, size: int, breathing: float) -> np.ndarray:
        return breath_envelope(size, breathing)

    def _jitter(self, samples: np.ndarray) -> np.ndarray:
        from model_providers.tts.audio_processing import apply_jitter

        return apply_jitter(samples, self._seed)
