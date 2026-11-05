import logging
import secrets
from typing import Any, Optional

from model_providers.tts.audio_processing import (
    breath_envelope,
    process_audio_bytes,
    process_samples,
)

logger = logging.getLogger(__name__)

AudioSamples = Any


class TTSPostProcessor:
    def __init__(self, seed: Optional[int] = None):
        self._seed = seed

    def apply(self, audio: bytes, emotion_params: Optional[dict] = None) -> bytes:
        return process_audio_bytes(audio, emotion_params, self._resolved_seed())

    def process(self, samples: AudioSamples, emotion_params: Optional[dict] = None) -> AudioSamples:
        return process_samples(samples, emotion_params, self._resolved_seed())

    def _breath_envelope(self, size: int, breathing: float) -> AudioSamples:
        return breath_envelope(size, breathing)

    def _resolved_seed(self) -> int:
        return self._seed if self._seed is not None else secrets.randbits(32)

    def _jitter(self, samples: AudioSamples) -> AudioSamples:
        from model_providers.tts.audio_processing import apply_jitter

        return apply_jitter(samples, self._resolved_seed())
