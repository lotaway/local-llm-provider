import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TTSPostProcessor:
    def __init__(self, seed: Optional[int] = None):
        self._seed = seed

    def apply(self, audio: bytes, emotion_params: Optional[dict] = None) -> bytes:
        samples = np.frombuffer(audio, dtype=np.float32)
        if samples.size == 0:
            return audio
        breathing = 0.5 + 0.25 * float((emotion_params or {}).get("energy", 0.5))
        envelope = self._breath_envelope(samples.size, breathing)
        processed = samples * envelope
        jittered = self._jitter(processed)
        return jittered.astype(np.float32).tobytes()


    def _breath_envelope(self, size: int, breathing: float) -> np.ndarray:
        envelope = np.ones(size, dtype=np.float32)
        attack = max(1, int(size * 0.05 * breathing))
        release = max(1, int(size * 0.08 * breathing))
        if attack:
            envelope[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
        if release and release < size:
            envelope[size - release :] = np.linspace(1.0, 0.0, release, dtype=np.float32)
        return envelope

    def _jitter(self, samples: np.ndarray) -> np.ndarray:
        if self._seed is None:
            return samples
        generator = np.random.default_rng(self._seed)
        return samples * (1.0 + generator.uniform(-0.02, 0.02, size=samples.size).astype(np.float32))
