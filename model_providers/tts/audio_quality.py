import hashlib
from typing import List

import numpy as np

FLOAT32_BYTES = 4


def duration_seconds(audio: bytes, sample_rate: int) -> float:
    return float(len(audio) / (sample_rate * FLOAT32_BYTES))


def is_duration_anomaly(audio: bytes, sample_rate: int, max_duration_seconds: float) -> bool:
    duration = duration_seconds(audio, sample_rate)
    return duration <= 0.0 or duration > max_duration_seconds


def digest(audio: bytes) -> str:
    return hashlib.sha256(audio).hexdigest()


def is_duplicate(audio: bytes, history: List[bytes]) -> bool:
    current = digest(audio)
    return any(digest(item) == current for item in history)


def pcm_from_wav(wav_bytes: bytes) -> bytes:
    """Extract float32 PCM from a WAV container when possible."""
    import io

    import soundfile as sf

    samples, _ = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    return np.asarray(samples, dtype=np.float32).tobytes()
