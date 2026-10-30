import numpy as np

DEFAULT_BREATHING = 0.5
BREATHING_SCALE = 0.25
ATTACK_RATIO = 0.05
RELEASE_RATIO = 0.08
JITTER_RANGE = 0.02


def apply_breathing(
    samples: np.ndarray,
    emotion_params: dict | None,
    breathing: float = DEFAULT_BREATHING,
) -> np.ndarray:
    if samples.size == 0:
        return samples
    energy = float((emotion_params or {}).get("energy", DEFAULT_BREATHING))
    breathing = breathing + BREATHING_SCALE * energy
    envelope = _breath_envelope(samples.size, breathing)
    return samples * envelope


def breath_envelope(size: int, breathing: float = DEFAULT_BREATHING) -> np.ndarray:
    return _breath_envelope(size, breathing)


def apply_jitter(samples: np.ndarray, seed: int | None) -> np.ndarray:
    if seed is None or samples.size == 0:
        return samples
    generator = np.random.default_rng(seed)
    noise = generator.uniform(-JITTER_RANGE, JITTER_RANGE, size=samples.size)
    return samples * (1.0 + noise.astype(np.float32))


def process_samples(
    samples: np.ndarray,
    emotion_params: dict | None = None,
    seed: int | None = None,
) -> np.ndarray:
    if samples.size == 0:
        return samples
    processed = apply_breathing(samples, emotion_params)
    return apply_jitter(processed, seed)


def process_audio_bytes(
    audio: bytes,
    emotion_params: dict | None = None,
    seed: int | None = None,
) -> bytes:
    samples = np.frombuffer(audio, dtype=np.float32)
    processed = process_samples(samples, emotion_params, seed)
    return processed.astype(np.float32).tobytes()


def _breath_envelope(size: int, breathing: float) -> np.ndarray:
    envelope = np.ones(size, dtype=np.float32)
    attack = max(1, int(size * ATTACK_RATIO * breathing))
    release = max(1, int(size * RELEASE_RATIO * breathing))
    if attack:
        envelope[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    if release and release < size:
        envelope[size - release :] = np.linspace(1.0, 0.0, release, dtype=np.float32)
    return envelope
