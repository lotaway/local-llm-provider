import numpy as np

from services.tts_post_processing import TTSPostProcessor


def test_post_processor_applies_breath_envelope():
    processor = TTSPostProcessor(seed=7)
    samples = np.ones(100, dtype=np.float32)

    processed = processor._breath_envelope(100, 0.5)

    assert processed[0] == 0.0
    assert processed[-1] == 0.0
    assert np.all(processed[1:-1] > 0.0)


def test_post_processor_keeps_silence_silent():
    processor = TTSPostProcessor(seed=7)

    assert processor.apply(b"\x00\x00\x00\x00", {"energy": 0.9}) == b"\x00\x00\x00\x00"


def test_post_processor_applies_deterministic_jitter():
    processor = TTSPostProcessor(seed=7)
    samples = np.full(10, 0.5, dtype=np.float32)

    first = processor._jitter(samples)
    second = TTSPostProcessor(seed=7)._jitter(samples)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, samples)
