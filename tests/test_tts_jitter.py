import numpy as np

from model_providers.tts.audio_processing import apply_jitter, process_samples


def test_jitter_is_no_longer_a_no_op_when_a_seed_is_passed():
    data = np.ones(64, dtype=np.float32)
    assert not np.array_equal(apply_jitter(data, 7), data)


def test_process_samples_applies_jitter_with_seed():
    data = np.ones(64, dtype=np.float32)
    assert not np.array_equal(process_samples(data, {}, 7), process_samples(data, {}))
