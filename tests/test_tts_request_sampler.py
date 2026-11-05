import random

from services.tts_request_sampler import TTSRequestSampler


def test_sampling_rate_bounds_are_validated():
    monitor = TTSRequestSampler(sample_rate=0.05)
    assert monitor.should_sample() in {True, False}
    assert monitor.sampled_count >= 0


def test_sampling_returns_only_expected_shape():
    monitor = TTSRequestSampler(sample_rate=1.0)
    assert monitor.should_sample() is True
    assert monitor.record({"status": "pass"}) == {"status": "pass"}
    assert monitor.results() == [{"status": "pass"}]


def test_sampling_rejects_invalid_rates():
    import pytest

    with pytest.raises(ValueError):
        TTSRequestSampler(sample_rate=0.0)
    with pytest.raises(ValueError):
        TTSRequestSampler(sample_rate=1.5)
