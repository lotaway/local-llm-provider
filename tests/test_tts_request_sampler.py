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
    with pytest.raises(ValueError):
        TTSRequestSampler(failed_sample_rate=1.5)


def test_stratified_sampling_guarantees_each_stratum_share():
    monitor = TTSRequestSampler(sample_rate=0.25)
    first = [monitor.should_sample({"status": "pass", "stratum": "user-a"}) for _ in range(8)]
    second = [monitor.should_sample({"status": "pass", "stratum": "user-b"}) for _ in range(8)]
    assert sum(first) == 2
    assert sum(second) == 2


def test_failed_requests_are_always_sampled(monkeypatch):
    monkeypatch.setattr("services.tts_request_sampler.random.random", lambda: 0.99)
    monitor = TTSRequestSampler(sample_rate=0.01)
    assert monitor.should_sample({"status": "reject", "stratum": "user-a"}) is True
    assert monitor.should_sample({"status": "pass", "duration_anomaly": True, "stratum": "user-a"}) is True
    assert monitor.should_sample({"status": "pass", "stratum": "user-a"}) is False
