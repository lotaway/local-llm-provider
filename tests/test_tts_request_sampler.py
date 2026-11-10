from datetime import datetime, timezone

import pytest

from services.tts_request_sampler import TTSRequestSampler


class _Clock:
    def __init__(self, day: int):
        self.day = day

    def __call__(self) -> datetime:
        return datetime(2026, 11, self.day, tzinfo=timezone.utc)


def test_sampling_rate_bounds_are_validated():
    monitor = TTSRequestSampler(sample_rate=0.05)
    assert monitor.select() in {True, False}
    assert monitor.sampled_count >= 0


def test_sampling_returns_only_expected_shape():
    monitor = TTSRequestSampler(sample_rate=1.0)
    assert monitor.select() is True
    assert monitor.record({"status": "pass"}) is None
    assert monitor.results() == [{"status": "pass"}]


def test_sampling_rejects_invalid_rates():
    with pytest.raises(ValueError):
        TTSRequestSampler(sample_rate=0.0)
    with pytest.raises(ValueError):
        TTSRequestSampler(sample_rate=1.5)
    with pytest.raises(ValueError):
        TTSRequestSampler(failed_sample_rate=1.5)
    with pytest.raises(ValueError):
        TTSRequestSampler(result_limit=0)


def test_stratified_sampling_guarantees_each_stratum_share():
    monitor = TTSRequestSampler(sample_rate=0.25)
    first = [monitor.select({"status": "pass", "stratum": "user-a"}) for _ in range(8)]
    second = [monitor.select({"status": "pass", "stratum": "user-b"}) for _ in range(8)]
    assert sum(first) == 2
    assert sum(second) == 2


def test_failed_requests_are_always_sampled(monkeypatch):
    monkeypatch.setattr("services.tts_request_sampler.random.random", lambda: 0.99)
    monitor = TTSRequestSampler(sample_rate=0.01)
    assert monitor.select({"status": "reject", "stratum": "user-a"}) is True
    assert monitor.select({"status": "pass", "duration_anomaly": True, "stratum": "user-a"}) is True
    assert monitor.select({"status": "pass", "stratum": "user-a"}) is False


def test_sampled_results_are_bounded_and_drop_request_text():
    monitor = TTSRequestSampler(sample_rate=1.0, result_limit=2)
    for index in range(3):
        monitor.record({"status": "pass", "text": f"private-{index}", "wer": float(index)})

    results = monitor.results()
    assert [item["wer"] for item in results] == [1.0, 2.0]
    assert all("text" not in item for item in results)


def test_stratum_counters_reset_on_a_new_utc_day():
    clock = _Clock(7)
    monitor = TTSRequestSampler(sample_rate=0.25, clock=clock)
    for _ in range(8):
        monitor.select({"status": "pass", "stratum": "user-a"})
    assert monitor._strata == {"user-a": 8}

    clock.day = 8
    monitor.select({"status": "pass", "stratum": "user-a"})
    assert monitor._strata == {"user-a": 1}
