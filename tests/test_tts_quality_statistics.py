from datetime import datetime, timezone

import threading
import time

from services.tts_quality_statistics import TTSQualityStatistics


class _FakeClock:
    def __init__(self, day: int):
        self.day = day

    def __call__(self):
        return datetime(2026, 11, self.day, 23, 59, tzinfo=timezone.utc)


def test_trend_statistics_aggregates_failures_and_percentiles():
    statistics = TTSQualityStatistics(window=20)
    for index in range(20):
        statistics.add({"status": "reject" if index % 4 == 3 else "pass", "wer": 0.1 * index, "dnsmos": {"overall": 3.9}, "speaker_similarity": 0.9})

    wer_stats = statistics.wer_statistics()
    assert wer_stats["count"] == 20.0
    assert wer_stats["mean"] > 0.0
    assert wer_stats["p90"] >= wer_stats["mean"]
    assert round(wer_stats["p95"], 6) == 1.8

    dnsmos_stats = statistics.dnsmos_statistics()
    assert dnsmos_stats["mean"] > 0.0

    assert statistics.failure_rate() == 5 / 20


def test_trend_statistics_rejects_invalid_window():
    from pytest import raises

    with raises(ValueError):
        TTSQualityStatistics(window=0)


def test_daily_statistics_separates_days_and_reports_anomaly_rate():
    clock = _FakeClock(7)
    statistics = TTSQualityStatistics(window=20, clock=clock)
    statistics.add({"status": "pass", "wer": 0.02, "dnsmos": {"overall": 4.2}, "speaker_similarity": 0.9})
    statistics.add({"status": "reject", "wer": 0.30, "dnsmos": {"overall": 2.1}, "duration_anomaly": True})
    clock.day = 8
    statistics.add({"status": "pass", "wer": 0.10, "dnsmos": {"overall": 3.8}, "speaker_similarity": 0.8})
    first = statistics.daily_statistics("2026-11-07")
    assert first["day"] == "2026-11-07"
    assert first["count"] == 2.0
    assert round(first["wer"]["mean"], 6) == 0.16
    assert round(first["dnsmos"]["mean"], 6) == 3.15
    assert first["speaker_similarity"]["count"] == 1.0
    assert first["failure_rate"] == 0.5
    assert first["anomaly_rate"] == 0.5
    second = statistics.daily_statistics("2026-11-08")
    assert second["count"] == 1.0
    assert second["wer"]["mean"] == 0.10
    assert second["anomaly_rate"] == 0.0


def test_daily_breakdown_returns_recent_days_in_order():
    clock = _FakeClock(6)
    statistics = TTSQualityStatistics(window=20, clock=clock)
    for day in (6, 7, 8):
        clock.day = day
        statistics.add({"status": "pass", "wer": float(day) / 100})

    breakdown = statistics.daily_breakdown(limit=2)
    assert [item["day"] for item in breakdown] == ["2026-11-07", "2026-11-08"]
    assert breakdown[-1]["wer"]["mean"] == 0.08
    assert statistics.daily_breakdown() == statistics.daily_breakdown(limit=7)
    assert statistics.daily_breakdown(limit=0) == []


def test_daily_statistics_defaults_to_current_day_and_handles_empty():
    clock = _FakeClock(9)
    statistics = TTSQualityStatistics(window=20, clock=clock)
    empty = statistics.daily_statistics()
    assert empty["day"] == "2026-11-09"
    assert empty["count"] == 0.0
    assert empty["anomaly_rate"] == 0.0
    assert empty["wer"]["p95"] == 0.0
    assert statistics.daily_breakdown() == []


def test_skipped_records_are_not_anomalies_but_evaluation_failures_are():
    statistics = TTSQualityStatistics(window=20, clock=_FakeClock(7))
    statistics.add({"status": "pass"})
    statistics.add({"status": "skipped", "reason": "evaluator_unavailable"})
    statistics.add({"status": "evaluation_failed"})
    statistics.add({"status": "reject", "duplicate": True})

    summary = statistics.daily_statistics("2026-11-07")
    assert summary["count"] == 4.0
    assert summary["failure_rate"] == 0.25
    assert summary["anomaly_rate"] == 0.5


def test_first_chunk_latency_is_summarized_without_missing_records():
    statistics = TTSQualityStatistics(window=20, clock=_FakeClock(7))
    statistics.add({"status": "pass", "first_chunk_ms": 120.0})
    statistics.add({"status": "pass", "first_chunk_ms": 180.0})
    statistics.add({"status": "pass"})
    summary = statistics.first_chunk_statistics()
    assert summary["count"] == 2.0
    assert summary["mean"] == 150.0
    assert statistics.daily_statistics()["first_chunk"]["p95"] == 180.0


def test_rolling_mean_ignores_records_without_the_metric():
    statistics = TTSQualityStatistics(window=10, clock=_FakeClock(7))
    statistics.add({"wer": 0.2})
    statistics.add({"status": "pass"})

    assert statistics.rolling_mean("wer") == 0.2


def test_reads_survive_concurrent_writes_without_mutation_errors():
    statistics = TTSQualityStatistics(window=50)
    stop = threading.Event()
    writer = threading.Thread(target=_write_until_stopped, args=(statistics, stop), daemon=True)
    writer.start()
    reads = 0
    try:
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            statistics.daily_breakdown()
            statistics.daily_statistics()
            statistics.wer_statistics()
            reads += 1
    finally:
        stop.set()
        writer.join(1.0)

    assert reads > 0


def _write_until_stopped(statistics, stop):
    while not stop.is_set():
        statistics.add({"status": "pass", "wer": 0.1, "dnsmos": {"overall": 4.0}})
