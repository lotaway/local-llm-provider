from services.tts_trend_statistics import TTSTrendStatistics


def test_trend_statistics_aggregates_failures_and_percentiles():
    statistics = TTSTrendStatistics(window=20)
    for index in range(20):
        statistics.add({"status": "reject" if index % 4 == 3 else "pass", "wer": 0.1 * index, "dnsmos": {"overall": 3.9}, "speaker_similarity": 0.9})

    wer_stats = statistics.wer_statistics()
    assert wer_stats["count"] == 20.0
    assert wer_stats["mean"] > 0.0
    assert wer_stats["p90"] >= wer_stats["mean"]

    dnsmos_stats = statistics.dnsmos_statistics()
    assert dnsmos_stats["mean"] > 0.0

    assert statistics.failure_rate() == 5 / 20


def test_trend_statistics_rejects_invalid_window():
    from pytest import raises

    with raises(ValueError):
        TTSTrendStatistics(window=0)
