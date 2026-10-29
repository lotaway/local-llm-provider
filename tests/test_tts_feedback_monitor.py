from services.tts_feedback_monitor import TTSFeedbackMonitor


def test_feedback_distribution_aggregates_regeneration_counts():
    monitor = TTSFeedbackMonitor()

    monitor.record({"rating": "positive", "complaint": None, "regenerate_count": 0})
    monitor.record({"rating": "negative", "complaint": "quality", "regenerate_count": 2})

    distribution = monitor.distribution()

    assert distribution["total"] == 2
    assert distribution["ratings"] == {"negative": 1, "positive": 1}
    assert distribution["complaints"] == {"quality": 1}
    assert distribution["avg_regenerate_count"] == 1.0
    assert monitor.complaint_rate() == 0.5


def test_feedback_monitor_defaults_for_empty_store():
    monitor = TTSFeedbackMonitor()

    assert monitor.distribution()["total"] == 0
    assert monitor.complaint_rate() == 0.0
