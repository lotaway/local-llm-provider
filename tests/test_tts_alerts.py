from services.tts_alerts import TTSTAlertService
from services.tts_quality_gateway import TTSSpeechQualityGateway


class FakeNotifier:
    def __init__(self):
        self.alerts = []

    def notify(self, result):
        self.alerts.append(result)


def test_alert_service_flags_failed_generation():
    service = TTSTAlertService()

    assert service.should_alert(
        {"status": "reject", "wer": 0.2, "dnsmos": {"overall": 3.5}, "speaker_similarity": 0.8}
    ) is True
    assert TTSSpeechQualityGateway is not None


def test_alert_service_flags_quality_regression():
    service = TTSTAlertService()
    result = {"status": "reject", "wer": 0.5, "dnsmos": {"overall": 3.0}, "speaker_similarity": 0.5}

    assert service.should_alert(result) is True
    assert service.check([result])[0]["status"] == "reject"


def test_alert_service_passes_clean_result():
    service = TTSTAlertService()

    assert service.should_alert({"status": "pass", "wer": 0.01, "dnsmos": {"overall": 4.0}, "speaker_similarity": 0.9}) is False
    assert service.check([]) == []


def test_alert_service_notifies_and_exposes_alerts():
    notifier = FakeNotifier()
    service = TTSTAlertService(notifier=notifier)

    result = {"status": "reject", "wer": 0.5, "dnsmos": {"overall": 3.0}, "speaker_similarity": 0.5}
    alerts = service.check([result])

    assert alerts == [result]
    assert len(notifier.alerts) == 1
    assert len(service.alerts()) == 1


def test_alert_service_reset_clears_alerts():
    service = TTSTAlertService()

    service.check([{"status": "reject", "wer": 0.5, "dnsmos": {"overall": 3.0}, "speaker_similarity": 0.5}])
    service.reset()

    assert service.alerts() == []
