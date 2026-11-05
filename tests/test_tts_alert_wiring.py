import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "services" / "tts_alert_rules.py"
COMPOSITION = ROOT / "bootstrap" / "tts_composition.py"


def _rules():
    spec = importlib.util.spec_from_file_location("tts_alert_rules_under_test", RULES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _thresholds():
    return (0.08, 3.8, 0.78)


def test_missing_quality_metrics_do_not_create_false_alerts():
    assert _rules().should_alert({"status": "pass"}, *_thresholds()) is False


def test_every_quality_threshold_can_trigger_an_alert():
    rules = _rules()
    assert rules.should_alert({"wer": 0.2}, *_thresholds()) is True
    assert rules.should_alert({"dnsmos": {"overall": 2.0}}, *_thresholds()) is True
    assert rules.should_alert({"speaker_similarity": 0.4}, *_thresholds()) is True


def test_rejected_results_still_alert():
    assert _rules().should_alert({"status": "reject"}, *_thresholds()) is True


def test_composition_root_injects_a_production_notifier():
    source = COMPOSITION.read_text(encoding="utf-8")
    assert "TTSAlertService(notifier=StructuredLogNotifier())" in source
    assert "preferences = TTSPreferenceStore(collection=build_preference_collection())" in source


def test_every_request_is_checked_against_the_alert_rules():
    source = (ROOT / "services" / "tts_service.py").read_text(encoding="utf-8")
    assert "self.alerts.check([result])" in source
    assert 'if status != "pass":' not in source


def test_startup_uses_the_tts_composition_root():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "from bootstrap.tts_composition import build_tts_service" in source
    assert "backend_globals.tts_service = build_tts_service()" in source
