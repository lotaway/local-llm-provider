import importlib.util
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = Path(__file__).resolve().parents[1] / "tts_estimate" / "report_summary.py"
spec = importlib.util.spec_from_file_location("tts_estimate_report_summary", MODULE_PATH)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def _result(passed=True, wer=0.05):
    metrics = SimpleNamespace(
        wer=wer,
        dnsmos_overall=4.0,
        dnsmos_signal=4.1,
        dnsmos_background=3.9,
        speaker_similarity=0.9,
    )
    return SimpleNamespace(metrics=metrics, is_passed=passed)


def test_run_evaluation_result_is_a_dict():
    report = summary.build_report(_result())
    assert report["status"] == "pass"
    assert report["wer"] == 0.05
    assert report["dnsmos"]["overall"] == 4.0


def test_batch_summary_is_aggregated_outside_the_loop():
    reports = [summary.build_report(_result()), summary.build_report(_result(False, 0.2))]
    aggregated = summary.summarize_reports(reports)
    assert aggregated["total"] == 2
    assert aggregated["passed"] == 1
    assert aggregated["rejected"] == 1
    assert aggregated["max_wer"] == 0.2
