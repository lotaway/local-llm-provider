import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY_NAMES = {
    "TTSTAlertService",
    "TTSSpeechQualityGateway",
    "TTSSRegressionSuite",
    "TTSTrendStatistics",
    "TTSSamplingMonitor",
}
EXCLUDED = {"__pycache__", ".git", "third_party", "llama.cpp"}


def _source_files():
    return [
        item for item in ROOT.rglob("*.py")
        if not set(item.parts) & EXCLUDED
    ]


def test_no_legacy_tts_class_names_remain():
    offenders = []
    for item in _source_files():
        try:
            tree = ast.parse(item.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in LEGACY_NAMES:
                offenders.append((item.name, node.name))
    assert offenders == []


def test_service_imports_the_unified_names():
    source = (ROOT / "services" / "tts_service.py").read_text(encoding="utf-8")
    assert "from services.tts_quality_statistics import TTSQualityStatistics" in source
    assert "from services.tts_request_sampler import TTSRequestSampler" in source
    assert "TTSTrendStatistics" not in source
    assert "TTSSamplingMonitor" not in source


def test_ci_runs_the_renamed_test_files():
    source = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "tests/test_tts_quality_statistics.py" in source
    assert "tests/test_tts_request_sampler.py" in source
    assert "tests/test_tts_trend_statistics.py" not in source
