import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUSINESS = ["services/tts_post_processing.py", "services/tts_audio_quality.py"]
ADAPTER = "model_providers/tts/audio_processing.py"


def _imports(relative):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def test_business_layer_does_not_import_numpy():
    for relative in BUSINESS:
        assert "numpy" not in _imports(relative), relative


def test_numpy_math_stays_in_the_adapter():
    assert "numpy" in _imports(ADAPTER)


def test_post_processor_uses_an_abstract_sample_type():
    source = (ROOT / "services" / "tts_post_processing.py").read_text(encoding="utf-8")
    assert "AudioSamples = Any" in source
    assert "np.ndarray" not in source
