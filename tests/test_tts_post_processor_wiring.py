import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "model_providers" / "tts" / "style_tts2_adapter.py"


def _source():
    return ADAPTER.read_text(encoding="utf-8")


def test_engine_postprocess_uses_the_injected_processor():
    source = _source()
    assert "postprocess=self.post_processor.process" in source
    assert "postprocess=process_samples" not in source


def test_adapter_constructs_the_post_processor():
    tree = ast.parse(_source())
    names = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "TTSPostProcessor" in names


def test_adapter_imports_the_service_facade():
    assert "from services.tts_post_processing import TTSPostProcessor" in _source()
