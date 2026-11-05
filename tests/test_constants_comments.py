import io
import tokenize
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "constants.py"


def _comments():
    source = SOURCE.read_text(encoding="utf-8")
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [token.string for token in tokens if token.type == tokenize.COMMENT]


def test_configuration_module_has_no_explanatory_comments():
    assert _comments() == []


def test_configuration_defaults_are_unchanged():
    source = SOURCE.read_text(encoding="utf-8")
    assert 'ENGINE_PREFERENCE = os.getenv("ENGINE_PREFERENCE", "auto")' in source
    assert 'VLLM_QUANTIZATION = os.getenv("VLLM_QUANTIZATION")' in source
