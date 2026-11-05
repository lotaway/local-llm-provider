import ast
import io
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    "services/abstraction_engine.py",
    "clients/agents/runtime_factory.py",
]
LOGGER_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}


def _handlers(relative):
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    return [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]


def _records_context(handler):
    for call in (node for node in ast.walk(handler) if isinstance(node, ast.Call)):
        if isinstance(call.func, ast.Attribute) and call.func.attr in LOGGER_METHODS:
            return True
    return False


def test_no_handler_swallows_an_exception():
    for relative in TARGETS:
        for handler in _handlers(relative):
            assert not any(isinstance(node, ast.Pass) for node in ast.walk(handler)), relative


def test_every_handler_records_exception_context():
    for relative in TARGETS:
        for handler in _handlers(relative):
            assert _records_context(handler), relative


def test_mcp_loading_failure_is_classified():
    source = (ROOT / "clients/agents/runtime_factory.py").read_text(encoding="utf-8")
    assert "MCP_LOAD_FAILURES = (ImportError, OSError, ValueError, TypeError, AttributeError)" in source
    assert "except MCP_LOAD_FAILURES as error:" in source


def test_abstraction_engine_has_no_explanatory_comments():
    source = (ROOT / "services/abstraction_engine.py").read_text(encoding="utf-8")
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    assert [token.string for token in tokens if token.type == tokenize.COMMENT] == []
