import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _init_function():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    return next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "init_molt_scheduler")


def test_startup_imports_current_memory_repo():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "from bootstrap.molt_factory import build_molt_controller, current_memory_repo" in source
    assert "logger.exception(\"Failed to start MOLT scheduler\")" in source


def _call_names():
    return {
        name
        for node in ast.walk(_init_function())
        if isinstance(node, ast.Call)
        for name in [getattr(node.func, "attr", None) or getattr(node.func, "id", None)]
        if name
    }


def test_startup_starts_controller_and_scheduler():
    calls = _call_names()
    assert "start" in calls
    assert "build_molt_controller" in calls
    assert "current_memory_repo" in calls
