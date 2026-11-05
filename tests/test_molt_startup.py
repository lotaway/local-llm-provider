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


def test_startup_starts_controller_and_scheduler():
    calls = [node.func.attr for node in ast.walk(_init_function()) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert "start" in calls
    assert "build_molt_controller" in calls
    assert "current_memory_repo" in calls
