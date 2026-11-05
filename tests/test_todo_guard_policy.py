import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts" / "todo_guard.py"
MAPPING = ROOT / "scripts" / "todo_test_map.json"


def _guard(name):
    spec = importlib.util.spec_from_file_location(name, GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_allow_empty_check_returns_zero_without_checked_todos():
    guard = _guard("todo_guard_a")
    args = argparse.Namespace(list=False, check=True, report=None, allow_empty=True)
    assert guard.dispatch(args, set()) == 0


def test_strict_check_still_rejects_an_empty_set():
    guard = _guard("todo_guard_b")
    args = argparse.Namespace(list=False, check=True, report=None, allow_empty=False)
    assert guard.dispatch(args, set()) == 1


def test_checked_references_are_still_resolved():
    guard = _guard("todo_guard_c")
    assert guard.referenced_tests(["- [x] fixed `tests/test_todo_guard.py`"]) == {"tests/test_todo_guard.py"}


def test_mapping_files_exist_and_run_in_ci():
    guard = _guard("todo_guard_d")
    paths = guard.mapped_tests()
    assert paths
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for item in paths:
        assert (ROOT / item).is_file(), item
        assert item in ci, item


def test_ci_check_uses_the_allow_empty_flag():
    source = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "python scripts/todo_guard.py --check --allow-empty" in source
