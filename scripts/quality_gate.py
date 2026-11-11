"""Baseline-tolerant quality gate.

Prevents regressions of the audited code-principle violations while the
existing debt is paid down. Every rule only fails on NEW violations or on
GROWTH of a recorded baseline entry; fixing violations lowers the counts.
Run ``python scripts/quality_gate.py --update-baseline`` to refresh the
baseline after paying down debt.

Rules:
1. Source files must not exceed MAX_FILE_LINES unless recorded in baseline.
2. Functions must not exceed MAX_FUNCTION_LINES unless recorded in baseline.
3. ``print(``, ``except ...: pass``, explanatory ``#`` comments and business-layer
   direct ``pymongo`` / ``numpy`` / ``redis`` imports are counted per file and
   must not exceed the recorded baseline. Tool directives (``# type:``,
   ``# noqa``, ...), contract/type documentation (``#:``, ``# Args:``, ...)
   and pure separator lines are exempt from the comment rule.
4. Checked TODO items are validated against pytest by scripts/todo_guard.py
   in the CI targeted-tests job.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_policy import growth_violations
BASELINE_PATH = Path(__file__).resolve().parent / "quality_baseline.json"
parse_errors = 0

EXCLUDED_DIRS = {
    "__pycache__",
    "build",
    "cache",
    "docs",
    "llama.cpp",
    "local_llm_provider.egg-info",
    "third_party",
    "tests",
}

ROOT_FILES = [
    "auth.py",
    "constants.py",
    "main.py",
    "permission_manager.py",
    "test_container_lazy.py",
    "verify_claude_skills.py",
]

MAX_FILE_LINES = 500
MAX_FUNCTION_LINES = 20
FORBIDDEN_IMPORT_MODULES = {"pymongo", "numpy", "redis"}
FORBIDDEN_IMPORT_DIRS = {"services", "controllers", "agents", "application", "domain"}
COMMENT_EXCLUSIONS = (
    "#!", "# -*-", "# coding", "# encoding", "# type:", "# noqa", "# noinspection",
    "# pylint:", "# ruff:", "# mypy:", "# fmt:", "# pragma:", "# isort:",
    "# TODO", "# FIXME", "#:", "# @param", "# @return", "# Args:", "# Returns:",
    "# Raises:", "# Contract:", "# Invariant:", "# Precondition", "# Postcondition",
)


def source_files() -> List[Path]:
    found: List[Path] = [ROOT / name for name in ROOT_FILES if (ROOT / name).exists()]
    for directory in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name not in EXCLUDED_DIRS):
        found.extend(sorted(directory.rglob("*.py")))
    return [path for path in found if path.is_file()]


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def function_qualname(node: ast.AST) -> str:
    parts: List[str] = []
    current: Any = node
    while isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        parts.append(current.name)
        current = current.parent if hasattr(current, "parent") else None
    return ".".join(reversed(parts))


def longest_functions(tree: ast.AST, rel: str) -> Iterable[Tuple[str, int]]:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            length = node.end_lineno - node.lineno + 1
            yield f"{rel}::{function_qualname(node)}", length
        elif isinstance(node, ast.AsyncFunctionDef):
            length = node.end_lineno - node.lineno + 1
            yield f"{rel}::{function_qualname(node)}", length


def pattern_counts(source: str) -> Dict[str, int]:
    import re

    bare_excepts = len(re.findall(r"except[^\n]*:\n\s*(?:#[^\n]*\n\s*)*pass\b", source))
    prints = len(re.findall(r"(?<![#\"\'])\bprint\(", source))
    forbidden_imports = 0
    for match in re.finditer(r"(?m)^\s*(?:import|from)\s+(pymongo|numpy|redis)\b", source):
        forbidden_imports += 1
    return {
        "print": prints,
        "bare_except_pass": bare_excepts,
        "forbidden_imports": forbidden_imports,
        "comments": comment_count(source),
    }


COMMENT_DIRECTIVE_PREFIXES = (
    "#!",
    "# -*-",
    "# type:",
    "# noqa",
    "# fmt:",
    "# isort:",
    "# pylint:",
    "# pragma:",
    "# Contract:",
    "# 契约:",
)


def comment_count(source: str) -> int:
    """Count explanatory ``#`` comments, skipping tool/type directives."""
    import io
    import tokenize

    count = 0
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            text = token.string.strip()
            if any(text.startswith(prefix) for prefix in COMMENT_DIRECTIVE_PREFIXES):
                continue
            count += 1
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return 0
    return count


def is_explanatory_comment(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith(COMMENT_EXCLUSIONS):
        return False
    body = stripped.lstrip("#").strip()
    return bool(body) and any(character.isalpha() for character in body)


def explanatory_comment_count(source: str) -> int:
    import io
    import tokenize

    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        return sum(
            1
            for token in tokens
            if token.type == tokenize.COMMENT and is_explanatory_comment(token.string)
        )
    except (tokenize.TokenError, IndentationError):
        return 0


def report_parse_error(rel: str) -> None:
    global parse_errors
    print(f"  - unparseable source file: {rel}")
    parse_errors += 1


def record_lines(source: str, rel: str, current: Dict[str, Dict[str, int]]) -> None:
    lines = len(source.splitlines())
    if lines > MAX_FILE_LINES:
        current["files"][rel] = lines


def record_functions(tree: ast.AST, rel: str, current: Dict[str, Dict[str, int]]) -> None:
    for key, length in longest_functions(tree, rel):
        if length > MAX_FUNCTION_LINES:
            current["functions"][key] = max(current["functions"].get(key, 0), length)


def record_patterns(source: str, rel: str, current: Dict[str, Dict[str, int]]) -> None:
    counts = pattern_counts(source)
    if counts["print"]:
        current["print"][rel] = counts["print"]
    if counts["bare_except_pass"]:
        current["bare_except_pass"][rel] = counts["bare_except_pass"]
    if counts["forbidden_imports"] and rel.split("/", 1)[0] in FORBIDDEN_IMPORT_DIRS:
        current["forbidden_imports"][rel] = counts["forbidden_imports"]


def record_comments(source: str, rel: str, current: Dict[str, Dict[str, int]]) -> None:
    comments = explanatory_comment_count(source)
    if comments:
        current["explanatory_comments"][rel] = comments


def measure_file(source: str, tree: ast.AST, rel: str, current: Dict[str, Dict[str, int]]) -> None:
    record_lines(source, rel, current)
    record_functions(tree, rel, current)
    record_patterns(source, rel, current)
    record_comments(source, rel, current)


def collect() -> Dict[str, Dict[str, int]]:
    current: Dict[str, Dict[str, int]] = {
        "files": {},
        "functions": {},
        "print": {},
        "bare_except_pass": {},
        "forbidden_imports": {},
        "explanatory_comments": {},
    }
    for path in source_files():
        rel = path.relative_to(ROOT).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            report_parse_error(rel)
            continue
        measure_file(source, tree, rel, current)
    return current


def flatten(current: Dict[str, Dict[str, int]]) -> Dict[str, str]:
    flat: Dict[str, str] = {}
    for rule, entries in current.items():
        for key, value in entries.items():
            flat[f"{rule}|{key}"] = str(value)
    return flat


def compare(current: Dict[str, Dict[str, int]], baseline: Dict[str, str]) -> List[str]:
    violations: List[str] = []
    flat = flatten(current)
    for key, value in sorted(flat.items()):
        recorded = baseline.get(key)
        if recorded is None:
            violations.append(f"new violation: {key} = {value}")
        elif int(value) > int(recorded):
            violations.append(f"grew: {key} = {value} (baseline {recorded})")
    if parse_errors:
        violations.append(f"{parse_errors} file(s) failed to parse")
    return violations


def read_baseline() -> Dict[str, str]:
    if not BASELINE_PATH.exists():
        return {}
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def require_baseline_approval(current: Dict[str, Dict[str, int]], baseline: Dict[str, str]) -> None:
    growth = growth_violations(flatten(current), baseline)
    if growth and os.getenv("LLP_BASELINE_CHANGE_APPROVED") != "1":
        sys.exit("baseline update blocked: " + "; ".join(growth))


def main() -> int:
    update_baseline = "--update-baseline" in sys.argv
    current = collect()
    baseline = read_baseline()
    violations = compare(current, baseline)

    if update_baseline:
        require_baseline_approval(current, baseline)
        BASELINE_PATH.write_text(json.dumps(flatten(current), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"baseline updated: {BASELINE_PATH}")
        return 0

    if violations:
        print("quality gate failed:")
        for item in violations:
            print(f"  - {item}")
        return 1
    print("quality gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
