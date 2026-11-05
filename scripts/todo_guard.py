"""Keep checked TODO items honest against their pytest results."""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Set

ROOT = Path(__file__).resolve().parents[1]
TODO_PATH = ROOT / "TODO.md"
MAPPING_PATH = Path(__file__).resolve().parent / "todo_test_map.json"
CHECKED_PREFIX = "- [x]"
REFERENCE = re.compile(r"`((?:tests|tts_estimate/tests)/[A-Za-z0-9_./-]+\.py)(?:::[^\s`]+)?`")


def emit(message: str, error: bool = False) -> None:
    print(message, file=sys.stderr if error else sys.stdout)


def checked_lines() -> List[str]:
    lines = TODO_PATH.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.lstrip().startswith(CHECKED_PREFIX)]


def referenced_tests(lines: Iterable[str]) -> Set[str]:
    found: Set[str] = set()
    for line in lines:
        for match in REFERENCE.finditer(line):
            found.add(Path(match.group(1)).as_posix())
    return found


def mapped_tests() -> Set[str]:
    if not MAPPING_PATH.is_file():
        return set()
    data = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    return set(data.get("tests", []))


def dangling_paths(paths: Iterable[str]) -> List[str]:
    return [path for path in sorted(paths) if not (ROOT / path).is_file()]


def case_paths(case: ET.Element) -> Set[str]:
    candidates: Set[str] = set()
    file_attr = case.get("file")
    if file_attr:
        candidates.add(Path(file_attr).as_posix())
    parts = (case.get("classname") or "").split(".")
    for size in range(len(parts), 0, -1):
        candidates.add("/".join(parts[:size]) + ".py")
    return candidates


def failing_cases(root: ET.Element) -> List[str]:
    failing = []
    for case in root.iter("testcase"):
        if any(case.find(tag) is not None for tag in ("failure", "error", "skipped")):
            failing.append(f"{case.get('classname', '?')}::{case.get('name', '?')}")
    return failing


def report_violations(report_path: Path, paths: Set[str]) -> List[str]:
    if not report_path.is_file():
        return [f"missing junit report: {report_path}"]
    try:
        root = ET.parse(report_path).getroot()
    except (ET.ParseError, OSError) as error:
        return [f"unreadable junit report {report_path}: {error}"]
    cases = list(root.iter("testcase"))
    covered = {path for path in paths if any(path in case_paths(case) for case in cases)}
    violations = [f"no testcase collected for {path}" for path in sorted(paths - covered)]
    violations.extend(f"{name} is not green" for name in failing_cases(root))
    return violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate checked TODO items against pytest")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="print referenced test files")
    group.add_argument("--check", action="store_true", help="verify references resolve")
    group.add_argument("--report", type=Path, help="verify a pytest junit XML report is green")
    parser.add_argument("--allow-empty", action="store_true", help="accept an empty checked TODO set")
    return parser.parse_args()


def dispatch(args: argparse.Namespace, paths: Set[str]) -> int:
    if args.list:
        for path in sorted(paths):
            emit(path)
        return 0
    if args.check:
        emit(f"[todo] {len(paths)} checked TODO test references resolved")
        return 0 if paths or args.allow_empty else 1
    violations = report_violations(args.report, paths)
    for item in violations:
        emit(f"[todo] {item}", error=True)
    if violations:
        return 1
    emit(f"[todo] {len(paths)} checked TODO references are green")
    return 0


def main() -> int:
    args = parse_args()
    paths = referenced_tests(checked_lines()) | mapped_tests()
    dangling = dangling_paths(paths)
    if dangling:
        for path in dangling:
            emit(f"[todo] dangling reference: {path}", error=True)
        return 1
    return dispatch(args, paths)


if __name__ == "__main__":
    raise SystemExit(main())
