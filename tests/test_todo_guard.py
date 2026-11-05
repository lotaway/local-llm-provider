import sys
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.todo_guard import (
    case_paths,
    checked_lines,
    dangling_paths,
    failing_cases,
    referenced_tests,
    report_violations,
)


def test_reference_extraction_ignores_unchecked_items():
    lines = [
        "- [x] fixed `tests/test_a.py::test_case` and `tests/test_b.py`",
        "- [ ] pending `tests/test_pending.py`",
    ]
    assert referenced_tests(lines) == {"tests/test_a.py", "tests/test_b.py"}


def test_checked_lines_only_returns_checkboxes():
    assert all(line.lstrip().startswith("- [x]") for line in checked_lines())
    assert referenced_tests(checked_lines())


def test_dangling_paths_are_detected(tmp_path):
    assert dangling_paths(["tests/test_todo_guard.py"]) == []
    assert dangling_paths(["tests/does_not_exist.py"]) == ["tests/does_not_exist.py"]


def test_case_paths_match_file_attribute_and_classname():
    case = ET.fromstring('<testcase classname="tests.test_example.TestThing" name="test_x" file="tests/test_example.py"/>')
    assert "tests/test_example.py" in case_paths(case)


def test_failing_cases_include_failures_and_skips():
    root = ET.fromstring(
        "<testsuite><testcase classname='tests.test_x' name='ok'/>"
        "<testcase classname='tests.test_x' name='bad'><failure/></testcase>"
        "<testcase classname='tests.test_x' name='skip'><skipped/></testcase></testsuite>"
    )
    assert len(failing_cases(root)) == 2


def test_report_requires_all_referenced_files_to_be_collected(tmp_path):
    report = tmp_path / "junit.xml"
    report.write_text(
        "<testsuites><testsuite><testcase classname='tests.test_a' name='test_ok'/></testsuite></testsuites>",
        encoding="utf-8",
    )
    assert report_violations(report, {"tests/test_a.py"}) == []
    violations = report_violations(report, {"tests/test_a.py", "tests/test_b.py"})
    assert any("tests/test_b.py" in item for item in violations)
