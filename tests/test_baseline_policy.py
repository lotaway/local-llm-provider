import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "scripts" / "baseline_policy.py"
GATE = ROOT / "scripts" / "quality_gate.py"


def _load(script, name):
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_new_baseline_entry_is_a_violation():
    found = _load(POLICY, "baseline_policy_a").growth_violations({"print|a.py": "2"}, {})
    assert found and "new baseline entry" in found[0]


def test_grown_baseline_entry_is_a_violation():
    found = _load(POLICY, "baseline_policy_b").growth_violations({"print|a.py": "3"}, {"print|a.py": "2"})
    assert found and "baseline grew" in found[0]


def test_reduced_baseline_entry_is_allowed():
    found = _load(POLICY, "baseline_policy_c").growth_violations({"print|a.py": "1"}, {"print|a.py": "2"})
    assert found == []


def test_quality_gate_blocks_unapproved_baseline_growth():
    os.environ.pop("LLP_BASELINE_CHANGE_APPROVED", None)
    gate = _load(GATE, "quality_gate_a")
    try:
        gate.require_baseline_approval({"print": {"a.py": 2}}, {})
    except SystemExit:
        return
    raise AssertionError("unapproved growth was not blocked")


def test_approval_allows_baseline_growth():
    os.environ["LLP_BASELINE_CHANGE_APPROVED"] = "1"
    try:
        _load(GATE, "quality_gate_b").require_baseline_approval({"print": {"a.py": 2}}, {})
    finally:
        os.environ.pop("LLP_BASELINE_CHANGE_APPROVED", None)
