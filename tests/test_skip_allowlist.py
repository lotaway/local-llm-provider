import importlib.util
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_skip_allowlist.py"
ALLOWLIST = ROOT / "scripts" / "test_skip_allowlist.json"


def _module():
    spec = importlib.util.spec_from_file_location("skip_allowlist_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_test_file_is_tracked():
    data = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
    assert _module().violations(data, date(2026, 11, 5)) == []


def test_allowlist_has_no_expired_entries():
    data = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
    assert _module().expired_entries(data, date(2026, 11, 5)) == []


def test_targeted_and_allowlisted_sets_do_not_overlap():
    module = _module()
    data = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
    assert not module.targeted_files() & module.allowlisted_files(data)
