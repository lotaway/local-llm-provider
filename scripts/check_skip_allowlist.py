import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = Path(__file__).resolve().parent / "test_skip_allowlist.json"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
TEST_DIR = ROOT / "tests"


def test_files() -> Set[str]:
    return {"tests/" + item.name for item in TEST_DIR.glob("test_*.py")}


def targeted_files() -> Set[str]:
    text = WORKFLOW.read_text(encoding="utf-8")
    return {"tests/" + name for name in re.findall(r"tests/(test_[A-Za-z0-9_]+\.py)", text)}


def allowlisted_files(data: Dict) -> Set[str]:
    return {item["path"] for item in data["skips"]}


def expired_entries(data: Dict, today: date) -> List[str]:
    return [item["path"] for item in data["skips"] if date.fromisoformat(item["review_by"]) < today]


def violations(data: Dict, today: date) -> List[str]:
    tracked = test_files()
    targeted = targeted_files()
    allowed = allowlisted_files(data)
    found = [f"untracked test: {item}" for item in sorted(tracked - targeted - allowed)]
    found.extend(f"overlapping test: {item}" for item in sorted(targeted & allowed))
    found.extend(f"allowlisted but missing: {item}" for item in sorted(allowed - tracked))
    found.extend(f"expired review: {item}" for item in expired_entries(data, today))
    found.extend(f"missing reason: {item['path']}" for item in data["skips"] if not item.get("reason"))
    return found


def main() -> int:
    data = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
    found = violations(data, date.today())
    for item in found:
        sys.stderr.write(f"[skip-policy] {item}\n")
    if found:
        return 1
    sys.stdout.write(f"[skip-policy] {len(test_files())} tests tracked\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
