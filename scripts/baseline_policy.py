import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).resolve().parent / "quality_baseline.json"
APPROVAL_ENV = "LLP_BASELINE_CHANGE_APPROVED"


def growth_violations(current: Dict[str, str], recorded: Dict[str, str]) -> List[str]:
    violations = []
    for key, value in sorted(current.items()):
        baseline = recorded.get(key)
        if baseline is None:
            violations.append(f"new baseline entry: {key} = {value}")
        elif int(value) > int(baseline):
            violations.append(f"baseline grew: {key} = {value} (was {baseline})")
    return violations


def baseline_at_ref(ref: str) -> Dict[str, str]:
    result = subprocess.run(
        ["git", "show", f"origin/{ref}:scripts/quality_baseline.json"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def main() -> int:
    ref = os.getenv("BASE_REF", "")
    if not ref:
        sys.stdout.write("baseline policy skipped: no pull request base ref\n")
        return 0
    try:
        recorded = baseline_at_ref(ref)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.stderr.write(f"baseline policy cannot read base: {error}\n")
        return 1
    current = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    found = growth_violations(current, recorded)
    if found and os.getenv(APPROVAL_ENV) != "1":
        for item in found:
            sys.stderr.write(f"[baseline] {item}\n")
        return 1
    sys.stdout.write("baseline policy passed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
