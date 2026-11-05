from pathlib import Path

from permission_manager import PermissionManager
from schemas.capability import Capability, CapabilityKind
from schemas.permission import (
    SAFETY_LEVEL_NAMES,
    PermissionNames,
    SafetyLevel,
    normalize_safety_level,
)

ROOT = Path(__file__).resolve().parents[1]


def test_safety_level_values_are_the_four_spec_levels():
    assert SAFETY_LEVEL_NAMES == ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert [level.name for level in SafetyLevel] == list(SAFETY_LEVEL_NAMES)


def test_capability_default_is_low():
    capability = Capability(id="skill.x", kind=CapabilityKind.SKILL, name="x", description="x")
    assert capability.safety_level == SafetyLevel.LOW.name
    assert capability.to_dict()["safety_level"] == "LOW"


def test_legacy_safe_is_normalized_to_low():
    assert normalize_safety_level("SAFE") == "LOW"
    assert normalize_safety_level(" high ") == "HIGH"
    assert normalize_safety_level("bogus") == "UNKNOWN"


def test_permission_manager_two_stage_approval():
    manager = PermissionManager()
    low = manager.check_permission(PermissionNames.LLM_QUERY)
    high = manager.check_permission(PermissionNames.MCP_FILE_WRITE)
    assert low["needs_human"] is False
    assert low["safety_level"] in SAFETY_LEVEL_NAMES
    assert high["needs_human"] is True
    assert high["safety_level"] in SAFETY_LEVEL_NAMES


def test_risk_prompt_matches_spec_levels():
    source = (ROOT / "clients" / "agents" / "risk_agent.py").read_text(encoding="utf-8")
    assert "safe|low|medium|high|critical" not in source
    assert "|".join(name.lower() for name in SAFETY_LEVEL_NAMES) in source
    assert "SAFE" not in source
    assert "risk_order = [name.lower() for name in SAFETY_LEVEL_NAMES]" in source
