"""Test: Verify skill loading system works correctly"""

import sys
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, "/Volumes/Extra/Projects/local-llm-provider")


def test_skill_registry_imports():
    """Test that all imports work"""
    print("Testing imports...")
    from skills import (
        SkillRegistry,
        SkillManifest,
        SkillTool,
        registry,
        get_registry,
        init_skills,
        create_skill_manifest,
        register_skill,
    )

    print("  All imports successful!")
    return True


def test_skill_manifest_creation():
    """Test creating a skill manifest"""
    print("\nTesting skill manifest creation...")

    from skills import SkillManifest, SkillTool

    tool = SkillTool(
        name="test_tool",
        description="A test tool",
        category="testing",
    )

    manifest = SkillManifest(
        name="test_skill",
        description="A test skill",
        version="1.0.0",
        category="testing",
        tools=[tool],
    )

    assert manifest.name == "test_skill"
    assert manifest.version == "1.0.0"
    assert len(manifest.tools) == 1
    assert manifest.tools[0].name == "test_tool"

    print("  Skill manifest creation successful!")
    return True


def test_chrome_skill_manifest():
    """Test that Chrome DevTools skill manifest exists"""
    print("\nTesting Chrome DevTools skill manifest...")

    from skills.chrome_devtools_skill import SKILL_MANIFEST, get_skill_instance

    assert SKILL_MANIFEST.name == "chrome_devtools"
    assert len(SKILL_MANIFEST.tools) > 0

    # Check all tools have required fields
    for tool in SKILL_MANIFEST.tools:
        assert tool.name is not None
        assert tool.description is not None

    print(f"  Chrome skill has {len(SKILL_MANIFEST.tools)} tools")

    # Test getting an instance
    skill = get_skill_instance()
    assert skill is not None
    print("  Chrome skill instance created successfully!")

    return True


def test_dynamic_loading():
    """Test dynamic skill loading"""
    print("\nTesting dynamic skill loading...")

    from skills import init_skills, registry

    # Clear any existing registrations
    registry.clear()

    # Load skills
    manifests = init_skills()

    assert len(manifests) > 0, "Expected at least one skill to be loaded"

    print(f"  Loaded {len(manifests)} skills:")
    for m in manifests:
        print(f"    - {m.name} v{m.version} ({len(m.tools)} tools)")

    # Verify registry has the skills
    assert len(registry.list_skills()) == len(manifests)

    # Test listing all tools
    all_tools = registry.list_all_tools()
    print(f"  Total tools: {len(all_tools)}")

    # Test finding skill for tool
    navigate_skill = registry.find_skill_for_tool("navigate_page")
    assert navigate_skill is not None
    print(f"  Tool 'navigate_page' belongs to skill: {navigate_skill}")

    return True


def test_skill_summaries():
    """Test skill summary generation"""
    print("\nTesting skill summaries...")

    from skills import registry

    summaries = registry.list_skill_summaries()
    assert len(summaries) > 0

    for s in summaries:
        assert "name" in s
        assert "description" in s

    print(f"  Generated {len(summaries)} skill summaries")

    return True


def run_tests():
    """Run all tests"""
    print("=" * 60)
    print("Skills Package - Test Suite")
    print("=" * 60)

    tests = [
        ("Imports", test_skill_registry_imports),
        ("Skill Manifest Creation", test_skill_manifest_creation),
        ("Chrome Skill Manifest", test_chrome_skill_manifest),
        ("Dynamic Loading", test_dynamic_loading),
        ("Skill Summaries", test_skill_summaries),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, f"ERROR: {e}"))

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)

    all_passed = True
    for name, result in results:
        status = "✓" if "PASS" in result else "✗"
        print(f"  {status} {name}: {result}")
        if "PASS" not in result:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
