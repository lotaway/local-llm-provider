import os
import shutil
import tempfile
import logging
from skills.skill_registry import init_skills, registry

# Configure logging
logging.basicConfig(level=logging.INFO)


def test_global_skill_loading():
    # 1. Create temp directory for global skills
    with tempfile.TemporaryDirectory() as temp_dir:
        skill_dir = os.path.join(temp_dir, "test-global-skill")
        os.makedirs(skill_dir)

        # 2. Create dummy SKILL.md
        skill_md_content = """---
name: test-global-skill
description: A test skill from global directory
version: 1.0.0
---
"""
        with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
            f.write(skill_md_content)

        # 3. Set environment variables
        os.environ["LLP_ENABLE_CLAUDE_GLOBAL"] = "1"
        os.environ["LLP_CLAUDE_SKILLS_DIR"] = temp_dir

        print(f"Testing with LLP_CLAUDE_SKILLS_DIR={temp_dir}")

        # 4. Clear registry and init skills
        registry.clear()
        manifests = init_skills()

        # 5. Verify
        found = False
        for m in manifests:
            if m.name == "test-global-skill":
                found = True
                print(f"SUCCESS: Found skill '{m.name}' from global directory")
                break

        if not found:
            print("FAILURE: Did not find 'test-global-skill'")
            print("Loaded skills:", [m.name for m in manifests])

        # 6. Test default disabled behavior
        os.environ["LLP_ENABLE_CLAUDE_GLOBAL"] = "0"
        registry.clear()
        manifests = init_skills()
        found = False
        for m in manifests:
            if m.name == "test-global-skill":
                found = True
                break

        if not found:
            print("SUCCESS: correctly ignored global skill when disabled")
        else:
            print("FAILURE: found global skill even when disabled")


if __name__ == "__main__":
    test_global_skill_loading()
