"""Skills Package - External capabilities that can be used by agents

Available Skills:
- ChromeDevToolsSkill: MCP-based browser automation for Chrome DevTools control

Usage:
    from skills import registry, init_skills, ChromeDevToolsSkill

    # Initialize all skills
    init_skills()

    # List available skills
    skills = registry.list_skills()

    # Get all tools from all skills
    tools = registry.list_all_tools()
"""

from .chrome_devtools_skill import ChromeDevToolsSkill, ChromeSkill
from .skill_registry import (
    SkillRegistry,
    SkillManifest,
    SkillTool,
    registry,
    get_registry,
    init_skills,
    create_skill_manifest,
    register_skill,
)

__all__ = [
    "ChromeDevToolsSkill",
    "ChromeSkill",
    "SkillRegistry",
    "SkillManifest",
    "SkillTool",
    "registry",
    "get_registry",
    "init_skills",
    "create_skill_manifest",
    "register_skill",
]
