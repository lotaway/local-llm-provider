"""Skill Registry - Dynamic skill loading and management system

This module provides a registry for dynamically loading and managing skills.
Skills are discovered at runtime and their capabilities are exposed to agents.
"""

import importlib.util
import inspect
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


@dataclass
class SkillTool:
    """A tool provided by a skill"""

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    category: str = "general"
    read_only: bool = False


@dataclass
class SkillManifest:
    """Manifest describing a skill's capabilities"""

    name: str
    description: str
    version: str = "1.0.0"
    category: str = "general"
    tools: List[SkillTool] = field(default_factory=list)
    module_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category,
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "category": t.category,
                    "read_only": t.read_only,
                }
                for t in self.tools
            ],
        }


class SkillRegistry:
    """Registry for managing skills dynamically"""

    _instance: Optional["SkillRegistry"] = None
    _skills: Dict[str, SkillManifest]
    _skill_instances: Dict[str, Any]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._skills = {}
            cls._instance._skill_instances = {}
        return cls._instance

    def register(self, manifest: SkillManifest, instance: Any = None):
        """Register a skill with the registry"""
        self._skills[manifest.name] = manifest
        if instance:
            self._skill_instances[manifest.name] = instance
        logger.info(f"Registered skill: {manifest.name} v{manifest.version}")

    def get_manifest(self, skill_name: str) -> Optional[SkillManifest]:
        """Get skill manifest by name"""
        return self._skills.get(skill_name)

    def get_skill_instance(self, skill_name: str) -> Optional[Any]:
        """Get skill instance by name"""
        return self._skill_instances.get(skill_name)

    def list_skills(self) -> List[SkillManifest]:
        """List all registered skills"""
        return list(self._skills.values())

    def list_skill_summaries(self) -> List[Dict[str, str]]:
        """List skill summaries for agent discovery"""
        return [
            {"name": s.name, "description": s.description}
            for s in self._skills.values()
        ]

    def list_all_tools(self) -> List[SkillTool]:
        """List all tools from all skills"""
        tools = []
        for skill in self._skills.values():
            tools.extend(skill.tools)
        return tools

    def find_skill_for_tool(self, tool_name: str) -> Optional[str]:
        """Find which skill provides a specific tool"""
        for skill in self._skills.values():
            if any(t.name == tool_name for t in skill.tools):
                return skill.name
        return None

    def clear(self):
        """Clear all registered skills"""
        self._skills = {}
        self._skill_instances = {}
        logger.info("Skill registry cleared")


def create_skill_manifest(
    name: str,
    description: str,
    version: str = "1.0.0",
    category: str = "general",
    tools: Optional[List[SkillTool]] = None,
) -> SkillManifest:
    """Helper to create a skill manifest"""
    return SkillManifest(
        name=name,
        description=description,
        version=version,
        category=category,
        tools=tools or [],
    )


def register_skill(manifest: SkillManifest, instance: Any = None):
    """Register a skill with the global registry"""
    registry = SkillRegistry()
    registry.register(manifest, instance)


def load_skills_from_directory(directory: str) -> List[SkillManifest]:
    """Dynamically load all skills from a directory"""
    manifests = []

    if not os.path.exists(directory):
        logger.warning(f"Skills directory not found: {directory}")
        return manifests

    for filename in os.listdir(directory):
        if filename.startswith("_") or filename.startswith("."):
            continue

        module_path = os.path.join(directory, filename)

        if filename.endswith(".py"):
            module_name = filename[:-3]
        elif os.path.isdir(module_path):
            module_name = filename
        else:
            continue

        # Skip skill_registry.py to avoid issues
        if module_name == "skill_registry":
            continue

        try:
            if filename.endswith(".py"):
                spec = importlib.util.spec_from_file_location(
                    module_name, os.path.join(directory, f"{module_name}.py")
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    if hasattr(module, "SKILL_MANIFEST"):
                        manifest = module.SKILL_MANIFEST
                        instance = None
                        if hasattr(module, "get_skill_instance"):
                            try:
                                instance = module.get_skill_instance()
                            except Exception:
                                pass
                        register_skill(manifest, instance)
                        manifests.append(manifest)
                        logger.info(f"Loaded skill: {manifest.name}")

                    elif hasattr(module, "register_skill"):
                        module.register_skill()
                        logger.info(f"Registered skill from: {module_name}")

        except Exception as e:
            logger.warning(f"Failed to load skill from {filename}: {e}")
            continue

    # Manually register Chrome DevTools skill if it wasn't loaded dynamically
    chrome_skill_names = ["chrome_devtools_skill", "chrome_devtools"]
    chrome_loaded = any(
        s in manifests for m in manifests for s in chrome_skill_names if m.name in s
    )

    if not chrome_loaded:
        try:
            from skills.chrome_devtools_skill import SKILL_MANIFEST, get_skill_instance

            instance = None
            try:
                instance = get_skill_instance()
            except Exception:
                pass
            register_skill(SKILL_MANIFEST, instance)
            manifests.append(SKILL_MANIFEST)
            logger.info(f"Registered Chrome DevTools skill: {SKILL_MANIFEST.name}")
        except Exception as e:
            logger.warning(f"Could not register Chrome DevTools skill: {e}")

    return manifests


registry = SkillRegistry()


def get_registry() -> SkillRegistry:
    """Get the global skill registry"""
    return registry


def init_skills():
    """Initialize all skills from the skills directory"""
    skills_dir = os.path.dirname(__file__)
    return load_skills_from_directory(skills_dir)
