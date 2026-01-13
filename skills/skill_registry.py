import yaml
import logging
import os
import re
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


@dataclass
class SkillTool:
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    category: str = "general"
    read_only: bool = False


@dataclass
class SkillManifest:
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
        self._skills[manifest.name] = manifest
        if instance:
            self._skill_instances[manifest.name] = instance
        logger.info(f"Registered skill: {manifest.name} v{manifest.version}")

    def get_manifest(self, skill_name: str) -> Optional[SkillManifest]:
        return self._skills.get(skill_name)

    def get_skill_instance(self, skill_name: str) -> Optional[Any]:
        return self._skill_instances.get(skill_name)

    def list_skills(self) -> List[SkillManifest]:
        return list(self._skills.values())

    def list_skill_summaries(self) -> List[Dict[str, str]]:
        return [
            {"name": s.name, "description": s.description}
            for s in self._skills.values()
        ]

    def list_all_tools(self) -> List[SkillTool]:
        tools = []
        for skill in self._skills.values():
            tools.extend(skill.tools)
        return tools

    def find_skill_for_tool(self, tool_name: str) -> Optional[str]:
        for skill in self._skills.values():
            if any(t.name == tool_name for t in skill.tools):
                return skill.name
        return None

    def clear(self):
        self._skills = {}
        self._skill_instances = {}
        logger.info("Skill registry cleared")


def parse_claude_skill(skill_path: str) -> Optional[SkillManifest]:
    skill_md_path = os.path.join(skill_path, "SKILL.md")
    if not os.path.exists(skill_md_path):
        return None

    try:
        with open(skill_md_path, "r", encoding="utf-8") as f:
            content = f.read()

        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if not frontmatter_match:
            logger.warning(f"No frontmatter found in {skill_md_path}")
            return None

        frontmatter_yaml = frontmatter_match.group(1)
        metadata = yaml.safe_load(frontmatter_yaml)

        name = metadata.get("name")
        description = metadata.get("description")
        version = metadata.get("version", "1.0.0")

        if not name or not description:
            logger.warning(f"Missing name or description in {skill_md_path}")
            return None

        tools = [
            SkillTool(
                name=name,
                description=description,
                category="claude-skill",
            )
        ]

        return SkillManifest(
            name=name,
            description=description,
            version=version,
            category="claude-skill",
            tools=tools,
            module_path=skill_path,
        )
    except Exception as e:
        logger.error(f"Error parsing Claude skill at {skill_path}: {e}")
        return None


def create_skill_manifest(
    name: str,
    description: str,
    version: str = "1.0.0",
    category: str = "general",
    tools: Optional[List[SkillTool]] = None,
) -> SkillManifest:
    return SkillManifest(
        name=name,
        description=description,
        version=version,
        category=category,
        tools=tools or [],
    )


def register_skill(manifest: SkillManifest, instance: Any = None):
    registry = SkillRegistry()
    registry.register(manifest, instance)


def load_skills_from_directory(directory: str) -> List[SkillManifest]:
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

            elif os.path.isdir(module_path):
                manifest = parse_claude_skill(module_path)
                if manifest:
                    register_skill(manifest)
                    manifests.append(manifest)
                    logger.info(f"Loaded Claude skill: {manifest.name}")
                else:
                    try:
                        spec = importlib.util.spec_from_file_location(
                            module_name, os.path.join(module_path, "__init__.py")
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
                                logger.info(f"Loaded directory skill: {manifest.name}")
                    except Exception as e:
                        logger.debug(f"Failed to load directory {module_name} as module: {e}")

        except Exception as e:
            logger.warning(f"Failed to load skill from {filename}: {e}")
            continue

    return manifests


registry = SkillRegistry()


def get_registry() -> SkillRegistry:
    return registry


def init_skills():
    skills_dir = os.path.dirname(__file__)
    all_manifests = load_skills_from_directory(skills_dir)
    claude_skills_dir = os.path.join(skills_dir, "claude-skills", "skills")
    if os.path.exists(claude_skills_dir):
        claude_manifests = load_skills_from_directory(claude_skills_dir)
        all_manifests.extend(claude_manifests)
    return all_manifests
