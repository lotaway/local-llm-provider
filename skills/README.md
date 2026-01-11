# Skills Package

Dynamic skill loading and management system for agent-based applications.

## Overview

This package provides a registry-based system for dynamically loading and managing skills. Skills are self-contained modules that expose tools/capabilities to agents at runtime.

## Architecture

```
skills/
├── __init__.py              # Package exports and initialization
├── skill_registry.py        # Core registry for skill management
├── chrome_devtools_skill.py # Chrome DevTools skill implementation
├── example_*.py             # Usage examples
└── README.md               # This file
```

## Key Concepts

### SkillManifest
A metadata structure describing a skill:
```python
@dataclass
class SkillManifest:
    name: str              # Unique skill identifier
    description: str       # Human-readable description
    version: str = "1.0.0"
    category: str = "general"
    tools: List[SkillTool] = field(default_factory=list)
```

### SkillTool
A tool/command provided by a skill:
```python
@dataclass
class SkillTool:
    name: str              # Tool identifier
    description: str       # Tool description
    category: str = "general"
    read_only: bool = False
```

### SkillRegistry
Singleton registry for managing all skills:
- `list_skills()` - Get all registered skill manifests
- `list_skill_summaries()` - Get name/description pairs for agent discovery
- `list_all_tools()` - Get all tools from all skills
- `find_skill_for_tool(tool_name)` - Find which skill provides a tool
- `get_skill_instance(skill_name)` - Get skill instance for execution

## Usage

### Basic Initialization

```python
from skills import init_skills, registry

# Initialize all skills from the skills directory
init_skills()

# List available skills
for skill in registry.list_skills():
    print(f"{skill.name}: {skill.description}")
```

### Dynamic Tool Discovery

```python
from skills import registry

# Get all tools from all skills
tools = registry.list_all_tools()

# Filter tools by category
browser_tools = registry.get_tools_by_category("navigation")

# Find which skill provides a tool
skill_name = registry.find_skill_for_tool("navigate_page")
```

### Integrating with Agents

```python
from skills import registry

class PlanningAgent:
    async def plan(self, user_query: str) -> Dict:
        # Get available skills info for LLM
        skills_info = []
        for skill in registry.list_skills():
            skills_info.append(f"- {skill.name}: {skill.description}")
            for tool in skill.tools:
                skills_info.append(f"  - {tool.name}: {tool.description}")
        
        # Use with LLM for intelligent planning...
        pass
```

## Adding a New Skill

1. Create a new Python file in the `skills/` directory:

```python
# skills/my_skill.py
from skills import SkillManifest, SkillTool, register_skill

TOOLS = [
    SkillTool(
        name="my_tool",
        description="Does something useful",
        category="utilities",
    ),
]

MANIFEST = SkillManifest(
    name="my_skill",
    description="A useful skill for doing things",
    version="1.0.0",
    category="utilities",
    tools=TOOLS,
)

def get_skill_instance():
    """Return a skill instance for execution"""
    return MySkill()

class MySkill:
    async def my_tool(self, param: str) -> str:
        """Execute the tool"""
        return f"Result: {param}"
```

2. The skill will be automatically discovered and registered by `init_skills()`

## Chrome DevTools Skill

The included `chrome_devtools_skill.py` provides browser automation capabilities:

### Categories

- **Navigation**: `navigate_page`, `list_pages`, `select_page`, `close_page`, `new_page`, `wait_for`
- **Input**: `click`, `click_at`, `hover`, `fill`, `fill_form`, `drag`, `upload_file`, `press_key`
- **Debugging**: `take_screenshot`, `take_snapshot`, `list_console_messages`, `get_console_message`, `evaluate_script`
- **Performance**: `performance_start_trace`, `performance_stop_trace`, `performance_analyze_insight`
- **Network**: `list_network_requests`, `get_network_request`
- **Emulation**: `emulate`, `resize_page`

### Usage Example

```python
import asyncio
from skills.chrome_devtools_skill import ChromeDevToolsSkill

async def example():
    async with ChromeDevToolsSkill() as skill:
        # Navigate to a page
        await skill.navigate_to("https://example.com")
        
        # Take a screenshot
        result = await skill.take_screenshot()
        
        # Get page snapshot
        snapshot = await skill.get_page_snapshot()

asyncio.run(example())
```

## Examples

See `example_dynamic_skill_loading.py` for a complete demonstration of the skill loading system.

See `example_chrome_skill_usage.py` for Chrome DevTools skill usage examples.

## API Reference

### skill_registry.py

- `SkillRegistry` - Registry class for managing skills
- `SkillManifest` - Dataclass for skill metadata
- `SkillTool` - Dataclass for tool metadata
- `create_skill_manifest()` - Helper to create skill manifests
- `register_skill()` - Register a skill with global registry
- `init_skills()` - Load all skills from skills directory
- `get_registry()` - Get the global registry instance

### chrome_devtools_skill.py

- `ChromeDevToolsSkill` - Main skill class for browser automation
- `ChromeSkill` - Alias for ChromeDevToolsSkill
- `SKILL_MANIFEST` - Pre-defined skill manifest
- `get_skill_instance()` - Get a skill instance

