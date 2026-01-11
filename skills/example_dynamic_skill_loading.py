"""Example: Dynamic Skill Loading and Usage

This example demonstrates how to:
1. Load skills dynamically from the skills directory
2. Discover available skills and tools
3. Use skills with agents for intelligent task planning
"""

import asyncio
from skills import (
    registry,
    init_skills,
    SkillManifest,
    SkillTool,
    get_registry,
)


async def demonstrate_skill_loading():
    """Demonstrate dynamic skill loading"""
    print("=" * 60)
    print("Dynamic Skill Loading Demo")
    print("=" * 60)

    # Initialize all skills from the skills directory
    print("\n1. Loading skills from directory...")
    manifests = init_skills()
    print(f"   Loaded {len(manifests)} skills:")

    for manifest in manifests:
        print(f"   - {manifest.name} v{manifest.version}")
        print(f"     Category: {manifest.category}")
        print(f"     Description: {manifest.description}")
        print(f"     Tools: {len(manifest.tools)}")

    # List all skills summaries (name + description only)
    print("\n2. Skill summaries for agent discovery:")
    summaries = registry.list_skill_summaries()
    for s in summaries:
        print(f"   - {s['name']}: {s['description']}")

    # List all tools from all skills
    print("\n3. All available tools:")
    tools = registry.list_all_tools()
    for tool in tools:
        print(f"   - [{tool.category}] {tool.name}: {tool.description}")

    # Find which skill provides a specific tool
    print("\n4. Finding skill for a tool:")
    skill_for_navigate = registry.find_skill_for_tool("navigate_page")
    print(f"   Tool 'navigate_page' is provided by: {skill_for_navigate}")

    skill_for_screenshot = registry.find_skill_for_tool("take_screenshot")
    print(f"   Tool 'take_screenshot' is provided by: {skill_for_screenshot}")


async def demonstrate_skill_selection():
    """Demonstrate how an agent would select a skill/tool"""
    print("\n" + "=" * 60)
    print("Skill Selection Demo for Agents")
    print("=" * 60)

    # This is how an intelligent agent would select the right tool
    user_request = "Take a screenshot of example.com"

    print(f"\nUser request: {user_request}")
    print("\nAgent would:")
    print("  1. List all available skills")
    print("  2. Find skills matching the request type (browser automation)")
    print("  3. Select tools from those skills")

    # Find skills matching browser automation
    browser_skills = [
        s
        for s in registry.list_skills()
        if "browser" in s.category.lower() or "chrome" in s.name.lower()
    ]

    print(f"\n  Found {len(browser_skills)} browser automation skills:")
    for skill in browser_skills:
        print(f"    - {skill.name}")
        screenshot_tools = [t for t in skill.tools if "screenshot" in t.name.lower()]
        for tool in screenshot_tools:
            print(f"      Using tool: {tool.name}")

    # Show tool selection process
    print("\n  Selected tool: take_screenshot")
    print("  Parameters needed: url (optional, uses current page if not provided)")


async def demonstrate_manifest_structure():
    """Show the structure of skill manifests"""
    print("\n" + "=" * 60)
    print("Skill Manifest Structure")
    print("=" * 60)

    skills = registry.list_skills()
    if skills:
        manifest = skills[0]
        print(f"\nManifest for '{manifest.name}':")
        print(f"  name: {manifest.name}")
        print(f"  description: {manifest.description}")
        print(f"  version: {manifest.version}")
        print(f"  category: {manifest.category}")
        print(f"  tools: {len(manifest.tools)}")

        print("\n  Tools:")
        for tool in manifest.tools[:3]:  # Show first 3
            print(f"    - name: {tool.name}")
            print(f"      category: {tool.category}")
            print(f"      description: {tool.description}")
            print(f"      read_only: {tool.read_only}")


async def main():
    """Run all demonstrations"""
    print("Skills Package - Dynamic Loading Example")
    print("-" * 60)

    await demonstrate_skill_loading()
    await demonstrate_skill_selection()
    await demonstrate_manifest_structure()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print(
        """
Integration with Agents:
1. Agent receives user query
2. Agent calls registry.list_skill_summaries() to see available skills
3. Agent analyzes query to determine needed skill type
4. Agent selects appropriate skill and tool
5. Agent calls skill.get_skill_instance() to get skill instance
6. Agent executes tool call

This enables:
- Dynamic tool discovery at runtime
- Easy addition of new skills (just add Python files)
- Clear separation between skill definition and usage
- Scalable architecture for multiple skill providers
"""
    )


if __name__ == "__main__":
    asyncio.run(main())
