"""Example: Using Chrome DevTools Skill with existing agents

This example demonstrates how to integrate the Chrome DevTools skill
with the existing agent system for browser automation tasks.
"""

import asyncio
from skills import ChromeDevToolsSkill


async def basic_example():
    """Basic usage of Chrome DevTools skill"""
    async with ChromeDevToolsSkill() as skill:
        # List available tools
        tools = skill.get_available_tools()
        print(f"Available tools: {len(tools)}")
        print(", ".join(tools[:5]), "...")

        # Navigate to a page
        result = await skill.navigate_to("https://example.com")
        print(f"Navigate result: {result.success}")

        # Take a screenshot
        result = await skill.take_screenshot()
        print(f"Screenshot result: {result.success}")

        # Get page snapshot to see elements
        result = await skill.get_page_snapshot()
        print(f"Snapshot result: {result.success}")
        print(f"Snapshot preview: {result.data[:200] if result.data else 'No data'}...")


async def automation_example():
    """Example: Automated form filling and interaction"""
    async with ChromeDevToolsSkill(headless=True) as skill:
        # Navigate to a login page
        await skill.navigate_to("https://example.com/login")

        # Get page snapshot to find form elements
        snapshot = await skill.get_page_snapshot()

        # Assuming we found the username and password fields
        # In real usage, you'd parse the snapshot to get element UIDs
        # await skill.fill_input("uid-123", "test@example.com")
        # await skill.fill_input("uid-456", "password123")

        # Click login button
        # await skill.click_element("uid-789")

        # Wait for navigation
        # await skill.execute_tool("wait_for", {"text": "Welcome"})

        print("Automation completed")


async def performance_example():
    """Example: Performance tracing workflow"""
    async with ChromeDevToolsSkill() as skill:
        # Start performance trace
        await skill.start_performance_trace(auto_stop=False)

        # Navigate to the page to analyze
        await skill.navigate_to("https://example.com")

        # Let the page load and interact...
        await asyncio.sleep(2)

        # Stop trace and get results
        result = await skill.stop_performance_trace()
        print(f"Performance trace: {result.success}")
        if result.data:
            print(f"Trace data: {result.data[:500]}...")


async def debugging_example():
    """Example: Debugging a page with console and network tools"""
    async with ChromeDevToolsSkill() as skill:
        # Navigate to page
        await skill.navigate_to("https://example.com")

        # Get console messages
        console_result = await skill.get_console_messages(types=["error", "warn"])
        print(f"Console messages: {console_result.data}")

        # List network requests
        network_result = await skill.list_network_requests()
        print(f"Network requests: {network_result.data}")


async def main():
    """Run all examples"""
    print("=" * 50)
    print("Chrome DevTools Skill - Example Usage")
    print("=" * 50)

    print("\n1. Basic Example")
    print("-" * 30)
    try:
        await basic_example()
    except Exception as e:
        print(f"Error in basic example: {e}")

    print("\n2. Performance Example")
    print("-" * 30)
    try:
        await performance_example()
    except Exception as e:
        print(f"Error in performance example: {e}")

    print("\n3. Debugging Example")
    print("-" * 30)
    try:
        await debugging_example()
    except Exception as e:
        print(f"Error in debugging example: {e}")


if __name__ == "__main__":
    asyncio.run(main())
