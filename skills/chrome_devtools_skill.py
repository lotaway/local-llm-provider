"""Chrome DevTools Skill - MCP-based browser automation capability for agents

This skill provides a Python interface to the Chrome DevTools MCP server,
allowing agents to control Chrome browser for automation, testing, and debugging tasks.

Usage:
    from skills import ChromeDevToolsSkill

    skill = ChromeDevToolsSkill()
    await skill.initialize()

    # Take a screenshot
    result = await skill.execute_tool("take_screenshot", {})

    # Navigate to a page
    result = await skill.execute_tool("navigate_page", {"type": "url", "url": "https://example.com"})

    # Take a snapshot to see page elements
    result = await skill.execute_tool("take_snapshot", {})
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, TextIO
from dataclasses import dataclass, field

from .skill_registry import SkillManifest, SkillTool, register_skill

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """MCP Tool definition"""

    name: str
    description: str
    category: str
    read_only: bool
    parameters: Dict[str, Any]


@dataclass
class ToolResult:
    """Result from tool execution"""

    success: bool
    data: Any = None
    error: Optional[str] = None
    content: Optional[List[Dict]] = None


class ChromeDevToolsSkill:
    """Skill for controlling Chrome browser via MCP protocol

    This skill wraps the chrome-devtools-mcp server and provides
    a Python interface for agents to execute browser automation tasks.
    """

    # Tool definitions matching chrome-devtools-mcp
    TOOLS = [
        # Input Automation (8 tools)
        ToolDefinition(
            name="click",
            description="Clicks on the provided element",
            category="input",
            read_only=False,
            parameters={
                "uid": {
                    "type": "string",
                    "description": "The uid of an element on the page from the page content snapshot",
                },
                "dblClick": {
                    "type": "boolean",
                    "description": "Set to true for double clicks",
                    "optional": True,
                },
            },
        ),
        ToolDefinition(
            name="click_at",
            description="Clicks at the provided coordinates",
            category="input",
            read_only=False,
            parameters={
                "x": {"type": "number", "description": "The x coordinate"},
                "y": {"type": "number", "description": "The y coordinate"},
                "dblClick": {
                    "type": "boolean",
                    "description": "Set to true for double clicks",
                    "optional": True,
                },
            },
        ),
        ToolDefinition(
            name="hover",
            description="Hover over the provided element",
            category="input",
            read_only=False,
            parameters={
                "uid": {
                    "type": "string",
                    "description": "The uid of an element on the page from the page content snapshot",
                }
            },
        ),
        ToolDefinition(
            name="fill",
            description="Type text into an input, text area or select an option from a <select> element",
            category="input",
            read_only=False,
            parameters={
                "uid": {"type": "string", "description": "The uid of an element"},
                "value": {"type": "string", "description": "The value to fill in"},
            },
        ),
        ToolDefinition(
            name="fill_form",
            description="Fill out multiple form elements at once",
            category="input",
            read_only=False,
            parameters={
                "elements": {
                    "type": "array",
                    "description": "Elements from snapshot to fill out",
                    "items": {
                        "type": "object",
                        "properties": {
                            "uid": {"type": "string"},
                            "value": {"type": "string"},
                        },
                    },
                }
            },
        ),
        ToolDefinition(
            name="drag",
            description="Drag an element onto another element",
            category="input",
            read_only=False,
            parameters={
                "from_uid": {
                    "type": "string",
                    "description": "The uid of the element to drag",
                },
                "to_uid": {
                    "type": "string",
                    "description": "The uid of the element to drop into",
                },
            },
        ),
        ToolDefinition(
            name="upload_file",
            description="Upload a file through a provided element",
            category="input",
            read_only=False,
            parameters={
                "uid": {
                    "type": "string",
                    "description": "The uid of the file input element",
                },
                "filePath": {
                    "type": "string",
                    "description": "The local path of the file to upload",
                },
            },
        ),
        ToolDefinition(
            name="press_key",
            description="Press a key or key combination",
            category="input",
            read_only=False,
            parameters={
                "key": {
                    "type": "string",
                    "description": "A key or combination (e.g., 'Enter', 'Control+A')",
                }
            },
        ),
        # Navigation (6 tools)
        ToolDefinition(
            name="list_pages",
            description="Get a list of pages open in the browser",
            category="navigation",
            read_only=True,
            parameters={},
        ),
        ToolDefinition(
            name="select_page",
            description="Select a page as a context for future tool calls",
            category="navigation",
            read_only=True,
            parameters={
                "pageId": {
                    "type": "number",
                    "description": "The ID of the page to select",
                },
                "bringToFront": {
                    "type": "boolean",
                    "description": "Whether to focus the page",
                    "optional": True,
                },
            },
        ),
        ToolDefinition(
            name="close_page",
            description="Closes the page by its index",
            category="navigation",
            read_only=False,
            parameters={
                "pageId": {
                    "type": "number",
                    "description": "The ID of the page to close",
                }
            },
        ),
        ToolDefinition(
            name="new_page",
            description="Creates a new page",
            category="navigation",
            read_only=False,
            parameters={
                "url": {"type": "string", "description": "URL to load in a new page"},
                "timeout": {
                    "type": "number",
                    "description": "Maximum wait time in milliseconds",
                    "optional": True,
                },
            },
        ),
        ToolDefinition(
            name="navigate_page",
            description="Navigates the currently selected page to a URL",
            category="navigation",
            read_only=False,
            parameters={
                "type": {
                    "type": "string",
                    "enum": ["url", "back", "forward", "reload"],
                    "optional": True,
                },
                "url": {
                    "type": "string",
                    "description": "Target URL",
                    "optional": True,
                },
                "ignoreCache": {
                    "type": "boolean",
                    "description": "Whether to ignore cache on reload",
                    "optional": True,
                },
                "timeout": {
                    "type": "number",
                    "description": "Maximum wait time in milliseconds",
                    "optional": True,
                },
            },
        ),
        ToolDefinition(
            name="wait_for",
            description="Wait for the specified text to appear on the selected page",
            category="navigation",
            read_only=True,
            parameters={
                "text": {"type": "string", "description": "Text to appear on the page"},
                "timeout": {
                    "type": "number",
                    "description": "Maximum wait time in milliseconds",
                    "optional": True,
                },
            },
        ),
        # Emulation (2 tools)
        ToolDefinition(
            name="emulate",
            description="Emulates various features on the selected page",
            category="emulation",
            read_only=False,
            parameters={
                "networkConditions": {
                    "type": "string",
                    "description": "Throttle network (e.g., 'Fast 3G')",
                    "optional": True,
                },
                "cpuThrottlingRate": {
                    "type": "number",
                    "description": "CPU slowdown factor (1-20)",
                    "optional": True,
                },
                "geolocation": {
                    "type": "object",
                    "description": "Geolocation to emulate",
                    "properties": {
                        "latitude": {
                            "type": "number",
                            "description": "Latitude between -90 and 90",
                        },
                        "longitude": {
                            "type": "number",
                            "description": "Longitude between -180 and 180",
                        },
                    },
                    "optional": True,
                },
            },
        ),
        ToolDefinition(
            name="resize_page",
            description="Resizes the selected page's window",
            category="emulation",
            read_only=False,
            parameters={
                "width": {"type": "number", "description": "Page width"},
                "height": {"type": "number", "description": "Page height"},
            },
        ),
        # Performance (3 tools)
        ToolDefinition(
            name="performance_start_trace",
            description="Starts a performance trace recording on the selected page",
            category="performance",
            read_only=True,
            parameters={
                "reload": {
                    "type": "boolean",
                    "description": "Whether to reload the page after starting trace",
                },
                "autoStop": {
                    "type": "boolean",
                    "description": "Whether to automatically stop after 5 seconds",
                },
            },
        ),
        ToolDefinition(
            name="performance_stop_trace",
            description="Stops the active performance trace recording",
            category="performance",
            read_only=True,
            parameters={},
        ),
        ToolDefinition(
            name="performance_analyze_insight",
            description="Provides detailed information on a specific Performance Insight",
            category="performance",
            read_only=True,
            parameters={
                "insightSetId": {
                    "type": "string",
                    "description": "The id for the specific insight set",
                },
                "insightName": {
                    "type": "string",
                    "description": "The name of the Insight",
                },
            },
        ),
        # Network (2 tools)
        ToolDefinition(
            name="list_network_requests",
            description="List all requests for the currently selected page",
            category="network",
            read_only=True,
            parameters={
                "pageSize": {
                    "type": "number",
                    "description": "Maximum number of requests to return",
                    "optional": True,
                },
                "pageIdx": {
                    "type": "number",
                    "description": "Page number (0-based)",
                    "optional": True,
                },
                "resourceTypes": {
                    "type": "array",
                    "description": "Filter by resource types",
                    "optional": True,
                },
                "includePreservedRequests": {
                    "type": "boolean",
                    "description": "Include preserved requests",
                    "optional": True,
                },
            },
        ),
        ToolDefinition(
            name="get_network_request",
            description="Gets a network request by its reqid",
            category="network",
            read_only=True,
            parameters={
                "reqid": {
                    "type": "number",
                    "description": "The reqid of the network request",
                    "optional": True,
                }
            },
        ),
        # Debugging (5 tools)
        ToolDefinition(
            name="take_screenshot",
            description="Take a screenshot of the page or element",
            category="debugging",
            read_only=False,
            parameters={
                "format": {
                    "type": "string",
                    "enum": ["png", "jpeg", "webp"],
                    "description": "Image format",
                    "optional": True,
                },
                "quality": {
                    "type": "number",
                    "description": "Compression quality (0-100)",
                    "optional": True,
                },
                "uid": {
                    "type": "string",
                    "description": "Element uid for element screenshot",
                    "optional": True,
                },
                "fullPage": {
                    "type": "boolean",
                    "description": "Take screenshot of full page",
                    "optional": True,
                },
                "filePath": {
                    "type": "string",
                    "description": "Path to save the screenshot",
                    "optional": True,
                },
            },
        ),
        ToolDefinition(
            name="take_snapshot",
            description="Take a text snapshot of the currently selected page based on the a11y tree",
            category="debugging",
            read_only=False,
            parameters={
                "verbose": {
                    "type": "boolean",
                    "description": "Include all available information",
                    "optional": True,
                },
                "filePath": {
                    "type": "string",
                    "description": "Path to save the snapshot",
                    "optional": True,
                },
            },
        ),
        ToolDefinition(
            name="list_console_messages",
            description="List all console messages for the currently selected page",
            category="debugging",
            read_only=True,
            parameters={
                "pageSize": {
                    "type": "number",
                    "description": "Maximum number of messages",
                    "optional": True,
                },
                "pageIdx": {
                    "type": "number",
                    "description": "Page number (0-based)",
                    "optional": True,
                },
                "types": {
                    "type": "array",
                    "description": "Filter by message types",
                    "optional": True,
                },
                "includePreservedMessages": {
                    "type": "boolean",
                    "description": "Include preserved messages",
                    "optional": True,
                },
            },
        ),
        ToolDefinition(
            name="get_console_message",
            description="Gets a console message by its ID",
            category="debugging",
            read_only=True,
            parameters={
                "msgid": {
                    "type": "number",
                    "description": "The msgid of a console message",
                }
            },
        ),
        ToolDefinition(
            name="evaluate_script",
            description="Evaluate a JavaScript function inside the currently selected page",
            category="debugging",
            read_only=False,
            parameters={
                "function": {
                    "type": "string",
                    "description": "A JavaScript function declaration",
                },
                "args": {
                    "type": "array",
                    "description": "Arguments to pass to the function",
                    "optional": True,
                },
            },
        ),
    ]

    def __init__(
        self,
        mcp_server_path: str = None,
        browser_url: str = None,
        headless: bool = False,
        channel: str = "stable",
        isolated: bool = False,
        timeout: int = 30,
    ):
        """Initialize Chrome DevTools skill

        Args:
            mcp_server_path: Path to the MCP server command or None to use npx
            browser_url: Connect to a running Chrome instance (e.g., http://127.0.0.1:9222)
            headless: Whether to run in headless mode
            channel: Chrome channel (stable, canary, beta, dev)
            isolated: Whether to use isolated temporary user data directory
            timeout: Default timeout for tool execution in seconds
        """
        self.mcp_server_path = mcp_server_path or "npx"
        self.mcp_args = ["-y", "chrome-devtools-mcp@latest"]
        self.browser_url = browser_url
        self.headless = headless
        self.channel = channel
        self.isolated = isolated
        self.timeout = timeout

        self.process: Optional[subprocess.Popen[str]] = None
        self._initialized = False
        self._tools_cache: Optional[List[ToolDefinition]] = None

    async def initialize(self) -> bool:
        """Initialize the MCP server connection

        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True

        try:
            # Build server arguments
            args = [self.mcp_server_path] + self.mcp_args

            if self.browser_url:
                args.extend(["--browser-url", self.browser_url])
            if self.headless:
                args.append("--headless")
            if self.channel and self.channel != "stable":
                args.extend(["--channel", self.channel])
            if self.isolated:
                args.append("--isolated")

            # Start MCP server process
            logger.info(f"Starting Chrome DevTools MCP server: {' '.join(args)}")

            self.process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait for server to initialize
            await asyncio.sleep(2)

            # Check if process is still running
            if self.process.poll() is not None:
                stderr = self.process.stderr.read()
                logger.error(f"MCP server failed to start: {stderr}")
                return False

            # Initialize by sending capabilities request
            success = await self._send_init_request()

            if success:
                self._initialized = True
                logger.info("Chrome DevTools MCP server initialized successfully")

            return success

        except Exception as e:
            logger.error(f"Failed to initialize Chrome DevTools skill: {e}")
            return False

    async def _send_init_request(self) -> bool:
        """Send MCP initialize request"""
        try:
            # MCP uses JSON-RPC 2.0
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "local-llm-provider", "version": "1.0.0"},
                },
            }

            response = await self._send_message(init_request)
            if response and "result" in response:
                logger.info(
                    f"MCP server capabilities: {response['result'].get('capabilities', {})}"
                )
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to send init request: {e}")
            return False

    async def _send_message(self, message: Dict) -> Dict:
        """Send a JSON-RPC message and get response"""
        if not self.process or self.process.poll() is not None:
            raise RuntimeError("MCP server process is not running")

        message_str = json.dumps(message) + "\n"

        try:
            # Write to stdin
            self.process.stdin.write(message_str)
            await self.process.stdin.drain()

            # Read response
            response_line = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.process.stdout.readline
            )

            if not response_line:
                raise RuntimeError("No response from MCP server")

            return json.loads(response_line.strip())

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise

    async def list_tools(self) -> List[ToolDefinition]:
        """List available tools from the MCP server

        Returns:
            List of tool definitions
        """
        if self._tools_cache:
            return self._tools_cache

        try:
            request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

            response = await self._send_message(request)

            if "result" in response and "tools" in response["result"]:
                # Parse and cache tools
                tools_data = response["result"]["tools"]
                self._tools_cache = [
                    ToolDefinition(
                        name=t["name"],
                        description=t.get("description", ""),
                        category=t.get("annotations", {}).get("category", "unknown"),
                        read_only=t.get("annotations", {}).get("readOnlyHint", False),
                        parameters=t.get("inputSchema", {}).get("properties", {}),
                    )
                    for t in tools_data
                ]

            return self._tools_cache or self.TOOLS

        except Exception as e:
            logger.warning(
                f"Failed to list tools from server, using cached definitions: {e}"
            )
            return self.TOOLS

    async def execute_tool(
        self, tool_name: str, params: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Execute a tool on the MCP server

        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool

        Returns:
            ToolResult with success status and data
        """
        if not self._initialized:
            if not await self.initialize():
                return ToolResult(
                    success=False, error="Failed to initialize MCP server"
                )

        try:
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": params or {}},
            }

            response = await self._send_message(request)

            if "error" in response:
                return ToolResult(
                    success=False,
                    error=response["error"].get("message", str(response["error"])),
                )

            # Parse content from result
            content = response.get("result", {}).get("content", [])
            text_content = self._parse_content(content)

            return ToolResult(success=True, data=text_content, content=content)

        except Exception as e:
            logger.error(f"Failed to execute tool {tool_name}: {e}")
            return ToolResult(success=False, error=str(e))

    def _parse_content(self, content: List[Dict]) -> str:
        """Parse content from tool response"""
        if not content:
            return ""

        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "image":
                # Handle base64 image data
                data = item.get("data", "")
                parts.append(f"[Image: {len(data)} bytes of base64 data]")
            elif item.get("type") == "resource":
                # Handle resource (like screenshots)
                resource = item.get("resource", {})
                if isinstance(resource, dict):
                    parts.append(f"[Resource: {resource.get('uri', 'unknown')}]")

        return "\n".join(parts)

    def _next_id(self) -> int:
        """Generate next request ID"""
        if not hasattr(self, "_request_id"):
            self._request_id = 10
        self._request_id += 1
        return self._request_id

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names

        Returns:
            List of tool names
        """
        return [tool.name for tool in self.TOOLS]

    def get_tool_by_name(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name

        Args:
            tool_name: Name of the tool

        Returns:
            ToolDefinition or None if not found
        """
        for tool in self.TOOLS:
            if tool.name == tool_name:
                return tool
        return None

    def get_tools_by_category(self, category: str) -> List[ToolDefinition]:
        """Get tools by category

        Args:
            category: Tool category (input, navigation, emulation, performance, network, debugging)

        Returns:
            List of tools in the category
        """
        return [tool for tool in self.TOOLS if tool.category == category]

    async def close(self):
        """Close the MCP server connection"""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error closing MCP process: {e}")
                self.process.kill()
            self.process = None
            self._initialized = False
            self._tools_cache = None
            logger.info("Chrome DevTools MCP server connection closed")

    def is_available(self) -> bool:
        """Check if the skill is available (can connect to MCP server)

        Returns:
            True if available, False otherwise
        """
        return self._initialized and self.process and self.process.poll() is None

    # Convenience methods for common tasks

    async def navigate_to(self, url: str, timeout: int = None) -> ToolResult:
        """Navigate to a URL

        Args:
            url: Target URL
            timeout: Optional timeout in milliseconds

        Returns:
            ToolResult
        """
        params = {"type": "url", "url": url}
        if timeout:
            params["timeout"] = timeout
        return await self.execute_tool("navigate_page", params)

    async def take_screenshot(
        self, format: str = "png", full_page: bool = False
    ) -> ToolResult:
        """Take a screenshot of the current page

        Args:
            format: Image format (png, jpeg, webp)
            full_page: Whether to capture full page

        Returns:
            ToolResult
        """
        return await self.execute_tool(
            "take_screenshot", {"format": format, "fullPage": full_page}
        )

    async def get_page_snapshot(self, verbose: bool = False) -> ToolResult:
        """Get a snapshot of the current page

        Args:
            verbose: Whether to include all available information

        Returns:
            ToolResult with page content
        """
        return await self.execute_tool("take_snapshot", {"verbose": verbose})

    async def click_element(self, uid: str, dbl_click: bool = False) -> ToolResult:
        """Click on an element by its UID

        Args:
            uid: Element UID from snapshot
            dbl_click: Whether to double click

        Returns:
            ToolResult
        """
        return await self.execute_tool("click", {"uid": uid, "dblClick": dbl_click})

    async def fill_input(self, uid: str, value: str) -> ToolResult:
        """Fill an input field

        Args:
            uid: Input element UID
            value: Value to fill

        Returns:
            ToolResult
        """
        return await self.execute_tool("fill", {"uid": uid, "value": value})

    async def start_performance_trace(self, auto_stop: bool = True) -> ToolResult:
        """Start a performance trace

        Args:
            auto_stop: Whether to automatically stop after 5 seconds

        Returns:
            ToolResult
        """
        return await self.execute_tool(
            "performance_start_trace", {"reload": False, "autoStop": auto_stop}
        )

    async def stop_performance_trace(self) -> ToolResult:
        """Stop the performance trace

        Returns:
            ToolResult
        """
        return await self.execute_tool("performance_stop_trace", {})

    async def get_console_messages(
        self, types: Optional[List[str]] = None
    ) -> ToolResult:
        """Get console messages

        Args:
            types: Filter by message types (log, debug, info, error, warn, etc.)

        Returns:
            ToolResult
        """
        params = {}
        if types:
            params["types"] = types
        return await self.execute_tool("list_console_messages", params)

    async def list_network_requests(self) -> ToolResult:
        """List network requests

        Returns:
            ToolResult
        """
        return await self.execute_tool("list_network_requests", {})

    async def evaluate_js(
        self, function: str, args: Optional[List[Dict[str, Any]]] = None
    ) -> ToolResult:
        """Evaluate JavaScript code

        Args:
            function: JavaScript function to evaluate
            args: Arguments to pass to the function

        Returns:
            ToolResult
        """
        params = {"function": function}
        if args:
            params["args"] = args
        return await self.execute_tool("evaluate_script", params)

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        return False


# Alias for convenience
ChromeSkill = ChromeDevToolsSkill


# Skill Manifest for dynamic registration
SKILL_MANIFEST = SkillManifest(
    name="chrome_devtools",
    description="Browser automation capability for Chrome DevTools control. Provides tools for page navigation, element interaction, screenshot capture, performance tracing, network monitoring, and JavaScript execution in Chrome browser.",
    version="1.0.0",
    category="browser_automation",
    tools=[
        # Navigation tools
        SkillTool(
            name="navigate_page",
            description="Navigate the currently selected page to a URL, back, forward, or reload",
            category="navigation",
        ),
        SkillTool(
            name="list_pages",
            description="Get a list of pages open in the browser",
            category="navigation",
        ),
        SkillTool(
            name="select_page",
            description="Select a page as a context for future tool calls",
            category="navigation",
        ),
        SkillTool(
            name="close_page",
            description="Closes the page by its index",
            category="navigation",
        ),
        SkillTool(
            name="new_page",
            description="Creates a new page with optional URL",
            category="navigation",
        ),
        SkillTool(
            name="wait_for",
            description="Wait for the specified text to appear on the selected page",
            category="navigation",
        ),
        # Input tools
        SkillTool(
            name="click",
            description="Click on a page element by its UID from snapshot",
            category="input",
        ),
        SkillTool(
            name="click_at",
            description="Click at specific X,Y coordinates",
            category="input",
        ),
        SkillTool(
            name="hover",
            description="Hover over a page element",
            category="input",
        ),
        SkillTool(
            name="fill",
            description="Type text into an input, text area or select an option from a <select> element",
            category="input",
        ),
        SkillTool(
            name="fill_form",
            description="Fill out multiple form elements at once",
            category="input",
        ),
        SkillTool(
            name="drag",
            description="Drag an element onto another element",
            category="input",
        ),
        SkillTool(
            name="upload_file",
            description="Upload a file through a file input element",
            category="input",
        ),
        SkillTool(
            name="press_key",
            description="Press a key or key combination",
            category="input",
        ),
        # Debugging tools
        SkillTool(
            name="take_screenshot",
            description="Take a screenshot of the page or a specific element",
            category="debugging",
        ),
        SkillTool(
            name="take_snapshot",
            description="Take a text snapshot of the currently selected page based on the a11y tree",
            category="debugging",
        ),
        SkillTool(
            name="list_console_messages",
            description="List all console messages for the currently selected page",
            category="debugging",
        ),
        SkillTool(
            name="get_console_message",
            description="Gets a console message by its ID",
            category="debugging",
        ),
        SkillTool(
            name="evaluate_script",
            description="Evaluate a JavaScript function inside the currently selected page",
            category="debugging",
        ),
        # Performance tools
        SkillTool(
            name="performance_start_trace",
            description="Starts a performance trace recording on the selected page",
            category="performance",
        ),
        SkillTool(
            name="performance_stop_trace",
            description="Stops the active performance trace recording",
            category="performance",
        ),
        SkillTool(
            name="performance_analyze_insight",
            description="Provides detailed information on a specific Performance Insight",
            category="performance",
        ),
        # Network tools
        SkillTool(
            name="list_network_requests",
            description="List all requests for the currently selected page",
            category="network",
        ),
        SkillTool(
            name="get_network_request",
            description="Gets a network request by its reqid",
            category="network",
        ),
        # Emulation tools
        SkillTool(
            name="emulate",
            description="Emulate various features on the selected page (network conditions, geolocation, CPU throttling)",
            category="emulation",
        ),
        SkillTool(
            name="resize_page",
            description="Resize the selected page's window",
            category="emulation",
        ),
    ],
)


def get_skill_instance() -> ChromeDevToolsSkill:
    """Get a new ChromeDevToolsSkill instance"""
    return ChromeDevToolsSkill()
