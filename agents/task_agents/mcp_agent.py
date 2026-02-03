"""MCP Task Agent - MCP tool invocation"""

from typing import Any, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.agent_base import BaseAgent, AgentResult, AgentStatus


class MCPTaskAgent(BaseAgent):
    """Agent for MCP tool invocation"""

    def __init__(self, llm_model, name: str = None):
        super().__init__(llm_model, name)
        self.tools = {}
        self.permission_manager = None
        self._register_default_tools()

    def _register_default_tools(self):
        self.register_tool(
            "read_file", self._read_file_tool, permission_name="mcp.read_file"
        )

    def _read_file_tool(self, query: str, task: str, context: Dict[str, Any]) -> str:
        file_map = context.get("file_map", {})
        target_file_id = None
        for file_id in file_map:
            if file_id in task:
                target_file_id = file_id
                break

        if not target_file_id:
            return "Error: Could not determine which file to read. Please specify the File ID."

        file_info = file_map.get(target_file_id)
        if not file_info:
            return f"Error: File ID {target_file_id} not found."

        file_path = file_info["path"]
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return f"Content of file '{file_info['name']}':\n\n{content}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def get_available_tools(self) -> list:
        return list(self.tools.keys())

    def register_tool(self, tool_name: str, tool_callable, permission_name: str = None):
        self.tools[tool_name] = {
            "callable": tool_callable,
            "permission": permission_name or f"mcp.{tool_name}",
        }
        self.logger.info(
            f"Registered MCP tool: {tool_name} with permission {permission_name or f'mcp.{tool_name}'}"
        )

    async def execute(
        self,
        input_data: Any,
        context: Dict[str, Any],
        private_context: Dict[str, Any],
        stream_callback=None,
    ) -> AgentResult:
        task = input_data if isinstance(input_data, dict) else {}
        tool_name = task.get("tool_name", "")
        task_description = task.get("description", "")
        original_query = context.get("original_query", "")

        if not tool_name:
            return AgentResult(
                status=AgentStatus.FAILURE,
                data=None,
                message="MCP任务失败：未指定工具名称",
            )

        if tool_name not in self.tools:
            available_mcp_tools = self.get_available_tools()
            if "file_map" in context and "read_file" not in available_mcp_tools:
                available_mcp_tools.append("read_file")

            mcp_tools_info = (
                f"可用的MCP工具: {', '.join(available_mcp_tools)}"
                if available_mcp_tools
                else "当前没有可用的MCP工具，请只使用LLM和RAG能力"
            )
            available_files = context.get("available_files", [])
            files_info = ""
            if available_files:
                files_info = (
                    "\n可用的文件 (使用 'read_file' 工具读取内容):\n"
                    + "\n".join(available_files)
                    + "\n"
                )

            return AgentResult(
                status=AgentStatus.NEEDS_RETRY,
                data={
                    "error": "tool_not_found",
                    "requested_tool": tool_name,
                    "available_mcp_tools": available_mcp_tools,
                    "original_task": task_description,
                    "suggestion": f"MCP工具 '{tool_name}' 未注册。{mcp_tools_info}\n{files_info}\n\n请重新规划任务，只使用可用的工具（LLM、RAG或已注册的MCP工具）。",
                },
                message=f"MCP工具 '{tool_name}' 不可用，需要重新规划（{mcp_tools_info}）",
                next_agent="planning",
            )

        tool_info = self.tools[tool_name]
        tool_callable = tool_info["callable"]
        permission_name = tool_info["permission"]
        if self.permission_manager:
            perm_result = self.permission_manager.check_permission(
                permission_name, context={"tool": tool_name, "task": task_description}
            )

            if not perm_result["allowed"]:
                return AgentResult(
                    status=AgentStatus.FAILURE,
                    data=None,
                    message=f"MCP任务失败：权限被拒绝 - {perm_result['reason']}",
                )

            if perm_result["needs_human"]:
                return AgentResult(
                    status=AgentStatus.NEEDS_HUMAN,
                    data={
                        "tool": tool_name,
                        "task": task_description,
                        "permission": permission_name,
                        "safety_level": perm_result["safety_level"],
                        "prompt": f"工具 '{tool_name}' 需要人工审批（安全级别: {perm_result['safety_level']}）",
                    },
                    message=f"需要人工审批：{perm_result['reason']}",
                    next_agent="planning",
                )

        try:
            # Execute tool - wrap in thread if it's sync
            import asyncio

            if asyncio.iscoroutinefunction(tool_callable):
                result = await tool_callable(
                    query=original_query, task=task_description, context=context
                )
            else:
                result = await asyncio.to_thread(
                    tool_callable,
                    query=original_query,
                    task=task_description,
                    context=context,
                )

            context["last_agent_type"] = "mcp"
            context["last_tool_name"] = tool_name
            context.setdefault("skills_used", []).append(f"mcp:{tool_name}")

            return AgentResult(
                status=AgentStatus.SUCCESS,
                data=result,
                message=f"MCP工具 '{tool_name}' 执行完成",
                next_agent="verification",
            )

        except Exception as e:
            self.logger.error(f"MCP task failed: {e}")
            return AgentResult(
                status=AgentStatus.FAILURE, data=None, message=f"MCP任务失败: {str(e)}"
            )
