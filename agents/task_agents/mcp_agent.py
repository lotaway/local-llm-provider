from typing import Any, Dict, List, Optional
import asyncio
import time
from agents.agent_base import BaseAgent, AgentResult, AgentStatus
from schemas.evolution_trace import ToolCallTrace

class MCPTaskAgent(BaseAgent):
    def __init__(self, llm_model, name: str = None):
        super().__init__(llm_model, name)
        self.tools = {}
        self.permission_manager = None
        self._register_default_tools()

    def _register_default_tools(self):
        self.register_tool(
            "read_file", self._read_file_tool, permission_name="mcp.read_file"
        )

    def register_tool(self, tool_name: str, tool_callable, permission_name: str = None):
        self.tools[tool_name] = {
            "callable": tool_callable,
            "permission": permission_name or f"mcp.{tool_name}",
        }

    def get_available_tools(self) -> list:
        return list(self.tools.keys())

    async def execute(
        self,
        input_data: Any,
        context: Dict[str, Any],
        private_context: Dict[str, Any],
        stream_callback=None,
    ) -> AgentResult:
        task = input_data if isinstance(input_data, dict) else {}
        name = task.get("tool_name", "")
        desc = task.get("description", "")
        
        if not name:
            return AgentResult(AgentStatus.FAILURE, None, "No tool name specified")

        if name not in self.tools:
            return self._handle_missing_tool(name, desc, context)

        perm_error = self._check_permissions(name, desc, context)
        if perm_error:
            return perm_error

        return await self._invoke_tool(name, desc, context)

    def _handle_missing_tool(self, name, desc, context) -> AgentResult:
        available = self.get_available_tools()
        return AgentResult(
            status=AgentStatus.NEEDS_RETRY,
            data={"error": "tool_not_found", "requested_tool": name, "available": available},
            message=f"Tool {name} not found",
            next_agent="planning"
        )

    def _check_permissions(self, name, desc, context) -> Optional[AgentResult]:
        if not self.permission_manager:
            return None
        
        perm_name = self.tools[name]["permission"]
        res = self.permission_manager.check_permission(perm_name, {"tool": name, "task": desc})
        
        if not res["allowed"]:
            return AgentResult(AgentStatus.FAILURE, None, f"Denied: {res['reason']}")
            
        if res["needs_human"]:
            return AgentResult(
                AgentStatus.NEEDS_HUMAN, 
                {"tool": name, "task": desc, "safety": res["safety_level"]},
                message=f"Approval required",
                next_agent="planning"
            )
        return None

    async def _invoke_tool(self, name, desc, context) -> AgentResult:
        tool_info = self.tools[name]
        start_time = time.time()
        
        try:
            obs = await self._call_tool_callable(tool_info["callable"], name, desc, context)
            latency = (time.time() - start_time) * 1000
            
            trace = ToolCallTrace(tool_name=name, arguments={"description": desc}, raw_observation=obs, latency_ms=latency)
            
            context["last_agent_type"] = "mcp"
            context.setdefault("skills_used", []).append(f"mcp:{name}")

            return AgentResult(AgentStatus.SUCCESS, obs, f"Tool {name} executed", next_agent="verification", tool_calls=[trace])
        except Exception as e:
            return AgentResult(AgentStatus.FAILURE, None, f"Execution error: {str(e)}")

    async def _call_tool_callable(self, callable_obj, name, desc, context) -> Any:
        query = context.get("original_query", "")
        if asyncio.iscoroutinefunction(callable_obj):
            return await callable_obj(query=query, task=desc, context=context)
        return await asyncio.to_thread(callable_obj, query=query, task=desc, context=context)

    def _read_file_tool(self, query: str, task: str, context: Dict[str, Any]) -> str:
        file_map = context.get("file_map", {})
        target = next((fid for fid in file_map if fid in task), None)
        
        if not target:
            return "Error: Document ID not found in task description"
            
        path = file_map[target]["path"]
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
