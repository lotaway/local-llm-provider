"""MCP Task Agent - MCP tool invocation"""

from typing import Any, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.agent_base import BaseAgent, AgentResult, AgentStatus


class MCPTaskAgent(BaseAgent):
    """Agent for MCP tool invocation"""
    
    def __init__(self, llm_model, name: str = None):
        """
        Initialize MCP task agent
        
        Args:
            llm_model: LocalLLModel instance
            name: Agent name
        """
        super().__init__(llm_model, name)
        self.tools = {}
        self.permission_manager = None  # Will be set by runtime
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default MCP tools"""
        # Placeholder for tool registration
        # Tools will be implemented in mcp_tools package
        pass
    
    def register_tool(self, tool_name: str, tool_callable, permission_name: str = None):
        """
        Register an MCP tool
        
        Args:
            tool_name: Name of the tool
            tool_callable: Callable function for the tool
            permission_name: Permission name for this tool (e.g., "mcp.web_search")
        """
        self.tools[tool_name] = {
            "callable": tool_callable,
            "permission": permission_name or f"mcp.{tool_name}"
        }
        self.logger.info(f"Registered MCP tool: {tool_name} with permission {permission_name or f'mcp.{tool_name}'}")
    
    def execute(self, input_data: Any, context: Dict[str, Any], stream_callback=None) -> AgentResult:
        """
        Execute MCP task with permission checking
        
        Args:
            input_data: Task definition with tool_name
            context: Runtime context
            stream_callback: Optional callback for streaming LLM outputs (not used by MCP)
            
        Returns:
            AgentResult with tool execution result
        """
        task = input_data if isinstance(input_data, dict) else {}
        tool_name = task.get("tool_name", "")
        task_description = task.get("description", "")
        original_query = context.get("original_query", "")
        
        if not tool_name:
            return AgentResult(
                status=AgentStatus.FAILURE,
                data=None,
                message="MCP任务失败：未指定工具名称"
            )
        
        if tool_name not in self.tools:
            return AgentResult(
                status=AgentStatus.FAILURE,
                data=None,
                message=f"MCP任务失败：工具 '{tool_name}' 未注册"
            )
        
        tool_info = self.tools[tool_name]
        tool_callable = tool_info["callable"]
        permission_name = tool_info["permission"]
        
        # Check permission if permission manager is available
        if self.permission_manager:
            perm_result = self.permission_manager.check_permission(
                permission_name,
                context={"tool": tool_name, "task": task_description}
            )
            
            if not perm_result["allowed"]:
                return AgentResult(
                    status=AgentStatus.FAILURE,
                    data=None,
                    message=f"MCP任务失败：权限被拒绝 - {perm_result['reason']}"
                )
            
            if perm_result["needs_human"]:
                return AgentResult(
                    status=AgentStatus.NEEDS_HUMAN,
                    data={
                        "tool": tool_name,
                        "task": task_description,
                        "permission": permission_name,
                        "safety_level": perm_result["safety_level"],
                        "prompt": f"工具 '{tool_name}' 需要人工审批（安全级别: {perm_result['safety_level']}）"
                    },
                    message=f"需要人工审批：{perm_result['reason']}",
                    next_agent="planning"
                )
        
        try:
            # Execute tool
            result = tool_callable(query=original_query, task=task_description, context=context)
            
            return AgentResult(
                status=AgentStatus.SUCCESS,
                data=result,
                message=f"MCP工具 '{tool_name}' 执行完成",
                next_agent="verification"
            )
            
        except Exception as e:
            self.logger.error(f"MCP task failed: {e}")
            return AgentResult(
                status=AgentStatus.FAILURE,
                data=None,
                message=f"MCP任务失败: {str(e)}"
            )
