"""Router Agent - Routes tasks to appropriate execution agents"""

from typing import Any, Dict
from .agent_base import BaseAgent, AgentResult, AgentStatus


class RouterAgent(BaseAgent):
    """Agent for routing tasks to appropriate execution agents"""
    
    def execute(self, input_data: Any, context: Dict[str, Any], stream_callback=None) -> AgentResult:
        """
        Route task to appropriate agent
        
        Args:
            input_data: Task definition from planning agent
            context: Runtime context
            stream_callback: Optional callback for streaming LLM outputs (not used by router)
            
        Returns:
            AgentResult with routing decision
        """
        if not isinstance(input_data, dict):
            return AgentResult(
                status=AgentStatus.FAILURE,
                data=None,
                message="路由失败：输入数据格式错误"
            )
        
        task = input_data
        agent_type = task.get("agent_type", "").lower()
        
        # Store current task in context
        context["current_task"] = task
        
        # Route based on agent type
        if agent_type == "llm":
            next_agent = "task_llm"
        elif agent_type == "rag":
            next_agent = "task_rag"
        elif agent_type == "mcp":
            next_agent = "task_mcp"
        else:
            return AgentResult(
                status=AgentStatus.FAILURE,
                data=None,
                message=f"路由失败：未知的agent类型 '{agent_type}'"
            )
        
        self.logger.info(f"Routing to {next_agent} for task: {task.get('description', 'unknown')}")
        
        return AgentResult(
            status=AgentStatus.SUCCESS,
            data=task,
            message=f"路由到 {next_agent}",
            next_agent=next_agent
        )
