from typing import Any, Dict
from .agent_base import BaseAgent, AgentResult, AgentStatus


class RouterAgent(BaseAgent):
    async def execute(
        self,
        input_data: Any,
        context: Dict[str, Any],
        private_context: Dict[str, Any],
        stream_callback=None,
    ) -> AgentResult:
        if not isinstance(input_data, dict):
            return AgentResult(
                status=AgentStatus.FAILURE,
                data=None,
                message="路由失败：输入数据格式错误",
            )

        task = input_data
        agent_type = task.get("agent_type", "").lower()
        context["current_task"] = task
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
                message=f"路由失败：未知的agent类型 '{agent_type}'",
            )

        self.logger.info(
            f"Routing to {next_agent} for task: {task.get('description', 'unknown')}"
        )

        return AgentResult(
            status=AgentStatus.SUCCESS,
            data=task,
            message=f"路由到 {next_agent}",
            next_agent=next_agent,
        )
