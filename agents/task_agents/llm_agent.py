"""LLM Task Agent - Direct LLM query execution"""

from typing import Any, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agents.agent_base import BaseAgent, AgentResult, AgentStatus


class LLMTaskAgent(BaseAgent):
    """Agent for direct LLM queries without RAG or tools"""

    async def execute(
        self,
        input_data: Any,
        context: Dict[str, Any],
        private_context: Dict[str, Any],
        stream_callback=None,
    ) -> AgentResult:
        """
        Execute LLM task

        Args:
            input_data: Task definition
            context: Runtime context
            stream_callback: Optional callback for streaming LLM outputs

        Returns:
            AgentResult with LLM response
        """
        task = input_data if isinstance(input_data, dict) else {}
        task_description = task.get("description", "")
        original_query = context.get("original_query", "")

        # Build prompt from task and context
        messages = [
            {
                "role": "system",
                "content": "你是一个有帮助的AI助手，请根据用户的问题提供准确、清晰的回答。",
            },
            {
                "role": "user",
                "content": f"""
原始问题：{original_query}
具体任务：{task_description}

请回答这个问题。
""",
            },
        ]

        try:
            response = await self._call_llm(
                messages,
                stream_callback=stream_callback,
                temperature=0.7,
                max_new_tokens=2000,
            )

            return AgentResult(
                status=AgentStatus.SUCCESS,
                data=response,
                message="LLM任务完成",
                next_agent="verification",
            )

        except Exception as e:
            self.logger.error(f"LLM task failed: {e}")
            return AgentResult(
                status=AgentStatus.FAILURE, data=None, message=f"LLM任务失败: {str(e)}"
            )
