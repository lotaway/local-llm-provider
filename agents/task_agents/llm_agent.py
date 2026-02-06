from typing import Any, Dict, List
from .agent_base import BaseAgent, AgentResult, AgentStatus

class LLMTaskAgent(BaseAgent):
    SYSTEM_PROMPT = "你是一个有帮助的AI助手，请提供准确、清晰的回答。"

    async def execute(
        self,
        input_data: Any,
        context: Dict[str, Any],
        private_context: Dict[str, Any],
        stream_callback=None,
    ) -> AgentResult:
        messages = self._build_task_messages(input_data, context)
        
        try:
            llm_res = await self._call_llm(messages, stream_callback)
            return self._create_success_result(llm_res, context)
        except Exception as e:
            return AgentResult(AgentStatus.FAILURE, None, f"LLMError: {str(e)}")

    def _build_task_messages(self, input_data: Any, context: Dict[str, Any]) -> List[Dict]:
        task = input_data if isinstance(input_data, dict) else {}
        desc = task.get("description", "")
        query = context.get("original_query", "")
        
        content = f"原始问题：{query}\n具体任务：{desc}"
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]

    def _create_success_result(self, llm_res: Dict[str, str], context: Dict[str, Any]) -> AgentResult:
        context["last_agent_type"] = "llm"
        context.setdefault("skills_used", []).append("llm")
        
        return AgentResult(
            AgentStatus.SUCCESS, 
            llm_res["response"], 
            "LLMTaskComplete", 
            next_agent="verification", 
            thought_process=llm_res["thought"]
        )
